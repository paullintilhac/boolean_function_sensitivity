# Using ../sensitivity/36test.py
# Functions whose Fourier degree is concentrated on higher weights are harder to learn for LSTMs with SGD

from pyhessian.hessian import hessian
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import random
import argparse
# from new_transformer import Transformer
from transformer import Transformer as Transformer2
from hardcoded_transformer import HardCodedTransformer
# from transformer_old import Transformer as Transformer3
from updated_transformer import Transformer as Transformer
import math



# ===== hessian_probe.py (refactored, single HVP) =====
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import contextlib

# ---------- helpers: flatten / unflatten ----------
@torch.no_grad()
def _flatten_params(params):
    return torch.cat([p.reshape(-1) for p in params])

def _unflatten_like(vec_flat, params):
    """Split a flat vector into a list shaped like params."""
    outs, i = [], 0
    for p in params:
        n = p.numel()
        outs.append(vec_flat[i:i+n].view_as(p))
        i += n
    return outs

@torch.no_grad()
def _param_flat_norm2(tensors):
    s = 0.0
    for t in tensors:
        if t is not None:
            s += float((t.detach()**2).sum())
    return s

# ---------- single, global HVP (list-in / list-out) ----------
def hvp_list(loss, params, vec_list):
    """
    Hessian–vector product for mean-MSE-like losses.
    Args:
      loss: scalar Tensor
      params: list[nn.Parameter]
      vec_list: list[Tensor] with same shapes as params
    Returns:
      hv_list: list[Tensor] (same shapes as params, zeros where unused)
      unused_idx: list[int] indices for which grad/hvp was None
    """
    # First gradient wrt params
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    safe = []
    for g, p in zip(grads, params):
        safe.append(torch.zeros_like(p) if g is None else g)

    # g·v
    gdotv = None
    for g, v in zip(safe, vec_list):
        term = (g * v).sum()
        gdotv = term if gdotv is None else (gdotv + term)

    # If graph is flat (perfect fit + masked path), safely return zeros
    if (gdotv is None) or (not gdotv.requires_grad):
        return [torch.zeros_like(p) for p in params], list(range(len(params)))

    hv_list = torch.autograd.grad(gdotv, params, retain_graph=True, allow_unused=True)
    out = []
    unused_idx = []
    for i, (h, p) in enumerate(zip(hv_list, params)):
        if h is None:
            out.append(torch.zeros_like(p))
            unused_idx.append(i)
        else:
            out.append(h)
    return out, unused_idx

# ---------- parameter collection & blocks ----------
def collect_hardcoded_params(model, include_bias=True):
    wanted = []
    def add(p):
        if p is not None and getattr(p, "requires_grad", False) and p.numel() > 0:
            wanted.append(p)

    add(model.attn1.in_proj_weight); add(model.attn2.in_proj_weight)
    if include_bias:
        add(model.attn1.in_proj_bias); add(model.attn2.in_proj_bias)

    add(model.mlp.fc1.weight); add(model.mlp.fc2.weight); add(model.mlp.fc_out.weight)
    if include_bias:
        add(model.mlp.fc1.bias); add(model.mlp.fc2.bias)
    return wanted

def _split_inproj(mha):
    W = mha.in_proj_weight
    B = mha.in_proj_bias if mha.in_proj_bias is not None else None
    E = mha.embed_dim
    Wq, Wk, Wv = W[:E], W[E:2*E], W[2*E:]
    bq = B[:E] if B is not None else None
    bk = B[E:2*E] if B is not None else None
    bv = B[2*E:] if B is not None else None
    return (Wq, bq), (Wk, bk), (Wv, bv)

def get_param_blocks(model):
    blocks = OrderedDict()
    (Wq1,bq1),(Wk1,bk1),(Wv1,bv1) = _split_inproj(model.attn1)
    blocks["attn1.Q"] = [Wq1] + ([bq1] if bq1 is not None else [])
    blocks["attn1.K"] = [Wk1] + ([bk1] if bk1 is not None else [])
    blocks["attn1.V"] = [Wv1] + ([bv1] if bv1 is not None else [])

    (Wq2,bq2),(Wk2,bk2),(Wv2,bv2) = _split_inproj(model.attn2)
    blocks["attn2.Q"] = [Wq2] + ([bq2] if bq2 is not None else [])
    blocks["attn2.K"] = [Wk2] + ([bk2] if bk2 is not None else [])
    blocks["attn2.V"] = [Wv2] + ([bv2] if bv2 is not None else [])

    blocks["mlp.fc1"]    = [model.mlp.fc1.weight, model.mlp.fc1.bias]
    blocks["mlp.fc2"]    = [model.mlp.fc2.weight, model.mlp.fc2.bias]
    blocks["mlp.fc_out"] = [model.mlp.fc_out.weight]
    return blocks

# ---------- probes ----------
def _probe_mse(model, xs, ys):
    with torch.no_grad():
        out = model(xs).squeeze(-1)
        return float(((out - ys) ** 2).mean().item())

def gauss_newton_trace_mse(model, inputs, params, n_samples=64, seed=0, scale=2.0):
    """
    Estimate trace(scale * J^T J) for mean-MSE outputs (perfect-fit friendly).
    """
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    traces = []

    for _ in range(n_samples):
        out = model(inputs).squeeze(-1)  # (B,)
        r = torch.empty_like(out, device=device).bernoulli_(0.5, generator=gen).mul_(2).add_(-1)
        grads = torch.autograd.grad(out, params, grad_outputs=r, retain_graph=False, allow_unused=True)
        norm2 = 0.0
        for g in grads:
            if g is not None:
                norm2 += (g**2).sum()
        traces.append(scale * norm2.detach())

    return torch.stack(traces).mean().item()

def hutchinson_trace(model, inputs, targets, params, n_samples=64, seed=0):
    """
    Hutchinson’s trace(H) with robust HVP (global hvp_list).
    """
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    def _loss_fn(out, tgt):
        return (out.squeeze(-1) - tgt).pow(2).mean()

    traces = []
    unused_counts = torch.zeros(len(params), dtype=torch.long, device=device)

    for _ in range(n_samples):
        model.zero_grad(set_to_none=True)
        out = model(inputs)
        loss = _loss_fn(out, targets)

        v = [torch.empty_like(p, device=device).bernoulli_(0.5, generator=gen).mul_(2).add_(-1)
             for p in params]

        hv, unused = hvp_list(loss, params, v)
        if len(unused) > 0:
            unused_idx = torch.tensor(unused, device=device, dtype=torch.long)
            unused_counts.index_add_(0, unused_idx, torch.ones_like(unused_idx, dtype=unused_counts.dtype))

        vtHv = sum((vi * hi).sum() for vi, hi in zip(v, hv))
        traces.append(vtHv.detach())

    tr_est = torch.stack(traces).mean().item()

    nz = (unused_counts > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(nz, int):
        nz = [nz]
    if len(nz) > 0:
        print(f"[Hutchinson] Unused param indices over {n_samples} samples:", nz)

    return tr_est

def power_iteration_top_eig(model, inputs, targets, params, iters=50, seed=0):
    """
    Approximate top eigenvalue of Hessian via power iteration using hvp_list.
    """
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    total_dim = sum(p.numel() for p in params)
    v_flat = torch.randn(total_dim, generator=rng, device=device)
    v_flat = v_flat / (v_flat.norm() + 1e-12)

    loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()
    model.zero_grad(set_to_none=True)
    out = model(inputs)
    base_loss = loss_fn(out, targets)

    lam = 0.0
    for _ in range(iters):
        v_list = _unflatten_like(v_flat, params)
        hv_list, _ = hvp_list(base_loss, params, v_list)
        hv_flat = _flatten_params([h.contiguous() for h in hv_list])

        lam = torch.dot(v_flat, hv_flat) / (v_flat.norm()**2 + 1e-12)
        v_flat = hv_flat / (hv_flat.norm() + 1e-12)

    return lam.item()

@torch.no_grad()
def attention_diag_stats(model, inputs):
    """Print pre-softmax logit ranges & entropies for attn1/attn2."""
    device = next(model.parameters()).device
    model.eval()

    x_ints = inputs.to(device)
    N = model.N; E = model.E
    shifts = torch.arange(N, device=device, dtype=torch.long)
    bits = ((x_ints.unsqueeze(-1) >> shifts) & 1).long()
    dat_bits = model.bit_embed(bits)
    pos_idx = model.pos_idx_base.unsqueeze(0).expand(x_ints.shape[0], -1)
    pos_vecs = model.pos_embed(pos_idx)
    zeros = torch.zeros(x_ints.shape[0], N, 3, device=device)
    X0 = torch.cat([pos_vecs, dat_bits, zeros], dim=-1)   # (B,N,E)

    def pre_softmax_logits(mha, Q_in, K_in):
        W = mha.in_proj_weight; B = mha.in_proj_bias
        E = mha.embed_dim
        if B is None:
            Q = F.linear(Q_in.transpose(0,1), W[:E], None).transpose(0,1)
            K = F.linear(K_in.transpose(0,1), W[E:2*E], None).transpose(0,1)
        else:
            Q = F.linear(Q_in.transpose(0,1), W[:E],      B[:E]).transpose(0,1)
            K = F.linear(K_in.transpose(0,1), W[E:2*E],  B[E:2*E]).transpose(0,1)
        logits = torch.einsum("lbe, mbe -> blm", Q, K)  # no 1/sqrt(d) in your kernel
        return logits

    logits1 = pre_softmax_logits(model.attn1, X0, X0)
    probs1  = logits1.softmax(dim=-1)
    H1 = -(probs1.clamp_min(1e-12)*probs1.clamp_min(1e-12).log()).sum(-1)
    print(f"[attn1] logits range: {logits1.min().item():.2f} .. {logits1.max().item():.2f} ; "
          f"entropy mean: {H1.mean().item():.3f}")

    Y1, _ = model.attn1(X0, X0, X0); X1 = X0 + Y1
    X2 = model.mlp(X1)

    logits2 = pre_softmax_logits(model.attn2, X2, X2)
    probs2  = logits2.softmax(dim=-1)
    H2 = -(probs2.clamp_min(1e-12)*probs2.clamp_min(1e-12).log()).sum(-1)
    print(f"[attn2] logits range: {logits2.min().item():.2f} .. {logits2.max().item():.2f} ; "
          f"entropy mean: {H2.mean().item():.3f}")

# ---------- scaling helpers (already safe with no_grad) ----------
def _q_slice(param, E):
    return param[:E, :]

@contextmanager
def scale_queries(model, which=("attn1","attn2"), factor=1.0):
    caches = {}
    try:
        with torch.no_grad():
            if "attn1" in which:
                E1 = model.attn1.embed_dim
                Wq1 = _q_slice(model.attn1.in_proj_weight, E1)
                caches["attn1"] = Wq1.detach().clone()
                Wq1.mul_(factor)
            if "attn2" in which:
                E2 = model.attn2.embed_dim
                Wq2 = _q_slice(model.attn2.in_proj_weight, E2)
                caches["attn2"] = Wq2.detach().clone()
                Wq2.mul_(factor)
        yield
    finally:
        with torch.no_grad():
            if "attn1" in caches:
                E1 = model.attn1.embed_dim
                _q_slice(model.attn1.in_proj_weight, E1).copy_(caches["attn1"])
            if "attn2" in caches:
                E2 = model.attn2.embed_dim
                _q_slice(model.attn2.in_proj_weight, E2).copy_(caches["attn2"])

@contextmanager
def scale_mlp_slopes(model, factor=1.0):
    cache = {
        "fc1_w": model.mlp.fc1.weight.detach().clone(),
        "fc2_w": model.mlp.fc2.weight.detach().clone(),
    }
    try:
        with torch.no_grad():
            model.mlp.fc1.weight.mul_(factor)
            model.mlp.fc2.weight.mul_(factor)
        yield
    finally:
        with torch.no_grad():
            model.mlp.fc1.weight.copy_(cache["fc1_w"])
            model.mlp.fc2.weight.copy_(cache["fc2_w"])

def analyze_hessian_blocks(model, inputs, targets, n_hutch=64, top_iters=60,
                           switch_tol=1e-8, seed=0, use_bias=True):
    """
    Uses GN when perfect (MSE < switch_tol) else Hutchinson with global hvp_list.
    Returns a list of (label, approx_trace) so the caller can sum or report.
    """
    device = next(model.parameters()).device
    model.train()
    params = collect_hardcoded_params(model, include_bias=use_bias)
    if len(params) == 0:
        print("[analyze] No trainable params collected — returning [].")
        return []

    loss_probe = _probe_mse(model, inputs, targets)
    print(f"[CHECK] Probe MSE(model vs driver targets): {loss_probe:.3e}")

    rows = []
    if loss_probe < switch_tol:
        tr_est = gauss_newton_trace_mse(model, inputs, params, n_samples=n_hutch, seed=seed, scale=2.0)
        print("Gauss–Newton trace estimate (perfect-fit):", tr_est)
        rows.append(("GN(total)", float(tr_est)))
    else:
        tr_est = hutchinson_trace(model, inputs, targets, params, n_samples=n_hutch, seed=seed)
        print("Hutchinson trace estimate:", tr_est)
        rows.append(("Hutch(total)", float(tr_est)))
    return rows
# ===== end PROBE =====


class SAM(torch.optim.Optimizer):
    """SAM wrapper around a base optimizer (e.g., AdamW)."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True):
        if rho <= 0.0:
            raise ValueError("rho must be > 0")
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
    @torch.no_grad()
    def _grad_norm(self):
        eps = 1e-12
        norms = []
        for group in self.param_groups:
            adaptive = group['adaptive']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                w = p.abs() if adaptive else 1.0
                norms.append((w * g).norm(p=2))
        if not norms:
            dev = self.param_groups[0]['params'][0].device
            return torch.tensor(0.0, device=dev) + eps
        return torch.norm(torch.stack(norms), p=2) + eps
    @torch.no_grad()
    def first_step(self, zero_grad=True):
        scale = self.param_groups[0]['rho'] / self._grad_norm()
        for group in self.param_groups:
            adaptive = group['adaptive']
            for p in group['params']:
                if p.grad is None: continue
                e = (p.abs() if adaptive else 1.0) * p.grad * scale
                p.add_(e)
                self.state[p]['e_w'] = e
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def zero_grad(self): self.base_optimizer.zero_grad()
    def step(self): raise NotImplementedError("Use first_step() and second_step()")


import os
import itertools
import time
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp, barrier
mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()
#from functools import partial

if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
    
def get_weight_norm(model):
       total_norm = 0.0
       for p in model.parameters():
           if p.requires_grad:
               param_norm = p.data.norm(2)
               total_norm += param_norm.item() ** 2
       return total_norm ** 0.5
    
def rboolf(N, width, deg,seed=None):
    if seed:
        torch.manual_seed(seed)
    coefficients = torch.randn(width).abs().to(device)
    #print("coefficients initial shape: " + str(coefficients.shape) + ", width: " + str(width))
    coefficients = (coefficients)/coefficients.pow(2).sum().sqrt()
    
    combs = torch.tensor(list(itertools.combinations(torch.arange(N), deg))).to(device)
    combs = combs[torch.randperm(len(combs))][:width] # Shuffled
    print("coefficients: "  + str(coefficients))
    print("combs: "  + str(combs))
    return (coefficients, combs)

def ddp_setup(rank, world_size,backend):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]= "23456"
    if backend == "gloo":
        init_process_group(backend="gloo",
                       init_method='tcp://127.0.0.1:23456',
                       rank=rank,
                       world_size=world_size,
                       timeout=datetime.timedelta(seconds=5400)
                      )
    else:
        init_process_group(backend="nccl",
                       rank=rank,
                       world_size=world_size,
                       timeout=datetime.timedelta(seconds=5400)
                      )        

class Trainer:
    def __init__(
            self,
            coeffs: torch.FloatTensor,
            combs: torch.FloatTensor,
            model:torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
            dir_name: str,
            width: int,
            deg: int,
            func: int,
            N: int,
            n_samples: int,
            backend:str,
            stop_loss:float,
            ln_eps:float,
            ln:bool,
            save_checkpoints: bool,
            f: float,
            d: int,
            h: int,
            dropout: float,
            wd: float,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = DDP(model,device_ids=[self.gpu_id])
        self.model.to(self.gpu_id)
        self.train_data=train_data
        self.optimizer = optimizer
        self.save_every=save_every
        self.ln_eps=ln_eps
        self.ln = ln
        self.wd = wd
        self.dir_name = dir_name  
        self.save_checkpoints = save_checkpoints
        self.dropout = dropout
        self.summary = pd.DataFrame(columns=
                                ["deg",
                                 "width",
                                 "func",
                                 "epoch",
                                 "train_loss",
                                 "val_loss",
                                 "batch_size",
                                 "lr",
                                 "n_samples",
                                 "func_val_test",
                                 "time_elapsed",
                                 "backend",
                                 "top_eig",
                                 "trace",
                                 "top_eig_train",
                                 "trace_train",
                                 "stop_loss",
                                 "ln_eps",
                                 "ln",
                                 "weight_norm",
                                  "l",
                                 "d",
                                 "f",
                                 "h",
                                 "dropout",
                                 "wd"])
        self.stop_loss = stop_loss
        self.epoch_loss = 0
        self.N = N
        self.func = func
        self.coeffs = coeffs.to(gpu_id)
        self.combs = combs.to(gpu_id)
        self.width=width
        self.deg = deg
        self.n_samples = n_samples
        self.d = d
        self.f = f
        self.h = h
        for batch in train_data:
            self.batch_size = len(batch)
            break
        self.lr = (
            optimizer.base_optimizer.param_groups[-1]['lr']
            if hasattr(optimizer, 'base_optimizer')
            else optimizer.param_groups[-1]['lr']
        )
        self.backend = backend
        #self.func.to(gpu_id)
        
    def func_batch(self, x):
        # x: 1D tensor of integers (can be on any device)
        x = torch.as_tensor(x, dtype=torch.long, device=self.gpu_id)
        shifts = torch.arange(self.N, device=self.gpu_id)          # 0..N-1, LSB-first
        bits01 = ((x.unsqueeze(-1) >> shifts) & 1).float()         # (B, N) in {0,1}
        bin_pm = (bits01 - 0.5) * 2.0                              # {-1,+1}
    
        # self.combs is shape (width, deg) on gpu_id already
        idx = self.combs.long()                                     # (W, D)
        # Gather (B, W, D) and product over D -> (B, W)
        comps = bin_pm[:, idx]                                      # advanced indexing
        comps = comps.prod(dim=2)                                   # (B, width)
    
        return comps @ self.coeffs                                  # (B,)

        
    def _run_batch(self, inputs, targets):
        # Same loss as elsewhere
        # loss_fn = lambda out, tgt: (out - tgt).pow(2).mean()
        loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()

        # ---- SAM path ----
        if hasattr(self.optimizer, "base_optimizer"):
            # 1) Compute grads at w and take the SAM ascent step to w~
            #    Use no_sync() to avoid an extra all-reduce in DDP on the first backward.
            with self.model.no_sync():
                out = self.model(inputs)
                loss = loss_fn(out, targets)
                loss.backward()
            self.optimizer.first_step(zero_grad=True)
    
            # 2) Compute grads at w~ and take the descent step (restoring weights)
            out = self.model(inputs)
            loss_perturbed = loss_fn(out, targets)
            loss_perturbed.backward()
            self.optimizer.second_step(zero_grad=True)
    
            # Return the original (unperturbed) loss for logging/averaging
            return loss.detach()
    
        # ---- Standard optimizer path ----
        out = self.model(inputs)
        loss = loss_fn(out, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()
    
    def _run_epoch(self,epoch):
        
        b_sz = len(next(iter(self.train_data)))
        epoch_loss = 0
        total_records = 0
        start_time = time.time()
        
        for idx, inputs in enumerate(self.train_data):
          #inputs.to(self.gpu_id)    
          targets =self.func_batch(inputs).to(self.gpu_id)
          batch_loss = self._run_batch(inputs, targets)
          epoch_loss+=batch_loss*float(len(inputs))
          total_records+=len(inputs)
          iteration = epoch*len(self.train_data)+idx+1
            
        epoch_loss/=float(total_records)
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        #print(f"Epoch time: {elapsed_time:.3f} seconds.")
        time_per_record_ms = float(elapsed_time*100)/float(total_records)
        #print(f"Epoch time: {elapsed_time:.3f} seconds. time per record (ms): {time_per_record_ms: .3f}")
        return epoch_loss

    def save_checkpoint(self,epoch,model_name):
        os.makedirs(os.path.join(self.dir_name, model_name), exist_ok=True)
        full_model_name = model_name+"/epoch-"+str(epoch)+".pt"
        ckp = self.model.module.state_dict()
        torch.save(ckp,os.path.join(self.dir_name, full_model_name))
        # loss_fn = lambda result, targets: (result-targets).pow(2).mean()
        loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()

        print(f"Epoch {epoch} | Training checkpoint saved at model_{epoch}.pt")

    def train(self,epochs: int):
        self.model.train()
        
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = self._run_epoch(epoch)
            
            if ((epoch % self.save_every)==0 and self.gpu_id==0) or (epoch_loss < self.stop_loss):
            # if ((((epoch+1) % self.save_every)==0 or epoch==0) and self.gpu_id==0):

                #print("inside conditional")
                if self.save_checkpoints:
                    self.save_checkpoint(epoch,"degree-"+str(self.deg)+"/width-"+str(self.width)+"/func-"+str(self.func))
                end_time = time.time()
                elapsed_time = round((end_time - start_time)/60,3) 

                #print("self.func: " + str(self.func))
                val_loss = self.validate(1000,model) 
                loss_fn = lambda result, targets: (result-targets).pow(2).mean()
                start_time_hessian = time.time()
                top_eig, trace = self.calc_hessian(copy.deepcopy(self.model.module), loss_fn=loss_fn, num_samples= 1000,device_id = self.gpu_id)
                top_eig_train, trace_train = self.calc_hessian(copy.deepcopy(self.model.module), loss_fn=loss_fn, num_samples= 1000,device_id = self.gpu_id, use_train=True)
                #weight_norm = 0
                weight_norm = get_weight_norm(self.model.module)
                #weight_norm = torch.linalg.norm(self.model.weight)
                #top_eig=0
                #trace = 0
                end_time_hessian = time.time()
                elapsed_time_hessian = round((end_time_hessian - start_time_hessian)/60,3) 
                print("elapsed time norm: " + str(elapsed_time_hessian))
                self.summary.loc[0] = {"deg":self.deg,
                                       "width":self.width,
                                       "func":self.func,
                                       "epoch":epoch,
                                       "train_loss":epoch_loss.cpu(),
                                       "val_loss":val_loss.cpu(),
                                      "batch_size": self.batch_size,
                                      "lr":self.lr,
                                      "n_samples":self.n_samples,
                                      "func_val_test":self.func_batch([2]).cpu(),
                                      "time_elapsed":elapsed_time,
                                      "backend":self.backend,
                                      "top_eig":top_eig,
                                      "trace":trace,
                                       "top_eig_train": top_eig_train,
                                       "trace_train": trace_train,
                                      "stop_loss": self.stop_loss,
                                      "ln_eps": self.ln_eps,
                                      "ln": self.ln,
                                      "weight_norm": weight_norm,
                                       "d":self.d,
                                       "f":self.f,
                                       "h":self.h,
                                       "dropout":self.dropout,
                                       "wd":self.wd
                                      }
               

                self.summary.to_csv(f"{self.dir_name}/summary.csv",mode='a', header=not os.path.exists(f"{self.dir_name}/summary.csv"), index=False)
                print(f" Epoch: {epoch}, TimeElapsed: {elapsed_time}, EpochLoss: {epoch_loss:.3f}, ValidationLoss: {val_loss:.3f}")
            flag = torch.zeros(1).to(self.gpu_id)
            if epoch_loss<self.stop_loss:
                 flag += 1
            all_reduce(flag, op=ReduceOp.SUM)
            if flag > 0:
                break
            barrier()
        # loss_fn = lambda result, targets: (result-targets).pow(2).mean()
        # top_eig = self.calc_hessian(copy.deepcopy(self.model.module), loss_fn=loss_fn, num_samples= 1000) 
        return

    # def validate(self, num_samples,test_model):
    #   test_model.eval()
    #   inputs = torch.tensor([random.randint(0, 2**self.N-1) for _ in range(num_samples)]).to(self.gpu_id)
    #   targets = self.func_batch(inputs).to(self.gpu_id)
    #   result = test_model(inputs).to(self.gpu_id)
    #   loss = (result - targets).pow(2).mean()
    #   return loss.detach().cpu()

    def validate(self, num_samples, test_model):
        test_model.eval()
        inputs  = torch.randint(0, 2**self.N, (num_samples,), device=self.gpu_id)
        targets = self.func_batch(inputs)                 # (B,)
        result  = test_model(inputs).squeeze(-1)          # (B,)
        return (result - targets).pow(2).mean().detach().cpu()
        
    def calc_hessian(self, model, loss_fn, num_samples,device_id, use_train=False):
        model.eval().to(self.gpu_id)
        if use_train:
            ds = getattr(self.train_data, "dataset", None)
            if isinstance(ds, torch.Tensor):
                inputs = ds[:min(num_samples, ds.shape[0])].to(self.gpu_id)
            else:
                collected, total = [], 0
                for batch in self.train_data:
                    batch_inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    take = min(batch_inputs.shape[0], num_samples - total)
                    collected.append(batch_inputs[:take])
                    total += take
                    if total >= num_samples: break
                inputs = torch.cat(collected, dim=0).to(self.gpu_id)
        else:
            inputs = torch.tensor([random.randint(0, 2**self.N-1) for _ in range(num_samples)]).to(self.gpu_id)
        targets = self.func_batch(inputs).to(self.gpu_id)
        data = (inputs, targets)
        hess_mod = hessian(model, loss_fn, data)
        for param in model.parameters(): param.grad = None
        top_eigs, top_eigVs = hess_mod.eigenvalues(maxIter = 200)
        top_eig = top_eigs[0]
        trace = hess_mod.trace()
        return top_eig, np.mean(trace)


    
def load_train_objs(wd,dropout,lr,num_samples, N, dim, h, f, rank, ln_eps, ln,coefs, combs, sam=False, sam_rho=0.05, asam=False):
        train_set = torch.tensor([random.randint(0, 2**N-1) for _ in range(int(num_samples))]).to(rank)
        hardcoded_model = HardCodedTransformer(N, combs, coefs,nonrep_mask = -.1)
        model = Transformer(dropout,N, dim, h, f, ln_eps, rank, ln)
        total_params = sum(p.numel() for p in model.parameters())
        #print(model)
        print("Trainable Model Parameter Count: " + str(total_params))
        hardcoded_total_params = sum(p.numel() for p in hardcoded_model.parameters())
        #print(model)
        print("Hardcoded Model Parameter Count: " + str(hardcoded_total_params))
        base_opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=wd)
        optimizer = SAM(model.parameters(), base_optimizer=base_opt, rho=sam_rho, adaptive=asam) if sam else base_opt
        return train_set, model, optimizer, hardcoded_model                

def addGaussianNoise(model, sigma, as_variance=True, skip_frozen=True, include_bias=True, seed=None):
    """
    Adds centered Gaussian noise to parameters in-place.

    Args:
      model: nn.Module (e.g., HardCodedTransformer)
      sigma: if as_variance=True, interpreted as variance; else as std dev
      as_variance: True -> use std = sqrt(sigma); False -> std = sigma
      skip_frozen: if True, only perturb params with requires_grad=True
      include_bias: if False, skip bias terms
      seed: optional int for reproducibility
    """
    std = math.sqrt(sigma) if as_variance else float(sigma)
    if seed is not None:
        # Use device-aware generator so CUDA noise is deterministic too
        device = next(model.parameters()).device
        g = torch.Generator(device=device).manual_seed(seed)
    else:
        g = None

    with torch.no_grad():
        for name, p in model.named_parameters():
            if skip_frozen and not p.requires_grad:
                continue
            if (not include_bias) and name.endswith(".bias"):
                continue
            # If you want to explicitly skip the fixed embeddings:
            if "pos_embed.weight" in name or "bit_embed.weight" in name:
                continue
            noise = torch.empty_like(p)
            noise = noise.normal_(mean=0.0, std=std, generator=g)
            p.add_(noise)
            
def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--dim2', type=int, default=22)
    parser.add_argument('--f', type=int, default=64)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--lr', type=str,default = "1e-5")
    parser.add_argument('--wd', type=float,default = .1)
    parser.add_argument('--dropout', type=float,default = .2)
    parser.add_argument('--backend',type=str, default = "gloo")
    parser.add_argument('--stop_loss', type=float,default = .02)
    parser.add_argument('--ln_eps', type=float,default = 1e-5)
    parser.add_argument('--ln', action='store_true')
    parser.add_argument('--save_checkpoints', action='store_true')
    parser.add_argument('--sam', action='store_true')
    parser.add_argument('--sam_rho', type=float, default=0.05)
    parser.add_argument('--asam', action='store_true')
    return parser.parse_args()

def main(rank, args,world_size,coefs,combs,main_dir,deg,width,i):
      #print("func in main: " + str(func))
      ddp_setup(rank,world_size,args.backend)
      # Create new directory to save results for the particular function
      #dir_name = os.path.join(main_dir, f"deg{deg}_width{width}_func{i}")
      #os.makedirs(dir_name, exist_ok=True)
        
      train_set,model,optimizer,hardcoded_model = load_train_objs(args.dropout,
                                                  args.wd,args.lr,
                                                  args.num_samples,
                                                  args.N,
                                                  args.dim,
                                                  args.h,
                                                  args.f,
                                                  rank,
                                                  args.ln_eps,
                                                  args.ln,
                                                  coefs,
                                                  combs,
                                                  sam=args.sam,
                                                  sam_rho=args.sam_rho,
                                                  asam=args.asam
                                                  )
      model.to(rank)
      hardcoded_model.to(rank)
   
      train_loader = DataLoader(
          train_set,
          shuffle=False,
          batch_size=args.bs,
          sampler = DistributedSampler(train_set)
      )
         
      trainer = Trainer(coefs,combs, model,
                        train_loader,
                        optimizer,
                        gpu_id=rank,
                        save_every=args.save_every,
                        dir_name= main_dir,
                        width=width,
                        deg=deg,
                        func=i,
                        N=args.N,
                        n_samples = args.num_samples,
                        backend = args.backend,
                        stop_loss = args.stop_loss,
                        ln_eps = args.ln_eps,
                        ln = args.ln,
                        save_checkpoints=args.save_checkpoints,
                        d=args.dim,
                        f=args.f,
                        h=args.h,
                        dropout=args.dropout,
                        wd=args.wd
                        )

    
      params = collect_hardcoded_params(hardcoded_model, include_bias=True)
    
      # (2) Build a small probe batch (same as you already do)
      xs = torch.randint(0, 2**args.N, (512,), device=device)
      with torch.no_grad():
        ys = trainer.func_batch(xs).to(device)
    
      # (3) If loss is near zero, use Gauss–Newton trace:
      hardcoded_model.eval()
      with torch.no_grad():
        out = hardcoded_model(xs)
        loss_probe = ((out.squeeze(-1) - ys)**2).mean().item()
      print(f"[CHECK] Probe MSE(model vs driver targets): {loss_probe}")
    
      if loss_probe < 1e-8:
        tr_est = gauss_newton_trace_mse(hardcoded_model, xs, params, n_samples=64, seed=0, scale=2.0)
        print("Gauss–Newton trace estimate (perfect-fit):", tr_est)
      else:
        tr_est = hutchinson_trace(hardcoded_model, xs, ys, params, n_samples=64, seed=0)
        print("Hutchinson trace estimate:", tr_est)
       
      # loss_fn = lambda result, targets: (result-targets).pow(2).mean()
      loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()
      hardcoded_model.eval()
      print("hardcoded model: " + str(hardcoded_model))
      #addGaussianNoise(hardcoded_model, .1)
      hardcoded_hessian = trainer.calc_hessian(hardcoded_model, loss_fn, num_samples=1000,device_id=rank)
      hardcoded_hessian_train = trainer.calc_hessian(hardcoded_model, loss_fn, num_samples=1000,device_id=rank, use_train=True)
      weight_norm = get_weight_norm(hardcoded_model)
      hardcoded_loss = trainer.validate(1000,hardcoded_model)
      print("hardcoded loss: " + str(hardcoded_loss))
      print("frobenius weight norm: " + str(weight_norm)) 
      print("hardcoded hessian stats: " + str(hardcoded_hessian))
      
      # Build a small probe batch
      Bprobe = 256
      xs = torch.randint(0, 2**args.N, (Bprobe,), device=device)
      # IMPORTANT: build targets with **same LSB-first convention** as the model:
      def targets_lsb(x_long):
        shifts = torch.arange(args.N, device=device)
        bits01 = ((x_long.unsqueeze(-1) >> shifts) & 1).float()
        bin_pm = (bits01 - 0.5) * 2.0
        idx = combs.long().to(device)         # (width, D)
        comps = bin_pm[:, idx].prod(dim=2)    # (B, width)
        return comps @ coefs.to(device)
      ys = targets_lsb(xs)
    
      # 1) Quick attention stats
      attention_diag_stats(hardcoded_model, xs)
    
      # 2) Blockwise traces + top eigenvalues
      analyze_hessian_blocks(hardcoded_model, xs, ys, n_hutch=64, top_iters=60)
      def _normalize_rows(rows):
        # Accept float, list of floats, list of (label, val), or list of dicts
        if isinstance(rows, (float, int)):
            return [("total", float(rows))]
        if isinstance(rows, list) and rows:
            if isinstance(rows[0], (float, int)):
                # list of floats
                return [(f"part{i}", float(v)) for i, v in enumerate(rows)]
            if isinstance(rows[0], dict):
                # list of dicts with a 'trace' field
                return [(d.get("name", f"part{i}"), float(d["trace"])) for i, d in enumerate(rows)]
            if isinstance(rows[0], (list, tuple)) and len(rows[0]) == 2:
                # already (label, value)
                return [(str(a), float(b)) for a, b in rows]
        # empty or unknown -> safe default
        return []

      
      # 3) Scaling experiments:
      for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
        with scale_queries(hardcoded_model, which=("attn1","attn2"), factor=s):
            rows = analyze_hessian_blocks(hardcoded_model, xs, ys, n_hutch=32, top_iters=40)
            rows = _normalize_rows(rows)
            tot_trace = sum(val for _, val in rows)
            print(f"[scale_queries factor={s}] total_trace ≈ {tot_trace:.1f}")
    
      for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
        with scale_mlp_slopes(hardcoded_model, factor=s):
            rows = analyze_hessian_blocks(hardcoded_model, xs, ys, n_hutch=32, top_iters=40)
            rows = _normalize_rows(rows)
            tot_trace = sum(val for _, val in rows)
            print(f"[scale_mlp_slopes factor={s}] total_trace ≈ {tot_trace:.1f}")
      _hc_df = pd.DataFrame([{
          "deg": trainer.deg,
          "width": trainer.width,
          "func": trainer.func,
          "top_eig": hardcoded_hessian[0],
          "trace": hardcoded_hessian[1],
          "top_eig_train": hardcoded_hessian_train[0],
          "trace_train": hardcoded_hessian_train[1],
          "frobenius_weight_norm": weight_norm,
          "test_loss": hardcoded_loss
      }])
      _hc_df.to_csv(f"{trainer.dir_name}/hardcoded_hessian.csv", index=False,mode='a', header=not os.path.exists(f"{trainer.dir_name}/hardcoded_hessian.csv"))
      print("trainer.func_batch([2, 3]): " + str(trainer.func_batch([2,3])))
      #trainer.train(args.epochs)
      barrier()
      print("finished training, cleaning up process group...")
      destroy_process_group()
      print("finished cleaning up process group")
      return

if __name__ == "__main__":
    arguments = parse_args()
    arguments.save_checkpoints = False
    print(arguments)
    losses = {}
    func_per_deg = arguments.repeat
    main_dir = f"HESSIAN_CALCS10"
    os.makedirs(main_dir, exist_ok=True)
    # with open("logs_width.txt", "a") as f:
    #   f.write("------------------------------------------\n")

    for i in range(1):
        for deg in range(1,5):
            losses[deg] = []
            #for width in range(1, arguments.N, 5):
            for width in [1,2,3,4,5]:
                start_time = time.time()
                #world_size = torch.cuda.device_count()
                #args["world_size"]=world_size 
                print(f"Generating: func {i}, deg {deg}, width {width}")
                seedNum = int(str(i)+str(deg)+str(width))
                (coefs, combs) = rboolf(arguments.N, width, deg,seed=seedNum)
                torch.save(coefs,os.path.join(main_dir, f"coefs_func{i}_deg{deg}_width{width}.pt"))
                torch.save(combs,os.path.join(main_dir, f"combs_func{i}_deg{deg}_width{width}.pt"))
                
                mp.set_start_method('spawn',force = True)

                torch.set_num_threads(1)
                mp.spawn(main,args=(arguments,arguments.world_size,coefs,combs,main_dir,deg,width,i,),nprocs=arguments.world_size,join=True)
                print("returned from mp.spwan")
                end_time = time.time()
        
                elapsed_time = round((end_time - start_time)/60,3)
                print("elapsed time for whole training process: " + str(elapsed_time))