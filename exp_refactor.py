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
import json
import glob
import shutil
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp, barrier
mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()
#from functools import partial

# ===== Configuration Ignore List for Checkpoint Loading =====
# Parameters in this list can differ between the current run and a saved checkpoint,
# and we will still load from that checkpoint. This allows resuming training with
# different hyperparameters (e.g., learning rate, weight decay) while keeping the
# same model weights.
#
# IMPORTANT: This list MUST NOT include:
#   - Sample size (n_samples, num_samples): Different dataset size would require different model state
#   - Batch size (bs, batch_size): Affects training dynamics and optimizer state
#   - Architectural parameters: N, dim (d), h, f, dropout (affects model architecture)
#   - Model structure: ln, ln_eps (affects layer normalization structure)
#   - SAM parameters: sam, sam_rho, asam (affects optimizer structure)
#
# Parameters that CAN be ignored (safe to differ):
#   - Learning rate (lr): Can be changed without affecting model weights
#   - Weight decay (wd): Can be changed without affecting model weights
#   - Stop loss (stop_loss): Just a training criterion, doesn't affect model
#   - Save frequency (save_every): Just a logging parameter
#   - Backend: Doesn't affect model weights
CONFIG_IGNORE_LIST = {
    'lr',           # Learning rate - can change during training
    'wd',           # Weight decay - can change during training
    'stop_loss',    # Training criterion - doesn't affect model
    'save_every',   # Logging frequency - doesn't affect model
    'backend',      # Distributed backend - doesn't affect model weights
    'epochs',       # Max epochs - doesn't affect model weights
}

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

# ===== Checkpoint Management Functions =====

def get_checkpoint_dir(base_dir, func, deg, width):
    """Get the checkpoint directory path for a specific function/deg/width."""
    return os.path.join(base_dir, "checkpoints", f"func{func}_deg{deg}_width{width}")

def get_config_dict(args, trainer=None):
    """Extract all relevant configuration parameters into a dictionary."""
    config = {
        'N': args.N,
        'dim': args.dim,
        'h': args.h,
        'f': args.f,
        'dropout': args.dropout,
        'num_samples': args.num_samples,
        'bs': args.bs,
        'lr': str(args.lr),  # Keep as string to match input format
        'wd': args.wd,
        'ln_eps': args.ln_eps,
        'ln': args.ln,
        'sam': args.sam,
        'sam_rho': args.sam_rho,
        'asam': args.asam,
        'stop_loss': args.stop_loss,
        'save_every': args.save_every,
        'backend': args.backend,
    }
    if trainer is not None:
        # Add any trainer-specific config that might not be in args
        config['batch_size'] = trainer.batch_size
    return config

def configs_match(config1, config2, ignore_list=CONFIG_IGNORE_LIST):
    """Check if two configs match, ignoring parameters in ignore_list."""
    for key in set(config1.keys()) | set(config2.keys()):
        if key in ignore_list:
            continue
        if key not in config1 or key not in config2:
            return False
        # Handle string vs float comparison for lr
        if key == 'lr':
            if str(config1[key]) != str(config2[key]):
                return False
        else:
            if config1[key] != config2[key]:
                return False
    return True

def save_checkpoint_with_config(checkpoint_dir, model, epoch, config, gpu_id=0):
    """Save checkpoint with config file. Only keeps the most recent checkpoint."""
    if gpu_id != 0:  # Only save on rank 0
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Remove old checkpoint files (keep only most recent)
    old_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    for old_ckpt in old_checkpoints:
        try:
            os.remove(old_ckpt)
        except:
            pass
    
    # Save model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    if hasattr(model, 'module'):  # DDP model
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    
    # Save config
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump({**config, 'epoch': epoch}, f, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_path} (epoch {epoch})")

def find_matching_checkpoint(base_dir, func, deg, width, current_config):
    """Find a checkpoint with matching configuration (ignoring CONFIG_IGNORE_LIST)."""
    checkpoint_dir = get_checkpoint_dir(base_dir, func, deg, width)
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    
    # Load saved config
    try:
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
    except:
        return None
    
    # Check if configs match (ignoring ignore list)
    if not configs_match(current_config, saved_config):
        return None
    
    # Find the most recent checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not checkpoints:
        return None
    
    # Get checkpoint with highest epoch number
    def get_epoch(path):
        try:
            basename = os.path.basename(path)
            epoch_str = basename.replace("checkpoint_epoch", "").replace(".pt", "")
            return int(epoch_str)
        except:
            return -1
    
    latest_checkpoint = max(checkpoints, key=get_epoch)
    return latest_checkpoint, saved_config.get('epoch', 0)
    
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
            sam: bool = False,
            sam_rho: float = 0.05,
            asam: bool = False,
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
                                 "wd",
                                 "N",
                                 "sam",
                                 "sam_rho",
                                 "asam"])
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
        self.sam = sam
        self.sam_rho = sam_rho
        self.asam = asam
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
    
        # self.combs is shape (width, deg) on gpu_id already
        idx = self.combs.long()                                     # (W, D)
        # Compute parity for each combination: 1 if even number of 1s, 0 if odd
        sum_bits = bits01[:, idx].sum(dim=2)                        # (B, W) - sum of bits in each combination
        comps = 1.0 - (sum_bits % 2.0)                              # (B, W) - parity in {0,1}
    
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
        start_time = time.time()
        
        # Training phase: model in train mode (with dropout)
        for idx, inputs in enumerate(self.train_data):
          #inputs.to(self.gpu_id)    
          targets =self.func_batch(inputs).to(self.gpu_id)
          batch_loss = self._run_batch(inputs, targets)
          iteration = epoch*len(self.train_data)+idx+1
        
        # Evaluation phase: calculate epoch loss with model in eval mode (no dropout)
        # This makes epoch loss comparable to validation loss
        self.model.eval()
        epoch_loss = 0
        total_records = 0
        loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()
        
        with torch.no_grad():
            for idx, inputs in enumerate(self.train_data):
                targets = self.func_batch(inputs).to(self.gpu_id)
                out = self.model(inputs)
                batch_loss = loss_fn(out, targets)
                epoch_loss += batch_loss * float(len(inputs))
                total_records += len(inputs)
        
        epoch_loss /= float(total_records)
        
        # Switch back to train mode for next epoch
        self.model.train()
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        #print(f"Epoch time: {elapsed_time:.3f} seconds.")
        time_per_record_ms = float(elapsed_time*100)/float(total_records)
        #print(f"Epoch time: {elapsed_time:.3f} seconds. time per record (ms): {time_per_record_ms: .3f}")
        return epoch_loss

    def save_checkpoint(self, epoch, model_name=None):
        """Save checkpoint with configuration. Only keeps most recent checkpoint."""
        if not self.save_checkpoints or self.gpu_id != 0:
            return
        
        # Build config dict from trainer attributes
        config = {
            'N': self.N,
            'dim': self.d,
            'h': self.h,
            'f': self.f,
            'dropout': self.dropout,
            'num_samples': self.n_samples,
            'bs': self.batch_size,
            'lr': str(self.lr),
            'wd': self.wd,
            'ln_eps': self.ln_eps,
            'ln': self.ln,
            'sam': self.sam,
            'sam_rho': self.sam_rho,
            'asam': self.asam,
            'stop_loss': self.stop_loss,
            'save_every': self.save_every,
            'backend': self.backend,
        }
        
        checkpoint_dir = get_checkpoint_dir(self.dir_name, self.func, self.deg, self.width)
        save_checkpoint_with_config(checkpoint_dir, self.model, epoch, config, self.gpu_id)
    
    def load_checkpoint_if_exists(self, args):
        """Load checkpoint if one exists with matching configuration."""
        if self.gpu_id != 0:  # Only load on rank 0, then broadcast
            return None, 0
        
        current_config = get_config_dict(args, self)
        result = find_matching_checkpoint(self.dir_name, self.func, self.deg, self.width, current_config)
        
        if result is None:
            return None, 0
        
        checkpoint_path, start_epoch = result
        print(f"Loading checkpoint from {checkpoint_path} (resuming from epoch {start_epoch})")
        
        try:
            state_dict = torch.load(checkpoint_path, map_location=f'cuda:{self.gpu_id}')
            self.model.module.load_state_dict(state_dict)
            print(f"Successfully loaded checkpoint from epoch {start_epoch}")
            return checkpoint_path, start_epoch
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None, 0

    def train(self, epochs: int, start_epoch: int = 0):
        """Train the model, optionally starting from a checkpoint epoch."""
        self.model.train()
        
        start_time = time.time()

        for epoch in range(start_epoch, epochs):
            epoch_loss = self._run_epoch(epoch)
            
            if ((epoch % self.save_every)==0 and self.gpu_id==0) or (epoch_loss < self.stop_loss):
            # if ((((epoch+1) % self.save_every)==0 or epoch==0) and self.gpu_id==0):

                #print("inside conditional")
                if self.save_checkpoints:
                    self.save_checkpoint(epoch)
                end_time = time.time()
                elapsed_time = round((end_time - start_time)/60,3) 

                #print("self.func: " + str(self.func))
                val_loss = self.validate(1000,self.model) 
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
                                       "wd":self.wd,
                                       "N":self.N,
                                       "sam":self.sam,
                                       "sam_rho":self.sam_rho,
                                       "asam":self.asam
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
        hardcoded_models=[]
        # Only "original" mode is supported (matches mathematical construction)
        hardcoded_model = HardCodedTransformer(N, combs, coefs, aggregator_idx=N, mode="original")
        hardcoded_total_params = sum(p.numel() for p in hardcoded_model.parameters())
        #print(model)
        print("Hardcoded Model Parameter Count: " + str(hardcoded_total_params))
        hardcoded_models.append(hardcoded_model)
            
        model = Transformer(dropout,N, dim, h, f, ln_eps, rank, ln)
        total_params = sum(p.numel() for p in model.parameters())
        #print(model)
        print("Trainable Model Parameter Count: " + str(total_params))
        
        base_opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=wd)
        optimizer = SAM(model.parameters(), base_optimizer=base_opt, rho=sam_rho, adaptive=asam) if sam else base_opt
        return train_set, model, optimizer, hardcoded_models               

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
        
      train_set,model,optimizer,hardcoded_models = load_train_objs(args.dropout,
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
      for hardcoded_model in hardcoded_models:
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
                        wd=args.wd,
                        sam=args.sam,
                        sam_rho=args.sam_rho,
                        asam=args.asam
                        )

      # loss_fn = lambda result, targets: (result-targets).pow(2).mean()
      loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()
      hardcoded_hessian_stats = []
      
      # Only "original" mode is supported (matches mathematical construction)
      hardcoded_model = hardcoded_models[0]
      print("hardcoded model: " + str(hardcoded_model))
      #addGaussianNoise(hardcoded_model, .1)
      hardcoded_hessian_train = trainer.calc_hessian(hardcoded_model, loss_fn, num_samples=1000,device_id=rank, use_train=True)
      addGaussianNoise(hardcoded_model, .0001)
      hardcoded_hessian_pert=trainer.calc_hessian(hardcoded_model, loss_fn, num_samples=1000,device_id=rank, use_train=True)

      weight_norm = get_weight_norm(hardcoded_model)
      hardcoded_loss = trainer.validate(1000,hardcoded_model)
      print("hardcoded loss: " + str(hardcoded_loss))
      print("frobenius weight norm: " + str(weight_norm)) 
      print("hardcoded hessian stats: " + str(hardcoded_hessian_pert))
  
      
      _hc_df = pd.DataFrame([{
          "deg": trainer.deg,
          "width": trainer.width,
          "func": trainer.func,
          "const_mode": "original",
          "top_eig_pert": round(hardcoded_hessian_pert[0],2),
          "trace_pert": round(hardcoded_hessian_pert[1],2),
          "top_eig_train": round(hardcoded_hessian_train[0],2),
          "trace_train": round(hardcoded_hessian_train[1],2),
          "frobenius_weight_norm": round(weight_norm,2),
          "test_loss": torch.round(hardcoded_loss,decimals=3)
      }])
      _hc_df.to_csv(f"{trainer.dir_name}/hardcoded_hessian.csv", index=False,mode='a', header=not os.path.exists(f"{trainer.dir_name}/hardcoded_hessian.csv"))
      print("trainer.func_batch([2, 3]): " + str(trainer.func_batch([2,3])))
      
      # Try to load checkpoint if one exists with matching configuration
      start_epoch = 0
      if rank == 0:  # Only check on rank 0
          checkpoint_path, start_epoch = trainer.load_checkpoint_if_exists(args)
          if checkpoint_path:
              # Broadcast that we loaded a checkpoint (simple barrier for now)
              # In a real DDP setup, you'd want to sync the state_dict across ranks
              print(f"Resuming training from epoch {start_epoch}")
      
      # Synchronize all ranks before starting training
      barrier()
      
      trainer.train(args.epochs, start_epoch=start_epoch)
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
    main_dir = f"HESSIAN_CALCS_105_FIX2"
    os.makedirs(main_dir, exist_ok=True)
    # with open("logs_width.txt", "a") as f:
    #   f.write("------------------------------------------\n")

    for i in range(10,20):
        for deg in [5,4,1,2,3]:
            losses[deg] = []
            #for width in range(1, arguments.N, 5):
            for width in [20,14,1,7]:
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
