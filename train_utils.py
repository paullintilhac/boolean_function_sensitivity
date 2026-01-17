# train_utils.py
import argparse
import datetime
import math
import os

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, barrier, ReduceOp


# ===================== SAM Optimizer =====================

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) wrapper around a base optimizer."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True):
        if rho <= 0.0:
            raise ValueError("rho must be > 0")
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def _grad_norm(self):
        """Compute ||w * g||_2 across all parameters.

        Returns:
            None if there are no gradients, otherwise a scalar tensor.
        """
        eps = 1e-12
        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                w = p.abs() if adaptive else 1.0
                n = (w * g).norm(p=2)
                if torch.isnan(n) or torch.isinf(n):
                    raise RuntimeError("SAM grad norm is NaN/inf")
                norms.append(n)
        if not norms:
            # no gradients: treat as "skip step"
            return None
        return torch.norm(torch.stack(norms), p=2) + eps

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """Ascent step: w <- w + e_w."""
        norm = self._grad_norm()
        if norm is None:
            # nothing to do; skip this SAM step
            return
        scale = self.param_groups[0]["rho"] / norm
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = (p.abs() if adaptive else 1.0) * p.grad * scale
                p.add_(e)
                self.state[p]["e_w"] = e
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """Descent step: restore w and apply base optimizer step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.get(p, None)
                if not state or "e_w" not in state:
                    continue  # skipped first_step: nothing to undo
                p.sub_(state["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()


# ===================== Basic utilities =====================

def get_weight_norm(model):
    """Frobenius norm of all trainable parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def rboolf(N, width, deg, seed=None):
    """
    Sample a random Boolean polynomial:
      - width many terms
      - each term has degree = deg
    Returns:
      coefficients: (width,) tensor, positive, L2-normalized
      combs:        (width, deg) tensor of indices in [0, N)
    On CPU; caller can move to device / rank as needed.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # random positive coefficients normalized
    coefficients = torch.randn(width).abs()
    coefficients = coefficients / coefficients.pow(2).sum().sqrt()

    # sample `width` random subsets of size `deg` without replacement
    combs_list = []
    for _ in range(width):
        combs_list.append(torch.randperm(N)[:deg])
    combs = torch.stack(combs_list, dim=0)

    print("coefficients:", coefficients)
    print("combs:", combs)
    return coefficients, combs


def addGaussianNoise(
    model, sigma, as_variance=True, skip_frozen=True, include_bias=True, seed=None
):
    """
    Adds centered Gaussian noise to parameters in-place.
    """
    std = math.sqrt(sigma) if as_variance else float(sigma)
    if seed is not None:
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
            if "pos_embed.weight" in name or "bit_embed.weight" in name:
                continue
            noise = torch.empty_like(p)
            noise = noise.normal_(mean=0.0, std=std, generator=g)
            p.add_(noise)


# ===================== DDP / arg parsing =====================

def ddp_setup(rank, world_size, backend="nccl"):
    """
    Initialize process group and set CUDA device for DDP.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "23456")

    torch.cuda.set_device(rank)

    if backend == "gloo":
        init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:23456",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=5400),
        )
    else:
        init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=5400),
        )
    barrier(device_ids=[rank])


def parse_args():
    parser = argparse.ArgumentParser(description="linear spectrum non boolean test.")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--dim2", type=int, default=22)
    parser.add_argument("--f", type=int, default=64)
    parser.add_argument("--h", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--lr", type=str, default="1e-5")
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--stop_loss", type=float, default=0.02)
    parser.add_argument("--ln_eps", type=float, default=1e-5)
    parser.add_argument("--ln", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--asam", action="store_true")
    return parser.parse_args()

