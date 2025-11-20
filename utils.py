# utils.py
import math
import random
import torch

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

if mps_avail:
    device = torch.device("mps")
elif cuda_avail:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


def get_weight_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def rboolf(N, width, deg, seed=None):
    """Sample `width` random degree-`deg` subsets without enumerating all C(N,deg)."""
    if seed is not None:
        torch.manual_seed(seed)

    # random positive coefficients normalized
    coefficients = torch.randn(width).abs().to(device)
    coefficients = coefficients / coefficients.pow(2).sum().sqrt()

    # sample `width` random subsets of size `deg` without replacement per row
    combs_list = []
    for _ in range(width):
        combs_list.append(torch.randperm(N)[:deg])
    combs = torch.stack(combs_list, dim=0).to(device)

    return coefficients, combs


def addGaussianNoise(
    model, sigma, as_variance=True, skip_frozen=True, include_bias=True, seed=None
):
    """
    Adds centered Gaussian noise to parameters in-place.
    """
    std = math.sqrt(sigma) if as_variance else float(sigma)
    if seed is not None:
        dev = next(model.parameters()).device
        g = torch.Generator(device=dev).manual_seed(seed)
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

