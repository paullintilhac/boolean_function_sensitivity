# Perturbation Hessian Experiment
# Calculates Hessian traces under Gaussian noise perturbations for HardCodedTransformer

from pyhessian.hessian import hessian
import copy
import numpy as np
import pandas as pd
import torch
import random
import math
import os
import itertools
from contextlib import contextmanager
from hardcoded_transformer import HardCodedTransformer, rboolf, func_batch


# ========== Device Setup with MPS Support ==========

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

if mps_avail:
    device = torch.device("mps")
    device_id = 0  # For compatibility with pyhessian patching
elif cuda_avail:
    device = torch.device("cuda:0")
    device_id = 0
else:
    device = torch.device("cpu")
    device_id = 0

print(f"Using device: {device}")


# ========== MPS Patching for pyhessian ==========

@contextmanager
def patch_pyhessian_for_mps():
    """
    Monkey patch torch functions to redirect 'cuda' device strings to MPS
    when using MPS device. This is needed because pyhessian library has
    hardcoded CUDA assumptions.
    """
    if not mps_avail:
        yield
        return
    
    # Store original functions
    original_tensor_cuda = torch.Tensor.cuda
    original_tensor_to = torch.Tensor.to
    original_randn = torch.randn
    original_randn_like = torch.randn_like
    original_randint_like = torch.randint_like
    original_zeros = torch.zeros
    original_ones = torch.ones
    original_empty = torch.empty
    original_empty_like = torch.empty_like
    
    def patched_cuda(self, device=None):
        # Redirect any cuda call to MPS (including None, 0, 'cuda', 'cuda:0', or cuda device objects)
        if device is None or device == 0 or device == 'cuda' or device == 'cuda:0' or \
           (isinstance(device, torch.device) and device.type == 'cuda'):
            return original_tensor_to(self, torch.device("mps"))
        # For any other case, also redirect to MPS if it's not explicitly a non-CUDA device
        # This is a safety measure since we're on MPS and shouldn't use CUDA
        return original_tensor_to(self, torch.device("mps"))
    
    def patched_to(self, *args, **kwargs):
        # Check if device is specified in args or kwargs
        if args:
            arg0 = args[0]
            if arg0 == 0 or arg0 == 'cuda' or arg0 == 'cuda:0' or \
               (isinstance(arg0, torch.device) and arg0.type == 'cuda'):
                return original_tensor_to(self, torch.device("mps"))
        if 'device' in kwargs:
            dev = kwargs['device']
            if dev == 0 or dev == 'cuda' or dev == 'cuda:0' or \
               (isinstance(dev, torch.device) and dev.type == 'cuda'):
                kwargs['device'] = torch.device("mps")
        return original_tensor_to(self, *args, **kwargs)
    
    def patched_randn(*args, **kwargs):
        if 'device' in kwargs:
            if kwargs['device'] == 0 or kwargs['device'] == 'cuda' or kwargs['device'] == 'cuda:0':
                kwargs['device'] = torch.device("mps")
        return original_randn(*args, **kwargs)
    
    def patched_randn_like(input, **kwargs):
        if 'device' in kwargs:
            if kwargs['device'] == 0 or kwargs['device'] == 'cuda' or kwargs['device'] == 'cuda:0':
                kwargs['device'] = torch.device("mps")
        return original_randn_like(input, **kwargs)
    
    def patched_randint_like(input, *args, **kwargs):
        # torch.randint_like(input, low=0, high, ..., device=...)
        if 'device' in kwargs:
            dev = kwargs['device']
            if dev == 0 or dev == 'cuda' or dev == 'cuda:0' or \
               (isinstance(dev, torch.device) and dev.type == 'cuda'):
                kwargs['device'] = torch.device("mps")
        return original_randint_like(input, *args, **kwargs)
    
    def patched_zeros(*args, **kwargs):
        if 'device' in kwargs:
            if kwargs['device'] == 0 or kwargs['device'] == 'cuda' or kwargs['device'] == 'cuda:0':
                kwargs['device'] = torch.device("mps")
        return original_zeros(*args, **kwargs)
    
    def patched_ones(*args, **kwargs):
        if 'device' in kwargs:
            if kwargs['device'] == 0 or kwargs['device'] == 'cuda' or kwargs['device'] == 'cuda:0':
                kwargs['device'] = torch.device("mps")
        return original_ones(*args, **kwargs)
    
    def patched_empty(*args, **kwargs):
        if 'device' in kwargs:
            if kwargs['device'] == 0 or kwargs['device'] == 'cuda' or kwargs['device'] == 'cuda:0':
                kwargs['device'] = torch.device("mps")
        return original_empty(*args, **kwargs)
    
    def patched_empty_like(input, **kwargs):
        if 'device' in kwargs:
            if kwargs['device'] == 0 or kwargs['device'] == 'cuda' or kwargs['device'] == 'cuda:0':
                kwargs['device'] = torch.device("mps")
        return original_empty_like(input, **kwargs)
    
    # Apply patches
    torch.Tensor.cuda = patched_cuda
    torch.Tensor.to = patched_to
    torch.randn = patched_randn
    torch.randn_like = patched_randn_like
    torch.randint_like = patched_randint_like
    torch.zeros = patched_zeros
    torch.ones = patched_ones
    torch.empty = patched_empty
    torch.empty_like = patched_empty_like
    
    try:
        yield
    finally:
        # Restore original functions
        torch.Tensor.cuda = original_tensor_cuda
        torch.Tensor.to = original_tensor_to
        torch.randn = original_randn
        torch.randn_like = original_randn_like
        torch.randint_like = original_randint_like
        torch.zeros = original_zeros
        torch.ones = original_ones
        torch.empty = original_empty
        torch.empty_like = original_empty_like


# ========== Helper Functions ==========

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
    # Ensure sigma is non-negative and use torch.sqrt for numerical stability
    sigma = max(float(sigma), 1e-20)  # Prevent domain errors from very small values
    if as_variance:
        # Use torch.sqrt to avoid math domain errors and ensure numerical stability
        std_tensor = torch.tensor(sigma, dtype=torch.float32)
        std = float(torch.sqrt(std_tensor).item())
    else:
        std = float(sigma)
    
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
            # Skip fixed embeddings
            if "pos_embed.weight" in name or "bit_embed.weight" in name:
                continue
            # Generate noise and add it - even if very small, we still add it for consistency
            noise = torch.empty_like(p)
            noise = noise.normal_(mean=0.0, std=std, generator=g)
            p.add_(noise)


def calc_hessian(model, loss_fn, inputs, targets, device_obj, trace_samples=50000):
    """
    Calculate Hessian trace and top eigenvalue for a model.
    
    Args:
        model: nn.Module
        loss_fn: loss function (input, target) -> loss
        inputs: input tensor
        targets: target tensor
        device_obj: torch.device to use
        trace_samples: number of Hutchinson samples for trace estimation (default: 50000)
                      Error scales as 1/sqrt(samples), so more samples = lower variance
    
    Returns:
        tuple: (top_eig, trace_mean)
    """
    model.eval().to(device_obj)
    inputs = inputs.to(device_obj)
    targets = targets.to(device_obj)
    data = (inputs, targets)
    
    # Use MPS patching if needed (context manager handles mps_avail check internally)
    with patch_pyhessian_for_mps():
        hess_mod = hessian(model, loss_fn, data)
        for param in model.parameters():
            param.grad = None
        top_eigs, top_eigVs = hess_mod.eigenvalues(maxIter=200)
        top_eig = top_eigs[0]
        # Increase number of samples for trace estimation to reduce variance
        # Hutchinson's method error scales as 1/sqrt(samples)
        trace = hess_mod.trace(maxIter=trace_samples)
        return float(top_eig), float(np.mean(trace))


# ========== Main Experiment ==========

def run_perturbation_experiment():
    """
    Run perturbation experiment over all parameter combinations.
    """
    # Experiment parameters
    degrees = [1,2,3,4]
    widths = [1, 7, 14, 20]
    T_values = [20, 40, 100, 200]
    func_indices = list(range(10))  # 0-9
    sigma_values = [1e-2]
    num_samples = 1000  # Training samples for Hessian calculation
    
    # Output file
    output_file = "perturbation_hessian_results.csv"
    
    # Check if file exists to determine if we should write header
    file_exists = os.path.exists(output_file)
    
    # Loss function
    loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()
    
    # Progress tracking
    total_combinations = len(degrees) * len(widths) * len(T_values) * len(func_indices) * len(sigma_values)
    completed = 0
    
    print(f"Starting perturbation experiment")
    print(f"Total combinations: {total_combinations}")
    print(f"Using device: {device}")
    print(f"Output file: {output_file}")
    print("-" * 80)
    
    # Results list
    results = []
    
    # Main loop
    for deg in degrees:
        for width in widths:
            for T in T_values:
                # Check if width is valid for this T (can't have more combinations than available)
                max_combinations = math.comb(T, deg)
                if width > max_combinations:
                    print(f"Skipping deg={deg}, width={width}, T={T}: width > max_combinations({max_combinations})")
                    continue
                
                for func_idx in func_indices:
                    # Generate random function
                    seed_num = int(str(func_idx) + str(deg) + str(width) + str(T))
                    torch.manual_seed(seed_num)
                    
                    try:
                        coefs, combs = rboolf(T, width, deg, seed=seed_num)
                        coefs = coefs.to(device)
                        combs = combs.to(device)
                        
                        # Create model
                        model = HardCodedTransformer(
                            N=T,
                            combs=combs,
                            coefs=coefs,
                            aggregator_idx=T-1,
                            mode="original",
                            mlp_soft_factor=0.25
                        ).to(device)
                        
                        # Generate training data
                        # For large T, 2**T can overflow, so generate random bits instead
                        # Generate T random bits per sample and interpret as integer
                        bits = torch.randint(0, 2, (num_samples, T), device=device, dtype=torch.long)
                        # Convert bits to integers: sum(bits[i] * 2^j for j, bits[i][j])
                        powers_of_2 = torch.pow(2, torch.arange(T, device=device, dtype=torch.long))
                        train_inputs = (bits * powers_of_2.unsqueeze(0)).sum(dim=1)
                        # func_batch can handle both numpy arrays and tensors
                        # Pass as tensors to avoid type issues
                        train_targets = func_batch(train_inputs.cpu(), coefs.cpu(), combs.cpu(), T)
                        train_targets = train_targets.to(device)
                        
                        # For each sigma value
                        for sigma in sigma_values:
                            try:
                                completed += 1
                                
                                # Calculate Hessian BEFORE perturbation (on original unperturbed model)
                                print(f"[{completed}/{total_combinations}] Calculating Hessian BEFORE perturbation: "
                                      f"deg={deg}, width={width}, T={T}, func={func_idx}, sigma={sigma}")
                                
                                # Create a fresh copy for the "before" calculation
                                model_before = copy.deepcopy(model)
                                model_before.eval().to(device)
                                
                                # Use more samples for trace estimation to reduce variance
                                # For small perturbations, we need many samples to detect small differences
                                # Scale samples inversely with sigma: need ~1/sigma^2 samples for relative error ~sigma
                                # Cap at reasonable maximum to avoid excessive computation
                                base_samples = 50000
                                trace_samples = min(max(base_samples, int(1.0 / max(sigma, 1e-12))), 1000000)
                                
                                # Use same seed for Hessian calculations to ensure reproducibility
                                torch.manual_seed(42)  # Fixed seed for Hessian trace estimation
                                top_eig_before, trace_before = calc_hessian(
                                    model_before, loss_fn, train_inputs, train_targets, device,
                                    trace_samples=trace_samples
                                )
                                
                                # Create a fresh copy for perturbation (must be separate from model_before)
                                model_perturbed = copy.deepcopy(model)
                                model_perturbed.eval().to(device)
                                
                                # Add Gaussian noise to the perturbed model
                                # Use a unique seed that depends on sigma to get consistent but different perturbations
                                perturbation_seed = seed_num + int(sigma * 1e10) % 1000000
                                addGaussianNoise(model_perturbed, sigma, as_variance=True, seed=perturbation_seed)
                                
                                # Calculate Hessian AFTER perturbation with same seed for fair comparison
                                torch.manual_seed(42)  # Same seed as before for fair comparison
                                print(f"[{completed}/{total_combinations}] Calculating Hessian AFTER perturbation: "
                                      f"deg={deg}, width={width}, T={T}, func={func_idx}, sigma={sigma} "
                                      f"(trace_samples={trace_samples})")
                                
                                top_eig_after, trace_after = calc_hessian(
                                    model_perturbed, loss_fn, train_inputs, train_targets, device,
                                    trace_samples=trace_samples
                                )
                                
                                # Calculate delta (should approach 0 as sigma -> 0)
                                trace_delta = trace_after - trace_before
                                
                                # Verify perturbation was actually applied (for debugging)
                                if sigma < 1e-10:
                                    # Check if parameters actually changed
                                    param_diff = sum((p1 - p2).abs().sum().item() 
                                                     for p1, p2 in zip(model_before.parameters(), 
                                                                       model_perturbed.parameters()))
                                    print(f"  Debug: sigma={sigma}, std={torch.sqrt(torch.tensor(sigma)).item():.2e}, "
                                          f"param_diff={param_diff:.2e}, trace_delta={trace_delta:.6f}")
                                
                                # Clean up models
                                del model_before, model_perturbed
                                
                                # Store result
                                result = {
                                    'deg': deg,
                                    'width': width,
                                    'T': T,
                                    'func': func_idx,
                                    'sigma': sigma,
                                    'trace_before': trace_before,
                                    'trace_after': trace_after,
                                    'trace_delta': trace_delta,
                                    'top_eig_before': top_eig_before,
                                    'top_eig_after': top_eig_after
                                }
                                results.append(result)
                                
                                # Save incrementally
                                df = pd.DataFrame([result])
                                df.to_csv(output_file, mode='a', header=not file_exists, index=False)
                                file_exists = True
                                
                                print(f"  Result: trace_before={trace_before:.4f}, trace_after={trace_after:.4f}, "
                                      f"delta={trace_delta:.4f}")
                                
                                # Clean up (already deleted above in the try block)
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                elif device.type == 'mps':
                                    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
                                
                            except Exception as e:
                                print(f"  ERROR at sigma={sigma}: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        # Clean up model
                        del model, coefs, combs
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        elif device.type == 'mps':
                            torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
                    
                    except Exception as e:
                        print(f"ERROR creating model for deg={deg}, width={width}, T={T}, func={func_idx}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
    
    print("-" * 80)
    print(f"Experiment completed! Results saved to {output_file}")
    print(f"Total combinations processed: {completed}/{total_combinations}")
    
    return results


if __name__ == "__main__":
    results = run_perturbation_experiment()

