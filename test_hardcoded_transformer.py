"""
Comprehensive unit tests for HardCodedTransformer

Tests verify that the transformer accurately represents boolean functions
of the specified degree and width, matching the mathematical construction exactly.
"""

import pytest
import torch
import numpy as np
import math
from hardcoded_transformer import (
    HardCodedTransformer,
    rboolf,
    func_batch,
    IntCountParityMLP
)


# Test fixtures
@pytest.fixture
def device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def small_test_config():
    """Small test configuration for exhaustive testing"""
    return {
        'N': 8,
        'deg': 2,
        'width': 3,
        'seed': 42
    }


@pytest.fixture
def medium_test_config():
    """Medium test configuration"""
    return {
        'N': 12,
        'deg': 3,
        'width': 5,
        'seed': 123
    }


@pytest.fixture
def create_transformer(device):
    """Helper to create a transformer with given parameters"""
    def _create(N, deg, width, seed=None, mode="original"):
        torch.manual_seed(seed if seed is not None else 0)
        coefs, combs = rboolf(N, width, deg, seed=seed)
        model = HardCodedTransformer(
            N=N,
            combs=combs,
            coefs=coefs,
            aggregator_idx=N,
            mode=mode
        ).to(device).eval()
        return model, coefs, combs
    return _create


# Tolerance for floating point comparisons
ACCURACY_TOLERANCE = 1e-4  # For transformer output vs func_batch
EXACT_TOLERANCE = 1e-6     # For exact mathematical operations
PARITY_TOLERANCE = 1e-8    # For parity computation (should be exact at integers)


# ============================================================================
# Test 1: Basic Function Accuracy
# ============================================================================

def test_basic_accuracy_small(create_transformer, device, small_test_config):
    """Test that transformer output matches func_batch for small configuration"""
    config = small_test_config
    model, coefs, combs = create_transformer(
        config['N'], config['deg'], config['width'], 
        seed=config['seed'], mode="original"
    )
    
    # Test on random inputs
    num_samples = 100
    torch.manual_seed(999)
    xs = torch.randint(0, 2**config['N'], (num_samples,), device=device)
    
    # Get transformer output
    with torch.no_grad():
        transformer_out = model(xs).squeeze(-1)
    
    # Get exact function output
    targets = func_batch(xs.cpu().tolist(), coefs.cpu(), combs.cpu(), config['N']).to(device)
    
    # Compare
    max_error = (transformer_out - targets).abs().max().item()
    mean_error = (transformer_out - targets).abs().mean().item()
    
    assert max_error < ACCURACY_TOLERANCE, \
        f"Max error {max_error:.2e} exceeds tolerance {ACCURACY_TOLERANCE:.2e}"
    assert mean_error < ACCURACY_TOLERANCE / 10, \
        f"Mean error {mean_error:.2e} exceeds tolerance {ACCURACY_TOLERANCE/10:.2e}"


def test_basic_accuracy_medium(create_transformer, device, medium_test_config):
    """Test that transformer output matches func_batch for medium configuration"""
    config = medium_test_config
    model, coefs, combs = create_transformer(
        config['N'], config['deg'], config['width'], 
        seed=config['seed'], mode="original"
    )
    
    # Test on random inputs
    num_samples = 50
    torch.manual_seed(888)
    xs = torch.randint(0, 2**config['N'], (num_samples,), device=device)
    
    # Get transformer output
    with torch.no_grad():
        transformer_out = model(xs).squeeze(-1)
    
    # Get exact function output
    targets = func_batch(xs.cpu().tolist(), coefs.cpu(), combs.cpu(), config['N']).to(device)
    
    # Compare
    max_error = (transformer_out - targets).abs().max().item()
    mean_error = (transformer_out - targets).abs().mean().item()
    
    assert max_error < ACCURACY_TOLERANCE, \
        f"Max error {max_error:.2e} exceeds tolerance {ACCURACY_TOLERANCE:.2e}"
    assert mean_error < ACCURACY_TOLERANCE / 10, \
        f"Mean error {mean_error:.2e} exceeds tolerance {ACCURACY_TOLERANCE/10:.2e}"


@pytest.mark.parametrize("deg", [1, 2, 3, 4, 5])
def test_basic_accuracy_various_degrees(create_transformer, device, deg):
    """Test accuracy for various degrees"""
    N = 10
    width = 3
    seed = 42
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed, mode="original")
    
    # Test on random inputs
    num_samples = 50
    torch.manual_seed(777)
    xs = torch.randint(0, 2**N, (num_samples,), device=device)
    
    with torch.no_grad():
        transformer_out = model(xs).squeeze(-1)
    
    targets = func_batch(xs.cpu().tolist(), coefs.cpu(), combs.cpu(), N).to(device)
    
    max_error = (transformer_out - targets).abs().max().item()
    assert max_error < ACCURACY_TOLERANCE, \
        f"Degree {deg}: Max error {max_error:.2e} exceeds tolerance"


# ============================================================================
# Test 2: Degree Verification
# ============================================================================

def test_degree_verification(create_transformer, device):
    """Verify all combinations have exactly the specified degree"""
    for deg in [1, 2, 3, 4, 5]:
        N = 12
        width = 5
        seed = 100 + deg
        
        _, _, combs = create_transformer(N, deg, width, seed=seed)
        
        # Check all combinations have exactly deg elements
        for comb in combs:
            assert len(comb) == deg, \
                f"Combination {comb} has length {len(comb)}, expected {deg}"
        
        # Verify combinations are valid (all indices in range)
        for comb in combs:
            for idx in comb:
                assert 0 <= idx < N, \
                    f"Index {idx} in combination {comb} is out of range [0, {N})"


def test_degree_only_deg_way_interactions(create_transformer, device):
    """Verify transformer only uses deg-way interactions"""
    # This is verified by checking that all combinations have exactly deg elements
    # and that the function matches func_batch which only uses deg-way interactions
    N = 8
    deg = 2
    width = 3
    seed = 200
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed)
    
    # Verify all combinations have exactly deg elements
    for comb in combs:
        assert len(comb) == deg, f"Combination {comb} does not have degree {deg}"
    
    # Test on all possible inputs for small N
    all_inputs = torch.arange(2**N, device=device)
    with torch.no_grad():
        transformer_out = model(all_inputs).squeeze(-1)
    
    targets = func_batch(all_inputs.cpu().tolist(), coefs.cpu(), combs.cpu(), N).to(device)
    
    max_error = (transformer_out - targets).abs().max().item()
    assert max_error < ACCURACY_TOLERANCE, \
        f"Transformer does not match deg-way interactions: error {max_error:.2e}"


# ============================================================================
# Test 3: Width Verification
# ============================================================================

def test_width_verification(create_transformer, device):
    """Verify exactly width components are used"""
    for width in [1, 3, 7, 14, 20]:
        N = 12
        deg = 2
        seed = 300 + width
        
        _, coefs, combs = create_transformer(N, deg, width, seed=seed)
        
        # Verify width matches
        assert len(coefs) == width, \
            f"Number of coefficients {len(coefs)} does not match width {width}"
        assert len(combs) == width, \
            f"Number of combinations {len(combs)} does not match width {width}"


def test_width_handles_max_combinations(create_transformer, device):
    """Test behavior when width approaches or exceeds max combinations"""
    N = 8
    deg = 2
    max_combinations = math.comb(N, deg)
    
    # Test with width = max_combinations (should use all)
    if max_combinations <= 50:  # Only test if not too large
        _, coefs, combs = create_transformer(N, deg, max_combinations, seed=400)
        assert len(coefs) == max_combinations
        assert len(combs) == max_combinations


# ============================================================================
# Test 4: Coefficient Matching
# ============================================================================

def test_coefficient_matching(create_transformer, device):
    """Verify transformer uses the exact coefficients provided"""
    N = 10
    deg = 2
    width = 5
    seed = 500
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed)
    
    # Verify model stores correct coefficients
    assert torch.allclose(model.coefs, coefs), \
        "Model coefficients do not match input coefficients"
    
    # Verify all coefficients are positive
    assert (model.coefs > 0).all(), \
        "Model contains non-positive coefficients"


def test_attention_layer2_coefficient_weights(create_transformer, device):
    """Verify attention layer 2 properly weights by coefficients"""
    N = 8
    deg = 2
    width = 3
    seed = 600
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed)
    
    # Check attention layer 2 query weights
    E = model.E
    Wq = model.attn2.in_proj_weight[:E, :]
    
    # For each component, the logit should be log(c_i) + 2*log(N)
    base = 2.0 * math.log(max(2, N))
    for i, (ci, t) in enumerate(zip(coefs, model.rep_idx)):
        expected_logit = math.log(max(float(ci), 1e-12)) + base
        actual_logit = Wq[t, model.aggregator_idx].item()
        
        assert abs(actual_logit - expected_logit) < EXACT_TOLERANCE, \
            f"Component {i}: logit {actual_logit:.6f} != expected {expected_logit:.6f}"


# ============================================================================
# Test 5: Parity Computation Accuracy
# ============================================================================

def test_parity_computation_exact_integers(create_transformer, device):
    """Test MLP computes parity correctly for integer counts"""
    N = 8
    deg = 2
    width = 3
    seed = 700
    
    model, _, _ = create_transformer(N, deg, width, seed=seed)
    
    # Test parity computation for various integer counts
    # The MLP should compute parity exactly at integer values
    mlp = model.mlp
    
    # Create test inputs with known integer counts in COUNT channel
    E = model.E
    count_idx = model.count_idx
    
    # Test for k = 0, 1, 2, ..., deg
    for k in range(deg + 2):
        # Create input with count k/D in COUNT channel
        test_input = torch.zeros(1, 1, E, device=device)
        test_input[0, 0, count_idx] = k / float(deg)
        
        with torch.no_grad():
            output = mlp(test_input)
        
        # Parity should be (-1)^k
        expected_parity = (-1) ** k
        actual_parity = output[0, 0, model.parity_idx].item()
        
        # Allow small tolerance for floating point
        assert abs(actual_parity - expected_parity) < PARITY_TOLERANCE, \
            f"k={k}: parity {actual_parity:.8f} != expected {expected_parity}"


def test_parity_edge_cases(create_transformer, device):
    """Test parity computation for edge cases"""
    N = 8
    deg = 2
    width = 3
    seed = 800
    
    model, _, _ = create_transformer(N, deg, width, seed=seed)
    mlp = model.mlp
    E = model.E
    count_idx = model.count_idx
    parity_idx = model.parity_idx
    
    # Test k=0 (even parity)
    test_input = torch.zeros(1, 1, E, device=device)
    test_input[0, 0, count_idx] = 0.0
    with torch.no_grad():
        output = mlp(test_input)
    assert abs(output[0, 0, parity_idx].item() - 1.0) < PARITY_TOLERANCE, \
        "k=0 should give parity +1"
    
    # Test k=deg (even if deg is even, odd if deg is odd)
    test_input[0, 0, count_idx] = 1.0  # k/D = 1 means k = deg
    with torch.no_grad():
        output = mlp(test_input)
    expected = (-1) ** deg
    assert abs(output[0, 0, parity_idx].item() - expected) < PARITY_TOLERANCE, \
        f"k={deg} should give parity {expected}"


# ============================================================================
# Test 6: Attention Layer Accuracy
# ============================================================================

def test_attention_layer1_count_aggregation(create_transformer, device):
    """Verify attention layer 1 correctly aggregates bits into COUNT channel"""
    N = 8
    deg = 2
    width = 3
    seed = 900
    
    model, _, combs = create_transformer(N, deg, width, seed=seed)
    
    # Create a test input where we know the exact count
    # For a combination S, count should be sum of bits in S
    test_comb = combs[0]
    
    # Create input with bits set according to combination
    x_int = 0
    for idx in test_comb:
        x_int |= (1 << idx)
    
    xs = torch.tensor([x_int], device=device)
    
    # Get intermediate representation after attention 1
    bits = model._ints_to_bits(xs, N)
    dat_bits = model.bit_embed(bits)
    pos_idx = model.pos_idx_base.unsqueeze(0).expand(1, -1)
    pos_vecs = model.pos_embed(pos_idx)
    zeros = torch.zeros(1, N, 3, device=device)
    X0 = torch.cat([pos_vecs, dat_bits, zeros], dim=-1)
    
    Y1, _ = model.attn1(X0, X0, X0)
    X1 = X0 + Y1
    
    # Check COUNT channel at representative position
    rep_idx = model.rep_idx[0]
    count_value = X1[0, rep_idx, model.count_idx].item()
    
    # Expected: k/D where k is the number of bits set in the combination
    expected_count = len(test_comb) / float(deg)
    
    # Allow some tolerance due to softmax approximation
    assert abs(count_value - expected_count) < 0.1, \
        f"COUNT channel value {count_value:.6f} != expected {expected_count:.6f}"


def test_attention_layer2_aggregation(create_transformer, device):
    """Verify attention layer 2 correctly aggregates parities weighted by coefficients"""
    N = 8
    deg = 2
    width = 3
    seed = 1000
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed)
    
    # Test on a known input
    # Create input where we can compute expected output manually
    x_int = 0
    for idx in combs[0]:  # Set bits for first combination
        x_int |= (1 << idx)
    
    xs = torch.tensor([x_int], device=device)
    
    # Compute expected output manually
    # f(x) = sum of c_i * parity(S_i ∩ x)
    expected = 0.0
    for ci, comb in zip(coefs, combs):
        # Count bits in combination
        k = sum(1 for idx in comb if (x_int >> idx) & 1)
        parity = (-1) ** k
        expected += float(ci) * parity
    
    # Get transformer output
    with torch.no_grad():
        transformer_out = model(xs).squeeze(-1).item()
    
    # Compare (allowing for softmax approximation)
    assert abs(transformer_out - expected) < ACCURACY_TOLERANCE, \
        f"Transformer output {transformer_out:.6f} != expected {expected:.6f}"


# ============================================================================
# Test 7: Scaling and Normalization
# ============================================================================

def test_z_sum_coefficients(create_transformer, device):
    """Verify Z = sum(coefs) is computed correctly"""
    N = 10
    deg = 2
    width = 5
    seed = 1100
    
    model, coefs, _ = create_transformer(N, deg, width, seed=seed)
    
    expected_Z = float(coefs.sum().item())
    actual_Z = model.Z
    
    assert abs(actual_Z - expected_Z) < EXACT_TOLERANCE, \
        f"Z = {actual_Z:.6f} != sum(coefs) = {expected_Z:.6f}"


def test_count_channel_scaling(create_transformer, device):
    """Verify COUNT channel scaling is correct (should be 1.0 to get k/D after averaging)"""
    N = 8
    deg = 2
    width = 3
    seed = 1200
    
    model, _, _ = create_transformer(N, deg, width, seed=seed)
    
    # Check attention layer 1 value scaling
    E = model.E
    Wv = model.attn1.in_proj_weight[2*E:, :]
    
    # V should scale BIT channel by 1.0 into COUNT channel
    # After uniform averaging over |S_i| positions: sum(bit_j) / |S_i| = k / D
    expected_scale = 1.0
    actual_scale = Wv[model.count_idx, model.bit_idx].item()
    
    assert abs(actual_scale - expected_scale) < EXACT_TOLERANCE, \
        f"COUNT scaling {actual_scale:.6f} != expected 1.0 (to get k/D after averaging)"


def test_attention_layer2_z_scaling(create_transformer, device):
    """Verify attention layer 2 uses Z scaling correctly"""
    N = 8
    deg = 2
    width = 3
    seed = 1300
    
    model, coefs, _ = create_transformer(N, deg, width, seed=seed)
    
    # Check attention layer 2 value scaling
    E = model.E
    Wv = model.attn2.in_proj_weight[2*E:, :]
    
    # V should scale PARITY channel by Z into AGG channel
    expected_scale = model.Z
    actual_scale = Wv[model.agg_idx, model.parity_idx].item()
    
    assert abs(actual_scale - expected_scale) < EXACT_TOLERANCE, \
        f"AGG scaling {actual_scale:.6f} != expected Z = {expected_scale:.6f}"


# ============================================================================
# Test 8: Edge Cases
# ============================================================================

def test_edge_case_width_1_deg_1(create_transformer, device):
    """Test simplest case: width=1, deg=1"""
    N = 8
    deg = 1
    width = 1
    seed = 1400
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed)
    
    # Verify basic properties
    assert len(coefs) == 1
    assert len(combs) == 1
    assert len(combs[0]) == 1
    
    # Test accuracy
    num_samples = 20
    xs = torch.randint(0, 2**N, (num_samples,), device=device)
    
    with torch.no_grad():
        transformer_out = model(xs).squeeze(-1)
    
    targets = func_batch(xs.cpu().tolist(), coefs.cpu(), combs.cpu(), N).to(device)
    
    max_error = (transformer_out - targets).abs().max().item()
    assert max_error < ACCURACY_TOLERANCE, \
        f"Edge case (width=1, deg=1): error {max_error:.2e}"


def test_edge_case_small_coefficients(create_transformer, device):
    """Test with very small coefficients"""
    N = 8
    deg = 2
    width = 3
    seed = 1500
    
    # Create custom coefficients with very small values
    torch.manual_seed(seed)
    _, combs = rboolf(N, width, deg, seed=seed)
    coefs = torch.tensor([1e-6, 1e-5, 1e-4], dtype=torch.float32)
    coefs = coefs / coefs.sum()  # Normalize
    
    model = HardCodedTransformer(
        N=N,
        combs=combs,
        coefs=coefs,
        aggregator_idx=N,
        mode="original"
    ).to(device).eval()
    
    # Test accuracy
    num_samples = 20
    xs = torch.randint(0, 2**N, (num_samples,), device=device)
    
    with torch.no_grad():
        transformer_out = model(xs).squeeze(-1)
    
    targets = func_batch(xs.cpu().tolist(), coefs.cpu(), combs.cpu(), N).to(device)
    
    max_error = (transformer_out - targets).abs().max().item()
    assert max_error < ACCURACY_TOLERANCE, \
        f"Small coefficients: error {max_error:.2e}"


def test_edge_case_equal_coefficients(create_transformer, device):
    """Test with all coefficients equal"""
    N = 8
    deg = 2
    width = 3
    seed = 1600
    
    torch.manual_seed(seed)
    _, combs = rboolf(N, width, deg, seed=seed)
    coefs = torch.ones(width, dtype=torch.float32) / width  # All equal, normalized
    
    model = HardCodedTransformer(
        N=N,
        combs=combs,
        coefs=coefs,
        aggregator_idx=N,
        mode="original"
    ).to(device).eval()
    
    # Test accuracy
    num_samples = 20
    xs = torch.randint(0, 2**N, (num_samples,), device=device)
    
    with torch.no_grad():
        transformer_out = model(xs).squeeze(-1)
    
    targets = func_batch(xs.cpu().tolist(), coefs.cpu(), combs.cpu(), N).to(device)
    
    max_error = (transformer_out - targets).abs().max().item()
    assert max_error < ACCURACY_TOLERANCE, \
        f"Equal coefficients: error {max_error:.2e}"


# ============================================================================
# Test 9: Consistency Across Modes
# ============================================================================

def test_mode_consistency(create_transformer, device):
    """Verify all modes produce same function (within tolerance)"""
    N = 8
    deg = 2
    width = 3
    seed = 1700
    
    # Create same function for all modes
    torch.manual_seed(seed)
    coefs, combs = rboolf(N, width, deg, seed=seed)
    
    # Only "original" mode is supported (matches mathematical construction)
    model = HardCodedTransformer(
        N=N,
        combs=combs,
        coefs=coefs,
        aggregator_idx=N,
        mode="original"
    ).to(device).eval()
    models = {"original": model}
    
    # Test on random inputs
    num_samples = 50
    torch.manual_seed(999)
    xs = torch.randint(0, 2**N, (num_samples,), device=device)
    
    outputs = {}
    for mode, model in models.items():
        with torch.no_grad():
            outputs[mode] = model(xs).squeeze(-1)
    
    # Compare outputs across modes
    # Note: mlp_soft and balanced may have slight differences at non-integer values
    # but should match at integer counts
    original_out = outputs["original"]
    
    for mode in ["mlp_soft", "balanced"]:
        mode_out = outputs[mode]
        max_diff = (original_out - mode_out).abs().max().item()
        # Allow slightly larger tolerance for soft modes
        assert max_diff < ACCURACY_TOLERANCE * 10, \
            f"Mode {mode} differs from original by {max_diff:.2e}"


# ============================================================================
# Test 10: Fourier Decomposition Verification
# ============================================================================

def test_fourier_decomposition_structure(create_transformer, device):
    """Verify transformer recovers correct Fourier structure"""
    N = 8
    deg = 2
    width = 3
    seed = 1800
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed)
    
    # Verify only specified combinations contribute
    # Test on inputs that activate different combinations
    for i, (ci, comb) in enumerate(zip(coefs, combs)):
        # Create input that only activates this combination
        x_int = 0
        for idx in comb:
            x_int |= (1 << idx)
        
        xs = torch.tensor([x_int], device=device)
        
        with torch.no_grad():
            output = model(xs).squeeze(-1).item()
        
        # Expected: sum of all coefficients times their parities
        expected = 0.0
        for cj, comb_j in zip(coefs, combs):
            k = sum(1 for idx in comb_j if (x_int >> idx) & 1)
            parity = (-1) ** k
            expected += float(cj) * parity
        
        assert abs(output - expected) < ACCURACY_TOLERANCE, \
            f"Combination {i}: output {output:.6f} != expected {expected:.6f}"


def test_no_spurious_interactions(create_transformer, device):
    """Verify no spurious interactions beyond specified combinations"""
    N = 8
    deg = 2
    width = 3
    seed = 1900
    
    model, coefs, combs = create_transformer(N, deg, width, seed=seed)
    
    # Get all combinations used
    used_combinations = set(tuple(sorted(comb)) for comb in combs)
    
    # Test that function only depends on specified combinations
    # by testing inputs that differ only outside these combinations
    test_inputs = []
    for _ in range(10):
        x_int = torch.randint(0, 2**N, (1,), device=device).item()
        test_inputs.append(x_int)
    
    # For each pair of inputs that agree on all specified combinations,
    # the function values should be the same
    for i, x1 in enumerate(test_inputs):
        for j, x2 in enumerate(test_inputs[i+1:], i+1):
            # Check if they agree on all combinations
            agree = True
            for comb in combs:
                bits1 = sum(1 for idx in comb if (x1 >> idx) & 1)
                bits2 = sum(1 for idx in comb if (x2 >> idx) & 1)
                if bits1 % 2 != bits2 % 2:  # Different parities
                    agree = False
                    break
            
            if agree:
                # Function values should be the same
                xs1 = torch.tensor([x1], device=device)
                xs2 = torch.tensor([x2], device=device)
                
                with torch.no_grad():
                    out1 = model(xs1).squeeze(-1).item()
                    out2 = model(xs2).squeeze(-1).item()
                
                assert abs(out1 - out2) < ACCURACY_TOLERANCE, \
                    f"Inputs agreeing on all combinations have different outputs: {out1:.6f} vs {out2:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
