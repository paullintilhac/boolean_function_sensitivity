# hardcoded_transformer.py
import math
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================
# Custom single-head MHA (no W_o), explicit Q/K/V splits
# =============================================================

class CustomMHA(nn.MultiheadAttention):
    """
    Single-head multihead attention where we expose Q/K/V via
    in_proj_weight/in_proj_bias and *do not* apply an output projection.

    We also do *not* apply the usual 1/sqrt(d) scaling in the attention
    logits, to keep the behavior aligned with the theoretical construction.
    """
    def __init__(self, embed_dim, num_heads=1, bias=True,
                 batch_first=True, dropout=0.0):
        super().__init__(embed_dim=embed_dim,
                         num_heads=num_heads,
                         bias=bias,
                         batch_first=batch_first,
                         dropout=dropout)
        # Ensure no out-projection
        self.out_proj = None

    def forward(self, query, key, value):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_out, attn_w = _mha_forward_no_wo(
            query, key, value,
            num_heads=self.num_heads,
            embed_dim_to_check=self.embed_dim,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            dropout_p=self.dropout,
            training=self.training,
            need_weights=True,
        )

        if self.batch_first and is_batched:
            return attn_out.transpose(1, 0), attn_w
        else:
            return attn_out, attn_w


def _mha_forward_no_wo(query, key, value, num_heads, embed_dim_to_check,
                       in_proj_weight, in_proj_bias,
                       dropout_p=0.0, training=True, need_weights=True):
    # Shapes: (L,B,E)
    Lq, B, E = query.shape
    Lk, Bk, Ek = key.shape
    Lv, Bv, Ev = value.shape
    assert B == Bk == Bv
    assert E == Ek == Ev == embed_dim_to_check
    assert Lk == Lv
    head_dim = E // num_heads
    assert head_dim * num_heads == E

    # Q from query, K from key, V from value
    if in_proj_bias is None:
        q = F.linear(query, in_proj_weight[0:E, :], None)
        k = F.linear(key,   in_proj_weight[E:2*E, :], None)
        v = F.linear(value, in_proj_weight[2*E:, :], None)
    else:
        q = F.linear(query, in_proj_weight[0:E, :],    in_proj_bias[0:E])
        k = F.linear(key,   in_proj_weight[E:2*E, :],  in_proj_bias[E:2*E])
        v = F.linear(value, in_proj_weight[2*E:, :],   in_proj_bias[2*E:])

    q = q.reshape(Lq, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lq, D)
    k = k.reshape(Lk, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lk, D)
    v = v.reshape(Lv, B * num_heads, E).transpose(0, 1)         # (B*H, Lk, E)

    if not training:
        dropout_p = 0.0

    # NOTE: no 1/sqrt(d) scaling — matches the theoretical construction
    attn_w = torch.bmm(q, k.transpose(-2, -1))                  # (B*H, Lq, Lk)
    attn_w = F.softmax(attn_w, dim=-1)
    if dropout_p > 0.0:
        attn_w = F.dropout(attn_w, p=dropout_p)

    attn_out = torch.bmm(attn_w, v)                             # (B*H, Lq, E)
    attn_out = attn_out.reshape(B, num_heads, Lq, E).sum(dim=1) # (B, Lq, E)
    attn_out = attn_out.transpose(0, 1)                         # (Lq, B, E)

    if need_weights:
        attn_w = attn_w.reshape(B, num_heads, Lq, Lk).mean(dim=1)  # (B, Lq, Lk)
        return attn_out, attn_w
    else:
        return attn_out, None


# =============================================================
# MLP matching exact mathematical construction
# Interpolates m(x) = 1/2(1 + sin(π(D_f*x + 1/2))) on domain {0, 1/D_f, ..., 1}
# =============================================================

class IntCountParityMLP(nn.Module):
    """
    Exact construction from mathematical specification:
    - Hidden dimension: 4(D_f+1) units
    - M_{i,t} = 0 for all t, i ∈ [T+1]
    - M_{i,t} = 1 for all t, i = T+2 (data channel)
    - Γ_{4i-3} = -h_i - 2/(4D_f)
    - Γ_{4i-2} = -h_i - 1/(4D_f)
    - Γ_{4i-1} = -h_i + 1/(4D_f)
    - Γ_{4i} = -h_i + 2/(4D_f)
    - F_{4i-3,T+2} = 4m(h_i)D_f
    - F_{4i-2,T+2} = -4m(h_i)D_f
    - F_{4i-1,T+2} = -4m(h_i)D_f
    - F_{4i,T+2} = 4m(h_i)D_f
    where h_i ∈ {0, 1/D_f, ..., (D_f-1)/D_f, 1} and m(h_i) = 1/2(1 + sin(π(D_f*h_i + 1/2)))
    """
    def __init__(self, embed_dim, D, data_idx):
        super().__init__()
        self.E, self.D = embed_dim, int(D)
        self.data_idx = data_idx

        # Hidden dimension: 4(D_f + 1) units as per construction
        H = 4 * (self.D + 1)

        self.fc1 = nn.Linear(self.E, H, bias=True)  # M^T and Γ
        self.fc2 = nn.Linear(H, 1, bias=False)      # F^T (no bias in F)
        self.fc_out = nn.Linear(1, self.E, bias=False)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # Initialize M: M_{i,t} = 0 for all t, i ∈ [T+1]; M_{i,t} = 1 for all t, i = T+2
            # In PyTorch Linear: output = input @ weight^T + bias
            # So fc1: H = X @ M^T + Γ, where M^T is fc1.weight and Γ is fc1.bias
            # M^T has shape (H, E), where H = 4(D+1)
            # M_{i,t} = 1 means M^T[unit_i, data_idx] = 1
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            
            # For each h_i value (i ∈ [0, D]), create 4 units
            for i in range(self.D + 1):
                h_i = i / float(self.D)  # h_i ∈ {0, 1/D, 2/D, ..., 1}
                
                # Compute m(h_i) = 1/2(1 + sin(π(D_f*h_i + 1/2)))
                m_h_i = 0.5 * (1.0 + math.sin(math.pi * (self.D * h_i + 0.5)))
                
                # Create 4 units for this h_i value
                base_idx = 4 * i
                
                # M^T: all 4 units read from data channel (data_idx)
                self.fc1.weight[base_idx + 0, self.data_idx] = 1.0  # M^T for unit 4i-3
                self.fc1.weight[base_idx + 1, self.data_idx] = 1.0  # M^T for unit 4i-2
                self.fc1.weight[base_idx + 2, self.data_idx] = 1.0  # M^T for unit 4i-1
                self.fc1.weight[base_idx + 3, self.data_idx] = 1.0  # M^T for unit 4i
                
                # Γ biases
                self.fc1.bias[base_idx + 0] = -h_i - 2.0 / (4.0 * self.D)  # Γ_{4i-3}
                self.fc1.bias[base_idx + 1] = -h_i - 1.0 / (4.0 * self.D)  # Γ_{4i-2}
                self.fc1.bias[base_idx + 2] = -h_i + 1.0 / (4.0 * self.D)  # Γ_{4i-1}
                self.fc1.bias[base_idx + 3] = -h_i + 2.0 / (4.0 * self.D)  # Γ_{4i}
            
            # Initialize F: F_{i,t} = 0 for all i, t ∈ [T+1]
            # F_{4i-3,T+2} = 4m(h_i)D_f, etc.
            # fc2: output = H @ F^T, where F^T is fc2.weight (shape: 1, H)
            self.fc2.weight.zero_()
            
            for i in range(self.D + 1):
                h_i = i / float(self.D)
                m_h_i = 0.5 * (1.0 + math.sin(math.pi * (self.D * h_i + 0.5)))
                
                base_idx = 4 * i
                # F^T weights (F writes to data channel)
                self.fc2.weight[0, base_idx + 0] = 4.0 * m_h_i * self.D  # F_{4i-3,T+2}
                self.fc2.weight[0, base_idx + 1] = -4.0 * m_h_i * self.D  # F_{4i-2,T+2}
                self.fc2.weight[0, base_idx + 2] = -4.0 * m_h_i * self.D  # F_{4i-1,T+2}
                self.fc2.weight[0, base_idx + 3] = 4.0 * m_h_i * self.D  # F_{4i,T+2}
            
            # fc_out: write to data channel only
            self.fc_out.weight.zero_()
            self.fc_out.weight[self.data_idx, 0] = 1.0

    def forward(self, X):
        # Forward pass: g_t = F^T (M^T b_t + Γ)_+
        # b_t is the input X (after attention, data channel contains k_t/D)
        H = F.relu(self.fc1(X))  # (M^T b_t + Γ)_+
        s = self.fc2(H)          # F^T (M^T b_t + Γ)_+
        Y = self.fc_out(s)       # Write to data channel (transforms k/D to parity)
        return X + Y  # Residual preserves bit and position dims for attn2



# =============================================================
# HardCodedTransformer (rewritten to avoid giant negative masks)
# =============================================================

class HardCodedTransformer(nn.Module):
    """
    Hard-coded transformer that implements a parity-based Fourier construction.

    This version is modified to more closely match the theoretical construction
    in which:
      - background logits are 0
      - active logits in layer 1 are 2*log(T)
      - active logits in layer 2 are log(c_t) + 2*log(T)

    There is NO large negative "nonrep_mask" in Q anymore, so the Frobenius
    norms stay moderate and grow more naturally with degree/width.
    """
    def __init__(self, N, combs, coefs, aggregator_idx=None,
                 nonrep_mask=0.0,  # kept for API compatibility, not used
                 mode="original"):
        """
        mode: only "original" is supported (matches mathematical construction)
        """
        super().__init__()
        assert mode == "original", "Only 'original' mode is supported"

        self.N = int(N)
        self.L = int(N + 1)  # sequence length = N+1 (T input positions + 1 CLS token)
        # combs → python lists
        if isinstance(combs, torch.Tensor):
            self.combs = [list(map(int, row.tolist())) for row in combs]
        else:
            self.combs = [list(map(int, row)) for row in combs]
        self.D = max((len(c) for c in self.combs), default=0)

        # store positive Fourier coefficients
        self.coefs = torch.as_tensor(coefs).float().cpu()
        if not (self.coefs > 0).all():
            raise ValueError("All Fourier coefficients must be positive for this initializer.")

        # allocate unique representative positions for each component
        self.rep_idx = self._choose_unique_reps(self.combs, self.N)
        self.aggregator_idx = int(self.N if aggregator_idx is None else aggregator_idx)  # CLS at position T (N)
        self.Z = float(self.coefs.sum().item())

        # channel indices: T+3 total (T+1 positional + 1 bit + 1 data)
        self.bit_idx = self.L      # bit z_t at index T+1
        self.data_idx = self.L + 1  # data channel: bit → k/D → parity → aggregated
        self.E = self.L + 2        # total embedding dim = T+3

        # Embeddings: position one-hot (T+1 dims) + bit embedding
        self.bit_embed = nn.Embedding(2, 1)
        self.pos_embed = nn.Embedding(self.L, self.L)  # (T+1) x (T+1) one-hot
        with torch.no_grad():
            self.bit_embed.weight.copy_(torch.tensor([[0.0], [1.0]], dtype=torch.float32))
            self.pos_embed.weight.copy_(torch.eye(self.L, dtype=torch.float32))
        self.register_buffer("pos_idx_base", torch.arange(self.L, dtype=torch.long))

        # Layers
        self.attn1 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.attn2 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.mlp   = IntCountParityMLP(self.E, self.D, self.data_idx, self.data_idx)

        # initialize attention kernels to match the construction
        self._init_attn1()
        self._init_attn2()

    @staticmethod
    def _choose_unique_reps(combs, N):
        used, reps = set(), []
        for comp in combs:
            chosen = None
            for idx in comp:
                if idx not in used:
                    chosen = idx
                    break
            if chosen is None:
                for idx in range(N):
                    if idx not in used:
                        chosen = idx
                        break
            used.add(chosen)
            reps.append(chosen)
        return reps

    def _init_attn1(self):
        """
        First attention layer:

        - Keys K are just positional one-hots (pass-through on the first L=T+1 channels).
        - Values V add the bit channel into data channel, so after averaging
          over a component we get k/D in data channel.
        - Position dimensions get zeroed (no residual after attn1).
        - Queries Q implement the logit pattern:

              a_{i,j} = 2 log(N)   if j in S_i
                      = 0          otherwise
        """
        E, N, L = self.E, self.N, self.L

        # K: pass-through of position one-hot (T+1 dimensions)
        Wk = torch.zeros(E, E)
        Wk[:L, :L] = torch.eye(L)

        # V: add BIT into data channel -> after averaging over D positions, data = k/D
        Wv = torch.zeros(E, E)
        Wv[self.data_idx, self.bit_idx] = 1.0

        # Q: background logits = 0.
        # For each component S (combs), we pick a representative position t and
        # set Q so that query at t and keys in S_i get logit 2 log(N).
        Wq = torch.zeros(E, E)
        scale = 2.0 * math.log(max(2, N))
        for comp, t in zip(self.combs, self.rep_idx):
            for j in comp:
                # row = key-position j, col = query-position t
                Wq[j, t] = scale

        with torch.no_grad():
            self.attn1.in_proj_weight.zero_()
            self.attn1.in_proj_weight[:E, :].copy_(Wq)
            self.attn1.in_proj_weight[E:2*E, :].copy_(Wk)
            self.attn1.in_proj_weight[2*E:, :].copy_(Wv)
            if self.attn1.in_proj_bias is not None:
                self.attn1.in_proj_bias.zero_()

    def _init_attn2(self):
        """
        Second attention layer:

        - Keys K are again positional one-hots (T+1 dimensions).
        - Values V send data channel (parity) into CLS position's data channel, scaled by Z = sum c_i
          so that the softmax-weighted sum recovers sum c_i * parity_i.
        - Queries Q implement the logit pattern:

              a_{CLS,t} = log(c_t) + 2 log(N)    for representative t
                         = 0                     otherwise

          i.e. softmax over representatives approximates c_t / sum c_l.
        """
        E, N, L = self.E, self.N, self.L

        # K: pass-through of position one-hot (T+1 dimensions)
        Wk = torch.zeros(E, E)
        Wk[:L, :L] = torch.eye(L)

        # V: send data channel (parity) into CLS position's data channel, scaled by Z
        Wv = torch.zeros(E, E)
        # Write to CLS position's data channel
        Wv[self.data_idx, self.data_idx] = self.Z

        # Q: background 0; aggregator (CLS at position T) queries each representative.
        Wq = torch.zeros(E, E)
        base = 2.0 * math.log(max(2, N))
        for ci, t in zip(self.coefs, self.rep_idx):
            Wq[t, self.aggregator_idx] = math.log(max(float(ci), 1e-12)) + base

        with torch.no_grad():
            self.attn2.in_proj_weight.zero_()
            self.attn2.in_proj_weight[:E, :].copy_(Wq)
            self.attn2.in_proj_weight[E:2*E, :].copy_(Wk)
            self.attn2.in_proj_weight[2*E:, :].copy_(Wv)
            if self.attn2.in_proj_bias is not None:
                self.attn2.in_proj_bias.zero_()

    @staticmethod
    def _ints_to_bits(x: torch.Tensor, N: int) -> torch.Tensor:
        device = x.device
        # MPS doesn't support right shift operator, so move to CPU for bit manipulation
        if device.type == 'mps':
            x_cpu = x.cpu()
            shifts = torch.arange(N, dtype=torch.long)  # LSB-first
            bits = ((x_cpu.unsqueeze(-1) >> shifts) & 1).long()
            return bits.to(device)
        else:
            shifts = torch.arange(N, device=device, dtype=torch.long)  # LSB-first
            return ((x.unsqueeze(-1) >> shifts) & 1).long()

    def forward(self, x_ints: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        x_ints = x_ints.to(dev)
        B = x_ints.shape[0]

        # bits: (B,N) in {0,1}
        bits = self._ints_to_bits(x_ints, self.N)
        dat_bits = self.bit_embed(bits)                          # (B,N,1)
        
        # Add CLS token: position T with bit=0
        cls_bit = torch.zeros(B, 1, 1, device=dev)                # CLS token bit = 0
        dat_bits = torch.cat([dat_bits, cls_bit], dim=1)         # (B,N+1,1)
        
        # Positional embeddings: T+1 positions (0 to T)
        pos_idx = self.pos_idx_base.unsqueeze(0).expand(B, -1)   # (B,N+1)
        pos_vecs = self.pos_embed(pos_idx)                       # (B,N+1,N+1)
        
        # Data channel starts empty (will be filled by attn1)
        zeros = torch.zeros(B, self.L, 1, device=dev)            # data channel
        X0 = torch.cat([pos_vecs, dat_bits, zeros], dim=-1)      # (B,N+1,E)

        # attn1 (NO residual): data channel accumulates k/D at representatives
        # Position dims get zeroed (no residual preserves them, but we're removing residual)
        Y1, _ = self.attn1(X0, X0, X0)
        X1 = Y1  # No residual: position dims zeroed, data channel has k/D

        # MLP (with residual): data channel transforms k/D -> parity
        # Residual preserves bit and position dims for attn2
        X2 = self.mlp(X1)

        # attn2 (no residual): aggregate parities at reps into CLS position's data channel
        Y2, _ = self.attn2(X2, X2, X2)
        X3 = Y2

        # read from CLS position's data channel
        return X3[:, self.aggregator_idx, self.data_idx].unsqueeze(-1)

    # ---------------- Q/K rebalancing that PRESERVES logits ----------------
    @torch.no_grad()
    def _rebalance_qk(self, mha: CustomMHA):
        """
        Rescale Q and K so that ||Q|| and ||K|| are balanced, without changing
        the logits Q K^T. This only adjusts norms, not behavior.
        """
        W = mha.in_proj_weight
        E = mha.embed_dim
        Wq = W[:E, :]
        Wk = W[E:2*E, :]

        Aq = float((Wq**2).sum().item()) + 1e-12
        Ak = float((Wk**2).sum().item()) + 1e-12
        a = (Ak / Aq) ** 0.25

        Wq.mul_(a)      # Q <- a Q
        Wk.mul_(1.0/a)  # K <- (1/a) K
        # logits ~ (aQ)·((K)/a)^T == Q·K^T  (unchanged)


# =============================================================
# Utilities + simple smoke test
# =============================================================

def rboolf(N, width, deg, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    coeffs = torch.randn(width).abs()
    coeffs = coeffs / coeffs.pow(2).sum().sqrt()
    # For large N, avoid computing all combinations - generate random ones directly
    # Use range(N) instead of torch.arange(N) for itertools.combinations
    all_combs = list(itertools.combinations(range(N), deg))
    if len(all_combs) <= width:
        # If we have fewer combinations than needed, just use all of them
        combs = torch.tensor(all_combs, dtype=torch.long)
    else:
        # Randomly sample width combinations
        selected_indices = torch.randperm(len(all_combs))[:width]
        combs = torch.tensor([all_combs[i] for i in selected_indices], dtype=torch.long)
    return coeffs, combs

def func_batch(x, coeffs, combs, N):
    x = torch.as_tensor(x, dtype=torch.long)
    coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
    combs = torch.as_tensor(combs, dtype=torch.long)
    shifts = torch.arange(N, dtype=torch.long)      # LSB-first
    bits01 = ((x.unsqueeze(-1) >> shifts) & 1).float()  # (B, N) in {0,1}
    
    # Compute parity for each combination: 1 if even number of 1s, 0 if odd
    comps = []
    for elem in combs:
        sum_bits = bits01[:, tuple(elem.long().tolist())].sum(dim=1)  # Sum of bits in combination
        parity = 1.0 - (sum_bits % 2.0)  # 1 if even, 0 if odd (parity in {0,1})
        comps.append(parity)
    comps = torch.stack(comps, dim=1)
    return comps @ coeffs

if __name__ == "__main__":
    torch.manual_seed(0)
    N = 12
    deg = 3
    width = 3
    num_samples = 128

    coeffs, combs = rboolf(N, width, deg)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xs = torch.randint(0, 2**N, (num_samples,), device=dev)

    # Only "original" mode is supported (matches mathematical construction)
    model = HardCodedTransformer(
        N, combs, coeffs,
        aggregator_idx=N,
        nonrep_mask=0.0,
        mode="original",
    ).to(dev).eval()

    targets = func_batch(xs.cpu().tolist(), coeffs.cpu(), combs.cpu(), N).to(dev)
    out = model(xs).squeeze(-1)
    loss = (out - targets).pow(2).mean().item()
    frob = sum(p.detach().norm().item()**2 for p in model.parameters()) ** 0.5
    print(f"[original] loss: {loss:.3e} | Frobenius weight norm: {frob:.3f}")

