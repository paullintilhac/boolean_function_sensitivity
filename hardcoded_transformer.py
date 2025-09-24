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
    def __init__(self, embed_dim, num_heads=1, bias=True, batch_first=True, dropout=0.0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias,
                         batch_first=batch_first, dropout=dropout)
        # ensure no out-proj is used
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
            dropout_p=self.dropout, training=self.training,
            need_weights=True
        )
        if self.batch_first and is_batched:
            return attn_out.transpose(1, 0), attn_w
        else:
            return attn_out, attn_w


def _mha_forward_no_wo(query, key, value, num_heads, embed_dim_to_check,
                       in_proj_weight, in_proj_bias, dropout_p=0.0, training=True,
                       need_weights=True):
    # Shapes: (L,B,E)
    Lq, B, E = query.shape
    Lk, Bk, Ek = key.shape
    Lv, Bv, Ev = value.shape
    assert B == Bk == Bv and E == Ek == Ev == embed_dim_to_check and Lk == Lv
    head_dim = E // num_heads
    assert head_dim * num_heads == E

    # Q from query, K from key, V from value
    if in_proj_bias is None:
        q = F.linear(query, in_proj_weight[0:E, :], None)
        k = F.linear(key,   in_proj_weight[E:2*E, :], None)
        v = F.linear(value, in_proj_weight[2*E: , :], None)
    else:
        q = F.linear(query, in_proj_weight[0:E, :],    in_proj_bias[0:E])
        k = F.linear(key,   in_proj_weight[E:2*E, :],  in_proj_bias[E:2*E])
        v = F.linear(value, in_proj_weight[2*E: , :],  in_proj_bias[2*E:])

    q = q.reshape(Lq, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lq, D)
    k = k.reshape(Lk, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lk, D)
    v = v.reshape(Lv, B * num_heads, E).transpose(0, 1)         # (B*H, Lk, E)

    if not training:
        dropout_p = 0.0

    # no 1/sqrt(d) scaling to match your working construction
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
# Integer-Count Parity MLP (exact at integer counts)
# =============================================================
class IntCountParityMLP(nn.Module):
    """
    COUNT channel holds integer k in [0..D].
    fc1 builds ramps r_t(x) = ReLU(x - t) for t in {-1,0,1,...,D,D+1}.
    fc2 computes y = sum_k v_k * (r_{k-1} - 2 r_k + r_{k+1}), with v_k = (-1)^(D-k).
    fc_out writes y into PARITY channel; residual is added in the caller.
    """
    def __init__(self, embed_dim, D, count_idx, parity_idx):
        super().__init__()
        self.E, self.D = embed_dim, int(D)
        self.count_idx, self.parity_idx = count_idx, parity_idx

        vals = torch.tensor([1.0 if ((self.D - k) % 2 == 0) else -1.0 for k in range(self.D + 1)],
                            dtype=torch.float32)
        self.register_buffer("grid_vals", vals)

        self.t_list = list(range(-1, self.D + 2))  # -1,0,...,D,D+1
        H = len(self.t_list)

        self.fc1 = nn.Linear(self.E, H, bias=True)
        self.fc2 = nn.Linear(H, 1, bias=True)
        self.fc_out = nn.Linear(1, self.E, bias=False)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # ramps along COUNT
            self.fc1.weight.zero_(); self.fc1.bias.zero_()
            for h, t in enumerate(self.t_list):
                self.fc1.weight[h, self.count_idx] = 1.0
                self.fc1.bias[h] = -float(t)

            # second difference with v_k
            self.fc2.weight.zero_(); self.fc2.bias.zero_()
            index = {t: i for i, t in enumerate(self.t_list)}
            for k in range(self.D + 1):
                v_k = float(self.grid_vals[k])
                i_m1 = index[k - 1]; i_0 = index[k]; i_p1 = index[k + 1]
                self.fc2.weight[0, i_m1] += +1.0 * v_k
                self.fc2.weight[0, i_0 ] += -2.0 * v_k
                self.fc2.weight[0, i_p1] += +1.0 * v_k

            # write only to PARITY
            self.fc_out.weight.zero_()
            self.fc_out.weight[self.parity_idx, 0] = 1.0

    def forward(self, X):
        H = F.relu(self.fc1(X))
        s = self.fc2(H)
        Y = self.fc_out(s)
        return X + Y

    # ---- exact slope softening that preserves outputs at integers ----
    @torch.no_grad()
    def soften_slopes(self, s: float):
        """
        Multiply the COUNT slope inside fc1 by 's' (<1 softens, >1 sharpens),
        while dividing fc2.weight by 's' so the *integer* outputs remain exact.
        This keeps hinge locations (x=t) fixed and preserves parity amplitude.
        """
        # scale count column + matching bias to keep hinges at integers
        self.fc1.weight[:, :] *= 1.0
        self.fc1.weight[:, self.count_idx] *= s
        self.fc1.bias[:] *= 0.0  # we’ll re-set biases to keep hinges at integers
        for h, t in enumerate(self.t_list):
            self.fc1.bias[h] = -float(t) * s  # hinge at x=t unchanged

        # scale fc2 weights inversely to preserve integer outputs
        self.fc2.weight[:] /= s


# =============================================================
# HardCodedTransformer with three modes:
#   - "original": exact construction (baseline)
#   - "mlp_soft": soften MLP slopes by factor s<1, preserving exactness at integers
#   - "balanced": mlp_soft + Q/K norm balancing that preserves logits
# =============================================================
class HardCodedTransformer(nn.Module):
    def __init__(self, N, combs, coefs, aggregator_idx=None, nonrep_mask=-40.0,
                 mode="original", mlp_soft_factor=0.25):
        """
        mode:
          - "original"
          - "mlp_soft"
          - "balanced"
        mlp_soft_factor: s in (0,1] used by "mlp_soft" and "balanced"
        """
        super().__init__()
        assert mode in ("original", "mlp_soft", "balanced")
        self.mode = mode
        self.soft_s = float(mlp_soft_factor)

        self.N = int(N); self.L = int(N)
        # combs → python lists
        if isinstance(combs, torch.Tensor):
            self.combs = [list(map(int, row.tolist())) for row in combs]
        else:
            self.combs = [list(map(int, row)) for row in combs]
        self.D = max((len(c) for c in self.combs), default=0)

        self.nonrep_mask = float(nonrep_mask)
        self.coefs = torch.as_tensor(coefs).float().cpu()
        if not (self.coefs > 0).all():
            raise ValueError("All Fourier coefficients must be positive for this initializer.")

        self.rep_idx = self._choose_unique_reps(self.combs, self.N)
        self.aggregator_idx = int(self.L - 1 if aggregator_idx is None else aggregator_idx)
        self.Z = float(self.coefs.sum().item())

        # channel indices
        self.bit_idx   = self.L
        self.count_idx = self.L + 1
        self.parity_idx= self.L + 2
        self.agg_idx   = self.L + 3
        self.E = self.L + 4

        # Embeddings (trainable but initialized to desired values)
        self.bit_embed = nn.Embedding(2, 1)
        self.pos_embed = nn.Embedding(self.L, self.L)
        with torch.no_grad():
            self.bit_embed.weight.copy_(torch.tensor([[0.0],[1.0]], dtype=torch.float32))
            self.pos_embed.weight.copy_(torch.eye(self.L, dtype=torch.float32))
        self.register_buffer("pos_idx_base", torch.arange(self.L, dtype=torch.long))

        # Layers
        self.attn1 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.attn2 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.mlp = IntCountParityMLP(self.E, self.D, self.count_idx, self.parity_idx)

        # initialize attention kernels
        self._init_attn1()
        self._init_attn2()

        # optional softening / balancing
        if self.mode in ("mlp_soft", "balanced"):
            # exact-preserving slope softening
            self.mlp.soften_slopes(self.soft_s)
        if self.mode == "balanced":
            # rebalance Q/K in both layers while *preserving* logits
            self._rebalance_qk(self.attn1)
            self._rebalance_qk(self.attn2)

    @staticmethod
    def _choose_unique_reps(combs, N):
        used, reps = set(), []
        for comp in combs:
            chosen = None
            for idx in comp:
                if idx not in used:
                    chosen = idx; break
            if chosen is None:
                for idx in range(N):
                    if idx not in used:
                        chosen = idx; break
            used.add(chosen)
            reps.append(chosen)
        return reps

    def _init_attn1(self):
        E, N = self.E, self.N
        # K: pass-through of position one-hot
        Wk = torch.zeros(E, E); Wk[:N, :N] = torch.eye(N)
        # V: add BIT into COUNT scaled by D -> COUNT = integer k after averaging
        Wv = torch.zeros(E, E); Wv[self.count_idx, self.bit_idx] = float(self.D)
        # Q: mask non-members, enable members for each representative
        Wq = torch.full((E, E), self.nonrep_mask)
        scale = 2.0 * math.log(max(2, N))
        for comp, t in zip(self.combs, self.rep_idx):
            for j in comp:
                Wq[j, t] = scale  # row=key j, col=query-pos t

        with torch.no_grad():
            self.attn1.in_proj_weight.zero_()
            self.attn1.in_proj_weight[:E, :].copy_(Wq)
            self.attn1.in_proj_weight[E:2*E, :].copy_(Wk)
            self.attn1.in_proj_weight[2*E:, :].copy_(Wv)
            if self.attn1.in_proj_bias is not None:
                self.attn1.in_proj_bias.zero_()

    def _init_attn2(self):
        E, N = self.E, self.N
        # K: pass-through of position one-hot
        Wk = torch.zeros(E, E); Wk[:N, :N] = torch.eye(N)
        # V: send PARITY into AGG scaled by Z to cancel softmax
        Wv = torch.zeros(E, E); Wv[self.agg_idx, self.parity_idx] = self.Z
        # Q: aggregator queries each representative with log(ci)+2logN, others masked
        Wq = torch.full((E, E), self.nonrep_mask)
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
        shifts = torch.arange(N, device=device, dtype=torch.long)  # LSB-first
        return ((x.unsqueeze(-1) >> shifts) & 1).long()

    def forward(self, x_ints: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        x_ints = x_ints.to(dev)
        B = x_ints.shape[0]

        bits = self._ints_to_bits(x_ints, self.N)                # (B,N)
        dat_bits = self.bit_embed(bits)                          # (B,N,1)
        pos_idx  = self.pos_idx_base.unsqueeze(0).expand(B, -1)  # (B,N)
        pos_vecs = self.pos_embed(pos_idx)                       # (B,N,N)
        zeros = torch.zeros(B, self.N, 3, device=dev)            # COUNT, PARITY, AGG
        X0 = torch.cat([pos_vecs, dat_bits, zeros], dim=-1)      # (B,N,E)

        # attn1 (residual): COUNT = integer k at reps
        Y1, _ = self.attn1(X0, X0, X0)
        X1 = X0 + Y1

        # MLP (residual inside): COUNT -> PARITY (exact at integers)
        X2 = self.mlp(X1)

        # attn2 (no residual): aggregate parities at reps into AGG
        Y2, _ = self.attn2(X2, X2, X2)
        X3 = Y2

        return X3[:, self.aggregator_idx, self.agg_idx].unsqueeze(-1)

    # ---------------- Q/K rebalancing that PRESERVES logits ----------------
    @torch.no_grad()
    def _rebalance_qk(self, mha: CustomMHA):
        W = mha.in_proj_weight
        E = mha.embed_dim
        Wq = W[:E, :]
        Wk = W[E:2*E, :]

        # minimize ||a Wq||^2 + ||(1/a) Wk||^2  -> a = (||Wk||^2 / ||Wq||^2)^(1/4)
        Aq = float((Wq**2).sum().item()) + 1e-12
        Ak = float((Wk**2).sum().item()) + 1e-12
        a = (Ak / Aq) ** 0.25

        Wq.mul_(a)      # Q <- a Q
        Wk.mul_(1.0/a)  # K <- (1/a) K
        # logits ~ (aQ)·((K)/a)^T == Q·K^T  (unchanged)
        # No need to touch V or biases
# =============================================================
# Utilities + smoke test
# =============================================================
def rboolf(N, width, deg, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    coeffs = torch.randn(width).abs()
    coeffs = coeffs / coeffs.pow(2).sum().sqrt()
    combs = torch.tensor(list(itertools.combinations(torch.arange(N), deg)))
    combs = combs[torch.randperm(len(combs))][:width]
    return coeffs, combs

def func_batch(x, coeffs, combs, N):
    x = torch.as_tensor(x, dtype=torch.long)
    shifts = torch.arange(N, dtype=torch.long)      # LSB-first
    bits01 = ((x.unsqueeze(-1) >> shifts) & 1).float()
    bin_pm = (bits01 - 0.5) * 2.0
    comps = [bin_pm[:, tuple(elem.long().tolist())].prod(dim=1) for elem in combs]
    comps = torch.stack(comps, dim=1)
    return comps @ coeffs

if __name__ == "__main__":
    torch.manual_seed(0)
    N = 12; deg = 3; width = 3
    num_samples = 128

    coeffs, combs = rboolf(N, width, deg)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xs = torch.randint(0, 2**N, (num_samples,), device=dev)

    for mode in ["original", "mlp_soft", "balanced"]:
        model = HardCodedTransformer(N, combs, coeffs, aggregator_idx=N-1,
                                     nonrep_mask=-40.0, mode=mode, mlp_soft_factor=0.25).to(dev).eval()
        targets = func_batch(xs.cpu().tolist(), coeffs.cpu(), combs.cpu(), N).to(dev)
        out = model(xs).squeeze(-1)
        loss = (out - targets).pow(2).mean().item()
        print(f"[{mode}] loss: {loss:.3e}")
