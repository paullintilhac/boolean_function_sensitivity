
# hardcoded_transformer_no_wo.py
import math
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================
# Custom single-head MHA (no out_proj) with explicit Q/K/V splits
# (No sqrt(d) scaling; mirrors the "working" construction.)
# =============================================================
class CustomMHA(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads=1, bias=True, batch_first=True, dropout=0.0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias,
                         batch_first=batch_first, dropout=dropout)
        # Explicitly remove W_o / out_proj usage
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

    # Q from query, K from key, V from value (stacked in-proj weights)
    if in_proj_bias is None:
        q = F.linear(query, in_proj_weight[0:E, :], None)
        k = F.linear(key,   in_proj_weight[E:2*E, :], None)
        v = F.linear(value, in_proj_weight[2*E: , :], None)
    else:
        q = F.linear(query, in_proj_weight[0:E, :],    in_proj_bias[0:E])
        k = F.linear(key,   in_proj_weight[E:2*E, :],  in_proj_bias[E:2*E])
        v = F.linear(value, in_proj_weight[2*E: , :],  in_proj_bias[2*E:])

    # (B*H, L, D) and (B*H, L, E)
    q = q.reshape(Lq, B * num_heads, head_dim).transpose(0, 1)
    k = k.reshape(Lk, B * num_heads, head_dim).transpose(0, 1)
    v = v.reshape(Lv, B * num_heads, E).transpose(0, 1)

    if not training:
        dropout_p = 0.0

    # NOTE: No 1/sqrt(d) scaling here (mirrors your working kernel)
    attn_w = torch.bmm(q, k.transpose(-2, -1))   # (B*H, Lq, Lk)
    attn_w = F.softmax(attn_w, dim=-1)
    if dropout_p > 0.0:
        attn_w = F.dropout(attn_w, p=dropout_p)

    attn_out = torch.bmm(attn_w, v)              # (B*H, Lq, E)
    attn_out = attn_out.reshape(B, num_heads, Lq, E).sum(dim=1).transpose(0, 1)  # (Lq,B,E)

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
    fc1 builds ramps r_t(x)=ReLU(x - t) for t in {-1,0,1,...,D,D+1}.
    fc2 computes y = sum_k v_k * (r_{k-1} - 2 r_k + r_{k+1}), with v_k = (-1)^(D-k).
    fc_out writes y into PARITY channel; we add residual inside.
    """
    def __init__(self, embed_dim, D, count_idx, parity_idx, values=None):
        super().__init__()
        self.E, self.D = embed_dim, int(D)
        self.count_idx, self.parity_idx = count_idx, parity_idx

        if values is None:
            vals = torch.tensor([1.0 if ((self.D - k) % 2 == 0) else -1.0 for k in range(self.D + 1)],
                                dtype=torch.float32)
        else:
            assert values.numel() == self.D + 1
            vals = values.float()
        self.register_buffer("grid_vals", vals)

        self.t_list = list(range(-1, self.D + 2))  # -1,0,...,D,D+1
        H = len(self.t_list)

        self.fc1 = nn.Linear(self.E, H, bias=True)
        self.fc2 = nn.Linear(H, 1, bias=True)
        self.fc_out = nn.Linear(1, self.E, bias=False)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # fc1 ramps along COUNT
            self.fc1.weight.zero_(); self.fc1.bias.zero_()
            for h, t in enumerate(self.t_list):
                self.fc1.weight[h, self.count_idx] = 1.0
                self.fc1.bias[h] = -float(t)

            # fc2: second difference with v_k
            self.fc2.weight.zero_(); self.fc2.bias.zero_()
            index = {t: i for i, t in enumerate(self.t_list)}
            for k in range(self.D + 1):
                v_k = float(self.grid_vals[k])
                i_m1 = index[k - 1]; i_0 = index[k]; i_p1 = index[k + 1]
                self.fc2.weight[0, i_m1] += +1.0 * v_k
                self.fc2.weight[0, i_0 ] += -2.0 * v_k
                self.fc2.weight[0, i_p1] += +1.0 * v_k

            # write only to PARITY channel
            self.fc_out.weight.zero_()
            self.fc_out.weight[self.parity_idx, 0] = 1.0

    def forward(self, X):
        H = F.relu(self.fc1(X))
        s = self.fc2(H)
        Y = self.fc_out(s)
        return X + Y


# =============================================================
# HardCodedTransformerPos (all-positive coefs), D inferred from combs
# Channels: [pos_onehot N] + BIT + COUNT + PARITY + AGG
# =============================================================
class HardCodedTransformerPos(nn.Module):
    def __init__(self, N, combs, coefs, aggregator_idx=None, nonrep_mask=-40.0):
        super().__init__()
        self.N = int(N); self.L = int(N)
        # Infer D from comb sizes
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

        # Embeddings
        self.bit_embed = nn.Embedding(2, 1)
        with torch.no_grad():
            self.bit_embed.weight.copy_(torch.tensor([[0.0],[1.0]], dtype=torch.float32))
        self.bit_embed.weight.requires_grad = False

        self.pos_embed = nn.Embedding(self.L, self.L)
        with torch.no_grad():
            self.pos_embed.weight.copy_(torch.eye(self.L, dtype=torch.float32))
        self.pos_embed.weight.requires_grad = False
        self.register_buffer("pos_idx_base", torch.arange(self.L, dtype=torch.long))

        # Layers
        self.attn1 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.attn2 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.mlp = IntCountParityMLP(self.E, self.D, self.count_idx, self.parity_idx)

        self._init_attn1()
        self._init_attn2()

        for p in self.parameters():
            if p is self.bit_embed.weight or p is self.pos_embed.weight:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

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
        Wk = torch.zeros(E, E); Wk[:N, :N] = torch.eye(N)
        # BIT -> COUNT scaled by D so COUNT equals integer k
        Wv = torch.zeros(E, E); Wv[self.count_idx, self.bit_idx] = float(self.D)
        # Full-column masking; members at 2*log N
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
        Wk = torch.zeros(E, E); Wk[:N, :N] = torch.eye(N)
        # PARITY -> AGG scaled by Z to cancel softmax normalization
        Wv = torch.zeros(E, E); Wv[self.agg_idx, self.parity_idx] = self.Z
        # Nonrep masked; reps at log(ci) + 2*log N
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

        # MLP (residual inside): COUNT -> PARITY
        X2 = self.mlp(X1)

        # attn2 (no residual): aggregate parities at reps into AGG
        Y2, _ = self.attn2(X2, X2, X2)
        X3 = Y2

        return X3[:, self.aggregator_idx, self.agg_idx].unsqueeze(-1)


# ========================== Utilities + demo ==========================
def rboolf(N, width, deg, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    coeffs = torch.randn(width).abs()
    coeffs = coeffs / coeffs.pow(2).sum().sqrt()
    combs = torch.tensor(list(itertools.combinations(torch.arange(N), deg)))
    combs = combs[torch.randperm(len(combs))][:width]
    print("coefficients:", coeffs)
    print("combs:", combs)
    return coeffs, combs

def makeBitTensor(x, N):
    # LSB-first to match model
    return [int((int(x) >> j) & 1) for j in range(N)]

def func_batch(x, coeffs, combs, N):
    x = torch.as_tensor(x, dtype=torch.long)
    shifts = torch.arange(N, dtype=torch.long)
    bits01 = ((x.unsqueeze(-1) >> shifts) & 1).float()     # (B, N) in {0,1}
    bin_pm = (bits01 - 0.5) * 2.0                          # {-1, +1}
    comps = [bin_pm[:, tuple(elem.long().tolist())].prod(dim=1) for elem in combs]
    comps = torch.stack(comps, dim=1)                      # (B, width)
    return comps @ coeffs

if __name__ == "__main__":
    torch.manual_seed(0)
    N = 12; deg = 3; width = 3
    num_samples = 64
    coeffs, combs = rboolf(N, width, deg)
    model = HardCodedTransformerPos(N, combs, coeffs, aggregator_idx=N-1)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    inputs = torch.randint(0, 2**N, (num_samples,), device=dev)
    targets = func_batch(inputs.cpu().tolist(), coeffs.cpu(), combs.cpu(), N).to(dev)
    out = model(inputs).squeeze(-1)
    print("targets[:10]:", targets[:10])
    print("out[:10]:", out[:10])
    mse = (out - targets).pow(2).mean().item()
    print("loss:", mse)
