# hardcoded_transformer_slope_quarter.py
import math
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Single-head MHA w/out W_o
# ---------------------------
class CustomMHA(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads=1, bias=True, batch_first=True, dropout=0.0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias,
                         batch_first=batch_first, dropout=dropout)
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
    Lq, B, E = query.shape
    Lk, Bk, Ek = key.shape
    Lv, Bv, Ev = value.shape
    assert B == Bk == Bv and E == Ek == Ev == embed_dim_to_check and Lk == Lv
    head_dim = E // num_heads
    assert head_dim * num_heads == E

    if in_proj_bias is None:
        q = F.linear(query, in_proj_weight[0:E, :], None)
        k = F.linear(key,   in_proj_weight[E:2*E, :], None)
        v = F.linear(value, in_proj_weight[2*E: , :], None)
    else:
        q = F.linear(query, in_proj_weight[0:E, :],    in_proj_bias[0:E])
        k = F.linear(key,   in_proj_weight[E:2*E, :],  in_proj_bias[E:2*E])
        v = F.linear(value, in_proj_weight[2*E: , :],  in_proj_bias[2*E:])

    q = q.reshape(Lq, B * num_heads, head_dim).transpose(0, 1)
    k = k.reshape(Lk, B * num_heads, head_dim).transpose(0, 1)
    v = v.reshape(Lv, B * num_heads, E).transpose(0, 1)

    if not training:
        dropout_p = 0.0

    logits = torch.bmm(q, k.transpose(-2, -1))   # no 1/sqrt(d) scaling (matches your working code)
    attn_w = F.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        attn_w = F.dropout(attn_w, p=dropout_p)

    attn_out = torch.bmm(attn_w, v)              # (B*H, Lq, E)
    attn_out = attn_out.reshape(B, num_heads, Lq, E).sum(dim=1).transpose(0, 1)  # (Lq,B,E)

    if need_weights:
        attn_w = attn_w.reshape(B, num_heads, Lq, Lk).mean(dim=1)  # (B, Lq, Lk)
        return attn_out, attn_w
    else:
        return attn_out, None


# ---------------------------------------------------------
# Trapezoid MLP on normalized COUNT s = k/D in [0,1]
# Slopes ±1/4 and plateau half-width 1/(4D); scaled by 16D
# so output at grid s=k/D is exactly ±1.
# ---------------------------------------------------------
class IntCountParityMLP_Trapezoid(nn.Module):
    """
    We build, for each k in {0..D}, a trapezoid centered at s=k/D with:
      - plateau radius w = 1/(4D)
      - slopes of magnitude slope = 1/4 on the flanks
    A trapezoid can be written via ReLU basis as:
      T_k(s) = c * [ (s - a)_+ - (s - b)_+ - (s - d)_+ + (s - e)_+ ]
    where a=c-2w, b=c-w, d=c+w, e=c+2w, c=k/D, and c (here) is a coefficient (poor reuse of 'c'),
    to produce derivative pattern +slope on [a,b], 0 on [b,d], -slope on [d,e], 0 outside.

    Height on the plateau equals c * (b - a) = c * w.
    We want ±1 at s=k/D, so choose c = (±1)/w. But that would make slopes ±1/w = 4D (too large).
    Instead, we *fix slopes* to ±1/4 by setting c = slope = 1/4, which gives plateau height (1/4)*w = 1/(16D).
    To still get ±1 on-grid, we multiply the *sum* by AMP = 16D.

    Net effect:
      - slopes in s remain ±1/4
      - plateau at s=k/D becomes ±1 after multiplying by AMP
    """
    def __init__(self, embed_dim, D, count_idx, parity_idx, slope=0.25):
        super().__init__()
        self.E, self.D = embed_dim, int(D)
        self.count_idx, self.parity_idx = count_idx, parity_idx
        self.slope = float(slope)
        self.w = 1.0 / (4.0 * max(1, self.D))   # plateau half-width
        self.AMP = 1.0 / (self.slope * self.w)  # = 16D when slope=1/4

        H = 4 * (self.D + 1)  # 4 ReLU basis per k
        self.fc1 = nn.Linear(self.E, H, bias=True)
        self.fc2 = nn.Linear(H, 1, bias=True)
        self.fc_out = nn.Linear(1, self.E, bias=False)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # fc1: ReLU(s - t) along COUNT; we treat COUNT as s in [0,1].
            self.fc1.weight.zero_(); self.fc1.bias.zero_()
            idx = 0
            for k in range(self.D + 1):
                c = k / max(1, self.D)
                a = c - 2*self.w
                b = c - self.w
                d = c + self.w
                e = c + 2*self.w
                for t in (a, b, d, e):
                    self.fc1.weight[idx, self.count_idx] = 1.0   # ReLU(s - t)
                    self.fc1.bias[idx] = -float(t)
                    idx += 1

            # fc2: combine as [+1, -1, -1, +1] * slope, then multiply by AMP and parity sign
            self.fc2.weight.zero_(); self.fc2.bias.zero_()
            idx = 0
            for k in range(self.D + 1):
                v_k = 1.0 if ((self.D - k) % 2 == 0) else -1.0   # desired parity at s=k/D
                # base trapezoid with slopes ±slope
                coeffs = torch.tensor([+self.slope, -self.slope, -self.slope, +self.slope])
                # scale to make plateau = ±1 (via AMP)
                coeffs = coeffs * (self.AMP * v_k)
                self.fc2.weight[0, idx:idx+4].copy_(coeffs)
                idx += 4

            # write only to PARITY channel
            self.fc_out.weight.zero_()
            self.fc_out.weight[self.parity_idx, 0] = 1.0

    def forward(self, X):
        # interpret COUNT channel as normalized s \in [0,1]
        H = F.relu(self.fc1(X))
        s = self.fc2(H)
        Y = self.fc_out(s)
        return X + Y


# ---------------------------------------------------------
# Hard-coded transformer using normalized COUNT (s=k/D)
# and trapezoid MLP (slopes ±1/4), no W_o anywhere.
# ---------------------------------------------------------
class HardCodedTransformer(nn.Module):
    def __init__(self, N, combs, coefs, aggregator_idx=None, nonrep_mask=-40.0):
        super().__init__()
        self.N = int(N); self.L = int(N)
        if isinstance(combs, torch.Tensor):
            self.combs = [list(map(int, row.tolist())) for row in combs]
        else:
            self.combs = [list(map(int, row)) for row in combs]
        # degree D inferred from combs
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

        # fixed embeddings
        self.bit_embed = nn.Embedding(2, 1)
        with torch.no_grad():
            self.bit_embed.weight.copy_(torch.tensor([[0.0],[1.0]], dtype=torch.float32))
        self.bit_embed.weight.requires_grad = False

        self.pos_embed = nn.Embedding(self.L, self.L)
        with torch.no_grad():
            self.pos_embed.weight.copy_(torch.eye(self.L, dtype=torch.float32))
        self.pos_embed.weight.requires_grad = False
        self.register_buffer("pos_idx_base", torch.arange(self.L, dtype=torch.long))

        # layers
        self.attn1 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.attn2 = CustomMHA(embed_dim=self.E, num_heads=1, batch_first=True)
        self.mlp = IntCountParityMLP_Trapezoid(self.E, self.D, self.count_idx, self.parity_idx, slope=0.25)

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
        # K: identity on position one-hots
        Wk = torch.zeros(E, E); Wk[:N, :N] = torch.eye(N)
        # V: BIT -> COUNT, **normalized** (mean), so just copy the bit (softmax will average over members)
        Wv = torch.zeros(E, E); Wv[self.count_idx, self.bit_idx] = 1.0  # NOTE: was D before; now 1.0
        # Q: reps strongly focus only on their component members
        Wq = torch.full((E, E), self.nonrep_mask)
        scale = 2.0 * math.log(max(2, N))
        for comp, t in zip(self.combs, self.rep_idx):
            for j in comp:
                Wq[j, t] = scale  # row = key pos j, col = query pos t

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
        # V: PARITY -> AGG; multiply by Z to convert softmax-weighted average to sum by coefficients
        Wv = torch.zeros(E, E); Wv[self.agg_idx, self.parity_idx] = self.Z
        # Q: aggregator attends to reps with logits ~ log(c_i) + 2logN
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

        # attn1: representative queries average over their component members -> COUNT ≈ k/D
        Y1, _ = self.attn1(X0, X0, X0)
        X1 = X0 + Y1

        # MLP: trapezoids with slopes ±1/4; internally scaled to hit ±1 at s=k/D
        X2 = self.mlp(X1)

        # attn2: aggregate parities at reps into AGG; multiply by Z to undo softmax normalization
        Y2, _ = self.attn2(X2, X2, X2)
        X3 = Y2

        return X3[:, self.aggregator_idx, self.agg_idx].unsqueeze(-1)


# ---------------------------
# Utilities + smoke test
# ---------------------------
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
    bits01 = ((x.unsqueeze(-1) >> shifts) & 1).float()    # (B, N) in {0,1}
    bin_pm = (bits01 - 0.5) * 2.0                         # {-1, +1}
    comps = [bin_pm[:, tuple(elem.long().tolist())].prod(dim=1) for elem in combs]
    comps = torch.stack(comps, dim=1)                     # (B, width)
    return comps @ coeffs

if __name__ == "__main__":
    torch.manual_seed(0)
    N = 12; deg = 3; width = 3
    num_samples = 256
    coeffs, combs = rboolf(N, width, deg)
    model = HardCodedTransformer(N, combs, coeffs, aggregator_idx=N-1)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    inputs = torch.randint(0, 2**N, (num_samples,), device=dev)
    targets = func_batch(inputs.cpu().tolist(), coeffs.cpu(), combs.cpu(), N).to(dev)
    out = model(inputs).squeeze(-1)
    mse = (out - targets).pow(2).mean().item()
    print("loss:", mse)
