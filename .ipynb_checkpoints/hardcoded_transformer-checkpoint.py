import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import random

# =========================
# MultiheadAttention with identity out-proj (use standard kernel)
# =========================
class CustomMHA(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads=1, bias=True, batch_first=True, dropout=0.0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias,
                         batch_first=batch_first, dropout=dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.out_proj.weight.copy_(torch.eye(embed_dim, dtype=torch.float32))

# =========================
# Integer-Count Parity MLP (COUNT -> PARITY on normalized grid)
# =========================
class IntCountParityMLP(nn.Module):
    """
    Triangular-wave interpolation at grid {0,1/D,...,1}:
      - fc1 computes ramps ReLU((COUNT/D) - (k/D))  (normalizes COUNT by D)
      - fc2 applies second-difference (+1,-2,+1) * v_k  (NO extra *D here)
      - v_k = (-1)^D * (-1)^k  (target = (-1)^(D-k))
      - fc_out writes to PARITY channel only; caller adds residual.
    """
    def __init__(self, embed_dim, D, count_idx, parity_idx):
        super().__init__()
        self.E, self.D = int(embed_dim), int(D)
        self.count_idx, self.parity_idx = int(count_idx), int(parity_idx)

        self.t_list = list(range(-1, self.D + 2))  # -1,0,1,...,D,D+1
        H = len(self.t_list)

        self.fc1 = nn.Linear(self.E, H, bias=True)
        self.fc2 = nn.Linear(H, 1, bias=True)
        self.fc_out = nn.Linear(1, self.E, bias=False)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            invD = 1.0 / float(self.D if self.D > 0 else 1)

            # fc1: ramps in normalized COUNT
            self.fc1.weight.zero_(); self.fc1.bias.zero_()
            for h, t in enumerate(self.t_list):
                kk = min(max(t, 0), self.D)  # clamp edges
                self.fc1.weight[h, self.count_idx] = invD          # COUNT/D
                self.fc1.bias[h] = - float(kk) * invD               # -(k/D)

            # fc2: (+1,-2,+1) * v_k   (no extra *D)
            self.fc2.weight.zero_(); self.fc2.bias.zero_()
            sD = -1.0 if (self.D % 2 == 1) else 1.0   # (-1)^D
            index = {t: i for i, t in enumerate(self.t_list)}
            for k in range(self.D + 1):
                v_k = (1.0 if (k % 2 == 0) else -1.0) * sD  # (-1)^k * (-1)^D = (-1)^(D-k)
                i_m1 = index[k - 1]; i_0 = index[k]; i_p1 = index[k + 1]
                self.fc2.weight[0, i_m1] += +1.0 * v_k
                self.fc2.weight[0, i_0 ] += -2.0 * v_k
                self.fc2.weight[0, i_p1] += +1.0 * v_k

            # write only to PARITY channel
            self.fc_out.weight.zero_()
            self.fc_out.weight[self.parity_idx, 0] = 1.0

    def forward(self, X):  # X: (B,N,E)
        H = F.relu(self.fc1(X))
        s = self.fc2(H)
        Y = self.fc_out(s)  # only PARITY row non-zero
        return Y            # caller will add residual


# =========================
# Hardcoded Transformer with separate channels
# =========================
class HardCodedTransformer(nn.Module):
    """
    Channels: [N pos one-hot] + BIT + COUNT + PARITY + AGG  => E = N + 4

    attn1: for each representative t, non-members hard-masked ONLY in that column;
           members at 2*log(N); V routes BIT->COUNT with factor D so COUNT = integer k at reps.

    MLP: COUNT->PARITY on normalized grid (hats peak at 1 when fed COUNT/D), residual add.

    attn2: aggregator attends only to reps with logits log(c_i)+2*log(N), nonreps hard-masked in
           the aggregator column; V routes PARITY->AGG with gain Z = sum_i c_i (to cancel softmax denom);
           NO residual here.

    Output: AGG channel at aggregator token.
    """
    def __init__(self, N, combs, coefs, aggregator_idx=None, nonrep_mask=-40.0, debug=False):
        super().__init__()
        self.N = int(N); self.L = int(N)
        if isinstance(combs, torch.Tensor):
            self.combs = [list(map(int, row.tolist())) for row in combs]
        else:
            self.combs = [list(map(int, row)) for row in combs]
        self.D = max((len(c) for c in self.combs), default=0)

        self.coefs = torch.as_tensor(coefs).float().cpu()
        if (self.coefs < 0).any():
            raise ValueError("Fourier coefficients (coefs) must be non-negative for this positive-only initializer.")
        self.rep_idx = [c[0] for c in self.combs]
        self.aggregator_idx = int(self.L - 1 if aggregator_idx is None else aggregator_idx)
        self.nonrep_mask = float(nonrep_mask)
        self.Z = float(self.coefs.sum().item())
        self.DEBUG = bool(debug)

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

        # freeze embeddings only
        for p in self.parameters():
            if p is self.bit_embed.weight or p is self.pos_embed.weight:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

    def _init_attn1(self):
        E, N = self.E, self.N
        # Keys: positional identity (first N dims)
        Wk = torch.zeros(E, E); Wk[:N, :N] = torch.eye(N)
        # Values: route BIT -> COUNT with factor D to convert avg to integer count
        Wv = torch.zeros(E, E); Wv[self.count_idx, self.bit_idx] = float(self.D)
        # Queries: per-column mask; members at scale, non-members at mask
        Wq = torch.zeros(E, E)
        scale = 2.0 * math.log(max(2, N))
        for comp in self.combs:
            t = comp[0]
            Wq[:N, t] = self.nonrep_mask
            for j in comp:
                Wq[j, t] = scale

        with torch.no_grad():
            self.attn1.in_proj_weight.zero_()
            self.attn1.in_proj_weight[:E, :].copy_(Wq)
            self.attn1.in_proj_weight[E:2*E, :].copy_(Wk)
            self.attn1.in_proj_weight[2*E:, :].copy_(Wv)
            if self.attn1.in_proj_bias is not None:
                self.attn1.in_proj_bias.zero_()

    def _init_attn2(self):
        E, N = self.E, self.N
        # Keys: positional identity
        Wk = torch.zeros(E, E); Wk[:N, :N] = torch.eye(N)
        # Values: route PARITY -> AGG with **gain Z** to cancel softmax denom
        Wv = torch.zeros(E, E); Wv[self.agg_idx, self.parity_idx] = self.Z
        # Queries: aggregator column — nonreps masked; reps at log(c_i)+2*log(N)
        Wq = torch.zeros(E, E)
        Wq[:N, self.aggregator_idx] = self.nonrep_mask
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

        bits = self._ints_to_bits(x_ints, self.N)                 # (B,N)
        dat_bits = self.bit_embed(bits)                           # (B,N,1) in {0,1}
        pos_idx  = self.pos_idx_base.unsqueeze(0).expand(B, -1)   # (B,N)
        pos_vecs = self.pos_embed(pos_idx)                        # (B,N,N)
        zeros = torch.zeros(B, self.N, 3, device=dev)             # COUNT, PARITY, AGG
        X0 = torch.cat([pos_vecs, dat_bits, zeros], dim=-1)       # (B,N,E)

        # attn1 residual: integer COUNT at rep positions
        Y1, _ = self.attn1(X0, X0, X0)
        X1 = X0 + Y1

        # MLP residual: COUNT -> PARITY exactly at integers
        Ymlp = self.mlp(X1)
        X2 = X1 + Ymlp

        # attn2 (NO residual): aggregate PARITY at reps into AGG at aggregator token
        Y2, _ = self.attn2(X2, X2, X2)
        X3 = Y2

        # final scalar
        return X3[:, self.aggregator_idx, self.agg_idx].unsqueeze(-1)


# =========================
# Utilities + smoke test
# =========================
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
    y = format(x, "b")
    y = ("0"*(N-len(y))) + y
    return [int(z) for z in list(y)]

def func_batch(x, coeffs, combs, N):
    # target parity = Π ((bit - .5)*2) ⇒ (-1)^(D-k)
    binaryTensor = ((torch.tensor([makeBitTensor(y,N) for y in x])-.5)*2)
    comps = []
    for elem in combs:
        res = torch.ones(len(x))
        for e in elem:
            res = res * binaryTensor[:, e]
        comps.append(res)
    comps = torch.transpose(torch.stack(comps),1,0)
    return torch.matmul(comps, coeffs)

if __name__ == "__main__":
    torch.manual_seed(0)
    N = 12
    D = 3
    width = 3
    num_samples = 16
    coefs, combs = rboolf(N, width, D)
    model = HardCodedTransformer(N, combs, coefs, aggregator_idx=N-1)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    model.eval()
    inputs = torch.tensor([random.randint(0, 2**N-1) for _ in range(num_samples)], device=dev)
    targets = func_batch(inputs.cpu().tolist(), coefs.cpu(), combs.cpu(), N).to(dev)
    result = model(inputs)
    print("targets:", targets[:10])
    print("result:", result[:10])
    loss = (result.squeeze() - targets).pow(2).mean()
    print("loss:", float(loss))
