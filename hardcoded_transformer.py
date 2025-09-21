import math
import torch
from torch import nn
import torch.nn.functional as F

# ------------------- Minimal single-head MHA (no out-proj, no extra temp) -------------------
class CustomMHA(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads=1, bias=True, batch_first=True, dropout=0.0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias,
                         batch_first=batch_first, dropout=dropout)
        self.out_proj = None  # we don't use it

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

        # ALWAYS returns (attn_out, attn_w)
        attn_out, attn_w = multi_head_attention_forward(
            query, key, value,
            num_heads=self.num_heads,
            embed_dim_to_check=self.embed_dim,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            out_proj_weight=None, out_proj_bias=None,
            dropout_p=self.dropout, training=self.training,
            need_weights=True, average_attn_weights=True
        )

        if self.batch_first and is_batched:
            return attn_out.transpose(1, 0), attn_w
        else:
            return attn_out, attn_w

def multi_head_attention_forward(query, key, value, num_heads, embed_dim_to_check,
                                 in_proj_weight, in_proj_bias,
                                 out_proj_weight, out_proj_bias,
                                 dropout_p=0.0, training=True, need_weights=True, average_attn_weights=True):
    """
    Expected shapes (batch_first=False here):
      query: (Lq, B, E)
      key:   (Lk, B, E)
      value: (Lk, B, E)
    Returns:
      attn_out: (Lq, B, E)
      attn_w:   (B, Lq, Lk) if need_weights else None
    """
    Lq, B, E = query.shape
    Lk, Bk, Ek = key.shape
    Lv, Bv, Ev = value.shape
    assert B == Bk == Bv, "batch mismatch"
    assert E == Ek == Ev == embed_dim_to_check, "embed dim mismatch"
    assert Lk == Lv, "key/value length mismatch"

    head_dim = E // num_heads
    assert head_dim * num_heads == E, "E must be divisible by num_heads"

    # Single shared linear for Q,K,V (we call attn(x,x,x) in this construction)
    proj = F.linear(query, in_proj_weight, in_proj_bias)  # (Lq,B,3E)
    q = proj[:, :, :E]             # (Lq,B,E)
    k = proj[:, :, E:2*E]          # (Lq,B,E)  <-- using query-length; OK since we pass x,x,x
    v = proj[:, :, 2*E:]           # (Lq,B,E)

    # reshape to (B*H, L, D) for q,k and (B*H, L, E) for v
    q = q.reshape(Lq, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lq, D)
    k = k.reshape(Lq, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lq, D)
    v = v.reshape(Lq, B * num_heads, E).transpose(0, 1)         # (B*H, Lq, E)

    if not training:
        dropout_p = 0.0

    # No extra temperature scaling—selectivity encoded in Wq/Wk
    q_scaled = q                                                  # (B*H, Lq, D)
    attn_w = torch.bmm(q_scaled, k.transpose(-2, -1))            # (B*H, Lq, Lq)
    attn_w = F.softmax(attn_w, dim=-1)
    if dropout_p > 0.0:
        attn_w = F.dropout(attn_w, p=dropout_p)

    attn_out = torch.bmm(attn_w, v)                              # (B*H, Lq, E)

    # merge heads -> (B, Lq, E), then transpose to (Lq, B, E)
    attn_out = attn_out.reshape(B, num_heads, Lq, E).sum(dim=1)  # (B, Lq, E)
    attn_out = attn_out.transpose(0, 1)                          # (Lq, B, E)

    if need_weights:
        # average over heads to (B, Lq, Lk). Here Lk == Lq in this construction.
        attn_w = attn_w.reshape(B, num_heads, Lq, Lq).mean(dim=1)
        return attn_out, attn_w
    else:
        return attn_out, None


# ------------------- Hand-coded 2-attn-layer construction with residuals -------------------
class HardCodedTransformer(nn.Module):
    """
    Tokens: N bit tokens + 1 readout token (L = N+1).
    Features per token = [pos (N+1 one-hot) | bit] => E = (N+1)+1.

    Attn-1:
      - Q1 for anchor p_i has +gamma on positions in S_i (component), 0 elsewhere.
      - K1 exposes positional one-hots (identity on pos).
      - V1 passes only the bit channel.
      - Residual preserves exact positional one-hots; we then overwrite bit at anchors with signed parity.

    Attn-2:
      - Single-row query at readout position.
      - K2 puts log|c_i| at anchor positions, 0 at non-anchors -> softmax weights ∝ |c_i|,
        but diluted by non-anchors (each contributes 1 in denominator).
      - V2 bit scale K = Z = (M + sum|c_i|), where M = L - T, so final readout = sum_i c_i * parity_i.

    Output:
      - Linear read of the readout token's bit channel.
    """
    def __init__(self, N, combs, coeffs, device="cpu", eps=1e-12):
        super().__init__()
        self.N = int(N)
        self.combs = [list(c) for c in combs]
        coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
        assert len(self.combs) == coeffs.numel(), "coeffs length must match number of components"
        assert torch.all(coeffs != 0), "coeffs must be nonzero"

        self.T = len(self.combs)                       # number of components
        self.signs = torch.sign(coeffs)                # (T,)
        self.mags  = torch.abs(coeffs)                 # (T,)
        self.sum_abs = float(self.mags.sum().item())
        self.sum_lin = float(self.mags.sum().item())   # = sum |c_i|
        self.eps = float(eps)
        self.device = torch.device(device)

        # choose distinct anchors p_i in each comb (non-overlapping makes this trivial)
        self.anchors = self._pick_anchors(self.combs, self.N)

        # feature layout
        self.E_POS = self.N + 1
        self.E = self.E_POS + 1
        self.idx_bit = self.E_POS
        self.L = self.N + 1
        self.readout_tok = self.N

        # Embeddings
        self.bit_embed = nn.Embedding(2, 1)
        with torch.no_grad():
            self.bit_embed.weight[:] = torch.tensor([[0.0],[1.0]])
        self.bit_embed.weight.requires_grad = False

        self.pos_embed = nn.Embedding(self.L, self.E_POS)
        with torch.no_grad():
            W = torch.zeros(self.L, self.E_POS)
            for j in range(self.N):
                W[j, j] = 1.0
            W[self.readout_tok, self.N] = 1.0
            self.pos_embed.weight[:] = W
        self.pos_embed.weight.requires_grad = False

        # ----- Attn-1: Q1 with gamma = 2 log L; K1 identity on pos; V1 passes bit channel -----
        self.attn1 = CustomMHA(embed_dim=self.E, num_heads=1, bias=True, batch_first=True, dropout=0.0)

        gamma = 2.0 * math.log(self.L)  # single global gap
        Wq1 = torch.zeros(self.E, self.E)
        for S, p in zip(self.combs, self.anchors):
            for j in S:
                Wq1[j, p] = gamma

        Wk1 = torch.zeros(self.E, self.E)
        for j in range(self.N):
            Wk1[j, j] = 1.0  # identity on positional block

        Wv1 = torch.zeros(self.E, self.E)
        Wv1[self.idx_bit, self.idx_bit] = 1.0  # only pass bit channel

        with torch.no_grad():
            self.attn1.in_proj_weight[:] = torch.cat([Wq1, Wk1, Wv1], dim=0)
            self.attn1.in_proj_weight.requires_grad = False
            if self.attn1.in_proj_bias is None:
                self.attn1.in_proj_bias = nn.Parameter(torch.zeros(3*self.E), requires_grad=False)
            else:
                self.attn1.in_proj_bias.requires_grad = False

        self.register_buffer("comp_sizes", torch.tensor([len(S) for S in self.combs], dtype=torch.float32))
        self.register_buffer("signs_buf", self.signs.clone())

        # ----- Attn-2: Q at readout; K2 anchors = log|c_i|, non-anchors = 0; V2 bit-scale K=Z -----
        self.attn2 = CustomMHA(embed_dim=self.E, num_heads=1, bias=True, batch_first=True, dropout=0.0)

        qrow = 0
        krow = qrow
        Wq2 = torch.zeros(self.E, self.E)
        Wq2[qrow, self.N] = 1.0  # use readout's position one-hot as the query

        Wk2 = torch.zeros(self.E, self.E)
        logmag = torch.log(self.mags + self.eps)  # anchors scores = log|c_i|
        # non-anchors remain 0
        for i, p in enumerate(self.anchors):
            Wk2[krow, p] = float(logmag[i].item())

        # Denominator Z = M + sum |c_i|; we set V2 bit scale K = Z
        M = (self.L - self.T)
        Z = M + self.sum_lin
        K = Z if Z != 0.0 else 1.0

        Wv2 = torch.zeros(self.E, self.E)
        Wv2[self.idx_bit, self.idx_bit] = K

        with torch.no_grad():
            self.attn2.in_proj_weight[:] = torch.cat([Wq2, Wk2, Wv2], dim=0)
            self.attn2.in_proj_weight.requires_grad = False
            if self.attn2.in_proj_bias is None:
                self.attn2.in_proj_bias = nn.Parameter(torch.zeros(3*self.E), requires_grad=False)
            else:
                self.attn2.in_proj_bias.requires_grad = False

        # readout linear: pick the bit channel at readout token
        self.out = nn.Linear(self.E, 1, bias=False)
        with torch.no_grad():
            w = torch.zeros(1, self.E)
            w[0, self.idx_bit] = 1.0
            self.out.weight[:] = w
        self.out.weight.requires_grad = False

    @staticmethod
    def _pick_anchors(combs, N):
        used = set()
        anchors = []
        for S in combs:
            cand = None
            for j in sorted(S):
                if j not in used and 0 <= j < N:
                    cand = j
                    break
            if cand is None:
                raise ValueError("Could not choose distinct anchor positions for all components.")
            used.add(cand)
            anchors.append(cand)
        return anchors

    def _int_to_bits(self, x: int):
        return torch.tensor([(x >> i) & 1 for i in reversed(range(self.N))], dtype=torch.long, device=self.device)

    def forward(self, x_ints):
        """
        x_ints: (B,) integers in [0, 2^N)
        returns: (B,1) real values = sum_i c_i * parity_i(x)
        """
        x_ints = x_ints.to(self.device)
        B = x_ints.shape[0]

        # Build inputs: pos one-hots + bit channel
        bits = torch.stack([self._int_to_bits(int(v)) for v in x_ints.tolist()], dim=0)     # (B,N)
        dat_bits = self.bit_embed(bits)                                                     # (B,N,1)
        bit_block = torch.cat([dat_bits, torch.zeros(B, 1, 1, device=self.device)], dim=1)  # readout bit=0
        pos = self.pos_embed(torch.arange(self.L, device=self.device).unsqueeze(0).repeat(B,1))  # (B,L,E_POS)
        x = torch.cat([pos, bit_block], dim=2)                                              # (B,L,E)

        # --- Attn-1 + residual ---
        y1, _ = self.attn1(x, x, x)          # only bit channel gets mixed
        z1 = x + y1                           # residual keeps exact pos one-hots (and adds mean bit to bit channel)

        # From y1's anchor bit values, compute counts -> parity -> signed parity
        anchor_means = y1[:, self.anchors, self.idx_bit]                 # (B,T), each ~ mean(bit[S_i])
        counts = torch.floor(anchor_means * self.comp_sizes.unsqueeze(0) + 0.5)  # nearest integer
        counts = torch.clamp(counts, 0, self.comp_sizes.max())
        parity = torch.remainder(counts, 2.0)                             # {0,1}
        signed_parity = parity * self.signs_buf.unsqueeze(0)              # {-1,0,1}

        # overwrite bit channel: zero everywhere; write signed parity at anchors
        z1[:, :, self.idx_bit] = 0.0
        z1[:, self.anchors, self.idx_bit] = signed_parity

        # --- Attn-2 + residual ---
        y2, _ = self.attn2(z1, z1, z1)  # keys as log|c_i| at anchors; V2 bit scaled by K=Z
        z2 = z1 + y2                     # readout bit in z1 is 0, so result is purely from attention mix

        # readout
        y = self.out(z2[:, self.readout_tok, :])  # (B,1)
        return y


# ----------------------------- quick sanity test --------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    device = "cpu"
    N = 12
    combs = [
        [0, 2, 5],
        [1, 7, 8],
        [3, 4, 9],
    ]  # non-overlapping degree-3 components

    T = len(combs)
    raw = torch.randn(T)
    coeffs = (raw / raw.norm()).tolist()  # sum of squares = 1

    model = HardCodedTransformer(N=N, combs=combs, coeffs=coeffs, device=device)

    # random batch
    B = 16
    x = torch.randint(0, 2**N, (B,), device=device)
    y_model = model(x).squeeze(-1)

    # ground truth
    def ground_truth(bits_vec):
        total = 0.0
        for S, c in zip(combs, coeffs):
            k = int(bits_vec[S].sum().item())
            total += float(c) * (k % 2)
        return total

    bits = torch.stack([model._int_to_bits(int(v)) for v in x.tolist()], dim=0)
    y_true = torch.tensor([ground_truth(b) for b in bits], dtype=torch.float32, device=device)

    print("x (ints):", x.tolist())
    print("model   :", [float(v) for v in y_model])
    print("true    :", [float(v) for v in y_true])
    print("max |err|:", (y_model - y_true).abs().max().item())
