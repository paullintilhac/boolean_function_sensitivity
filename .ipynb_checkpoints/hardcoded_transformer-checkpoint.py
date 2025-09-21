import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Custom MHA (layout-safe)
# -------------------------

class CustomMHA(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads=1, bias=True, batch_first=True, dropout=0.0):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias,
                         batch_first=batch_first, dropout=dropout)
        # Disable the standard out_proj so we only use in-proj blocks
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
    Shapes (batch_first=False here):
      query: (Lq, B, E), key: (Lk, B, E), value: (Lk, B, E)
      returns attn_out: (Lq, B, E), attn_w: (B, Lq, Lk)
    """
    Lq, B, E = query.shape
    Lk, Bk, Ek = key.shape
    Lv, Bv, Ev = value.shape
    assert B == Bk == Bv, "Batch mismatch in MHA"
    assert E == Ek == Ev == embed_dim_to_check, "Embed dim mismatch"
    assert Lk == Lv, "Key/Value length mismatch"

    head_dim = E // num_heads
    assert head_dim * num_heads == E, "E must be divisible by num_heads"

    # Single shared linear for Q,K,V
    proj = F.linear(query, in_proj_weight, in_proj_bias)  # (Lq,B,3E)
    q = proj[:, :, :E]
    k = proj[:, :, E:2*E]
    v = proj[:, :, 2*E:]

    # reshape to (B*H, L, D) for q,k and (B*H, L, E) for v
    q = q.reshape(Lq, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lq, D)
    k = k.reshape(Lk, B * num_heads, head_dim).transpose(0, 1)  # (B*H, Lk, D)
    v = v.reshape(Lv, B * num_heads, E).transpose(0, 1)         # (B*H, Lk, E)

    if not training:
        dropout_p = 0.0

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


# -----------------------------------------
# Hard-coded Transformer (Hessian-friendly)
# -----------------------------------------

class HardCodedTransformer(nn.Module):
    """
    Two self-attention layers with fixed positional + bit embeddings.
    * Only embeddings are frozen (excluded from Hessian).
    * Attention in-proj weights/biases + readout are trainable (included in Hessian).
    * Device-safe tensor creation (indices, arange, zeros).
    """

    def __init__(self, N, combs=None, coeffs=None, num_heads=1, use_residual=True):
        super().__init__()
        self.N = N
        self.L = N          # number of tokens (positions)
        self.use_residual = use_residual

        # --- Fixed embeddings (EXCLUDED from Hessian) ---
        # Bit embedding: 0 -> 0, 1 -> 1
        self.bit_embed = nn.Embedding(2, 1)
        with torch.no_grad():
            w = torch.zeros(2, 1)
            w[1, 0] = 1.0
            self.bit_embed.weight.copy_(w)
        self.bit_embed.weight.requires_grad = False

        # Positional "one-hot" embedding (identity)
        self.pos_embed = nn.Embedding(self.L, self.L)
        with torch.no_grad():
            self.pos_embed.weight.copy_(torch.eye(self.L))
        self.pos_embed.weight.requires_grad = False

        # Base positional indices (buffer follows .to(device))
        self.register_buffer("pos_idx_base", torch.arange(self.L, dtype=torch.long))

        embed_dim = self.L + 1  # pos one-hot + bit-channel

        # --- Attention layers (INCLUDED in Hessian) ---
        self.attn1 = CustomMHA(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True, dropout=0.0)
        self.attn2 = CustomMHA(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True, dropout=0.0)

        # --- Readout (INCLUDED in Hessian) ---
        self.readout = nn.Linear(embed_dim, 1, bias=True)

        # Optional: initialize from combs/coeffs if provided
        if combs is not None and coeffs is not None:
            self.initialize_from_combs(combs, coeffs)

        # Ensure all non-embedding parameters require grad (Hessian includes them)
        for p in self.parameters():
            # embeddings are already frozen; others should require grad
            if p is self.bit_embed.weight or p is self.pos_embed.weight:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

    # --- Utilities ---

    def _dev(self):
        return next(self.parameters()).device

    @staticmethod
    def _ints_to_bits(x: torch.Tensor, N: int) -> torch.Tensor:
        """
        x: (B,) int tensor on the correct device
        returns: (B,N) long tensor of 0/1
        """
        device = x.device
        shifts = torch.arange(N, device=device, dtype=torch.long)
        bits = ((x.unsqueeze(-1) >> shifts) & 1).long()
        return bits

    # --- Optional initializer matching your construction ---
    def initialize_from_combs(self, combs, coeffs):
        """
        Example initializer that sets in-proj weights to a deterministic pattern
        based on (combs, coeffs). It copies the values with no_grad, then
        IMMEDIATELY re-enables gradients so pyhessian can differentiate w.r.t. them.
        """
        E = self.L + 1
        dev = self._dev()

        # Start from zeros
        Wq1 = torch.zeros(E, E, device=dev)
        Wk1 = torch.zeros(E, E, device=dev)
        Wv1 = torch.zeros(E, E, device=dev)

        # Simple pattern (you can replace with your exact construction):
        # - Keys output the positional one-hot (first L dims)
        # - Values output only the bit channel (last dim)
        with torch.no_grad():
            # K: pick positional part
            Wk1[:self.L, :self.L] = torch.eye(self.L, device=dev)
            # V: pick bit channel into last dim
            Wv1[self.L, self.L] = 1.0

            # Q: for each position t, make query emphasize positions in its comb (mask)
            # Use a mask of ones over the component; optionally scale by 2*log L
            logL = 2.0 * math.log(max(2, self.L))
            for comb in combs:
                # choose a representative index for the comb: first element
                t = comb[0]
                for j in comb:
                    Wq1[j, j] += logL  # build mask via diagonal accumulation

        # Copy into the packed in-proj (Q,K,V concatenated)
        with torch.no_grad():
            self.attn1.in_proj_weight[:E, :] = Wq1
            self.attn1.in_proj_weight[E:2*E, :] = Wk1
            self.attn1.in_proj_weight[2*E:, :] = Wv1
            if self.attn1.in_proj_bias is not None:
                self.attn1.in_proj_bias.zero_()

        # Re-enable grads (so Hessian sees them)
        self.attn1.in_proj_weight.requires_grad_(True)
        if self.attn1.in_proj_bias is not None:
            self.attn1.in_proj_bias.requires_grad_(True)

        # attn2: recombination via coeffs (again, toy layout—replace with your exact one)
        Wq2 = torch.zeros(E, E, device=dev)
        Wk2 = torch.zeros(E, E, device=dev)
        Wv2 = torch.zeros(E, E, device=dev)

        with torch.no_grad():
            # Let K be identity on positions again
            Wk2[:self.L, :self.L] = torch.eye(self.L, device=dev)
            # Value carries the (pos+bit) to the readout; here we keep it simple
            Wv2.copy_(torch.eye(E, device=dev))

            # Queries put weight proportional to |coeff| at the representative indices
            # (you can switch to 2*log|c_i| if desired)
            for comb, c in zip(combs, coeffs):
                t = comb[0]  # representative
                Wq2[t, t] += 2.0 * math.log(max(1e-6, abs(float(c))))

        with torch.no_grad():
            self.attn2.in_proj_weight[:E, :] = Wq2
            self.attn2.in_proj_weight[E:2*E, :] = Wk2
            self.attn2.in_proj_weight[2*E:, :] = Wv2
            if self.attn2.in_proj_bias is not None:
                self.attn2.in_proj_bias.zero_()

        self.attn2.in_proj_weight.requires_grad_(True)
        if self.attn2.in_proj_bias is not None:
            self.attn2.in_proj_bias.requires_grad_(True)

        # Readout init (trainable)
        with torch.no_grad():
            self.readout.weight.zero_()
            self.readout.bias.zero_()
        self.readout.weight.requires_grad_(True)
        if self.readout.bias is not None:
            self.readout.bias.requires_grad_(True)

    # --- Forward (no @torch.no_grad): full graph preserved ---
    def forward(self, x_ints: torch.Tensor) -> torch.Tensor:
        """
        x_ints: (B,) int tensor; can be CPU or GPU; moved to module device
        Returns: (B,1)
        """
        dev = self._dev()
        x_ints = x_ints.to(dev)

        B = x_ints.shape[0]
        N = self.N

        # Build (B,N) bit indices on device
        bits = self._ints_to_bits(x_ints, N)               # (B,N) long on dev

        # Embeddings on same device
        dat_bits = self.bit_embed(bits)                    # (B,N,1)
        pos_idx  = self.pos_idx_base.unsqueeze(0).expand(B, -1)  # (B,N) long on dev
        pos_vecs = self.pos_embed(pos_idx)                 # (B,N,N)

        # Concatenate to (B,N,N+1)
        x = torch.cat([pos_vecs, dat_bits], dim=-1)        # (B,N, N+1)

        # Self-attn 1
        y1, _ = self.attn1(x, x, x)
        x1 = x + y1 if self.use_residual else y1

        # Self-attn 2
        y2, _ = self.attn2(x1, x1, x1)
        x2 = x1 + y2 if self.use_residual else y2

        # Pool + readout (adjust if you previously used a different readout)
        x_pool = x2.mean(dim=1)                            # (B, N+1)
        out = self.readout(x_pool)                         # (B,1)
        return out


# -------------------------
# Optional smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 12
    combs = [[0,1,2], [3,4,5], [6,7,8]]
    coeffs = [0.5, -0.8, 0.3]

    model = HardCodedTransformer(N=N, combs=combs, coeffs=coeffs, num_heads=1, use_residual=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    # Check grad flags: only embeddings frozen
    frozen = [name for name,p in model.named_parameters() if not p.requires_grad]
    trainable = [name for name,p in model.named_parameters() if p.requires_grad]
    print("Frozen params:", frozen)
    print("Trainable params (Hessian will see these):", trainable)

    x = torch.randint(0, 2**N, (16,), dtype=torch.long, device=dev)
    y = model(x)
    print("Output:", y.shape, y.device)

    # Quick grad check
    y.sum().backward()
    nz = sum((p.grad is not None) and (p.grad.abs().sum() > 0) for p in model.parameters())
    print("Nonzero grads across params:", nz)
