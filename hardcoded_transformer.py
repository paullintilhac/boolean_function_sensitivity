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
