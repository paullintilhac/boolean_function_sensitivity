import torch
from torch import nn
import torch.nn.functional as F
import math


device = "cuda" if torch.cuda.is_available() else "cpu"
class AttentionBlock(nn.Module):
    
    def __init__(self, hidden_dim, ff_dim, num_heads, LNeps, N):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.attn = CustomMHA(hidden_dim, num_heads, bias=False, batch_first=True, N=N)
        # self.attn = nn.MultiheadAttention(hidden_dim, num_heads, bias=False, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=LNeps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=LNeps)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
            )
        
    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.linear(x))
        return x

class Transformer(torch.nn.Module):
    
    def __init__(self, N, hidden_dim, num_heads, num_layers, ff_dim, LNeps):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.h = num_heads
        self.l = num_layers
        self.ff_dim = ff_dim
        self.LNeps = LNeps

        # Layers
        self.embeddings = torch.nn.Embedding(2, hidden_dim//2)
        hidden_dim = N + hidden_dim//2

        # self.positional_embeddings = torch.nn.Embedding(N, hidden_dim//2)
        # self.positional_embeddings = torch.eye(N, N)
        self.transformer = nn.Sequential(*[AttentionBlock(hidden_dim=hidden_dim, ff_dim=ff_dim, num_heads=num_heads, LNeps=LNeps, N=N) for _ in range(num_layers)])        
        # Layers/Networks
        # self.mlp_head = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, ff_dim), 
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(ff_dim, hidden_dim)
        # )
        # self.output_proj = nn.Parameter(torch.randn((N, hidden_dim)), requires_grad=True).to(device)
        self.output_proj = nn.Linear(N*hidden_dim, 1, bias=False)
    
        
    def makeBitTensor(self, x, N):
        y = format(x, "b")
        y = ("0"*(N-len(y))) + y
        return [int(z) for z in list(y)]
    
    
    def forward(self, x):    

        batch_size = x.shape[0]
        inputNum = torch.LongTensor([ self.makeBitTensor(num, self.N) for num in x]).to(device)
        # positional = torch.LongTensor(list(range(0, self.N))).unsqueeze(1).expand(-1, batch_size).T.to(device)
        # pos, dat = self.positional_embeddings(positional), self.embeddings(inputNum)
        pos, dat = torch.eye(self.N, self.N).to(device).unsqueeze(0).repeat(batch_size, 1, 1), self.embeddings(inputNum)
        x = torch.cat([pos, dat], dim=2)
        x = self.transformer(x)
        # x = self.mlp_head(x)
        x = self.output_proj(x.view(x.shape[0], -1))

        return x
    
class CustomMHA(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, bias, batch_first, N):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias, batch_first=batch_first)
        self.N = N

    def forward(self, query, key, value):
        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # Actual Calculation
        attn_output, attn_output_weights = multi_head_attention_forward(query, key, value, 
                                                num_heads=self.num_heads, N=self.N, embed_dim_to_check=self.embed_dim, 
                                                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, 
                                                out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, 
                                                dropout_p=self.dropout, training=self.training, need_weights=True,
                                                average_attn_weights=True)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

def multi_head_attention_forward(query, key, value, num_heads, N, embed_dim_to_check,
                                 in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                                 dropout_p=0, training=True, need_weights=True, average_attn_weights=True):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        num_heads: parallel attention heads.
        N: size of our boolean domain
        in_proj_weight, in_proj_bias: input projection weight and bias.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        dropout_p: probability of an element to be zeroed.
        training: apply dropout if is ``True``.
        need_weights: output attn_output_weights.

    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # # Target Length, Batch Size, Embed Dim
    # tgt_len, bsz, embed_dim = query.size()
    # assert (embed_dim == embed_dim_to_check), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    # # allow MHA to have different sizes for the feature dimension
    # assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    # # Divide the embedding into certain number of heads
    # head_dim = embed_dim // num_heads
    # assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    # # Scaling = \sqrt{d_k}. This is where we add our 2log(N)
    # scaling = 2*math.log(N)*float(head_dim) ** -0.5 
    # scaling = float(head_dim) ** -0.5 

    # # For self.attention, the projections q, k, v using in_proj are calculated and then split
    # if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
    #     # self-attention
    #     q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    # # Same as scaling qk.T after multiplication
    # q = q * scaling

    # # Re-arranging to make batch matrix multiplication easier. 
    # # You end up with shapes of (bsz*num_heads, tgt_len, head_dim) that can be batch matrix-multiplied
    # q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # if k is not None:
    #     k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    # if v is not None:
    #     v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # # Source Length
    # src_len = k.size(1)

    # # The batch matrix multiplication
    # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    # # Self-explanatory
    # attn_output_weights = F.softmax(
    #     attn_output_weights, dim=-1)
    # if training and dropout_p > 0:
    #     attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # # Each of the bsz*num_heads output_weights is multiplied by V
    # attn_output = torch.bmm(attn_output_weights, v)
    # assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]

    # # Concatenates the different heads together
    # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    # if need_weights:
    #     # average attention weights over heads
    #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #     if average_attn_weights:
    #         attn_output_weights = attn_output_weights.mean(dim=1)
    #     return attn_output, attn_output_weights
    # else:
    #     return attn_output, None

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.


    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape


    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    
    #
    # compute in-projection
    #
    
    E = query.size(-1)
    proj = F.linear(query, in_proj_weight, in_proj_bias)
    proj = (
                proj.unflatten(-1, (3, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
    q, k, v = proj[0], proj[1], proj[2]

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E)) * 2*math.log(N)

        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, False
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return attn_output, None