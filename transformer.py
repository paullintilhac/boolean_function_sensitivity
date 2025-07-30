import torch
from torch import nn
import torch.nn.functional as F
import math

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


class AttentionBlock(nn.Module):
    
    def __init__(self, hidden_dim, output_dim, ff_dim, num_heads, LNeps, N, dropout, ln):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.attn = CustomMHA(hidden_dim, output_dim, num_heads, bias=False, batch_first=True, N=N,dropout=dropout)
        self.skip = hidden_dim == output_dim
        if ln:
            self.norm1 = nn.LayerNorm(output_dim, eps=LNeps)
            self.norm2 = nn.LayerNorm(output_dim, eps=LNeps)
        self.linear = nn.Sequential(
            nn.Linear(output_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, output_dim)
            )
        self.ln = ln
        
    def forward(self, x):
        if self.ln:
            if self.skip: 
                x = self.norm1(x + self.attn(x, x, x)[0])
            else:
                x = self.norm1(self.attn(x,x,x)[0])
            x = self.norm2(x + self.linear(x))
        else: 
            if self.skip: 
                x = x + self.attn(x, x, x)[0]
            else:
                x = self.attn(x,x,x)[0]
            x = x + self.linear(x)
        return x

class Transformer(torch.nn.Module):
    
    def __init__(self,dropout, N, hidden_dim, proj_dim, output_dim, num_heads, num_layers, ff_dim, LNeps,rank,ln,dim_red):

        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layaers = num_layers
        self.ff_dim = ff_dim
        self.LNeps = LNeps
        self.rank = rank
        self.dropout = dropout
        self.dim_red = dim_red
        # Data embedding
        self.embeddings = torch.nn.Embedding(2, hidden_dim)
        if hidden_dim == 2:
            self.embeddings.weight = nn.Parameter(torch.eye(hidden_dim), requires_grad=False)
        
        # hidden_dim = N + hidden_dim

        # Positional Embedding
        if self.dim_red:
            self.random_projection = 1/torch.sqrt(torch.tensor(proj_dim)) * torch.randn((N, proj_dim), requires_grad=False).to(rank)
        else:
            print("dim red is flase, setting proj dim from " +str(self.proj_dim)+" to " + str(self.N))
            self.proj_dim = self.N
        print("proj_dim: " + str(self.proj_dim)+", N: " + str(self.N) + ", hidden_dim: " + str(hidden_dim))
        self.hidden_dim = self.proj_dim + self.hidden_dim
        
        

        self.transformer = AttentionBlock(hidden_dim=self.hidden_dim, output_dim=output_dim, ff_dim=ff_dim, num_heads=num_heads, LNeps=LNeps, N=N,dropout=dropout,ln=ln)       

        self.output_proj = nn.Parameter(torch.randn((N, output_dim)), requires_grad=True)
        
        self.output_proj.to(rank)

        
    def makeBitTensor(self, x, N):
        y = format(x, "b")
        y = ("0"*(N-len(y))) + y
        return [int(z) for z in list(y)]
    
    
    def forward(self, x):    

        batch_size = x.shape[0]
        inputNum = x.to(self.rank).long()
        #inputNum = torch.LongTensor([ self.makeBitTensor(num, self.N) for num in x]).to(self.rank)
        pos = torch.eye(self.N, self.N).to(self.rank).unsqueeze(0).repeat(batch_size, 1, 1)
        if self.dim_red:
            pos = torch.matmul(pos, self.random_projection)
        dat = self.embeddings(inputNum)
        x = torch.cat([pos, dat], dim=2)



        x = self.transformer(x)
        x.to(self.rank)
        x = torch.tensordot(x , self.output_proj)
        return x
    
class CustomMHA(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, new_dim, num_heads, bias, batch_first, N,dropout):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias, batch_first=batch_first,dropout=dropout)
        self.out_proj = None
        self.N = N
        self.in_proj_weight = nn.Parameter(self.in_proj_weight[:2*embed_dim + new_dim , :])

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
                                                out_proj_weight=None, out_proj_bias=None, 
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
    
    proj = F.linear(query, in_proj_weight, in_proj_bias)
    # print(in_proj_weight.shape) # (2*embed_dim+new_dim, hidden_dim)
    # print(proj.shape) # (N, batch_size, 2*embed_dim+new_dim)
    q, k, v = proj[:, :, :embed_dim], proj[:, :, embed_dim:2*embed_dim], proj[:, :, 2*embed_dim:]
    new_dim = v.shape[-1]
    # proj = (
    #             proj.unflatten(-1, (3, embed_dim))
    #             .unsqueeze(0)
    #             .transpose(0, -2)
    #             .squeeze(-2)
    #             .contiguous()
    #         )
    # q, k, v = proj[0], proj[1], proj[2]
    # print(q.shape, k.shape, v.shape)

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, new_dim).transpose(0, 1)

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
        attn_output = attn_output.transpose(0, 1)
        # attn_output = (
        #     attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        # )

        # attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        # attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights