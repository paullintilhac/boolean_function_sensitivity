import torch
from torch import nn


device = "cuda"
class AttentionBlock(nn.Module):
    
    def __init__(self, hidden_dim, ff_dim, num_heads, LNeps):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, bias=False, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=LNeps, bias=True)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=LNeps, bias=True)
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
        self.transformer = torch.nn.Sequential(*[AttentionBlock(hidden_dim=hidden_dim, ff_dim=ff_dim, num_heads=num_heads, LNeps=LNeps) for _ in range(num_layers)])        
        # Layers/Networks
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ff_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, hidden_dim)
        )
        self.output_proj = torch.randn((N, hidden_dim), requires_grad=True).to(device)
    
        
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
        x = torch.tensordot(x , self.output_proj)

        return x