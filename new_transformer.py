import torch
from torch import nn

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


class Attention(nn.Module):
    
    def __init__(self, hidden_dim, output_dim, N, dropout=0.25):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, output_dim, bias=False)
        self.dp  = nn.Dropout(dropout)

        
    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        A = torch.softmax(q @ k.transpose(-1, -2) / (self.hidden_dim**0.5), dim=-1)  # noqa: N806
        y = self.dp(A) @ v
        return y

class Transformer(torch.nn.Module):
    
    def __init__(self,dropout, N, hidden_dim, hidden_dim2, num_layers, ff_dim, LNeps, rank):

        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.l = num_layers
        self.ff_dim = ff_dim
        self.LNeps = LNeps
        self.rank = rank
        self.dropout = dropout
        self.hidden_dim2 = hidden_dim2
        # Layers
        self.embeddings = torch.nn.Embedding(2, hidden_dim)
        hidden_dim = N + hidden_dim
            

        self.attention = nn.Sequential(*[Attention(hidden_dim=hidden_dim, N=N, output_dim=hidden_dim2, dropout=self.dropout) for _ in range(num_layers)])        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim2, ff_dim), 
            # torch.nn.Dropout(self.dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, hidden_dim2),
            # torch.nn.Dropout(dropout),
        )
        # LayerNorm
        self.ln1 = torch.nn.LayerNorm(hidden_dim2, eps=LNeps)
        self.ln2 = torch.nn.LayerNorm(hidden_dim2, eps=LNeps)

        # Output layer
        self.out = torch.nn.Linear(N, 1, bias=False)

        
    def makeBitTensor(self, x, N):
        y = format(x, "b")
        y = ("0"*(N-len(y))) + y
        return [int(z) for z in list(y)]
    
    
    def forward(self, x):    

        batch_size = x.shape[0]
        inputNum = torch.LongTensor([ self.makeBitTensor(num, self.N) for num in x]).to(self.rank)
        pos = torch.eye(self.N, self.N).to(self.rank).unsqueeze(0).repeat(batch_size, 1, 1)
        dat = self.embeddings(inputNum)
        x = torch.cat([pos, dat], dim=2)

        x = self.ln1(self.attention(x))
        x = self.ln2(self.mlp(x))
        x = x.squeeze(-1)
        print(x.shape)

        x = self.out(x)
        return x
    

if __name__ == "__main__":

    # Create a model
    print("Creating model")
    N = 10
    model = Transformer(dropout=0.1, N=N, hidden_dim=15, num_layers=1, ff_dim=32, LNeps=1e-5, rank=device, hidden_dim2=1)
    model.to(device)

    # Data:
    x = torch.randint(0, 2**N-1, (5,)).to(device)

    # Forward pass
    print(model(x).shape)
