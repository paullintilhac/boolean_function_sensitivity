# Using ../sensitivity/36test.py
# Functions whose Fourier degree is concentrated on higher weights are harder to learn for LSTMs with SGD

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from transformer import Transformer
import os

device = "cuda"
if torch.cuda.is_available():
    print("GPU is available")
    print("device count: " + str(torch.cuda.device_count()))
else:
    print("GPU is not available")

def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--f', type=int, default=64)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--i', type=int, default=300000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=100)


    return parser.parse_args()

def fitNetwork(function, N, batch_size):
  model = Transformer(N, args.dim, args.h, args.l, args.f, 1e-5).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000006, weight_decay=0.1)
  
  movAvg = 0
  lossesAfterIterations = [] 
  for iteration in range(args.i):
    
    optimizer.zero_grad()

    # Generate training data
    inputs = torch.tensor([random.randint(0, 2**N-1) for _ in range(batch_size)]).to(device)
    targets = torch.FloatTensor([float(function(x)) for x in inputs]).to(device)

    result = model(inputs)
    
    loss = (result - targets).pow(2).mean()
    movAvg = 0.99 * movAvg + (1-0.99) * (float(loss))
    
    if not iteration % 1000:
      lossesAfterIterations.append(loss.item())
      print(f"Iteration: {iteration}, Average Loss: {movAvg}")
      with open("logs.txt", "a") as f:
          f.write(f"Iteration: {iteration}, Average Loss: {movAvg}\n")
    
    (loss).backward()
    optimizer.step()

    if movAvg < 0.01:
      break
  return model

def fitNetworkBatch(function, N, batch_size):
  model = Transformer(N, args.dim, args.h, args.l, args.f, 1e-5).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000006, weight_decay=0.1)
  train_dataset = torch.load("input_dataset.pt")
  movAvg = 0
  lossesAfterIterations = [] 
  optimizer.zero_grad()

  train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
  )
  for epoch in range(0, 1000):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
      #print(batch)
      inputs = batch[0]
      targets = batch[1]
      result = model(inputs)
      loss = (result - targets).pow(2).mean()
      movAvg = 0.99 * movAvg + (1-0.99) * (float(loss))
      print("Loss:", loss.item())
      loss.backward()
      optimizer.step()
 
    
    lossesAfterIterations.append(loss.item())
    with open("logs.txt", "a") as f:
      f.write(f"epoch: {epoch}, Average Loss: {movAvg}\n")

    if movAvg < 0.01:
      break
  return model
def fitNetwork_old(function, N,batch_size):
   hidden_size = args.dim
   embeddings = torch.nn.Embedding(2, hidden_size//2).cuda()
   positional_embeddings = torch.nn.Embedding(N, hidden_size//2).cuda()
   qrnn = torch.nn.TransformerEncoder(encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=args.h, dim_feedforward=args.f, dropout=0.0, activation='relu'), num_layers=args.l).cuda()
   
   output = torch.nn.Linear(hidden_size, 1, bias=False).cuda()
  # Flatten and dot
   
   tanh = torch.nn.Tanh()
   
   def parameters():
     for x in [embeddings, qrnn, output, positional_embeddings]:
       for y in x.parameters():
          yield y
   
   optimizer = torch.optim.AdamW(parameters(), lr=0.00003, weight_decay=0.1)

   movAvg = 0
   lossesAfterIterations = [] 
   for iteration in range(args.i):
     optimizer.zero_grad()
     inputs = [random.randint(0, 2**N-1) for _ in range(batch_size)]
     targets = torch.FloatTensor([float(function(x)) for x in inputs]).cuda()
 #   print(targets.mean())
 #   quit()
     inputNum = torch.LongTensor([makeBitTensor(x,N) for x in inputs]).cuda().t()
     positional = torch.LongTensor(list(range(0, N))).unsqueeze(1).expand(-1, batch_size).cuda()
     inputTensorEmbed = torch.cat([embeddings(inputNum), positional_embeddings(positional)], dim=2)
     hidden = qrnn(inputTensorEmbed)[0]
     result = (output(hidden)).view(-1)
#    print(result.size())
     result = result.view(batch_size)
     loss = (result - targets).pow(2).mean()
     movAvg = 0.99 * movAvg + (1-0.99) * (float(loss))
     #if iteration % 10 == 0 and True:
       #print(iteration, movAvg / (1-0.99**(iteration+1)), N, sum([float(x.data.pow(2).sum()) for x in parameters() if x.grad is not None]))
     (loss).backward()
     optimizer.step()
#     print(iteration, abs(log(iteration+1)/log(10) % 1))
#     print(iteration, movAvg)
     if movAvg < 0.003 and False:
        lossesAfterIterations.append(movAvg)
        lossesAfterIterations.append(movAvg)
        lossesAfterIterations.append(movAvg)
        lossesAfterIterations.append(movAvg)
        lossesAfterIterations.append(movAvg)
        lossesAfterIterations.append(movAvg)
        lossesAfterIterations.append(movAvg)
        break 
     if (iteration-1) % 10000 == 0:
        print("saving...")
        lossesAfterIterations.append(movAvg)
        print("lossesAfterIterations: " + str(lossesAfterIterations))
        print(iteration, movAvg)
   return lossesAfterIterations 


def rboolf_old(N, deg=2):
   coefficients = torch.randn(N,N).cuda()
   mask = torch.zeros(N, N, dtype=torch.uint8).cuda()
    
   indices = torch.randperm(N*N)[:N]  # Shuffle flattened indices and pick first N
   rows = indices // N  # Convert to 2D row indices
   cols = indices % N   # Convert to 2D column indices
   
   # Set selected indices to one
   mask[rows, cols] = 1
   coefficients = (coefficients * mask)
   coefficients = (coefficients-coefficients.mean())/ coefficients.pow(2).sum().sqrt()
   
   def function(x):
       binary = formatted_binary = f"{x:0{N}b}"
       c = torch.zeros(N,N)
       r = 0
       for w in range(N):
           i, j = rows[w], cols[w]
    #       for i in range(N):
    #          for j in range(N):
           c[i,j] = 1 if binary[i] == binary[j] else -1
       r = (coefficients*c.cuda()).sum()
       return r
   return function

def rboolf(N, deg):
    coefficients = torch.randn(N).to(device)
    coefficients = (coefficients-coefficients.mean())/coefficients.pow(2).sum().sqrt()
    combs = torch.combinations(torch.arange(N+1), r=deg, with_replacement=True)
    combs = combs[torch.randperm(combs.size()[0])][:N].to(device) # Shuffled

    def func(x):
        binary = f"{x:0{N}b}"+"0"
        comps = []
        for elem in combs:
            res = 1
            for e in elem:
                bit = 1 if int(binary[e]) else -1
                res *= bit
            comps.append(res)
        return torch.dot(coefficients, torch.tensor(comps, dtype=torch.float32).to(device))
    return func

   

def main(args):
  losses = {}
  test_batch = 1000
  func_per_deg = args.repeat
  with open("logs2.txt", "a") as f:
    f.write("------------------------------------------\n")
  for deg in [3]:
    losses[deg] = []
    for i in range(func_per_deg):
      print(f"Degree: {deg}, Func: {(i+1)}/{func_per_deg}")
      func = rboolf(args.N, deg)
      model = fitNetworkBatch(func, args.N, args.bs)
      inputs = torch.tensor([random.randint(0, 2**args.N-1) for _ in range(test_batch)]).to(device)
      targets = torch.FloatTensor([float(func(x)) for x in inputs]).to(device)
      result = model(inputs).to(device)
      loss = (result - targets).pow(2).mean()
      losses[deg].append(loss.item())
      print(f"\nReported Loss: {loss.item()}\n")
      with open("logs.txt", "a") as f:
          f.write(f"Degree: {deg}, Func: {(i+1)}/{func_per_deg}")
          f.write(f"\nReported Loss: {loss.item()}\n")
    
      # func = rboolf(args.N, deg)
      # model = fitNetwork(func, args.N, args.bs)
      # inputs = torch.tensor([random.randint(0, 2**args.N-1) for _ in range(test_batch)]).to(device)
      # targets = torch.FloatTensor([float(func(x)) for x in inputs]).to(device)
      # result = model(inputs).to(device)
      # loss = (result - targets).pow(2).mean()
      # losses[deg].append(loss.item())
      # print(f"\nReported Loss: {loss.item()}\n")
  print(losses)
  
  import pandas as pd
  df = pd.DataFrame.from_dict(losses)
  dir_name = f"{args.N}_{args.dim}_{args.l}_{args.h}_{args.f}"
  os.makedirs(dir_name, exist_ok=True)
  df.to_csv(f"{dir_name}/results_layernorm.csv")

if __name__ == "__main__":
    args = parse_args()
    main(args)   
