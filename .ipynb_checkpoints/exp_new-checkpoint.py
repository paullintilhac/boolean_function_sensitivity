# Using ../sensitivity/36test.py
# Functions whose Fourier degree is concentrated on higher weights are harder to learn for LSTMs with SGD

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import random
import argparse
from transformer import Transformer
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("GPU is available")
    print("device count: " + str(torch.cuda.device_count()))
else:
    print("GPU is not available")

def fitNetwork(function, loader, N, epochs, dir_name):
    model = Transformer(N, args.dim, args.h, args.l, args.f, 1e-5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000006, weight_decay=0.1)
    
    movAvg = 0
    summary = pd.DataFrame(columns=["iter", "loss"])
    # dir_name = f"{args.N}_{args.dim}_{args.l}_{args.h}_{args.f}"

    for epoch in range(epochs):    
        for idx, inputs in enumerate(loader):
          
          targets = torch.FloatTensor([float(function(x)) for x in inputs]).to(device)
    
          result = model(inputs)
          
          loss = (result - targets).pow(2).mean()
          movAvg = 0.99 * movAvg + (1-0.99) * (float(loss.detach()))
          
          iteration = epoch*len(loader)+idx+1
          if (iteration) % 10 == 0:
            summary.loc[len(summary)] = {"iter":iteration, "loss":movAvg}
            summary.to_csv(f"{dir_name}/curr_func.csv")

          if (iteration) % 1000 == 0:
            val_loss = validate(model, function, num_samples=10000)
            print(f"Iteration: {iteration}, Loss: {loss.detach():.3f}, Validation Loss: {val_loss:.3f}")
            path = os.path.join(dir_name, f"model_{iteration}.pt")
            torch.save(model.state_dict(), path)  
          
          (loss).backward()
          optimizer.step()
          optimizer.zero_grad()
    
          if movAvg < 0.01:
            break
    return model, summary

def rboolf(N, width, deg):
    coefficients = torch.randn(width).to(device)
    coefficients = (coefficients-coefficients.mean())/coefficients.pow(2).sum().sqrt()
    combs = torch.combinations(torch.arange(N+1), r=deg, with_replacement=True)
    combs = combs[torch.randperm(combs.size()[0])][:width].to(device) # Shuffled

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
    return func, (coefficients, combs)

def validate(model, func, num_samples=1000):
      inputs = torch.tensor([random.randint(0, 2**args.N-1) for _ in range(num_samples)]).to(device)
      targets = torch.FloatTensor([float(func(x)) for x in inputs]).to(device)
      result = model(inputs).to(device)
      loss = (result - targets).pow(2).mean()
      return loss.detach()

def generate_dataset(num_samples, N, batch_size):
    num_samples = 10000
    inputs = torch.tensor([random.randint(0, 2**N-1) for _ in range(num_samples)]).to(device)
    train_loader = DataLoader(inputs, shuffle=True, batch_size=batch_size)
    return train_loader 

def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--f', type=int, default=64)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--num_samples', type=float, default=100000)
    parser.add_argument('--repeat', type=int, default=100)


    return parser.parse_args()

def main(args):
    # summary = pd.DataFrame(columns=["deg", "width", "func", "iter", "loss"])
    losses = {}
    test_batch = 10000
    func_per_deg = args.repeat
    main_dir = f"{args.N}_{args.dim}_{args.l}_{args.h}_{args.f}"

  # with open("logs_width.txt", "a") as f:
  #   f.write("------------------------------------------\n")
  
    for deg in [2,3,4,5]:
        losses[deg] = []
        for width in range(1, args.N, 3):
            for i in range(func_per_deg):
              print(f"Currently fitting: func {i}, deg {deg}, width {width}")
              # Create new directory to save results for the particular function
              dir_name = os.path.join(main_dir, f"func{i}_deg{deg}_width{width}")
              os.makedirs(dir_name, exist_ok=True)
      
              # Generate function and save its coefficients
              func, (coeffs, combs) = rboolf(args.N, width, deg)
              torch.save(coeffs, f"{dir_name}/func_coeffs.pt")
              torch.save(combs, f"{dir_name}/func_combs.pt")
        
              # Generate the training dataset
              train_loader = generate_dataset(args.num_samples, args.N, args.bs)
              torch.save(train_loader,dir_name+"/train_dataloader.pt")

              # Fit the model
              model, func_summary   = fitNetwork(func, train_loader, epochs=args.epochs, N=args.N, dir_name=dir_name)
              func_summary["deg"]   = deg
              func_summary["width"] = width
              func_summary["func"]  = i
              # func_summary = func_summary[summary.columns.tolist()]
              # summary = pd.concat([summary, func_summary])
              # summary.to_csv(f"{main_dir}/test.csv")
              summary_csv = f"{main_dir}/summary.csv"
              func_summary.to_csv(summary_csv, mode='a', header=not os.path.exists(summary_csv), index=False)

        
              # Get Test Loss
              model.eval()
              loss = validate(model=model, func=func, num_samples=10000)
              losses[deg].append(loss.item())
              # print(f"\nReported Loss: {loss.item()}\n")
        
              # # Write to Logs
              # with open("logs_width.txt", "a") as f:
              #     f.write(f"\nReported Loss: {loss.item()}\n")
    print(losses)
  
    # Save to File
    df = pd.DataFrame.from_dict(losses)
    os.makedirs(dir_name, exist_ok=True)
    df.to_csv(f"{dir_name}/results_layernorm.csv")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)   
