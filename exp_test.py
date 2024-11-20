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
import itertools
import time

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
    
def fitNetwork(function, loader, N, epochs, dir_name,n_devices):
    #dist.init_process_group(backend='nccl', init_method='env://', rank=n_devices, world_size=n_devices)

    lr = 5e-5 
    weight_decay = .1
  
    model = torch.nn.DataParallel(Transformer(N, args.dim, args.h, args.l, args.f, 1e-5),device_ids=range(n_devices))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    movAvg = 0
    summary = pd.DataFrame(columns=["epoch","iter", "loss"])
    # dir_name = f"{args.N}_{args.dim}_{args.l}_{args.h}_{args.f}"
    if not cuda_avail:
        model.to(device)
    else:    
        model.cuda()
        
    for epoch in range(epochs):   
        epoch_loss = 0
        total_records = 0
        #print("loader[0]: " + str(loader[0]))
        batch_losses = []

        start_time = time.time()

        for idx, inputs in enumerate(loader):
          #print("inputs shape: "  + str(inputs.shape))
          #print("idx: " + str(idx))
          
          if not cuda_avail:
              inputs.to(device)
          else:    
              inputs.to('cuda', non_blocking=True)

          targets = torch.FloatTensor([float(function(x)) for x in inputs]).to(device)
          result = model(inputs)
          #loss = -(result*targets).mean()
          loss =  (result-targets).pow(2).mean()
          batch_losses.append(loss.detach().cpu())
          epoch_loss+=loss.detach().cpu()*float(len(inputs))
          total_records+=len(inputs)
          (loss).backward()
          optimizer.step()
          optimizer.zero_grad()
        
          iteration = epoch*len(loader)+idx+1
          # epoch_loss = epoch_loss.detach().cpu()
        #print("batch losses head: " +str(batch_losses[:5]))
        #print("mean of batch losses: " + str(np.mean(np.array(batch_losses))))
        epoch_loss/=float(total_records)
        # Your code here
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_per_record_ms = float(elapsed_time*100)/float(total_records)
        print(f"Epoch time: {elapsed_time:.3f} seconds. time per record (ms): {time_per_record_ms: .3f}")
        if epoch_loss < 0.01:
          break	
        if (epoch) % 5 == 0:
            #print("iter: " + str(iteration) + ", loss: " +str(movAvg))
            print("iterations: " + str(iteration) + ", total records: " + str(total_records) + ", epoch: " + str(epoch) + ", epoch_loss: " + str(epoch_loss))

            summary.loc[len(summary)] = {"epoch":epoch,"iter":iteration, "loss":epoch_loss}
            summary.to_csv(f"{dir_name}/curr_func.csv")

        if (epoch) % 20 == 0:
            val_loss = validate(model, function, num_samples=1000)
            val_loss2 = validate(model, function, num_samples=1000)
            print("val loss2: " + str(val_loss2))
            print(f"Iterations: {iteration}, Epoch: {epoch}, EpochLoss: {epoch_loss:.3f}, ValidationLoss: {val_loss:.3f}, TotalRecords: {total_records:.3f}")
            path = os.path.join(dir_name, f"model_{epoch}.pt")
            torch.save(model.state_dict(), path)  
    dist.destroy_process_group()

    return model, summary

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


def rboolf(N, width, deg):
    coefficients = torch.randn(width).to(device)
    coefficients = (coefficients-coefficients.mean())/coefficients.pow(2).sum().sqrt()
    
    combs = torch.tensor(list(itertools.combinations(torch.arange(N+1), deg))).to(device)
    combs = combs[torch.randperm(len(combs))][:width] # Shuffled

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
      model.eval()
      inputs = torch.tensor([random.randint(0, 2**args.N-1) for _ in range(num_samples)]).to(device)
      targets = torch.FloatTensor([float(func(x)) for x in inputs]).to(device)
      result = model(inputs).to(device)
      loss = (result - targets).pow(2).mean()
      return loss.detach()

def generate_dataset(num_samples, N, batch_size):
    inputs = torch.tensor([random.randint(0, 2**N-1) for _ in range(int(num_samples))])
    train_loader = DataLoader(inputs, shuffle=True, batch_size=batch_size,num_workers = 128,pin_memory=True)
    return train_loader

def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--n_devices', type=int, default=1)
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
    func_per_deg = args.repeat
    main_dir = f"N{args.N}_HidDim{args.dim}_L{args.l}_H{args.h}_FFDim{args.f}_lr6e6_s10k_b64"
    os.makedirs(main_dir, exist_ok=True)
  # with open("logs_width.txt", "a") as f:
  #   f.write("------------------------------------------\n")
    print("generating dataset with " + str(args.num_samples)+" records. ")
    train_loader = generate_dataset(args.num_samples, args.N, args.bs)
    torch.save(train_loader,main_dir+"/train_dataloader.pt")
    for i in range(func_per_deg):
        for deg in [5]:
            losses[deg] = []
            #for width in range(1, args.N, 3):
            for width in [16]:
              print(f"Generating: func {i}, deg {deg}, width {width}")

              # Create new directory to save results for the particular function
              dir_name = os.path.join(main_dir, f"deg{deg}_width{width}_func{i}")
              os.makedirs(dir_name, exist_ok=True)
      
              # Generate function and save its coefficients
              #func = rboolf_old(args.N,  deg)

              func, (coeffs, combs) = rboolf(args.N, width, deg)
              torch.save(coeffs, f"{dir_name}/func_coeffs.pt")
              torch.save(combs, f"{dir_name}/func_combs.pt")
              # Generate the training dataset
              
              print(f"fitting function: func {i}, deg {deg}, width {width}")

              # Fit the model
              
              #torch.multiprocessing.spawn(fitNetwork, args=(func,train_loader_,), nprocs=world_size)
              model, func_summary   = fitNetwork(func, train_loader, epochs=args.epochs, N=args.N, dir_name=dir_name,n_devices=args.n_devices)
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
              loss = validate(model=model, func=func, num_samples=1000)
              losses[deg].append(loss.item())
              # print(f"\nReported Loss: {loss.item()}\n")
        
              # # Write to Logs
              # with open("logs_width.txt", "a") as f:
              #     f.write(f"\nReported Loss: {loss.item()}\n")
    print(losses)
  
    # Save to File
    df = pd.DataFrame.from_dict(losses)
    os.makedirs(dir_name, exist_ok=True)
    df.to_csv(f"{main_dir}/results_layernorm_test.csv")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)   
