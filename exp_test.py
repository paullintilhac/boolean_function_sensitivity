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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


def rboolf_full(seed,N, width, deg,x):
    torch.manual_seed(seed)
    coefficients = torch.randn(width).to(device)
    #print("coefficients initial shape: " + str(coefficients.shape) + ", width: " + str(width))
    coefficients = (coefficients-coefficients.mean())/coefficients.pow(2).sum().sqrt()
    combs = torch.tensor(list(itertools.combinations(torch.arange(N+1), deg))).to(device)
    combs = combs[torch.randperm(len(combs))][:width] # Shuffled
    binary = f"{x:0{N}b}"+"0"
    comps = []
    for elem in combs:
        res = 1
        for e in elem:
            bit = 1 if int(binary[e]) else -1
            res *= bit
        comps.append(res)
        #print("coeffiients: " + str(coefficients) + ", comps shape: " + str(comps))
    return torch.dot(coefficients, torch.tensor(comps, dtype=torch.float32).to(device))
    
def rboolf(N, width, deg):
    coefficients = torch.randn(width).to(device)
    #print("coefficients initial shape: " + str(coefficients.shape) + ", width: " + str(width))
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
        #print("coeffiients: " + str(coefficients) + ", comps shape: " + str(comps))
        return torch.dot(coefficients, torch.tensor(comps, dtype=torch.float32).to(device))
    return func, (coefficients, combs)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]= "12355"
    init_process_group(backend="nccl",rank=rank, world_size=world_size)
    

class Trainer:
    def __init__(
            self,
            func,
            model:torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
            dir_name: str,
            width: int,
            deg: int,
            N: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = DDP(model,device_ids=[self.gpu_id])
        self.model.to(self.gpu_id)
        self.train_data=train_data
        self.optimizer = optimizer
        self.save_every=save_every
        self.dir_name = dir_name    
        self.summary = pd.DataFrame(columns=["epoch","train_loss","val_loss"])
        self.epoch_loss = 0
        self.N = N
        self.func = func

    def _run_batch(self,inputs, targets):
        self.optimizer.zero_grad()
        inputs.to(self.gpu_id)
        result = self.model(inputs)
        #loss = -(result*targets).mean()
        loss =  (result-targets).pow(2).mean()
        (loss).backward()
        self.optimizer.step()
        return loss.detach().cpu()
    
    def _run_epoch(self,epoch):
        
        #b_sz = len(next(iter(self.train_data))[0])
        b_sz = len(next(iter(self.train_data)))
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        epoch_loss = 0
        total_records = 0
        start_time = time.time()
        
        for idx, inputs in enumerate(self.train_data):
          inputs.to(self.gpu_id)          
          targets = torch.FloatTensor([float(self.func(x)) for x in inputs]).to(self.gpu_id)
          batch_loss = self._run_batch(inputs, targets)
          epoch_loss+=batch_loss*float(len(inputs))
          total_records+=len(inputs)
          iteration = epoch*len(self.train_data)+idx+1
            
        epoch_loss/=float(total_records)
        # Your code here
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"Epoch time: {elapsed_time:.3f} seconds.")
        time_per_record_ms = float(elapsed_time*100)/float(total_records)
        print(f"Epoch time: {elapsed_time:.3f} seconds. time per record (ms): {time_per_record_ms: .3f}")
        return epoch_loss

    def save_checkpoint(self,epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp,os.path.join(self.dir_name, f"model_{epoch}.pt"))
        print(f"Epoch {epoch} | Training checkpoint saved at model_{epoch}.pt")

    def train(self,epochs: int):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = self._run_epoch(epoch)
            #print("remainder: " + str(epoch % self.save_every))
            if (epoch % self.save_every)==0 and self.gpu_id==0:
                #print("inside conditional")
                self.save_checkpoint(epoch)
                #print("self.func: " + str(self.func))
                val_loss = self.validate(1000)
                self.summary.loc[len(self.summary)] = {"epoch":epoch, "train_loss":epoch_loss,"val_loss":val_loss}
                self.summary.to_csv(f"{self.dir_name}/curr_func.csv")
                print(f" Epoch: {epoch}, EpochLoss: {epoch_loss:.3f}, ValidationLoss: {val_loss:.3f}")
            
    

    def validate(self, num_samples):
      self.model.eval()
      inputs = torch.tensor([random.randint(0, 2**self.N-1) for _ in range(num_samples)]).to(self.gpu_id)
      targets = torch.FloatTensor([float(self.func(x)) for x in inputs]).to(self.gpu_id)
      result = self.model(inputs).to(self.gpu_id)
      loss = (result - targets).pow(2).mean()
      return loss.detach().cpu()
    
def load_train_objs(num_samples, N, dim,h,l,f,rank):
        train_set = torch.tensor([random.randint(0, 2**N-1) for _ in range(int(num_samples))]).to(rank)
        lr = 6e-6 
        weight_decay = .1
        model = Transformer(N, dim, h, l, f, 1e-5,rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        return train_set, model, optimizer                


def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--world_size', type=int, default=8)
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

def main(rank, args,world_size,func,main_dir,deg,width,i):
      print("func in main: " + str(func))
      print(func(256))
      ddp_setup(rank,world_size)
      # Create new directory to save results for the particular function
      dir_name = os.path.join(main_dir, f"deg{deg}_width{width}_func{i}")
      os.makedirs(dir_name, exist_ok=True)
        
      # Generate function and save its coefficients
      #func = rboolf_old(args.N,  deg)
      #print("generating dataset with " + str(args.num_samples)+" records. ")
      train_set,model,optimizer = load_train_objs(args.num_samples,args.N,args.dim,args.h,args.l,args.f,rank)
      model.to(rank)
      #print("epochs: " + str(args.epochs) + ", bs: " + str(args.bs))
      train_loader = DataLoader(
          train_set,
          shuffle=False,
          batch_size=args.bs,
          sampler = DistributedSampler(train_set)
      )
             
      trainer = Trainer(func, model, train_loader,optimizer,gpu_id=rank,save_every=10,dir_name= dir_name,width=width,deg=deg,N=args.N)
      trainer.train(args.epochs)
      destroy_process_group()

global func     
from functools import partial

if __name__ == "__main__":
    arguments = parse_args()
    print(arguments)
    
    losses = {}
    func_per_deg = arguments.repeat
    main_dir = f"N{arguments.N}_HidDim{arguments.dim}_L{arguments.l}_H{arguments.h}_FFDim{arguments.f}_refactor"
    os.makedirs(main_dir, exist_ok=True)
  # with open("logs_width.txt", "a") as f:
  #   f.write("------------------------------------------\n")
    
    for i in range(func_per_deg):
        for deg in [5]:
            losses[deg] = []
            #for width in range(1, args.N, 3):
            for width in [16]:
                #world_size = torch.cuda.device_count()
                #args["world_size"]=world_size
                seed=4
                print(f"Generating: func {i}, deg {deg}, width {width}")
                partial_func = partial(rboolf_full,seed,arguments.N, width, deg)
                mp.spawn(main,args=(arguments,arguments.world_size,partial_func,main_dir,deg,width,i,),nprocs=arguments.world_size)
    