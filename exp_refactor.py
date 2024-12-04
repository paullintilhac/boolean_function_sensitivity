# Using ../sensitivity/36test.py
# Functions whose Fourier degree is concentrated on higher weights are harder to learn for LSTMs with SGD

from pyhessian import hessian
import copy
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
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()
#from functools import partial

if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

def rboolf(N, width, deg,seed=None):
    if seed:
        torch.manual_seed(seed)
    coefficients = torch.randn(width).to(device)
    #print("coefficients initial shape: " + str(coefficients.shape) + ", width: " + str(width))
    coefficients = (coefficients-coefficients.mean())/coefficients.pow(2).sum().sqrt()
    
    combs = torch.tensor(list(itertools.combinations(torch.arange(N), deg))).to(device)
    combs = combs[torch.randperm(len(combs))][:width] # Shuffled
    return (coefficients, combs)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]= "12355"
    init_process_group(backend="gloo",
                       init_method='tcp://127.0.0.1:23456',
                       rank=rank,
                       world_size=world_size,
                       timeout=datetime.timedelta(seconds=5400)
                      )

class Trainer:
    def __init__(
            self,
            coeffs: torch.FloatTensor,
            combs: torch.FloatTensor,
            model:torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
            dir_name: str,
            width: int,
            deg: int,
            func: int,
            N: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = DDP(model,device_ids=[self.gpu_id])
        self.model.to(self.gpu_id)
        self.train_data=train_data
        self.optimizer = optimizer
        self.save_every=save_every
        self.dir_name = dir_name    
        self.summary = pd.DataFrame(columns=["deg","width","func","epoch","train_loss","val_loss","batch_size","lr","func_val_test", "top_eig", "trace"])
        self.epoch_loss = 0
        self.N = N
        self.func = func
        self.coeffs = coeffs.to(gpu_id)
        self.combs = combs.to(gpu_id)
        self.width=width
        self.deg = deg
        for batch in train_data:
            self.batch_size = len(batch)
            break
        self.lr = optimizer.param_groups[-1]['lr']
        #self.func.to(gpu_id)
    
    def makeBitTensor(self, x, N):
        y = format(x, "b")
        y = ("0"*(N-len(y))) + y
        return [int(z) for z in list(y)]
        
    def func_batch(self,x):
        binaryTensor = ((torch.tensor([self.makeBitTensor(y,self.N) for y in x])-.5)*2)
        comps = []
        #print("self.combs length: " + str(len(self.combs)))
        for elem in self.combs:
            res = torch.tensor([1]*len(x))
            for e in elem:
                bitCol = binaryTensor[:,e]
                res = torch.mul(res, bitCol)
            comps.append(res)
        comps = torch.transpose(torch.stack(comps),1,0).to(self.gpu_id)
        return torch.matmul(comps, self.coeffs).to(self.gpu_id)
        
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
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        epoch_loss = 0
        total_records = 0
        start_time = time.time()
        
        for idx, inputs in enumerate(self.train_data):
          #inputs.to(self.gpu_id)    
          targets =self.func_batch(inputs).to(self.gpu_id)
          batch_loss = self._run_batch(inputs, targets)
          epoch_loss+=batch_loss*float(len(inputs))
          total_records+=len(inputs)
          iteration = epoch*len(self.train_data)+idx+1
            
        epoch_loss/=float(total_records)
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        #print(f"Epoch time: {elapsed_time:.3f} seconds.")
        time_per_record_ms = float(elapsed_time*100)/float(total_records)
        #print(f"Epoch time: {elapsed_time:.3f} seconds. time per record (ms): {time_per_record_ms: .3f}")
        return epoch_loss

    def save_checkpoint(self,epoch):
        ckp = self.model.module.state_dict()
        torch.save(ckp,os.path.join(self.dir_name, f"model_{epoch}.pt"))
        loss_fn = lambda result, targets: (result-targets).pow(2).mean()
        print(f"Epoch {epoch} | Training checkpoint saved at model_{epoch}.pt")

    def train(self,epochs: int):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = self._run_epoch(epoch)
            
            #print("remainder: " + str(epoch % self.save_every))
            if ((epoch % self.save_every)==0 and self.gpu_id==0) or (epoch_loss < 0.02):
                #print("inside conditional")
                # self.save_checkpoint(epoch)
                #print("self.func: " + str(self.func))
                val_loss = self.validate(1000) 
                loss_fn = lambda result, targets: (result-targets).pow(2).mean()
                top_eig, trace = self.calc_hessian(copy.deepcopy(self.model.module), loss_fn=loss_fn, num_samples= 1000) 
                self.summary.loc[0] = {"deg":self.deg,
                                                       "width":self.width,
                                                       "func":self.func,
                                                       "epoch":epoch,
                                                       "train_loss":epoch_loss.cpu(),
                                                       "val_loss":val_loss.cpu(),
                                                      "batch_size": self.batch_size,
                                                      "lr":self.lr,
                                                      "func_val_test":self.func_batch([2]).cpu(),
                                                      "top_eig":top_eig, 
                                                      "trace":trace}
                #print(f"appending to {self.dir_name}/summary.csv")
                self.summary.to_csv(f"{self.dir_name}/summary.csv",mode='a', header=not os.path.exists(f"{self.dir_name}/summary.csv"), index=False)
                print(f" Epoch: {epoch}, EpochLoss: {epoch_loss:.3f}, ValidationLoss: {val_loss:.3f}")
                if epoch_loss < 0.02:
                    return
        # loss_fn = lambda result, targets: (result-targets).pow(2).mean()
        # top_eig = self.calc_hessian(copy.deepcopy(self.model.module), loss_fn=loss_fn, num_samples= 1000) 
        return

    def validate(self, num_samples):
      self.model.eval()
      inputs = torch.tensor([random.randint(0, 2**self.N-1) for _ in range(num_samples)]).to(self.gpu_id)
      targets = self.func_batch(inputs).to(self.gpu_id)
      result = self.model(inputs).to(self.gpu_id)
      loss = (result - targets).pow(2).mean()
      return loss.detach().cpu()

    def calc_hessian(self, model, loss_fn, num_samples):
        model.eval().to(self.gpu_id)
        inputs = torch.tensor([random.randint(0, 2**self.N-1) for _ in range(num_samples)]).to(self.gpu_id)
        targets = self.func_batch(inputs).to(self.gpu_id)
        data = (inputs, targets)        

        # Estimate using PyHessian -- very good
        hess_mod = hessian(model, loss_fn, data, cuda=True)
        for param in model.parameters():
            param.grad = None
        top_eigs, top_eigVs = hess_mod.eigenvalues(maxIter = 200)
        top_eig = top_eigs[0] 
        trace = hess_mod.trace()


        # Manual Calculation -- to double-check (does not work -- returns all zeroes, need to validate)
        # from torch.autograd.functional import hessian as hessian2
        # def func(params): # Loss function, but in terms of model parameters, for hessian calculation

        #     # Set the parameters in the model (GPT assist)
        #     idx = 0
        #     for param in model.parameters():
        #         param_length = param.numel()
        #         param.data = params[idx:idx + param_length].view_as(param)
        #         idx += param_length

        #     output = model(inputs)
        #     loss = loss_fn(output, targets)
        #     return loss
        # params = torch.cat([p.flatten() for p in model.parameters()])

        # hess = hessian2(func, params)
        # for p in model.parameters():
        #     print(p)
        
        # eigvals = torch.linalg.eigvals(hess).abs()
        # top_eig2 = torch.topk(eigvals, 1)[0]
        
        return top_eig, trace






    
def load_train_objs(wd,dropout,lr,num_samples, N, dim,h,l,f,rank):
        train_set = torch.tensor([random.randint(0, 2**N-1) for _ in range(int(num_samples))]).to(rank)

        model = Transformer(dropout,N, dim, h, l, f, 1e-5,rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=wd)
        return train_set, model, optimizer                


def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--f', type=int, default=64)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--lr', type=str,default = "1e-5")
    parser.add_argument('--wd', type=float,default = .1)
    parser.add_argument('--dropout', type=float,default = .2)
    parser.add_argument('--repeat', type=int, default=100)


    return parser.parse_args()

def main(rank, args,world_size,coefs,combs,main_dir,deg,width,i):
      #print("func in main: " + str(func))
      ddp_setup(rank,world_size)
      # Create new directory to save results for the particular function
      #dir_name = os.path.join(main_dir, f"deg{deg}_width{width}_func{i}")
      #os.makedirs(dir_name, exist_ok=True)
        
      # Generate function and save its coefficients
      #func = rboolf_old(args.N,  deg)
      #print("generating dataset with " + str(args.num_samples)+" records. ")
      train_set,model,optimizer = load_train_objs(args.dropout, args.wd,args.lr,args.num_samples,args.N,args.dim,args.h,args.l,args.f,rank)
      total_params = sum(p.numel() for p in model.parameters())
      print("Model Parameter Count: " + str(total_params))
      model.to(rank)
      #print("epochs: " + str(args.epochs) + ", bs: " + str(args.bs))
      train_loader = DataLoader(
          train_set,
          shuffle=False,
          batch_size=args.bs,
          sampler = DistributedSampler(train_set)
      )
             
      trainer = Trainer(coefs,combs, model,
                        train_loader,
                        optimizer,
                        gpu_id=rank,
                        save_every=args.save_every,
                        dir_name= main_dir,
                        width=width,
                        deg=deg,
                        func=i,
                        N=args.N)
      print("trainer.func_batch([2, 3]): " + str(trainer.func_batch([2,3])))
      trainer.train(args.epochs)
      print("finished training, cleaning up process group...")
      destroy_process_group()
      
      print("finished cleaning up process group")
      return

if __name__ == "__main__":
    arguments = parse_args()
    print(arguments)
    
    losses = {}
    func_per_deg = arguments.repeat
    main_dir = f"N{arguments.N}_HidDim{arguments.dim}_L{arguments.l}_H{arguments.h}_FFDim{arguments.f}_4k_hessiantest"
    os.makedirs(main_dir, exist_ok=True)
    # with open("logs_width.txt", "a") as f:
    #   f.write("------------------------------------------\n")
    for i in [0,1]:
    # for i in range(func_per_deg):
        #for deg in [2]:
        for deg in range(5,6):
            losses[deg] = []
            #for width in range(1, arguments.N, 5):
            for width in [20,14,7,1]:

            #for width in [1]:
                start_time = time.time()
                #world_size = torch.cuda.device_count()
                #args["world_size"]=world_size
                print(f"Generating: func {i}, deg {deg}, width {width}")
                seedNum = int(str(i)+str(deg)+str(width))
                (coefs, combs) = rboolf(arguments.N, width, deg,seed=seedNum)
                mp.set_start_method('spawn',force = True)

                torch.set_num_threads(1)
                mp.spawn(main,args=(arguments,arguments.world_size,coefs,combs,main_dir,deg,width,i,),nprocs=arguments.world_size,join=True)
                print("returned from mp.spwan")
                end_time = time.time()
        
                elapsed_time = round((end_time - start_time)/60,3)
                print("elapsed time for whole training process: " + str(elapsed_time))
    