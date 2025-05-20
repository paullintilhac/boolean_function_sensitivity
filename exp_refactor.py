# Using ../sensitivity/36test.py
# Functions whose Fourier degree is concentrated on higher weights are harder to learn for LSTMs with SGD

from pyhessian.hessian import hessian
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import random
import argparse
from new_transformer import Transformer
from transformer import Transformer as Transformer2
from transformer_old import Transformer as Transformer3
import os
import itertools
import time
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp, barrier,get_rank, get_world_size, broadcast
import numpy as np, math, collections, itertools
from tqdm import tqdm

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()
#from functools import partial

if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
    
def get_weight_norm(model):
       total_norm = 0.0
       for p in model.parameters():
           if p.requires_grad:
               param_norm = p.data.norm(2)
               total_norm += param_norm.item() ** 2
       return total_norm ** 0.5

def rboolf(N, width, deg,seed=None):
    if seed:
        torch.manual_seed(seed)
    coefficients = torch.randn(width).to(device)
    #print("coefficients initial shape: " + str(coefficients.shape) + ", width: " + str(width))
    coefficients = (coefficients)/coefficients.pow(2).sum().sqrt()
    print("coefficients: " + str(coefficients))
    combs = torch.tensor(list(itertools.combinations(torch.arange(N), deg))).to(device)
    combs = combs[torch.randperm(len(combs))][:width] # Shuffled
    return (coefficients, combs)
from torch.fft import fft  # we will use a shortcut fast hadamard via bit-reordering

    
def ddp_restriction_tester(f, n, k, eps, delta=1/3,
                           max_s=12, device=None):
    rank  = get_rank()                     # 0 … world_size-1
    world = get_world_size()

    s     = min(max_s, math.ceil(math.log2(4*k/eps)))
    T     = math.ceil(math.log(1/delta))
    print("s: " + str(s) + ", T: " + str(T))
    rej   = torch.zeros(1, device=device)
    pow2_full = (1 << torch.arange(n, device=device, dtype=torch.long))

    for _ in range(T):
        
        if rank == 0:
            S = torch.tensor(random.sample(range(n), s),
                             dtype=torch.int64,
                             device=rank)        # <<< on GPU
            fixed_bits = torch.randint(0, 2, (n-s,),
                                       dtype=torch.uint8,
                                       device=rank)          # <<< on GPU
        else:
            S = torch.empty(s,  dtype=torch.int64,  device=device)   # <<< GPU
            fixed_bits = torch.empty(n-s, dtype=torch.uint8, device=device)

        broadcast(S, 0); broadcast(fixed_bits, 0)

        num   = 1 << s
        chunk = torch.arange(rank, num, world, dtype=torch.long).to(device)
        X     = torch.zeros(chunk.numel(), n, dtype=torch.uint8, device=device)
        for j, bit in enumerate(S):
            X[:, bit] = ((chunk >> j) & 1).to(torch.uint8)
        X[:, [i for i in range(n) if i not in S.tolist()]] = fixed_bits.to(device)
        ints = (X.long() * pow2_full).sum(dim=1)
        vals = f(ints).float().to(device)                           # (chunk,)
        # gather vals into a length-num vector “coeff”
        coeff = torch.zeros(num, dtype=torch.float32, device=device)
        #print("chunk: " + str(chunk) + ", vals: " + str(vals))
        coeff.index_add_(0, chunk, vals)
        all_reduce(coeff, op=ReduceOp.SUM)            # aggregate across GPUs
        
        coeff = hadamard_transform(coeff).abs_()**2   # energy
        print("coeff: " + str(coeff[:10]))
        total = coeff.sum()
        head  = coeff.topk(k).values.sum() if k+1 < num else torch.tensor(0., device=device)
        tail  = total - head

        print("tail: " + str(tail))
        if tail > eps * total / 2:
            rej += 1
    all_reduce(rej, op=ReduceOp.MAX)
    return (rej.item() == 0)    
import random
def hadamard_transform(vals):
    """Fast Walsh–Hadamard (in-place) for length power-of-two tensor (CPU or CUDA)."""
    h = 1
    n = vals.shape[0]
    while h < n:
        # pairwise butterfly:  vals[i], vals[i+h]  →  (a+b, a−b)
        vals = vals.view(-1, 2*h)
        first, second = vals[:, :h].clone(), vals[:, h:2*h].clone()
        vals[:, :h] = first + second
        vals[:, h:2*h] = first - second
        vals = vals.view(n)
        h <<= 1
    vals /= math.sqrt(n)          # make it orthonormal ‖f‖₂ = ‖ĥ‖₂
    return vals       
    
def ddp_setup(rank, world_size,backend):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]= "23456"
    if backend == "gloo":
        init_process_group(backend="gloo",
                       init_method='tcp://127.0.0.1:23456',
                       rank=rank,
                       world_size=world_size,
                       timeout=datetime.timedelta(seconds=5400)
                      )
    else:
        init_process_group(backend="nccl",
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
            n_samples: int,
            backend:str,
            stop_loss:float,
            ln_eps:float,
            ln:bool,
            save_checkpoints: bool,
            f: float,
            d: int,
            l: int,
            h: int,
            dropout: float,
            wd: float
            #test_dataloader:DataLoader
    ) -> None:
        self.gpu_id = gpu_id
        self.model = DDP(model,device_ids=[self.gpu_id])
        self.model.to(self.gpu_id)
        self.train_data=train_data
        print("length of train dataloader in init: " + str(len(train_data)))
        #self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.save_every=save_every
        self.ln_eps=ln_eps
        self.ln = ln
        self.wd = wd
        self.dir_name = dir_name  
        self.save_checkpoints = save_checkpoints
        self.dropout = dropout
        self.summary = pd.DataFrame(columns=
                                ["deg",
                                 "width",
                                 "func",
                                 "epoch",
                                 "train_loss",
                                 "val_loss",
                                 "batch_size",
                                 "lr",
                                 "n_samples",
                                 "func_val_test",
                                 "time_elapsed",
                                 "backend",
                                 "top_eig",
                                 "trace",
                                 "stop_loss",
                                 "ln_eps",
                                 "ln",
                                 "weight_norm",
                                  "l",
                                 "d",
                                 "f",
                                 "h",
                                 "dropout",
                                 "wd"])
        self.stop_loss = stop_loss
        self.epoch_loss = 0
        self.N = N
        self.func = func
        self.coeffs = coeffs.to(gpu_id)
        self.combs = combs.to(gpu_id)
        self.width=width
        self.deg = deg
        self.n_samples = n_samples
        self.l = l
        self.d = d
        self.f = f
        self.h = h
        for batch in train_data:
            self.batch_size = len(batch)
            break
        self.lr = optimizer.param_groups[-1]['lr']
        self.backend = backend
        #self.func.to(gpu_id)

    def makeBitTensor(self,x, N):
        y = format(x, "b")
        y = ("0"*(N-len(y))) + y
        return [int(z) for z in list(y)]    
    

    # def YZ19_sparse_tester(self, N, k, eps,H):
    #     """
    #     One-sided tester:  returns True  (ACCEPT) if f is k-sparse,
    #                       returns False (REJECT) if ℓ2-tail > eps.
    #     Uses m = 64 * k/eps^4 * ln(12/eps) queries.  (δ = 1/3 by default.)
    #     """
    #     H = H.to(self.gpu_id)
    #     # -- parameters from Theorem 3.1 (FOCS'19) --------------------------
    #     d, _        = H.shape


    #     q     = int(math.ceil(64 * k * eps**-4 * math.log(12/eps)))  # sample budget

    #     print("q: " + str(q) + ", d: " + str(d))
    #     # 1. pick a random linear hash  H : F2^n -> F2^d
    #     #    represented by a d×n binary matrix chosen uniformly.
    #     print("H dimension: " + str(H.shape))
    #     print("head of H: " + str(H[:5,:5]))
    #     # 2. draw q random x; bucket by  Hx
    #     num_buckets  = 1 << d
    #     pow2        = (1 << torch.arange(d, dtype=torch.long, device=self.gpu_id))  # 1,2,4,…
    #     print("pow2 shape: " + str(pow2.shape))
    #     sums   = torch.zeros(num_buckets, dtype=torch.float32, requires_grad=False).to(self.gpu_id)
    #     counts = torch.zeros(num_buckets, dtype=torch.float32, requires_grad=False).to(self.gpu_id)
        
    #     print("number of batches: " + str(len(self.test_dataloader)))
    #     for idx, batch in tqdm(enumerate(self.test_dataloader)):
    #         #print("batch dimension: " + str(batch.shape))
    #         binaryTensor = ((torch.tensor([self.makeBitTensor(y,N) for y in batch], requires_grad=False))).to(torch.float16).to(self.gpu_id)
    #         #print("H: " + str(H))
    #         #print("binaryTensor: " + str(binaryTensor))
    #         prod = (binaryTensor @ torch.transpose(H,0,1)).to(torch.int32)
    #         b = (prod.to(torch.int32) &1 ).to(torch.uint8) 
    #         b = (b.long() * pow2).sum(dim=1)
    #         #print("b shape: " + str(b.shape))
    #         outputs = self.model(batch).to(self.gpu_id)
    #         sums.index_add(0,   b, outputs)
    #         counts.index_add(0, b, torch.ones_like(outputs, requires_grad=False))
    #     # 3. estimate bucket energies  |g_H(u)|^2  via sample mean
    #     all_reduce(sums, op=ReduceOp.SUM)
    #     all_reduce(counts, op=ReduceOp.SUM)
    #     means         = torch.zeros_like(sums, requires_grad=False)
    #     nonempty      = counts > 0
    #     means[nonempty] = sums[nonempty] / counts[nonempty]

    #     energies = (means*means).detach().cpu().numpy()
    #     tail_energy = energies[np.argpartition(-energies, k)[k:]].sum()
    #     print("tail_energy: "  + str(tail_energy))
    #     if tail_energy > 0.75*eps:               # 0.75 factor = constant slack in proof
    #         return False                  # REJECT – evidence tail energy large
    #     else:
    #         return True
    
        
    def func_batch(self,x):
        binaryTensor = ((torch.tensor([self.makeBitTensor(y,self.N) for y in x])-.5)*2)
        #binaryTensor = ((torch.tensor([makeBitTensor(y,self.N) for y in x])))
        #print("binaryTensor: " + str(binaryTensor))
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
        criterion = torch.nn.MSELoss()
        loss =  criterion(result, targets)
        # loss =(result-targets).pow(2).mean()
        (loss).backward()
        self.optimizer.step()
        return loss.detach().cpu()
    
    def _run_epoch(self,epoch):
        
        b_sz = len(next(iter(self.train_data)))
        epoch_loss = 0
        total_records = 0
        start_time = time.time()
        #print("number of batches: "  + str(len(self.train_data)))
        for idx, inputs in enumerate(self.train_data):
          #print("length of inputs in training: " + str((inputs.shape)))
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

    def save_checkpoint(self,epoch,model_name):
        os.makedirs(os.path.join(self.dir_name, model_name), exist_ok=True)
        full_model_name = model_name+"/epoch-"+str(epoch)+".pt"
        ckp = self.model.module.state_dict()
        torch.save(ckp,os.path.join(self.dir_name, full_model_name))
        loss_fn = lambda result, targets: (result-targets).pow(2).mean()
        print(f"Epoch {epoch} | Training checkpoint saved at model_{epoch}.pt")

    def train(self,epochs: int):
        self.model.train()
        
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = self._run_epoch(epoch)
            
            if ((epoch % self.save_every)==0 and self.gpu_id==0) or (epoch_loss < self.stop_loss):
            # if ((((epoch+1) % self.save_every)==0 or epoch==0) and self.gpu_id==0):

                #print("inside conditional")
                if self.save_checkpoints:
                    self.save_checkpoint(epoch,"degree-"+str(self.deg)+"/width-"+str(self.width)+"/func-"+str(self.func))
                end_time = time.time()
                elapsed_time = round((end_time - start_time)/60,3) 

                #print("self.func: " + str(self.func))
                val_loss = self.validate(1000) 
                loss_fn = lambda result, targets: (result-targets).pow(2).mean()
                start_time_hessian = time.time()
                top_eig, trace = self.calc_hessian(copy.deepcopy(self.model.module), loss_fn=loss_fn, num_samples= 1000,device_id = self.gpu_id)
                #weight_norm = 0
                weight_norm = get_weight_norm(self.model.module)
                #weight_norm = torch.linalg.norm(self.model.weight)
                #top_eig=0
                #trace = 0
                end_time_hessian = time.time()
                elapsed_time_hessian = round((end_time_hessian - start_time_hessian)/60,3) 
                print("elapsed time norm: " + str(elapsed_time_hessian))
                self.summary.loc[0] = {"deg":self.deg,
                                       "width":self.width,
                                       "func":self.func,
                                       "epoch":epoch,
                                       "train_loss":epoch_loss.cpu(),
                                       "val_loss":val_loss.cpu(),
                                      "batch_size": self.batch_size,
                                      "lr":self.lr,
                                      "n_samples":self.n_samples,
                                      "func_val_test":self.func_batch([2]).cpu(),
                                      "time_elapsed":elapsed_time,
                                      "backend":self.backend,
                                      "top_eig":top_eig,
                                      "trace":trace,
                                      "stop_loss": self.stop_loss,
                                      "ln_eps": self.ln_eps,
                                      "ln": self.ln,
                                      "weight_norm": weight_norm,
                                       "l":self.l,
                                       "d":self.d,
                                       "f":self.f,
                                       "h":self.h,
                                       "dropout":self.dropout,
                                       "wd":self.wd
                                      }
                
               

                self.summary.to_csv(f"{self.dir_name}/summary.csv",mode='a', header=not os.path.exists(f"{self.dir_name}/summary.csv"), index=False)
                print(f" Epoch: {epoch}, TimeElapsed: {elapsed_time}, EpochLoss: {epoch_loss:.3f}, ValidationLoss: {val_loss:.3f}")
            flag = torch.zeros(1).to(self.gpu_id)
            if epoch_loss<self.stop_loss:
                 flag += 1
            all_reduce(flag, op=ReduceOp.SUM)
            if flag > 0:
                break
            barrier()
        
        
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

    def calc_hessian(self, model, loss_fn, num_samples,device_id):
        model.eval().to(self.gpu_id)
        inputs = torch.tensor([random.randint(0, 2**self.N-1) for _ in range(num_samples)]).to(self.gpu_id)
        targets = self.func_batch(inputs).to(self.gpu_id)
        data = (inputs, targets)        

        # Estimate using PyHessian -- very good
        hess_mod = hessian(model, loss_fn, data, device=device_id)
        for param in model.parameters():
            param.grad = None
        top_eigs, top_eigVs = hess_mod.eigenvalues(maxIter = 200)
        top_eig = top_eigs[0] 
        trace = hess_mod.trace()
        
        return top_eig, np.mean(trace)


    
def load_train_objs(wd,dropout,lr,num_samples, N, dim, dim2, h, l, f, rank, ln_eps, ln):
        train_set = torch.tensor([random.randint(0, 2**N-1) for _ in range(int(num_samples))]).to(rank)    

        model = Transformer2(dropout,N, dim, dim2, h, l, f, ln_eps,rank,ln)
        total_params = sum(p.numel() for p in model.parameters())
        print(model)
        print("Model_Mid Parameter Count: " + str(total_params))
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=wd)
        return train_set, model, optimizer                


def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--dim2', type=int, default=20)
    parser.add_argument('--f', type=int, default=64)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--lr', type=str,default = "1e-5")
    parser.add_argument('--wd', type=float,default = .1)
    parser.add_argument('--dropout', type=float,default = .2)
    parser.add_argument('--backend',type=str, default = "gloo")
    parser.add_argument('--stop_loss', type=float,default = .02)
    parser.add_argument('--ln_eps', type=float,default = 1e-5)
    parser.add_argument('--ln', action='store_true')
    parser.add_argument('--save_checkpoints', action='store_true')


    return parser.parse_args()

def main(rank, args,world_size,coefs,combs,main_dir,deg,width,i):
      #print("func in main: " + str(func))
      ddp_setup(rank,world_size,args.backend)
      # Create new directory to save results for the particular function
      #dir_name = os.path.join(main_dir, f"deg{deg}_width{width}_func{i}")
      #os.makedirs(dir_name, exist_ok=True)
      k=2
      eps = .01
    
      #d     = int(math.ceil( 2*math.log2( 24*k/eps ) ))            # hash output bits

      #q     = int(math.ceil(64 * k * eps**-4 * math.log(12/eps)))  # sample budget

      train_set,model,optimizer = load_train_objs(args.dropout,
                                                  args.wd,args.lr,
                                                  args.num_samples,
                                                  args.N,
                                                  args.dim,
                                                  args.dim2,
                                                  args.h,
                                                  args.l,
                                                  args.f,
                                                  rank,
                                                  args.ln_eps,
                                                  args.ln
                                                  )
      model.to(rank)
      train_loader = DataLoader(
          train_set,
          shuffle=False,
          batch_size=args.bs,
          sampler = DistributedSampler(train_set)
      )
      # #print("train length in main: " + str(len(train_loader)) +  ", len(train_loader[0]): "+ str(len(train_loader[0])))
      # test_dataloader = DataLoader(
      #     test_set,
      #     shuffle=False,
      #     batch_size=2048,
      #     sampler = DistributedSampler(test_set)
      # )   
      trainer = Trainer(coefs,combs, model,
                        train_loader,
                        optimizer,
                        gpu_id=rank,
                        save_every=args.save_every,
                        dir_name= main_dir,
                        width=width,
                        deg=deg,
                        func=i,
                        N=args.N,
                        n_samples = args.num_samples,
                        backend = args.backend,
                        stop_loss = args.stop_loss,
                        ln_eps = args.ln_eps,
                        ln = args.ln,
                        save_checkpoints=args.save_checkpoints,
                        l=args.l,
                        d=args.dim,
                        f=args.f,
                        h=args.h,
                        dropout=args.dropout,
                        wd=args.wd
                        #test_dataloader=test_dataloader
                        )
      trainer.model.eval()
      print("trainer.func_batch([2, 3]): " + str(trainer.func_batch([2,3])))
      trainer.train(args.epochs)
      barrier()
      print("about to run sparsity property test")
      delta = .01
      # trials= int(math.ceil(math.log(1/delta)))

      # sparsity = True
      # for counter in range(trials):
      #     H = torch.tensor(np.random.randint(0, 2, size=(d, args.N)), requires_grad=False).to(torch.float16)

      #     if not trainer.YZ19_sparse_tester( 20, 2, .1, H):
      #         sparsity= False
      def f(X):
        with torch.no_grad():
            return trainer.model(X.to(device)).view(-1)
      sparsity = ddp_restriction_tester(f,args.N, k, eps, delta, device=rank)
      print("sparsity: " + str(sparsity))
      print("finished training, cleaning up process group...")
      destroy_process_group()
      print("finished cleaning up process group")
      return

if __name__ == "__main__":
    arguments = parse_args()
    arguments.save_checkpoints = False
    print(arguments)
    losses = {}
    func_per_deg = arguments.repeat
    main_dir = f"HYPERPARAM_TESTS_MECHINTERP"
    os.makedirs(main_dir, exist_ok=True)
    # with open("logs_width.txt", "a") as f:
    #   f.write("------------------------------------------\n")

    for i in [2]:
        for deg in [2]:
            losses[deg] = []
            #for width in range(1, arguments.N, 5):
            for width in [3]:
                start_time = time.time()
                #world_size = torch.cuda.device_count()
                #args["world_size"]=world_size 
                print(f"Generating: func {i}, deg {deg}, width {width}")
                seedNum = int(str(i)+str(deg)+str(width))
                (coefs, combs) = rboolf(arguments.N, width, deg,seed=seedNum)
                torch.save(coefs,os.path.join(main_dir, f"coefs_func{i}_deg{deg}_width{width}.pt"))
                torch.save(combs,os.path.join(main_dir, f"combs_func{i}_deg{deg}_width{width}.pt"))
                
                mp.set_start_method('spawn',force = True)

                torch.set_num_threads(1)
                mp.spawn(main,args=(arguments,arguments.world_size,coefs,combs,main_dir,deg,width,i,),nprocs=arguments.world_size,join=True)
                print("returned from mp.spwan")
                end_time = time.time()
        
                elapsed_time = round((end_time - start_time)/60,3)
                print("elapsed time for whole training process: " + str(elapsed_time))