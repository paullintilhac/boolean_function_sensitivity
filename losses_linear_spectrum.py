# Using ../sensitivity/36test.py
# Functions whose Fourier degree is concentrated on higher weights are harder to learn for LSTMs with SGD

import numpy as np
import torch
import random
from math import log
import math
import sys
import os
import argparse
__file__ = "linear_spectrum_small"

print("done loading")
mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()


if mps_avail:
  device = torch.device("mps")
elif cuda_avail:
   device = torch.device("cuda")
print("cuda is available?" + str(cuda_avail) + ", mps available? " + str(mps_avail))
TEMPERATURE = 1

print ('argument list', sys.argv)
print("after argument list")

def parse_args():
    parser = argparse.ArgumentParser(description='linear spectrum non boolean test.')
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--h', type=int, default=2)
    parser.add_argument('--i', type=int, default=300000)
    return parser.parse_args()

print("before args partse")
args = parse_args()
print("after args parse")
print("args: " + str(args))
folder_string = "d_"+str(args.d)+"-f_"+str(args.f)+"-l_"+str(args.l) + "-h_"+str(args.h)
#if not os.path.exists(folder_string):
#    os.mkdir(folder_string)
print("folder_string: " + folder_string)
#print(weights)

# https://wiki.python.org/moin/BitManipulation
def parityOf(int_type):
        # now print the int in binary
#        print(int_type, "{0:b}".format(int_type))
# #      parity__ = (degree(int_type) % 2)
 #       parity_ = len([y for y in "{0:b}".format(int_type) if y == "1"]) % 2
        parity = 0
        while (int_type):
            parity = ~parity
            int_type = int_type & (int_type - 1)
#        print(parity, parity_, parity__)
        assert parity in [-1, 0]
        parity = 2*parity + 1
        return(parity)


def degree(subset):
   return len([y for y in "{0:b}".format(subset) if y == "1"])


import math
# randomize the function using the orthonormal basis

import torch
import random

hidden_size = args.d
batch_size = 2

def makeBitTensor(x, N):
  y = format(x, "b")
  y = ("0"*(N-len(y))) + y
  return [int(z) for z in list(y)]


def fitNetwork(function, N):
   embeddings = torch.nn.Embedding(2, hidden_size//2).to(device)
   positional_embeddings = torch.nn.Embedding(N, hidden_size//2).to(device)
   qrnn = torch.nn.TransformerEncoder(encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=args.h, dim_feedforward=args.f, dropout=0.0, activation='relu'), num_layers=args.l).to(device)
   
   output = torch.nn.Linear(hidden_size, 1, bias=False).to(device)
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
     targets = torch.FloatTensor([float(function(x)) for x in inputs]).to(device)
 #   print(targets.mean())
 #   quit()
     inputNum = torch.LongTensor([makeBitTensor(x,N) for x in inputs]).to(device).t()
     positional = torch.LongTensor(list(range(0, N))).unsqueeze(1).expand(-1, batch_size).to(device)
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


import random
myID = random.randint(1000,10000000)
with open(f"losses_{__file__}_{myID}.csv", "w") as outFile:
  #print(",".join(["AverageDegree", "Iterations", "Weights1", "Weights2", "PerturbedLoss", "Acc100", "Acc1000", "Acc10000", "Acc100000"]), file=outFile)
  for _ in range(10000):
   
   N = 30 #random.randint(2,30)
   averageDegree = N
   coefficients = torch.randn(N,N).to(device)
   mask = torch.zeros(N, N, dtype=torch.uint8).to(device)
    
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
       r = (coefficients*c.to(device)).sum()
       return r
   loss = fitNetwork(function, N)
   print(loss, averageDegree, loss)
   print(",".join([str(x) for x in (loss)]), file=outFile)
   outFile.flush()


# Try different average degress, instead of just pairs. Still maintain low degree. Say deg < 5-10.
# Vary the proportion of hidden elements that is pos. emb.
# MLP width start big
# keep d fixed, N fixed
# vary l (initially 1), vary f (3*df+1) a bit
# h = 1
# Get rid of layer norm, get more boolean functions, matrix dot product, pos. enc. 

# Generate a bunch of functions, train transformer on them, report generalization error
