# training.py
from pyhessian.hessian import hessian
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import random
import math
import argparse
import contextlib
import time
import os
import itertools
import datetime
import signal
import sys
import warnings

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed import (
    init_process_group,
    destroy_process_group,
    all_reduce,
    ReduceOp,
    barrier,
)

# models
from hardcoded_transformer import HardCodedTransformer
from updated_transformer import Transformer as Transformer

# local utils
from train_utils import (
    SAM,
    get_weight_norm,
    rboolf,
    addGaussianNoise,
    ddp_setup,
    parse_args,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.distributed\.distributed_c10d",
)

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

_shutting_down = False


def _graceful_exit(signum, frame):
    global _shutting_down
    if _shutting_down:
        os._exit(1)
    _shutting_down = True
    with contextlib.suppress(Exception):
        if dist.is_initialized():
            dist.destroy_process_group()
    sys.exit(0)


signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)

if mps_avail:
    device = torch.device("mps")
elif cuda_avail:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Trainer:
    def __init__(
        self,
        coeffs: torch.FloatTensor,
        combs: torch.FloatTensor,
        model: torch.nn.Module,
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
        backend: str,
        stop_loss: float,
        ln_eps: float,
        ln: bool,
        save_checkpoints: bool,
        f: float,
        d: int,
        h: int,
        dropout: float,
        wd: float,
        run_id: float,
    ) -> None:
        self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)
        model = model.to(self.gpu_id)
        self.model = DDP(model, device_ids=[self.gpu_id], output_device=self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.ln_eps = ln_eps
        self.ln = ln
        self.wd = wd
        self.dir_name = dir_name
        self.save_checkpoints = save_checkpoints
        self.dropout = dropout
        self.summary = pd.DataFrame(
            columns=[
                "deg",
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
                "top_eig_train",
                "trace_train",
                "stop_loss",
                "ln_eps",
                "ln",
                "weight_norm",
                "l",
                "d",
                "f",
                "h",
                "dropout",
                "wd",
            ]
        )
        self.stop_loss = stop_loss
        self.epoch_loss = 0
        self.N = N
        self.func = func
        self.coeffs = coeffs.to(gpu_id)
        self.combs = combs.to(gpu_id)
        self.width = width
        self.deg = deg
        self.n_samples = n_samples
        self.d = d
        self.f = f
        self.h = h
        for batch in train_data:
            self.batch_size = len(batch)
            break
        self.lr = (
            optimizer.base_optimizer.param_groups[-1]["lr"]
            if hasattr(optimizer, "base_optimizer")
            else optimizer.param_groups[-1]["lr"]
        )
        self.backend = backend
        self.run_id = run_id

    def func_batch(self, x):
        """
        x:
          - (B,) long  of packed ints (for N <= 62)
          - (B,N) long of bits        (for N > 62)
        """
        x = torch.as_tensor(x, device=self.gpu_id)
        if x.dim() == 2:
            # already bits
            bits01 = x.float()
        else:
            x = x.long()
            shifts = torch.arange(self.N, device=self.gpu_id)
            bits01 = ((x.unsqueeze(-1) >> shifts) & 1).float()

        bin_pm = (bits01 - 0.5) * 2.0  # {-1,+1}

        idx = self.combs.long()  # (W,D)
        comps = bin_pm[:, idx]   # (B,W,D)
        comps = comps.prod(dim=2)
        return comps @ self.coeffs

    def _run_batch(self, inputs, targets):
        loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt).pow(2).mean()

        if hasattr(self.optimizer, "base_optimizer"):
            self.optimizer.zero_grad()
            with self.model.no_sync():
                out = self.model(inputs)
                loss = loss_fn(out, targets)
                loss.backward()
            self.optimizer.first_step(zero_grad=True)

            out = self.model(inputs)
            loss_perturbed = loss_fn(out, targets)
            loss_perturbed.backward()

            self.optimizer.second_step(zero_grad=True)
            return loss.detach()

        out = self.model(inputs)
        loss = loss_fn(out, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def _run_epoch(self, epoch):
        sampler = getattr(self.train_data, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        total_records = 0

        for idx, inputs in enumerate(self.train_data):
            inputs = inputs.to(self.gpu_id, non_blocking=True)
            targets = self.func_batch(inputs)
            batch_loss = self._run_batch(inputs, targets)

            epoch_loss += batch_loss * float(len(inputs))
            total_records += len(inputs)

        epoch_loss /= float(total_records)
        return epoch_loss

    def save_checkpoint(self, epoch, model_name):
        os.makedirs(os.path.join(self.dir_name, model_name), exist_ok=True)
        full_model_name = model_name + "/epoch-" + str(epoch) + ".pt"
        ckp = self.model.module.state_dict()
        torch.save(ckp, os.path.join(self.dir_name, full_model_name))
        print(f"Epoch {epoch} | Training checkpoint saved at {full_model_name}")

    def train(self, epochs: int):
        self.model.train()
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = self._run_epoch(epoch)

            if (
                self.gpu_id == 0
                and (((epoch % self.save_every) == 0) or (epoch_loss < self.stop_loss))
            ):
                if self.save_checkpoints:
                    self.save_checkpoint(
                        epoch,
                        f"degree-{self.deg}/width-{self.width}/func-{self.func}",
                    )

                end_time = time.time()
                elapsed_time = round((end_time - start_time) / 60, 3)

                val_loss = self.validate(1000, self.model)
                loss_fn = lambda result, targets: (result - targets).pow(2).mean()

                start_time_hessian = time.time()
                top_eig, trace = self.calc_hessian(
                    copy.deepcopy(self.model.module),
                    loss_fn=loss_fn,
                    num_samples=1000,
                    device_id=self.gpu_id,
                )
                top_eig_train, trace_train = self.calc_hessian(
                    copy.deepcopy(self.model.module),
                    loss_fn=loss_fn,
                    num_samples=1000,
                    device_id=self.gpu_id,
                    use_train=True,
                )

                weight_norm = get_weight_norm(self.model.module)

                end_time_hessian = time.time()
                elapsed_time_hessian = round(
                    (end_time_hessian - start_time_hessian) / 60, 3
                )
                print(
                    f"[RANK {self.gpu_id}] elapsed time (Hessian): {elapsed_time_hessian}"
                )

                self.summary.loc[0] = {
                    "deg": self.deg,
                    "width": self.width,
                    "func": self.func,
                    "epoch": epoch,
                    "train_loss": epoch_loss.cpu(),
                    "val_loss": val_loss.cpu(),
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "n_samples": self.n_samples,
                    "func_val_test": self.func_batch(
                        torch.randint(0, 2, (1, self.N))
                    ).cpu(),
                    "time_elapsed": elapsed_time,
                    "backend": self.backend,
                    "top_eig": top_eig,
                    "trace": trace,
                    "top_eig_train": top_eig_train,
                    "trace_train": trace_train,
                    "stop_loss": self.stop_loss,
                    "ln_eps": self.ln_eps,
                    "ln": self.ln,
                    "weight_norm": weight_norm,
                    "d": self.d,
                    "f": self.f,
                    "h": self.h,
                    "dropout": self.dropout,
                    "wd": self.wd,
                }

                self.summary.to_csv(
                    f"{self.dir_name}/summary_{self.run_id}.csv",
                    mode="a",
                    header=not os.path.exists(f"{self.dir_name}/summary.csv"),
                    index=False,
                )
                print(
                    f"[RANK {self.gpu_id}] Epoch: {epoch}, "
                    f"TimeElapsed: {elapsed_time}, "
                    f"EpochLoss: {epoch_loss:.3f}, "
                    f"ValidationLoss: {val_loss:.3f}"
                )

            flag = torch.zeros(1, device=self.gpu_id)
            if epoch_loss < self.stop_loss:
                flag += 1
            all_reduce(flag, op=ReduceOp.SUM)
            if flag > 0:
                break
            barrier()

        return

    def validate(self, num_samples, test_model):
        test_model.eval()
        if self.N <= 62:
            inputs = torch.randint(0, 2 ** self.N, (num_samples,), device=self.gpu_id)
        else:
            inputs = torch.randint(0, 2, (num_samples, self.N), device=self.gpu_id)
        targets = self.func_batch(inputs)
        result = test_model(inputs).squeeze(-1)
        return (result - targets).pow(2).mean().detach().cpu()

    def calc_hessian(self, model, loss_fn, num_samples, device_id, use_train=False):
        dev = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        m = model.to(dev).eval()

        if use_train:
            ds = getattr(self.train_data, "dataset", None)
            if isinstance(ds, torch.Tensor):
                x = ds[: min(num_samples, ds.shape[0])].to(dev)
            else:
                xs, n = [], 0
                for b in self.train_data:
                    b = b[0] if isinstance(b, (list, tuple)) else b
                    k = min(b.shape[0], num_samples - n)
                    xs.append(b[:k])
                    n += k
                    if n >= num_samples:
                        break
                x = torch.cat(xs, 0).to(dev)
        else:
            if self.N <= 62:
                x = torch.randint(0, 2 ** self.N, (num_samples,), device=dev)
            else:
                x = torch.randint(0, 2, (num_samples, self.N), device=dev)

        y = self.func_batch(x).to(dev)

        H = hessian(m, loss_fn, (x, y))
        for p in m.parameters():
            p.grad = None
        top_eig = H.eigenvalues(maxIter=200)[0][0]
        trace = H.trace()
        return float(top_eig), float(np.mean(trace))


def load_train_objs(
    wd,
    dropout,
    lr,
    num_samples,
    N,
    dim,
    h,
    f,
    rank,
    ln_eps,
    ln,
    coefs,
    combs,
    sam=False,
    sam_rho=0.05,
    asam=False,
):
    # For N <= 62, keep packed ints; for N > 62, use bit vectors
    if N <= 62:
        train_set = torch.randint(
            low=0,
            high=2 ** N,
            size=(int(num_samples),),
            dtype=torch.long,
        ).to(rank)
    else:
        train_set = torch.randint(
            low=0,
            high=2,
            size=(int(num_samples), N),
            dtype=torch.long,
        ).to(rank)

    hardcoded_models = []
    for mode in ["original", "mlp_soft", "balanced"]:
        hardcoded_model = HardCodedTransformer(
            N,
            combs,
            coefs,
            aggregator_idx=N - 1,
            nonrep_mask=-40.0,
            mode=mode,
            mlp_soft_factor=0.25,
        )
        hardcoded_total_params = sum(p.numel() for p in hardcoded_model.parameters())
        print("Hardcoded Model Parameter Count:", hardcoded_total_params)
        hardcoded_models.append(hardcoded_model)

    model = Transformer(dropout, N, dim, h, f, ln_eps, rank, ln)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable Model Parameter Count:", total_params)

    base_opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=wd)
    optimizer = (
        SAM(model.parameters(), base_optimizer=base_opt, rho=sam_rho, adaptive=asam)
        if sam
        else base_opt
    )
    return train_set, model, optimizer, hardcoded_models


def main(rank, args, world_size, coefs, combs, main_dir, deg, width, i, run_id):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        torch.cuda.set_device(rank)
    else:
        torch.cuda.set_device(rank)

    coefs = coefs.to(rank)
    combs = combs.to(rank)
    ddp_setup(rank, world_size, args.backend)

    train_set, model, optimizer, hardcoded_models = load_train_objs(
        args.wd,
        args.dropout,
        args.lr,
        args.num_samples,
        args.N,
        args.dim,
        args.h,
        args.f,
        rank,
        args.ln_eps,
        args.ln,
        coefs,
        combs,
        sam=args.sam,
        sam_rho=args.sam_rho,
        asam=args.asam,
    )

    model.to(rank)
    for hardcoded_model in hardcoded_models:
        hardcoded_model.to(rank)

    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=args.bs,
        sampler=DistributedSampler(train_set),
    )

    trainer = Trainer(
        coefs,
        combs,
        model,
        train_loader,
        optimizer,
        gpu_id=rank,
        save_every=args.save_every,
        dir_name=main_dir,
        width=width,
        deg=deg,
        func=i,
        N=args.N,
        n_samples=args.num_samples,
        backend=args.backend,
        stop_loss=args.stop_loss,
        ln_eps=args.ln_eps,
        ln=args.ln,
        save_checkpoints=args.save_checkpoints,
        d=args.dim,
        f=args.f,
        h=args.h,
        dropout=args.dropout,
        wd=args.wd,
        run_id=run_id,
    )

    loss_fn = lambda out, tgt: (out.squeeze(-1) - tgt.to(out.device)).pow(2).mean()
    hardcoded_hessian_stats = []

    print("hardcoded model:", hardcoded_models[0])
    for j, mode in enumerate(["original", "mlp_soft", "balanced"]):
        hardcoded_model = hardcoded_models[j]
        hardcoded_hessian_stats = trainer.calc_hessian(
            hardcoded_model, loss_fn, num_samples=1000, device_id=rank
        )
        hardcoded_hessian_train = trainer.calc_hessian(
            hardcoded_model,
            loss_fn,
            num_samples=1000,
            device_id=rank,
            use_train=True,
        )

        weight_norm = get_weight_norm(hardcoded_model)
        hardcoded_loss = trainer.validate(1000, hardcoded_model)
        print("hardcoded loss:", hardcoded_loss)
        print("frobenius weight norm:", weight_norm)
        print("hardcoded hessian stats:", hardcoded_hessian_stats)

        _hc_df = pd.DataFrame(
            [
                {
                    "deg": trainer.deg,
                    "width": trainer.width,
                    "func": trainer.func,
                    "const_mode": mode,
                    "top_eig": round(hardcoded_hessian_stats[0], 2),
                    "trace": round(hardcoded_hessian_stats[1], 2),
                    "top_eig_train": round(hardcoded_hessian_train[0], 2),
                    "trace_train": round(hardcoded_hessian_train[1], 2),
                    "frobenius_weight_norm": round(weight_norm, 2),
                    "test_loss": torch.round(hardcoded_loss, decimals=3),
                }
            ]
        )
        _hc_df.to_csv(
            f"{trainer.dir_name}/hardcoded_hessian.csv",
            index=False,
            mode="a",
            header=not os.path.exists(f"{trainer.dir_name}/hardcoded_hessian.csv"),
        )

    print("trainer.func_batch([2, 3]):", trainer.func_batch(torch.randint(0, 2, (2, trainer.N))))
    trainer.train(args.epochs)
    barrier()
    print("finished training, cleaning up process group...")
    destroy_process_group()
    print("finished cleaning up process group")
    return


if __name__ == "__main__":
    arguments = parse_args()
    arguments.save_checkpoints = False
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print("time at start of job:", run_id)
    print(arguments)
    losses = {}
    func_per_deg = arguments.repeat
    main_dir = f"/scratch/plintilhac/HESSIAN_CALCS22"
    os.makedirs(main_dir, exist_ok=True)

    for i in range(4, 12):
        for deg in [5, 4, 3, 2, 1]:
            losses[deg] = []
            for width in [20, 14, 7, 1]:
                start_time = time.time()
                print(f"Generating: func {i}, deg {deg}, width {width}")
                seedNum = int(str(i) + str(deg) + str(width))
                (coefs, combs) = rboolf(arguments.N, width, deg, seed=seedNum)

                torch.save(
                    coefs,
                    os.path.join(
                        main_dir, f"coefs_func{i}_deg{deg}_width{width}.pt"
                    ),
                )
                torch.save(
                    combs,
                    os.path.join(
                        main_dir, f"combs_func{i}_deg{deg}_width{width}.pt"
                    ),
                )

                mp.set_start_method("spawn", force=True)
                torch.set_num_threads(1)
                mp.spawn(
                    main,
                    args=(
                        arguments,
                        arguments.world_size,
                        coefs,
                        combs,
                        main_dir,
                        deg,
                        width,
                        i,
                        run_id,
                    ),
                    nprocs=arguments.world_size,
                    join=True,
                )
                print("returned from mp.spawn")
                end_time = time.time()

                elapsed_time = round((end_time - start_time) / 60, 3)
                print(
                    "elapsed time for whole training process:", elapsed_time
                )

