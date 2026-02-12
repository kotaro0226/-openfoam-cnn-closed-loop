import os
import importlib
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
import torch.distributed as dist

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from args import args
from constants import FieldPredictionType
from dataloaders import create_sequences
from energy import EnergyMonitor
from helpers import scale, compute_memory, get_calling_filename
from plotting import plot_ML_outputs, plot_learning_curve

from helpers import setup_rank_print
setup_rank_print()

outfilename = f"subsampled_{args.fileprefix}.npz"

prec_dict = {"int8": torch.int8,
             "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float,
             "fp64": torch.float64}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # may slow down training, but ensures determinism

class NoContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

class NoScaler:
    """ Dummy class that provides the trivial protocol of PyTorch gradient scaler."""
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

class Trainer:
    """ Handles setup for DDP, sampler, dataloader and provides the training loop.
    """
    def __init__(self, args, X_train, Y_train, X_test, Y_test):
        """
        Initialize the distributed environment for DDP using environment variables
        provided by Slurm, set up model and dataset.
        """
        self.X_test = X_test
        self.Y_test = Y_test
        self.to_plot = args.plot

        # Global rank of the current process
        self.rank = int(os.environ['SLURM_PROCID']) if "SLURM_PROCID" in os.environ else 0
        # Total number of tasks
        self.world_size = int(os.environ['SLURM_NTASKS']) if "SLURM_NTASKS" in os.environ else 1
        if self.world_size == 1:
            print("Trainer: running on one process")
        if 'MASTER_ADDR' in os.environ:
            if self.rank == 0:
                print("Trainer: master address is " + os.environ["MASTER_ADDR"])
        else:
            raise "Need the environment variable MASTER_ADDR to be set to the IP address of the master process!"
        self.master_addr = os.environ['MASTER_ADDR']  # Address of the master node
        #self.master_port = os.environ['MASTER_PORT']  # Port of the master node

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            if self.rank == 0:
                print('Trainer: We have a GPU!')
            # Initialize the process group
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
            torch.cuda.set_device(self.rank % torch.cuda.device_count())  # Assign GPU based on rank
            self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
            # Verify GPU setup
            print(f"Trainer: Rank {self.rank}: Using GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name()}")
        else:
            if self.rank == 0:
                print('Trainer: CPU only.')
            dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size)

        print(f"Trainer: Rank {self.rank}: Device set to {self.device}")
        
        # Setup data loaders with DistributedSampler
        self.train_sampler = DistributedSampler(TensorDataset(X_train, Y_train), num_replicas=self.world_size, rank=self.rank, shuffle=args.shuffle)

        self.train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch, sampler=self.train_sampler)
        if self.rank == 0:
            print(f"Trainer: batch size: {args.batch}")
    
        input_shape = X_train.shape[1:]
        output_shape = Y_train.shape[1:] if len(Y_train.shape) > 1 else 1
        model_module = importlib.import_module('archs.' + args.arch)
        model = model_module.build_model(input_shape, output_shape=output_shape, window=args.window).to(self.device)

        print(f"Trainer: Rank {self.rank}: Model moved to {self.device}")
        # device_ids must be None for CPUs.
        device_ids = [self.device.index] if torch.cuda.is_available() else None
        # Wrap the model with DistributedDataParallel
        self.model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

        # Initialize lists to keep track of losses (recorded only on rank 0)
        self.train_loss_history = []
        self.val_loss_history = []
        self.last_eval_Y = None
        self.last_ref_Y = None

    def __del__(self):
        # Plot training diagnostics
        if self.rank == 0 and self.to_plot and self.last_eval_Y is not None:
            plot_learning_curve(self.train_loss_history, self.val_loss_history)
            plot_ML_outputs(self.last_eval_Y[0, :].to(dtype=torch.float).cpu().numpy().reshape(-1, 1),
                            self.last_ref_Y[0, :].to(dtype=torch.float).cpu().numpy().reshape(-1, 1))

        # Save the model only on rank 0
        if self.rank == 0:
            model_path = f"models/{args.arch}"
            os.makedirs(model_path, exist_ok=True)
            torch.save(self.model.state_dict(), f"{model_path}/{args.fileprefix}_model.pth")

        dist.destroy_process_group()

    def training_loop(self):

        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Create a scheduler that monitors the validation loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True, threshold=1e-4
        )

        criterion = nn.MSELoss()

        dev_type = self.device.type

        ctx = NoContext() if args.mxp_mode == "none" else \
                torch.autocast(device_type=dev_type, dtype=prec_dict[args.precision])
        scaler = NoScaler() if args.mxp_mode in ("none", "noscale") else \
                torch.amp.GradScaler(dev_type)
        precision = args.precision if args.mxp_mode != "none" else "default"
        if self.rank == 0:
            print(f"Trainer: Running with {args.mxp_mode} mixed-precision strategy, {precision} precision, on device type {dev_type}.")

        for epoch in range(args.epochs):
            self.model.train()
            self.train_sampler.set_epoch(epoch)  # Shuffle data for this epoch
            running_loss = 0.0
            num_batches = 0
            start_epoch = time.perf_counter()

            for i, (batch_X, batch_Y) in enumerate(self.train_loader):
                start_iter = time.perf_counter()
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                # print(f"Rank {self.rank}: Batch moved to {self.device}")
                optimizer.zero_grad()
                # ctx currently controls precision of operations
                with ctx:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_Y)
                # For mxp training, scaler scales loss to reduce chance of underflow in gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                running_loss += loss.item()
                num_batches += 1
                scaler.update()
                end_iter = time.perf_counter()
                #print(f"Epoch {epoch+1}, Iteration {i+1}/{len(self.train_loader)} took {end_iter - start_iter:.4f} seconds")
            end_epoch = time.perf_counter()

            #if self.rank == 0:
            #    print(f"Epoch {epoch+1} took {end_epoch - start_epoch:.2f} seconds in total")

            # Compute average training loss for this epoch
            epoch_train_loss = running_loss / num_batches
            if self.rank == 0:
                self.train_loss_history.append(epoch_train_loss)

            # Compute validation loss over the test_loader
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                X_test = self.X_test.to(self.device)
                Y_test = self.Y_test.to(self.device)
                with ctx:
                    Y_test_ML = self.model(X_test)
                    loss = criterion(Y_test_ML, Y_test)
                val_loss += loss.item()
                val_batches += 1

            epoch_val_loss = val_loss / val_batches
            if self.rank == 0:
                self.val_loss_history.append(epoch_val_loss)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1:3d}/{args.epochs:3d} - Train Loss: {epoch_train_loss:.6e} | Val Loss: {epoch_val_loss:.6e} | LR: {current_lr:.6e}", flush=True)
                #print(f"Epoch {epoch+1:3d}/{args.epochs:3d} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}", flush=True)

            # Update the scheduler with the validation loss
            scheduler.step(epoch_val_loss)

        self.last_eval_Y = Y_test_ML
        self.last_ref_Y = Y_test
        if self.rank == 0: 
            print(f"num batches during training {num_batches}")

    def eval(self):
        """Evaluate the test set and print the final loss."""
        self.model.eval()
        criterion = nn.MSELoss()
        dev_type = self.device.type
        # Use the appropriate context for mixed precision
        ctx = NoContext() if args.mxp_mode == "none" else \
            torch.autocast(device_type=dev_type, dtype=prec_dict[args.precision])

        with torch.no_grad():
            X_test = self.X_test.to(self.device)
            Y_test = self.Y_test.to(self.device)
            with ctx:
                outputs = self.model(X_test)
                final_loss = criterion(outputs, Y_test)
        if self.rank == 0:
            print(f"\033[92m \U0001F680 Evaluation on test set: Loss = {final_loss.item():.6e}\033[0m")


def main():
    """
    Main function to initialize data, parse arguments, and start the DDP training.
    """
    # Set the random seed
    #set_seed(42) 

    # Preprocess data
    data = np.load(os.path.join(args.output_dir, outfilename))

    try:
        total_memory = compute_memory(data)
        print(f"\nTotal Estimated Memory Usage: {total_memory['bytes']} bytes "
              f"({total_memory['MB']:.2f} MB, {total_memory['GB']:.4f} GB)")
    except:
        print("problem computing memory")

    X, Y = data['X'], data['Y']
    print(f"X: {X.shape}; Y: {Y.shape}", flush=True) # X: [T, [X,Y,Z]-or-NSAMPLES, C]; Y: [T, [X,Y,Z]-or-NSAMPLES, C]


    # Convert timeseries to sequences
    if args.sequence:
        X, Y = create_sequences(X, Y, args)
        print(f"After sequence X: {X.shape}; Y: {Y.shape}", flush=True)

    # Expand dims if no time dimension exists.
    if len(X.shape) in [3, 5]:
        X = np.expand_dims(X, axis=1)  # X becomes (B, 1, 32, 32, 32, 4)
    if len(Y.shape) == 5:
        Y = np.expand_dims(Y, axis=1)  # Y becomes (B, 1, 32, 32, 32, 1)
    print(f"After dim exp X: {X.shape}; Y: {Y.shape}", flush=True)
    
    if args.method == "full":
        # Transpose from (B, 1, 32, 32, 32, 4) to (B, 1, 4, 32, 32, 32)
        X = X.transpose(0, 1, 5, 2, 3, 4)
        Y = Y.transpose(0, 1, 5, 2, 3, 4)
    else:
        # Other method's transpose for X (and Y if needed) goes here.
        X = X.transpose(0, 1, 3, 2)
        if len(Y.shape) > 4:
            Y = Y.transpose(0, 1, 5, 2, 3, 4)
        else:
            Y = Y.transpose(0, 1, 3, 2)

    print(f"After transpose of vars: X: {X.shape}; Y: {Y.shape}", flush=True)

    # train:val split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=args.shuffle)

    # Scale the data
    scaler_x = eval(args.xscaler)()
    X_train = scale(scaler_x.fit_transform, X_train)
    X_test = scale(scaler_x.transform, X_test)
    if args.yscaler != 'None':
        scaler_y = eval(args.yscaler)()
        Y_train = scale(scaler_y.fit_transform, Y_train)
        Y_test = scale(scaler_y.transform, Y_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    print(f"X_train: {X_train.shape}; X_test: {X_test.shape}", flush=True)
    print(f"Y_train: {Y_train.shape}; Y_test: {Y_test.shape}", flush=True)

    # Determine world size and launch workers
    #world_size = int(os.environ['SLURM_NTASKS'])
    #rank = int(os.environ['SLURM_PROCID'])
    #main_worker(rank, world_size, args, X_train, Y_train, X_test, Y_test)

    trainer = Trainer(args, X_train, Y_train, X_test, Y_test)

    dist.barrier()
    if os.path.exists("/sys/cray/pm_counters"):
        if trainer.rank == 0:
            em = EnergyMonitor(get_calling_filename())
            em.start()

    trainer.training_loop()

    if os.path.exists("/sys/cray"):
        dist.barrier()
        if trainer.rank == 0:
            em.end()
            print("Aggregating energy reports across nodes:")
            em.aggregate()

    trainer.eval()

if __name__ == "__main__":
    main()
