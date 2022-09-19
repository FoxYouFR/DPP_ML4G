import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision import datasets
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import os

print(f"Cuda is available: {torch.cuda.is_available()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")
print(f"NCCL is available: {torch.distributed.is_nccl_available()}")
print(f"MPI is available: {torch.distributed.is_mpi_available()}")
print(f"GLOO is available: {torch.distributed.is_gloo_available()}")

# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 2
# number of epochs
n_epochs = 2
# learning rate
learning_rate = 0.01
# Number of passage before printing each time
print_every = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def step(model, images, labels, criterion, optimizer):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return outputs, loss
    
def train(local_rank, args):
    # the rank is the unique ID given to a process (e.g. 2 nodes with 4 GPUs each will yield ranks = [0,1,2,3,4,5,6,7])
    # the local rank has the same purpose but only for the device
    rank = args["node_rank"] * args["gpus"] + local_rank
    backend = args["backend"]
    print(f"Using {backend} backend")
    dist.init_process_group(backend, world_size=args["world_size"], rank=rank)
    
    model = Net()
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    torch.cuda.set_device(rank)
    
    data_path = "./data/"
    transform_train = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    # We also need to distribute the training dataset indicating the world size and the rank of the process
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args["world_size"], rank=rank, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                            pin_memory=True, sampler=train_sampler, drop_last=True)
    
    print("Batch size: {}".format(args["batch_size"]))
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for e in range(args["epochs"]):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            model.train()
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            outputs, loss = step(model, images, labels, criterion, optimizer)
            
        loss /= args["world_size"]
        print(f"GPU [{rank}] epoch [{e+1}] train loss is: {loss.item()}")

if __name__ == "__main__":
    args = {
        "nodes": 1, # Number of connected devices
        "gpus": 2, # Number of GPU available per device
        "node_rank": 0, # Rank of the node (should be different for each device)
        "epochs": 2,
        "batch_size": 20,
        "backend": "gloo" # must choose between nccl, gloo and mpi (only gloo available here)
    }
    args["world_size"] = args["gpus"] * args["nodes"] # the world is the group that contains all the processes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "20480"
    mp.spawn(train, nprocs=args["gpus"], args=(args,))