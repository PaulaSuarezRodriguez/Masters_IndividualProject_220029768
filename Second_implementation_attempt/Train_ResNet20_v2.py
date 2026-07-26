import Config
from Config import set_seed
set_seed()
from Config import set_device
from ResNet20 import _weights_init, LambdaLayer ,BasicBlock, ResNet20
import torch
import torchao
import torchvision
import Load_cifar100
from Load_cifar100 import train_loader, val_loader, test_loader
from Load_cifar100 import outdoors_specialised_models_subset, nature_specialised_models_subset, people_specialised_models_subset, animals_specialised_models_subset, objects_and_vehicles_specialised_models_subset
import numpy as np
import torch.ao.quantization as quantization
from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset

avalanche_train_dataset = AvalancheDataset(
    train_dataset
)

scenario = nc_benchmark(
    nature_dataset,
    n_experiences=3
)
resnet20_global_model = ResNet20()