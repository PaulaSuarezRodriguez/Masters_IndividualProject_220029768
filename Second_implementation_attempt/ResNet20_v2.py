# Implementation from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
import Config
from Config import set_seed
set_seed()
from Config import set_device
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

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module): # the block type used in ResNet20 by the original paper and the ResNet20 implementation in the referenced repository, as well as most ResNet20 implementations (given the other block type found in ResNet, "Bottleneck", is not used in ResNet20)
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init) # included by the referenced repository to follow the implementation of ResNet paper exactly; i.e. the usage of Kaiming initialization.

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1) # if it causes problems, change to out=torch.flatten(out, 1)
        out = self.linear(out)
        return out


########################################################################################################################################

# Implementation from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
import Config
from Config import set_seed
set_seed()
from Config import set_device
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

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module): # the block type used in ResNet20 by the original paper and the ResNet20 implementation in the referenced repository, as well as most ResNet20 implementations (given the other block type found in ResNet, "Bottleneck", is not used in ResNet20)
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init) # included by the referenced repository to follow the implementation of ResNet paper exactly; i.e. the usage of Kaiming initialization.

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1) # if it causes problems, change to out=torch.flatten(out, 1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    resnet20_global_model = ResNet20(num_classes=100)
    outdoors_specialised_model = ResNet20(num_classes=10)
    nature_specialised_model = ResNet20(num_classes=15)
    people_specialised_model = ResNet20(num_classes=5)
    animals_specialised_model = ResNet20(num_classes=45)
    objects_and_vehicles_specialised_model = ResNet20(num_classes=25)

