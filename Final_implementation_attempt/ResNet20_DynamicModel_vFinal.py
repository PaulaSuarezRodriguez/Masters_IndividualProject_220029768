# Implementation inspired by code from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
from Final_implementation_attempt.Config_vFinal import set_seed
set_seed()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#from avalanche.benchmarks import CLExperience
#from avalanche.models.dynamic_modules import IncrementalClassifier, DynamicModule
#from avalanche.models.dynamic_modules import DynamicModule

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer_DynamicModel(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer_DynamicModel, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_DynamicModel(nn.Module):
    """
    BasicBlock used in ResNet20 for CIFAR-style datasets.
    """
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_DynamicModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)#, track_running_stats=False)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)#, track_running_stats=False)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer_DynamicModel(
                    lambda x:
                    F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0
                    )
                )

            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm2d(planes)#, track_running_stats=False)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20_DynamicModel(nn.Module):
    def __init__(self, num_classes:int):
        super(ResNet20_DynamicModel, self).__init__()
        self.num_classes = num_classes
        self.in_features = 64
        self.in_planes = 16

        # Feature extractor (fixed; does not expand)
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)#, track_running_stats=False)
        self.layer1 = self._make_layer(16,3,stride=1)
        self.layer2 = self._make_layer(32,3,stride=2)
        self.layer3 = self._make_layer(64,3,stride=2)


        # Dynamic classifier
        self.linear = nn.Linear(in_features=self.in_features,out_features=self.num_classes)

        # Preserve your original Kaiming initialization.
        self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_DynamicModel(self.in_planes,planes,stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out,out.size()[3])
        out = out.view(out.size(0),-1)
        # IncrementalClassifier
        out = self.linear(out)
        return out

