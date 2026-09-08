# Implementation inspired by code from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
from avalanche.benchmarks import CLExperience
from Twelfth_implementation_attempt.Config_v12 import set_seed
set_seed()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#from avalanche.models.dynamic_modules import IncrementalClassifier, DynamicModule
#from avalanche.models.dynamic_modules import DynamicModule

# ============================================================================
# OPTION 1: ResNet20_Compact (Recommended - Minimal Changes)
# ============================================================================
# Reduces: 16→8, 32→16, 64→32 channels (50% reduction)
# This is the MINIMAL change version - just scale all channels down

########################################################################################################################

# ============================================================================
# OPTION 2: ResNet20_Compact_XL (Very Minimal - Just 25% Reduction)
# ============================================================================
# If you want even less change: 16→12, 32→24, 64→48 channels

########################################################################################################################

# ============================================================================
# OPTION 3: ResNet20_Compact_S (Most Aggressive - 75% Reduction)
# ============================================================================
# For maximum regularization: 16→4, 32→8, 64→16 channels

########################################################################################################################

# Compact ResNet20 for Small Datasets (2000 samples)
# Minimal changes from original ResNet20_DynamicModel_v12.py
# Just a scaled-down version to reduce overfitting

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
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

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
                    nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ============================================================================
# OPTION 1: ResNet20_Compact (Recommended - Minimal Changes)
# ============================================================================
# Reduces: 16→8, 32→16, 64→32 channels (50% reduction)
# This is the MINIMAL change version - just scale all channels down

class ResNet20_Compact(nn.Module):
    """
    Compact ResNet20 for small datasets (2000 samples).

    Changes from original ResNet20_DynamicModel_v12:
    - Layer1: 16 → 8 channels
    - Layer2: 32 → 16 channels
    - Layer3: 64 → 32 channels
    - Final features: 64 → 32

    Everything else identical. This reduces model size by ~50%
    to prevent overfitting on small datasets.
    """

    def __init__(self, num_classes: int):
        super(ResNet20_Compact, self).__init__()
        self.num_classes = num_classes
        self.in_features = 32  # Changed: 64 → 32
        self.in_planes = 8  # Changed: 16 → 8

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)  # Changed: 16 → 8
        self.bn1 = nn.BatchNorm2d(8)  # Changed: 16 → 8

        self.layer1 = self._make_layer(8, 3, stride=1)  # Changed: 16 → 8
        self.layer2 = self._make_layer(16, 3, stride=2)  # Changed: 32 → 16
        self.layer3 = self._make_layer(32, 3, stride=2)  # Changed: 64 → 32

        # Dynamic classifier - same interface
        self.linear = nn.Linear(in_features=self.in_features, out_features=self.num_classes)

        # Preserve original initialization
        self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_DynamicModel(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ============================================================================
# OPTION 2: ResNet20_Compact_XL (Very Minimal - Just 25% Reduction)
# ============================================================================
# If you want even less change: 16→12, 32→24, 64→48 channels

class ResNet20_Compact_XL(nn.Module):
    """
    Compact ResNet20 with smaller reduction (25% instead of 50%).

    Use this if 50% reduction seems too aggressive.
    Changes from original:
    - Layer1: 16 → 12 channels
    - Layer2: 32 → 24 channels
    - Layer3: 64 → 48 channels
    - Final features: 64 → 48
    """

    def __init__(self, num_classes: int):
        super(ResNet20_Compact_XL, self).__init__()
        self.num_classes = num_classes
        self.in_features = 48  # Changed: 64 → 48
        self.in_planes = 12  # Changed: 16 → 12

        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, bias=False)  # Changed: 16 → 12
        self.bn1 = nn.BatchNorm2d(12)  # Changed: 16 → 12

        self.layer1 = self._make_layer(12, 3, stride=1)  # Changed: 16 → 12
        self.layer2 = self._make_layer(24, 3, stride=2)  # Changed: 32 → 24
        self.layer3 = self._make_layer(48, 3, stride=2)  # Changed: 64 → 48

        self.linear = nn.Linear(in_features=self.in_features, out_features=self.num_classes)

        self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_DynamicModel(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ============================================================================
# OPTION 3: ResNet20_Compact_S (Most Aggressive - 75% Reduction)
# ============================================================================
# For maximum regularization: 16→4, 32→8, 64→16 channels

class ResNet20_Compact_S(nn.Module):
    """
    Very compact ResNet20 for maximum regularization on tiny datasets.

    Changes from original (75% reduction):
    - Layer1: 16 → 4 channels
    - Layer2: 32 → 8 channels
    - Layer3: 64 → 16 channels
    - Final features: 64 → 16

    Use only if overfitting is severe.
    """

    def __init__(self, num_classes: int):
        super(ResNet20_Compact_S, self).__init__()
        self.num_classes = num_classes
        self.in_features = 16  # Changed: 64 → 16
        self.in_planes = 4  # Changed: 16 → 4

        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)  # Changed: 16 → 4
        self.bn1 = nn.BatchNorm2d(4)  # Changed: 16 → 4

        self.layer1 = self._make_layer(4, 3, stride=1)  # Changed: 16 → 4
        self.layer2 = self._make_layer(8, 3, stride=2)  # Changed: 32 → 8
        self.layer3 = self._make_layer(16, 3, stride=2)  # Changed: 64 → 16

        self.linear = nn.Linear(in_features=self.in_features, out_features=self.num_classes)

        self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_DynamicModel(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ============================================================================
# USAGE
# ============================================================================
"""
Drop-in replacement for ResNet20_DynamicModel:

OPTION 1 (Recommended - 50% size reduction):
    from ResNet20_Compact import ResNet20_Compact
    model = ResNet20_Compact(num_classes=10)

OPTION 2 (Conservative - 25% size reduction):
    from ResNet20_Compact import ResNet20_Compact_XL
    model = ResNet20_Compact_XL(num_classes=10)

OPTION 3 (Aggressive - 75% size reduction):
    from ResNet20_Compact import ResNet20_Compact_S
    model = ResNet20_Compact_S(num_classes=10)

That's it. Use exactly as you would the original ResNet20_DynamicModel.
Everything else stays the same.
"""


'''if __name__ == "__main__":
    dummy_resnet20_dm_global_model = ResNet20_Compact(num_classes=100)
    dummy_resnet20_dm_global_model = ResNet20_Compact_XL(num_classes=100)
    dummy_resnet20_dm_global_model = ResNet20_Compact_S(num_classes=100)
    
    dummy_outdoors_dm_specialised_model = ResNet20_Compact(num_classes=10)
    dummy_outdoors_dm_specialised_model = ResNet20_Compact_XL(num_classes=10)
    dummy_outdoors_dm_specialised_model = ResNet20_Compact_S(num_classes=10)
    
    dummy_nature_dm_specialised_model = ResNet20_Compact(num_classes=15)
    dummy_nature_dm_specialised_model = ResNet20_Compact_XL(num_classes=15)
    dummy_nature_dm_specialised_model = ResNet20_Compact_S(num_classes=15)
    
    dummy_people_dm_specialised_model = ResNet20_Compact(num_classes=5)
    dummy_people_dm_specialised_model = ResNet20_Compact_XL(num_classes=5)
    dummy_people_dm_specialised_model = ResNet20_Compact_S(num_classes=5)

    dummy_animals_dm_specialised_model = ResNet20_Compact(num_classes=45)
    dummy_animals_dm_specialised_model = ResNet20_Compact_XL(num_classes=45)
    dummy_animals_dm_specialised_model = ResNet20_Compact_S(num_classes=45)

    dummy_objects_and_vehicles_dm_specialised_model = ResNet20_Compact(num_classes=25)
    dummy_objects_and_vehicles_dm_specialised_model = ResNet20_Compact_XL(num_classes=25)
    dummy_objects_and_vehicles_dm_specialised_model = ResNet20_Compact_S(num_classes=25)'''

