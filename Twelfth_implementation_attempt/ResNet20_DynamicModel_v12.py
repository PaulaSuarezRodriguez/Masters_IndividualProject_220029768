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
        self.bn2 = nn.BatchNorm2d(planes)#,track_running_stats=False)

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

        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)#, track_running_stats=False)
        self.layer1 = self._make_layer(16,3,stride=1)
        self.layer2 = self._make_layer(32,3,stride=2)
        self.layer3 = self._make_layer(64,3,stride=2)

        # ---------------------------------------------------------
        # Dynamic classifier
        # ---------------------------------------------------------
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

'''
    def add_classes(self, num_new_classes: int):

        if num_new_classes <= 0:
            return []

        old_num_classes = self.num_classes
        new_num_classes = old_num_classes + num_new_classes

        old_classifier = self.linear#self.classifier

        new_classifier = nn.Linear(
            int(self.in_features),
            int(new_num_classes),
            #bias=self.bias,
        ).to(
            device=old_classifier.weight.device
            if old_classifier is not None
            else None,
            dtype=old_classifier.weight.dtype
            if old_classifier is not None
            else None,
        )

        if old_classifier is not None:

            with torch.no_grad():

                new_classifier.weight[
                    :old_num_classes
                ].copy_(
                    old_classifier.weight
                )

                if self.bias:
                    new_classifier.bias[
                        :old_num_classes
                    ].copy_(
                        old_classifier.bias
                    )

        self.linear = new_classifier # self.classifier = new_classifier
        self.num_classes = new_num_classes

        return list(
            new_classifier.parameters()
        )
'''


'''
class IncrementalClassifier(DynamicModule):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(
        self,
        in_features,
        initial_out_features=2,
        masking=True,
        mask_value=-1000,
        **kwargs,
    ):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__(**kwargs)
        self.masking = masking
        self.mask_value = mask_value

        self.classifier = torch.nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        super().adaptation(experience)
        device = self._adaptation_device
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = experience.classes_in_this_experience
        new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)

        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units
                self.active_units = torch.zeros(
                    new_nclasses, dtype=torch.int8, device=device
                )
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[list(curr_classes)] = 1

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b

    def forward(self, x, **kwargs):
        """compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        out = self.classifier(x)
        if self.masking:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask=mask, value=self.mask_value)
        return out
'''
'''class IncrementalClassifier(DynamicModule):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(
        self,
        in_features,
        initial_out_features=2,
        masking=True,
        mask_value=-1000,
        **kwargs,
    ):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__(**kwargs)
        self.masking = masking
        self.mask_value = mask_value

        self.classifier = torch.nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        super().adaptation(experience)
        device = self._adaptation_device
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = experience.classes_in_this_experience
        new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)

        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units
                self.active_units = torch.zeros(
                    new_nclasses, dtype=torch.int8, device=device
                )
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[list(curr_classes)] = 1

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b

    def forward(self, x, **kwargs):
        """compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        out = self.classifier(x)
        if self.masking:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask=mask, value=self.mask_value)
        return out'''


if __name__ == "__main__":
    dummy_resnet20_dm_global_model = ResNet20_DynamicModel(num_classes=100)
    dummy_outdoors_dm_specialised_model = ResNet20_DynamicModel(num_classes=10)
    dummy_nature_dm_specialised_model = ResNet20_DynamicModel(num_classes=15)
    dummy_people_dm_specialised_model = ResNet20_DynamicModel(num_classes=5)
    dummy_animals_dm_specialised_model = ResNet20_DynamicModel(num_classes=45)
    dummy_objects_and_vehicles_dm_specialised_model = ResNet20_DynamicModel(num_classes=25)

