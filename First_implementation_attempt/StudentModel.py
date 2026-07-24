import torchinfo

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
"""
PyTorch translation of a QKeras student model.

Notes on the translation:
- QConv2D / QDense / QDepthwiseConv2D / QSeparableConv2D are QKeras layers that
  apply fake-quantization to weights and biases during training (e.g.
  "stochastic_ternary", "quantized_po2(4)", "quantized_bits(3,0,1)"). This
  translation reproduces the *architecture* (layer shapes, connectivity,
  activations) using standard full-precision PyTorch layers. It does NOT
  reproduce the quantization behavior itself.
    - If you need equivalent quantization-aware training in PyTorch (e.g. for
      an hls4ml/FINN deployment target), look at Brevitas
      (https://github.com/Xilinx/brevitas), which is the closest PyTorch
      analogue to QKeras and integrates with the same hls4ml toolflow.
- Keras Conv2D/QConv2D layers default to padding='valid' (no padding) unless
  padding='same' is specified. I've preserved that distinction below:
  padding=0 for layers with no explicit padding argument in the original code,
  padding=1 (with matching kernel_size=3, stride=1) for the two
  QDepthwiseConv2D layers, which explicitly set padding='same'.
- Keras BatchNormalization's `momentum` parameter is the weight given to the
  OLD running statistic (new_running = momentum*old + (1-momentum)*batch),
  which is the opposite convention from PyTorch's `momentum` (new_running =
  (1-momentum)*old + momentum*batch). Keras momentum=0.9 corresponds to
  PyTorch momentum=0.1 (the PyTorch default), which is what's used below.
- Keras `scale=False` in BatchNormalization disables the learnable gamma
  (scale) parameter but keeps beta (shift). PyTorch's BatchNorm2d/1d only
  expose a single `affine` flag that controls both together, so there's no
  direct equivalent flag. If matching this exactly matters for your use case,
  freeze `.weight` to a constant 1 and set `.weight.requires_grad = False`
  after construction; this is left out below to keep the translation simple.
- The two "Add()" residual connections in the original only work because the
  branch being merged back in (the depthwise conv branch) still has the same
  channel count and spatial size as the branch that went through the
  1x1 conv + BN — this is preserved below.
- Shapes assume a single-channel 32x32 input, matching the original
  `input_shape_gray = (32, 32, 1)`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from wandb import summary


class StudentModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        bn_kwargs = dict(momentum=0.1, eps=1e-4)  # Keras momentum 0.9 -> PyTorch 0.1

        # --- Stage 1 ---
        # 32x32x1 -> valid 3x3 convs shrink spatial dims by 2 each time
        self.conv2d_AE = nn.Conv2d(1, 96, kernel_size=3, padding=0)   # 32 -> 30
        self.conv2d_AD = nn.Conv2d(96, 96, kernel_size=3, padding=0)  # 30 -> 28
        self.conv2d_A = nn.Conv2d(96, 24, kernel_size=3, padding=0)   # 28 -> 26

        # QDepthwiseConv2D(padding='same') -> keep 26x26, 24 channels
        self.depthconv_1 = nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24)
        # (BN after depthconv_1 was commented out in the original — omitted here)

        self.conv2d_B = nn.Conv2d(48, 24, kernel_size=1)  # concat(24+24)=48 -> 24
        self.bn_B = nn.BatchNorm2d(24, **bn_kwargs)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 26x26 -> 13x13

        # --- Stage 2 ---
        self.conv2d_C = nn.Conv2d(24, 96, kernel_size=3, padding=0)  # 13 -> 11
        self.bn_C = nn.BatchNorm2d(96, **bn_kwargs)

        self.depthconv_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96)  # same padding
        self.bn_dc2 = nn.BatchNorm2d(96, **bn_kwargs)  # BN here IS active in the original

        self.conv2d_D = nn.Conv2d(192, 96, kernel_size=1)  # concat(96+96)=192 -> 96
        self.bn_D = nn.BatchNorm2d(96, **bn_kwargs)

        # QSeparableConv2D = depthwise conv followed by pointwise (1x1) conv
        self.sep_depthwise = nn.Conv2d(96, 96, kernel_size=3, padding=0, groups=96)  # 11 -> 9
        self.sep_pointwise = nn.Conv2d(96, 24, kernel_size=1)

        flatten_dim = 24 * 9 * 9  # = 1944

        self.dense1 = nn.Linear(flatten_dim, 256)
        self.dense2 = nn.Linear(256, 256)
        self.bn_dense2 = nn.BatchNorm1d(256, **bn_kwargs)

        self.dense3 = nn.Linear(256, num_classes)
        self.bn_dense3 = nn.BatchNorm1d(num_classes, **bn_kwargs)

    def forward(self, x):
        # x: (B, 1, 32, 32)
        # NOTE: the input QActivation("quantized_bits(8, 7, alpha=1)") is a
        # fake-quantization of the raw input and has no float-model
        # equivalent; it's omitted here (equivalent to identity for a
        # full-precision model).

        x = self.conv2d_AE(x)
        x = F.relu(x)          # act_1AE: quantized_relu(2)
        x = self.conv2d_AD(x)
        x = torch.sigmoid(x)   # act_1AD: sigmoid
        x = self.conv2d_A(x)

        y = self.depthconv_1(x)
        x = torch.cat([x, y], dim=1)

        x = F.relu(x)           # act_1
        x = self.conv2d_B(x)
        x = self.bn_B(x)
        x = x + y               # residual add
        x = F.relu(x)           # act_2
        x = self.pool(x)

        x = self.conv2d_C(x)
        x = self.bn_C(x)

        y = self.depthconv_2(x)
        y = self.bn_dc2(y)
        x = torch.cat([x, y], dim=1)

        x = F.relu(x)           # act_3
        x = self.conv2d_D(x)
        x = self.bn_D(x)
        x = x + y                # residual add
        x = F.relu(x)            # act_4

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = F.relu(x)             # act_5

        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.bn_dense2(x)
        x = F.relu(x)             # act_7
        x = self.dense3(x)
        x = self.bn_dense3(x)
        x = F.softmax(x, dim=1)   # NOTE: drop this if training with
                                  # nn.CrossEntropyLoss, which expects raw
                                  # logits and applies log-softmax internally

        return x


if __name__ == "__main__":
    model = StudentModel(num_classes=10)
    dummy = torch.randn(2, 1, 32, 32)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, 10)

    # Uncomment to print full architecture summary:
    print(model)
    torchinfo.summary(model, input_data=dummy)