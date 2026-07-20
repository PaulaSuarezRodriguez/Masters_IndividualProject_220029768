import torchvision
import Load_cifar100
import numpy as np

quantized_mobilenetv3 = torchvision.models.quantization.mobilenet_v3_large(pretrained=False)
print(quantized_mobilenetv3.classifier)