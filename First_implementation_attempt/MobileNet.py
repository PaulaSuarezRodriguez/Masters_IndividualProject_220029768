import torchvision

quantized_mobilenetv3 = torchvision.models.quantization.mobilenet_v3_large(pretrained=True)

print(quantized_mobilenetv3.classifier)


