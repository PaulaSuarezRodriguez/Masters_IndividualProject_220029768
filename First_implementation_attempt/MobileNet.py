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
#TorchAO -- from torchao import quantize_
#TorchAO -- from torchao.quantization.quant_api import Int4WeightOnlyConfig
#TorchAO -- quantize_(mobilenetv3_large, Int4WeightOnlyConfig(group_size=32))
#TODO: it may be that this TorchAO version (0.17.0) even though the recommended for my Torch version (2.11.0+cu128) is unstable and tutorials are not as widely available. Check into this if I ever decide to use TorchAO


'''mobilenetv3_small = torchvision.models.mobilenet_v3_small(pretrained=False)
mobilenetv3_large = torchvision.models.mobilenet_v3_large(pretrained=False)
mobilenetv3_small_pretrainedTrue = torchvision.models.mobilenet_v3_small(pretrained=True) #same size as when pretrained=False, according to torchinfo's summary
mobilenetv3_large_pretrainedTrue = torchvision.models.mobilenet_v3_large(pretrained=True)''' #same size as when pretrained=False, according to torchinfo's summary
'''summary(mobilenetv3_small,input_size=(1, 3, 32, 32))
print("mobilenetv3_small pretrained=False\n", mobilenetv3_small.classifier)
summary(mobilenetv3_large,input_size=(1, 3, 32, 32))
print("mobilenetv3_large pretrained=False\n", mobilenetv3_large.classifier)
summary(mobilenetv3_small_pretrainedTrue,input_size=(1,3,224,224)) #same size as when pretrained=False, according to torchinfo's summary
print("mobilenetv3_small pretrained=True IMAGENET INPUT SIZE \n", mobilenetv3_small_pretrainedTrue.classifier) #same size as when pretrained=False, according to torchinfo's summary
summary(mobilenetv3_large_pretrainedTrue,input_size=(1,3,224,224)) #same size as when pretrained=False, according to torchinfo's summary
print("mobilenetv3_large pretrained=True IMAGENET INPUT SIZE\n", mobilenetv3_large_pretrainedTrue.classifier) #same size as when pretrained=False, according to torchinfo's summary
'''

'''quantized_mobilenetv3 = torchvision.models.quantization.mobilenet_v3_large(pretrained=True, quantize=False) #TODO: change quantize back to False
quantized_mobilenetv3.train() #TODO: This .train() belongs to Avalanche. LEARN HOW TO USE AVALANCHE training AND WHETHER IS COMPATIBLE WITH QUANTISATION. "Definition of a learning experience based on a :class:`GenericCLScenario` instance. This experience implementation uses the generic experience-patterns assignment defined in the :class:`GenericCLScenario` instance. Instances of this class are usually obtained from a benchmark stream"
quantized_mobilenetv3.qconfig = quantization.get_default_qat_qconfig("x86")
quantization.prepare_qat(quantized_mobilenetv3, inplace=True)
# >>> Insert training loop/training logic here.
# convert and eval logic:
quantized_mobilenetv3.eval()
quantized_mobilenetv3 = quantization.convert(quantized_mobilenetv3)
print(quantized_mobilenetv3)'''
#TODO: RESEARCH! if using pretrained=True, prepare model to accept input size of CIFAR100. Research how to do that!.

###############################################################################################################################################
# Even though the best model for MCU is probably a quantized one, QAT via import torch.ao.quantization as quantization may not be
# as easily integrated with Avalanche and since I must first learn how to use Avalanche, let me use mobilenetv3_small_pretrainedTrue
# as the baseline models for now, until I learn how to use Avalanche and combine it with normal torchvision training.
# Also, if no longer using the quantize=True for the quantized model because if quantize=True then the model does not lend
# itself for further training (although it would be a good idea to quickly code this and verify that indeed quantize=True
# does not allow for the model to be further trained) and therefore, I am using import torch.ao.quantization as quantization for
# training/QAT/quantization then why use torchvision.models.quantization at all if it forces me to use mobilenet_v3_large as opposed to small?
# TODO: Verify all of this to be true and (am I really only using torchAO and no longer torchvision.models.quantization and so will it have no
#  effects if I prescind from torchvision.models.quantization OOOOOR do I need torchvision.models.quantization to load a mobilenet that lends
#  itself to quantization and further training BUT ALSO COMES WITH PRETRAINED WEIGHTS!!!!???? Is pretrained=True only a possibility if I use
#  torchvision.models.quantization models???? FIND OUT!!!!!!)


# temporary baseline models until I learn to use Avalanche and how it combines with torchvision:
mobilenetv3_small_pretrainedTrue = torchvision.models.mobilenet_v3_small(pretrained=True)
print(mobilenetv3_small_pretrainedTrue)

'''resnet18 = torchvision.models.resnet18(pretrained=False)
print(resnet18)
summary(resnet18,input_size=(1, 3, 32, 32))
print("resnet18 pretrained=False\n")'''
'''convnext_tiny = torchvision.models.convnext_tiny(pretrained=False)
print(convnext_tiny)
summary(convnext_tiny,input_size=(1, 3, 32, 32))
print("convnext_tiny pretrained=False\n")'''


alexnet = torchvision.models.res(pretrained=False)
print(alexnet)
summary(alexnet, input_size=(1, 3, 32, 32))
print("vgg16 pretrained=False\n")


