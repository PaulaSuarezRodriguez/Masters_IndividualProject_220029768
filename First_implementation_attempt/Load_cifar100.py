import Config
from Config import SEED
from Config import set_device
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
from torchvision.io import decode_image
import numpy as np
import torchvision
from torch.utils.data import Dataset
import pickle

cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

cifar100_superclasses = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"], # medium-sized mammals, but in code is called "medium_mammals"
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
}

'''#Code from https://github.com/ryanchankh/cifar100coarse and https://pytorch.org/tutorials/beginner/data_loading_tutorial.html and https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# Also useful, https://discuss.pytorch.org/t/how-to-get-name-of-files-while-loading-images-in-pytorch/135736
class Cifar100Coarse(Dataset):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None):
        self.base_dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=None
        )
        self.transform = transform
        self.target_transform = target_transform
        self.coarse_labels = self._sparse2coarse(self.base_dataset.targets) #call _sparse2coarse from within __innit__ such that it is automatically applied when creating a Cifar100Coarse object

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, fine_label = self.base_dataset[idx]  # image is a PIL.Image, fine_label is int
        coarse_label = self.coarse_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            fine_label = self.target_transform(fine_label)
            coarse_label = self.target_transform(coarse_label)

        return image, fine_label, coarse_label

    @staticmethod
    def _sparse2coarse(targets):
        """Map fine label indices (0-99) to coarse label indices (0-19)."""
        coarse_map = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                 3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                 6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                 0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                 5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
                                10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
                                 2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        return coarse_map[np.array(targets)]
        
        
print(train_dataset.coarse_labels)
image, fine_label, coarse_label = train_dataset[5]
print(image, fine_label, coarse_label)
    #print(train_dataset.base_dataset.data[i], train_dataset.coarse_labels[i], train_dataset.base_dataset.targets[i])'''

# I discovered the CIFAR-100 dataset offered by torchvision does offer coarse labels, as in its source code
# (https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html) it specified that its CIFAR-100 dataset
# is that from https://cave.cs.toronto.edu/kriz/cifar.html , which does contain the coarse labels in a file "meta".
# At the same time, in https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html how
# classtorchvision.datasets.CIFAR100 only returns images and targets and in its source code, how the images and targets
# are extracted, giving an idea of how the coarse labels could also be extracted from this file
# (https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py). Moreover, pytorch tutorial
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files explains
# how a custom Dataset class is needed for extracting labels in those torchvision datasets where labeels are stored in a separate file.
# From developer forums such as https://stackoverflow.com/questions/74020318/pytorch-cifar100-dataset-coarse-and-fine-label-class,
# https://discuss.pytorch.org/t/does-the-cifar100-dataset-of-pytorch-provide-coarse-labels/10858 and succesful implementations such
# as https://github.com/DocsaidLab/cifar100_training_demo/blob/main/dataset.py and https://github.com/xiaodongww/pytorch/blob/master/cifarDataset.py#L54
# I verified the existence of coarse labels in the CIFAR-100 dataset offered by torchvision and used as inspiration for devising my own Cifar100Coarse dataset.

class Cifar100Coarse(Dataset):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None):
        self.base_dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=None)
        self.transform = transform
        self.target_transform = target_transform

        filename = "train" if train else "test"
        with open(f"{root}/cifar-100-python/{filename}", "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        self.coarse_labels = np.array(raw["coarse_labels"]) # Read coarse_labels directly from the pickle

        with open(f"{root}/cifar-100-python/meta", "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        self.fine_label_names = meta["fine_label_names"]
        self.coarse_label_names = meta["coarse_label_names"]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, fine_label = self.base_dataset[idx]
        coarse_label = self.coarse_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            fine_label = self.target_transform(fine_label)
            coarse_label = self.target_transform(coarse_label)

        return image, fine_label, coarse_label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

train_dataset = Cifar100Coarse(root='./data', train=True, download=True, transform=transform)
test_dataset = Cifar100Coarse(root='./data', train=False, download=True, transform=transform)

for i in range(len(train_dataset)):
    image, fine_label, coarse_label = train_dataset[i]
    print(f"{image.shape}\n"
          f"Fine label number: {fine_label}, Fine label name: {train_dataset.fine_label_names[fine_label]} \n"
          f"Coarse label number: {coarse_label}, Coarse label name: {train_dataset.coarse_label_names[coarse_label]} \n\n\n")

    if train_dataset.fine_label_names[fine_label] not in cifar100_superclasses[train_dataset.coarse_label_names[coarse_label]]:
        raise ValueError(
            f"Incorrect mapping for data sample {i}: "
            f"An image sample of fine label '{train_dataset.fine_label_names[fine_label]}' has incorrectly been assigned the coarse label '{train_dataset.coarse_label_names[coarse_label]}'"
        )

val_size = 5000  # 10% of the 50000 that make up the training set
train_size = len(train_dataset) - val_size

train_dataset, val_dataset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)



if __name__ == "__main__":
    '''print(cifar100_train.data, cifar100_train.classes)

    images, fine_labels, coarse_labels = next(iter(train_loader))
    print(images[:5])  # [128, 3, 32, 32]
    print(fine_labels[:5])
    print(coarse_labels[:5])'''


