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

cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

#Code from https://github.com/ryanchankh/cifar100coarse and https://pytorch.org/tutorials/beginner/data_loading_tutorial.html and https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
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
        self.coarse_labels = self._sparse2coarse(self.base_dataset.targets)

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







if __name__ == "__main__":
    print(cifar100_train.data, cifar100_train.classes)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    ])

    train_dataset = Cifar100Coarse(root='./data', train=True, download=True, transform=transform)
    test_dataset = Cifar100Coarse(root='./data', train=False, download=True, transform=transform)

    val_size = 5000  # 10% of the 50000 that make up the training set
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    images, fine_labels, coarse_labels = next(iter(train_loader))
    print(images.shape)  # [128, 3, 32, 32]
    print(fine_labels[:5])
    print(coarse_labels[:5])
