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
from torch.utils.data import Subset
import pickle
import matplotlib.pyplot as plt
import avalanche
from avalanche.benchmarks import nc_benchmark # DEPRECATED but still in existance within "deprecated" folder in Avalanche 0.6.0 GitHub (https://github.com/ContinualAI/avalanche/blob/v0.6.0/avalanche/benchmarks/scenarios/deprecated/generators.py )
###from avalanche.benchmarks.generators import
from avalanche.benchmarks import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
#from avalanche.benchmarks.utils import make_classification_dataset # DEPRECATED! This is from v0.3.0 and the latest stable version / the one I am using is v0.6.0
#from avalanche.benchmarks.utils import make_tensor_classification_dataset # # DEPRECATED! This is from v0.3.1 and only in the docs of up to v0.5.0. The latest stable version / the one I am using is v0.6.0
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
DEBUG = False #TODO: Remember to change DEBUG to False if running experiments.

import os
print(os.getcwd())

cifar100_train = torchvision.datasets.CIFAR100(root='../data', train=True, download=True)
cifar100_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True)

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

similar_superclasses = {
    "aquatic": ["aquatic_mammals","fish"],
    "nature": ["flowers","trees","large_natural_outdoor_scenes"],
    "animals": ["large_carnivores","large_omnivores_and_herbivores","medium_mammals","small_mammals","reptiles","insects","non-insect_invertebrates"],
    "people": ["people"],
    "objects": ["household_furniture","household_electrical_devices","food_containers"],
    "transport": ["vehicles_1","vehicles_2"],
    "man-made": ["large_man-made_outdoor_things"],
    "food": ["fruit_and_vegetables"]
}

specialised_models_superclasses = {
    "outdoors":["large_man-made_outdoor_things", "large_natural_outdoor_scenes"],
    "nature": ["flowers","trees", "fruit_and_vegetables"],
    "people": ["people"],
    "animals": ["aquatic_mammals","fish", "large_carnivores", "large_omnivores_and_herbivores", "medium_mammals", "small_mammals", "reptiles", "insects", "non-insect_invertebrates"],
    "objects_and_vehicles": ["household_furniture","household_electrical_devices","food_containers", "vehicles_1","vehicles_2"],
}

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


#TODO: Remember to change DEBUG to False if running experiments.
if DEBUG:
    train_dataset = Cifar100Coarse(root='../data', train=True, download=True)
    test_dataset = Cifar100Coarse(root='../data', train=False, download=True)
    for i in range(len(train_dataset)):
        image, fine_label, coarse_label = train_dataset[i]
        print(#f"{image.shape}\n"
              f"Fine label number: {fine_label}, Fine label name: {train_dataset.fine_label_names[fine_label]} \n"
              f"Coarse label number: {coarse_label}, Coarse label name: {train_dataset.coarse_label_names[coarse_label]} \n\n\n")
        pil_img = image
        orig_w, orig_h = pil_img.size
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(pil_img, interpolation="nearest")
        ax.axis("off")
        plt.title(f"Coarse label:\n{train_dataset.coarse_label_names[coarse_label]}\n Fine label:\n{train_dataset.fine_label_names[fine_label]}", fontsize=8)
        plt.tight_layout()
        #if train_dataset.coarse_label_names[coarse_label] == "large_man-made_outdoor_things" or train_dataset.coarse_label_names[coarse_label] == "large_natural_outdoor_scenes":
            #plt.show()
        plt.show()
        plt.close()
        if train_dataset.fine_label_names[fine_label] not in cifar100_superclasses[train_dataset.coarse_label_names[coarse_label]]:
            raise ValueError(
                f"Incorrect mapping for data sample {i}: "
                f"An image sample of fine label '{train_dataset.fine_label_names[fine_label]}' has incorrectly been assigned the coarse label '{train_dataset.coarse_label_names[coarse_label]}'"
            )



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

train_dataset = Cifar100Coarse(root='../data', train=True, download=True, transform=transform)
test_dataset = Cifar100Coarse(root='../data', train=False, download=True, transform=transform)

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


#TODO: Let the split_by_coarse_label be for now and pre-train the global model first using the entire train dataset
def split_by_coarse_label(dataset, coarse_label_name):
    indices = []
    for idx, (_, _, coarse_label) in enumerate(dataset):
        if dataset.dataset.coarse_label_names[coarse_label] == coarse_label_name:
            indices.append(idx)
    return Subset(dataset, indices)


def split_by_similar_superclasses(dataset, similar_superclasses_group):
    """to be executed once per similar_superclasses_group/element in dictionary similar_superclasses to group samples from train_dataset into their corresponding superclass_similarity_groups.
    The reason coarse classes aren't the elements being grouped is due to the torchvision.datasets.Cifar100 having no notion of coarse labels or of
    any one training sample belonging to anything other than a fine-label. The only notion of coarse labels is generated by my custom code"""
    indices = []
    for idx, (_, _, coarse_label) in enumerate(dataset):
        #coarse_name = dataset.dataset.coarse_label_names[coarse_label]
        #if coarse_name in similar_superclasses_group:
        if dataset.dataset.coarse_label_names[coarse_label] in similar_superclasses_group:
            indices.append(idx)
    return Subset(dataset, indices)


def split_by_specialised_models_superclasses(dataset, specialised_models_superclasses_group):#, similar_superclasses_group):
    indices = []
    for idx, (_, _, coarse_label) in enumerate(dataset):
        if dataset.dataset.coarse_label_names[coarse_label] in specialised_models_superclasses_group:#[similar_superclasses_group]:
            indices.append(idx)
    return Subset(dataset, indices)


def split_by_indices_specialised_models_superclasses(dataset, specialised_models_superclasses):
    indices = []

    for specialised_models_superclasses_group in specialised_models_superclasses.values():
        for idx, (_, _, coarse_label) in enumerate(dataset):
            if dataset.dataset.coarse_label_names[coarse_label] in specialised_models_superclasses_group:
                indices.append(idx)

    return indices


'''def split_by_specialised_models_superclasses_using_avalanche(cifar100_train, cifar100_test, num_classes_per_group = [10, 15, 5, 45, 25]):#num_classes meaning num_classes_per_group as in for the specialised model for which we are creating a subset, how many classes does its specialised group of coarse classes / specialised subset contain? SAME AS num_classes IN class ResNet20 !!!
    avl_train = make_avalanche_dataset(cifar100_train) #(train_dataset)#(cifar100_train) #DEPRECATED make_classification_dataset(cifar100_train)  # .targets auto-detected (fine labels)
    avl_train = AvalancheDataset(
        [avl_train],
        data_attributes=[DataAttribute(train_dataset.dataset.coarse_labels.tolist(), "coarse_label")],
    )
    avl_test = make_avalanche_dataset(cifar100_test) #(test_dataset)#(cifar100_test) #DEPRECATED make_classification_dataset(cifar100_test)
    avl_test = AvalancheDataset(
        [avl_test],
        data_attributes=[DataAttribute(test_dataset.coarse_labels.tolist(), "coarse_label")],
    )

    scenario = nc_benchmark(
        train_dataset=avl_train, test_dataset=avl_test, # test_dataset=avl_test, val_dataset=avl_val,
        n_experiences=len(specialised_models_superclasses),
        task_labels=False,
        fixed_class_order=specialised_models_indices_all_superclasses,
        per_exp_classes={i: n for i, n in enumerate(num_classes_per_group)},
        class_ids_from_zero_in_each_exp=True,
    )
    return scenario.train_stream
    #exp.dataset.coarse_label # should exist post-split, alongside exp.dataset.targets'''

def split_by_specialised_models_superclasses_using_avalanche(cifar100_train_param, cifar100_test_param, train_dataset_param, test_dataset_param, specialised_models_indices_all_superclasses_param, num_classes_per_group = [10, 15, 5, 45, 25]):
    #class_order, num_classes_per_group = build_class_order(specialised_models_superclasses)

    avl_train = make_avalanche_dataset(cifar100_train_param) # make_classification_dataset(cifar100_train)  # .targets auto-detected
    avl_train = ClassificationDataset(
        [avl_train],
        data_attributes=[DataAttribute(train_dataset_param.dataset.coarse_labels.tolist(), "coarse_label"),
                         DataAttribute(train_dataset_param.indices, "fine_label")],
    )

    avl_test = make_avalanche_dataset(test_dataset_param) #make_classification_dataset(cifar100_test)
    avl_test = ClassificationDataset(
        [avl_test],
        data_attributes=[DataAttribute(test_dataset_param.coarse_labels.tolist(), "coarse_label"),
                         DataAttribute(test_dataset_param.base_dataset.targets, "fine_label")],
    )

    scenario = class_incremental_benchmark({"train": avl_train, "test":avl_test}, class_order=specialised_models_indices_all_superclasses_param, num_experiences=5, num_classes_per_exp={i: n for i, n in enumerate(num_classes_per_group)})
    return scenario.streams.train_stream

################## Calls to the functions that produce structures that need to be imported to other .py files: ############################################################################################################

# Code for function call to specialised_models_indices_all_superclasses such that it can serve as argument for parameter "fixed_class_order=" in avalanche's nc_benchmark for splitting train dataset by specialised_models_superclasses IN A MANNER THAT WILL NOT CAUSE A "RE-MAPPING OF LABELS" ERROR; outside of if __name__ == "__main__" so that it is executed and imported to other .py files whenever imported from other .py files
specialised_models_indices_all_superclasses = (
    split_by_indices_specialised_models_superclasses(
        train_dataset,
        specialised_models_superclasses
    )
)

#split_by_specialised_models_superclasses_using_avalanche(cifar100_train, cifar100_test, specialised_models_indices_all_superclasses, num_classes_per_group = [10, 15, 5, 45, 25])
split_by_specialised_models_superclasses_using_avalanche(cifar100_train, cifar100_test, train_dataset, test_dataset, specialised_models_indices_all_superclasses, num_classes_per_group = [10, 15, 5, 45, 25])
###########################################################################################################################################################################################################################

if __name__ == "__main__":
    '''# Code for testing train_loader
    images, fine_labels, coarse_labels = next(iter(train_loader))
    print(images[:5])  # [128, 3, 32, 32]
    print(fine_labels[:5])
    print(coarse_labels[:5])'''

    '''# Code for splitting train dataset by coarse label
    aquatic_mammals_train_subset = split_by_coarse_label(train_dataset, "aquatic_mammals")
    fish_train_subset = split_by_coarse_label(train_dataset, "fish")
    flowers_train_subset = split_by_coarse_label(train_dataset, "flowers")
    food_containers_train_subset = split_by_coarse_label(train_dataset, "food_containers")
    fruit_and_vegetables_train_subset = split_by_coarse_label(train_dataset, "fruit_and_vegetables")
    household_electrical_devices_train_subset = split_by_coarse_label(train_dataset, "household_electrical_devices")
    household_furniture_train_subset = split_by_coarse_label(train_dataset, "household_furniture")
    insects_train_subset = split_by_coarse_label(train_dataset, "insects")
    large_carnivores_train_subset = split_by_coarse_label(train_dataset, "large_carnivores")

    for (image, fine_label, coarse_label) in aquatic_mammals_train_subset:
        print(f"{image.shape}\n"
              f"type(aquatic_mammals_train_subset): {type(aquatic_mammals_train_subset)}\n"
              #f"Image tensor: {image},"
              f"Fine label number: {fine_label}, Fine label name: {aquatic_mammals_train_subset.dataset.dataset.fine_label_names[fine_label]} \n"
              f"Coarse label number: {coarse_label}, Coarse label name: {aquatic_mammals_train_subset.dataset.dataset.coarse_label_names[coarse_label]} \n\n\n")'''

    '''# Code for splitting train dataset by similar_superclass
    aquatic_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["aquatic"])
    nature_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["nature"])
    animals_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["animals"])
    humans_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["people"])
    objects_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["objects"])
    transport_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["transport"])
    man_made_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["man-made"])
    food_superclass_group_subset = split_by_similar_superclasses(train_dataset, similar_superclasses["food"])

    for (image, fine_label, coarse_label) in aquatic_superclass_group_subset:
        print(f"{image.shape}\n"
              f"type(aquatic_superclass_group_subset): {type(aquatic_superclass_group_subset)}\n"
              #f"Image tensor: {image},"
              f"Fine label number: {fine_label}, Fine label name: {aquatic_superclass_group_subset.dataset.dataset.fine_label_names[fine_label]} \n"
              f"Coarse label number: {coarse_label}, Coarse label name: {aquatic_superclass_group_subset.dataset.dataset.coarse_label_names[coarse_label]} \n\n\n")'''

    # Before realising this split would lead to issues regarding re-mapping of labels due to PyTorch models handling of data samples, this was Code for splitting train dataset by specialised_models_superclasses, outside of if __name__ == "__main__" so that it is executed and imported to other .py files whenever imported from other .py files
    outdoors_specialised_models_subset = split_by_specialised_models_superclasses(train_dataset, specialised_models_superclasses["outdoors"])
    nature_specialised_models_subset = split_by_specialised_models_superclasses(train_dataset,specialised_models_superclasses["nature"])
    people_specialised_models_subset = split_by_specialised_models_superclasses(train_dataset,specialised_models_superclasses["people"])
    animals_specialised_models_subset = split_by_specialised_models_superclasses(train_dataset,specialised_models_superclasses["animals"])
    objects_and_vehicles_specialised_models_subset = split_by_specialised_models_superclasses(train_dataset,specialised_models_superclasses["objects_and_vehicles"])

    '''# Code for TESTING splitting train dataset by specialised_models_superclasses; inside if __name__ == "__main__" so that it is not executed when this .py file and the ready-to-be-used-for-training-subsets are imported to other .py files
    for specialised_models_subset in [nature_specialised_models_subset, people_specialised_models_subset, animals_specialised_models_subset, objects_and_vehicles_specialised_models_subset]:
        for (image, fine_label, coarse_label) in specialised_models_subset:
            print(f"{image.shape}\n"
                  f"type({specialised_models_subset}): {type(specialised_models_subset)}\n"
                  #f"Image tensor: {image},"
                  f"Fine label number: {fine_label}, Fine label name: {specialised_models_subset.dataset.dataset.fine_label_names[fine_label]} \n"
                  f"Coarse label number: {coarse_label}, Coarse label name: {specialised_models_subset.dataset.dataset.coarse_label_names[coarse_label]} \n\n\n")'''

    print(outdoors_specialised_models_subset.indices)
    print(nature_specialised_models_subset.indices)
    print(people_specialised_models_subset.indices)
    print(animals_specialised_models_subset.indices)
    print(objects_and_vehicles_specialised_models_subset.indices)
    print("\n\n\n\n\n\n\n\n\n\n")

    '''# Code for TESTING specialised_models_indices_all_superclasses such that it can serve as argument for parameter "fixed_class_order=" in avalanche's nc_benchmark for splitting train dataset by specialised_models_superclasses IN A MANNER THAT WILL NOT CAUSE A "RE-MAPPING OF LABELS" ERROR; inside if __name__ == "__main__" so that it is not executed when this .py file and the ready-to-be-used-for-training-subsets are imported to other .py files
    specialised_models_indices_all_superclasses = (
        split_by_indices_specialised_models_superclasses(
            train_dataset,
            specialised_models_superclasses
        )
    )'''
    print(specialised_models_indices_all_superclasses)




# TODO: 1) finish testing if split_by_similar_superclasses works by inspecting outcomes of calling it in the for loop

#TODO: 2) finish similar_superclasses_grouping. This is for further classifying into broader groups. Is the below a valid example of that?
# I think not, but double-check
