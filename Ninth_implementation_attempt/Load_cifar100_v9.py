from Ninth_implementation_attempt.Config_v9 import set_seed

set_seed()
#set_device() # TODO: FIX ERROR! Why does set_device lead to error? ValueError: Expected a torch.device with a specified index or an integer, but got:cuda
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import Subset
import pickle
import matplotlib.pyplot as plt
###from avalanche.benchmarks.generators import
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
#from avalanche.benchmarks import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
#from avalanche.benchmarks.utils import make_classification_dataset # DEPRECATED! This is from v0.3.0 and the latest stable version / the one I am using is v0.6.0
#from avalanche.benchmarks.utils import make_tensor_classification_dataset # # DEPRECATED! This is from v0.3.1 and only in the docs of up to v0.5.0. The latest stable version / the one I am using is v0.6.0
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.scenarios.dataset_scenario import split_validation_class_balanced
from torchvision import transforms
#from avalanche.benchmarks.utils import make_classification_dataset
DEBUG = False #TODO: Remember to change DEBUG to False if running experiments.

import os
print(os.getcwd())

cifar100_train = torchvision.datasets.CIFAR100(root='../data', train=True, download=True)
cifar100_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True)

print("Type cifar100_train")
print(type(cifar100_train))

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

# Specifics of this version:
# https://github.com/ContinualAI/avalanche/blob/master/avalanche/benchmarks/utils/classification_dataset.py#L86 and https://avalanche-api.continualai.org/en/latest/generated/avalanche.benchmarks.class_incremental_benchmark.html#avalanche.benchmarks.class_incremental_benchmark
# why this combination? it seems class_incremental_benchmark takes ClassificationDatasets, which are a class inheriting from AvalancheDataset. Since I cannot
# manage to fix the error that the make_avalanche_dataset and class_incremental_benchmark combination of VERSION_4 gives me, I will try
# using ClassificationDatasets and class_incremental_benchmark in this, VERSION_5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

train_dataset = Cifar100Coarse(root='../data', train=True, download=True, transform=transform)
test_dataset = Cifar100Coarse(root='../data', train=False, download=True, transform=transform)

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
        # coarse_name = dataset.dataset.coarse_label_names[coarse_label]
        # if coarse_name in similar_superclasses_group:
        if dataset.dataset.coarse_label_names[coarse_label] in similar_superclasses_group:
            indices.append(idx)
    return Subset(dataset, indices)


# !!!!!!! CORRECTS def split_by_indices_specialised_models_superclasses(dataset, specialised_models_superclasses) FROM PREVIOUS VERSIONS TO WORK
# WITHOUT object Subset / train dataset already split by random_split creating Subset objects AND to return the fine-grained labels of all classes
# within specialised_models_superclasses in the order that results from iterating through specialised_models_superclasses as needed for class_order
# of class_incremental_benchmark
def split_by_indices_specialised_models_superclasses(dataset, specialised_models_superclasses): #function provided by ChatGPT
    # Map each fine class (0-99) to its coarse class (0-19)
    fine_to_coarse = {}

    for _, fine_label, coarse_label in dataset:
        if fine_label not in fine_to_coarse:
            fine_to_coarse[fine_label] = coarse_label
        if len(fine_to_coarse) == 100:
            break

    indices = []

    for group in specialised_models_superclasses.values():
        for fine_label, coarse_label in fine_to_coarse.items():
            if dataset.coarse_label_names[coarse_label] in group:
                indices.append(fine_label)

    return indices

################## Calls to the functions that produce structures that need to be imported to other .py files: ############################################################################################################
# Code for function call to specialised_models_indices_all_superclasses
# such that it can serve as argument for parameter "class_order" in avalanche's class_incremental_benchmark for splitting train dataset by
# specialised_models_superclasses IN A MANNER THAT WILL NOT CAUSE A "RE-MAPPING OF LABELS" ERROR; outside of if __name__ == "__main__" so
# that it is executed and imported to other .py files whenever imported from other .py files
print("Print type(train_dataset):\n", type(train_dataset))
print("Print type(test_dataset):\n", type(test_dataset))

num_classes_per_group = [10, 15, 5, 45, 25]

specialised_models_indices_all_superclasses = (
    split_by_indices_specialised_models_superclasses(
        train_dataset,
        specialised_models_superclasses
    )
)
print("Print type(train_dataset):\n", type(train_dataset))
print("Print type(test_dataset):\n", type(test_dataset))
# Former def split_by_specialised_models_superclasses_using_avalanche_VERSION_4(cifar100_train_param, cifar100_test_param, train_dataset_param, test_dataset_param, specialised_models_indices_all_superclasses_param, num_classes_per_group = [10, 15, 5, 45, 25]): ,
# which in turn was a replacement for def split_by_specialised_models_superclasses_using_avalanche(...) FROM PREVIOUS VERSIONS TO WORK WITHOUT object Subset / train dataset already split by random_split creating Subset objects
avl_train = make_avalanche_dataset(train_dataset)  # make_classification_dataset(cifar100_train)  # .targets auto-detected
print("Type avl_train BEFORE being passed to ClassificationDataset", type(avl_train))
avl_train = ClassificationDataset(
    [avl_train],
    data_attributes=[DataAttribute(train_dataset.coarse_labels.tolist(), "coarse_label"),
                     DataAttribute(train_dataset.base_dataset.targets, "targets")], #"fine_label")],
)

avl_test = make_avalanche_dataset(test_dataset)  # make_classification_dataset(cifar100_test)
print("Type avl_test BEFORE being passed to ClassificationDataset", type(avl_test))
avl_test = ClassificationDataset(
    [avl_test],
    data_attributes=[DataAttribute(test_dataset.coarse_labels.tolist(), "coarse_label"),
                     DataAttribute(test_dataset.base_dataset.targets, "targets")],# "fine_label")],
)

print("type(test_dataset.coarse_labels)", type(test_dataset.coarse_labels))
print("type(test_dataset.coarse_labels.tolist())", type(test_dataset.coarse_labels.tolist()))
print("type(test_dataset.base_dataset.targets)", type(test_dataset.base_dataset.targets))

print("Type avl_train", type(avl_train))
print("Type avl_test", type(avl_test))
print("Print avl_train:\n", avl_train)
print("Print avl_test:\n", avl_test)
print("Print len(specialised_models_indices_all_superclasses):", len(specialised_models_indices_all_superclasses))
print("Print specialised_models_indices_all_superclasses:")
print(specialised_models_indices_all_superclasses)


scenario_train_test = class_incremental_benchmark({"train": avl_train, "test": avl_test},
                                       class_order=specialised_models_indices_all_superclasses, #num_experiences=5,
                                       num_classes_per_exp=num_classes_per_group) #, seed=SEED) ValueError: Can't set `seed` if a fixed `class_order` is given.
                                                                                # num_classes_per_exp={i: n for i, n in enumerate(num_classes_per_group)})

print("Print scenario_train_test:\n", scenario_train_test, "\n")
print("Print scenario_train_test.streams:\n", scenario_train_test.streams, "\n")
print("Print type(scenario_train_test.streams):\n", type(scenario_train_test.streams), "\n")
train_stream_all_specialised_models = scenario_train_test.streams["train"]
print("Print train_stream_all_specialised_models:\n", train_stream_all_specialised_models, "\n")
print("Print train_stream_all_specialised_models.name:\n", train_stream_all_specialised_models.name, "\n")
print("Print train_stream_all_specialised_models.set_stream_info:\n", train_stream_all_specialised_models.set_stream_info, "\n")
print("Print train_stream_all_specialised_models.benchmark:\n", train_stream_all_specialised_models.benchmark, "\n")
print("Print train_stream_all_specialised_models.exps_iter:\n", train_stream_all_specialised_models.exps_iter, "\n")
test_stream_all_specialised_models = scenario_train_test.streams["test"]

# Train subsets for each specialised model BEFORE train-val split
outdoors_specialised_models_full_train_subset = train_stream_all_specialised_models[0]
nature_specialised_models_full_train_subset = train_stream_all_specialised_models[1]
people_specialised_models_full_train_subset = train_stream_all_specialised_models[2]
animals_specialised_models_full_train_subset = train_stream_all_specialised_models[3]
objects_and_vehicles_specialised_models_full_train_subset = train_stream_all_specialised_models[4]

print("Print outdoors_specialised_models_full_train_subset:\n", outdoors_specialised_models_full_train_subset)
print("type outdoors_specialised_models_full_train_subset:\n", type(outdoors_specialised_models_full_train_subset), "\n")
print("Print nature_specialised_models_full_train_subset:\n", nature_specialised_models_full_train_subset)
print("type nature_specialised_models_full_train_subset:\n", type(nature_specialised_models_full_train_subset), "\n")
print("Print people_specialised_models_full_train_subset:\n", people_specialised_models_full_train_subset)
print("type people_specialised_models_full_train_subset:\n", type(people_specialised_models_full_train_subset), "\n")
print("Print animals_specialised_models_full_train_subset:\n", animals_specialised_models_full_train_subset)
print("type animals_specialised_models_full_train_subset:\n", type(animals_specialised_models_full_train_subset), "\n")
print("Print objects_and_vehicles_specialised_models_full_train_subset:\n", objects_and_vehicles_specialised_models_full_train_subset)
print("type objects_and_vehicles_specialised_models_full_train_subset:\n", type(objects_and_vehicles_specialised_models_full_train_subset), "\n")


# Test subsets for each specialised model BEFORE train-val split
outdoors_specialised_models_test_subset_before_train_val_split = test_stream_all_specialised_models[0]
nature_specialised_models_test_subset_before_train_val_split = test_stream_all_specialised_models[1]
people_specialised_models_test_subset_before_train_val_split = test_stream_all_specialised_models[2]
animals_specialised_models_test_subset_before_train_val_split = test_stream_all_specialised_models[3]
objects_and_vehicles_specialised_models_test_subset_before_train_val_split = test_stream_all_specialised_models[4]

print("Print outdoors_specialised_models_test_subset_before_train_val_split:\n", outdoors_specialised_models_test_subset_before_train_val_split)
print("type outdoors_specialised_models_test_subset_before_train_val_split:\n", type(outdoors_specialised_models_test_subset_before_train_val_split), "\n")
print("Print nature_specialised_models_test_subset_before_train_val_split:\n", nature_specialised_models_test_subset_before_train_val_split)
print("type nature_specialised_models_test_subset_before_train_val_split:\n", type(nature_specialised_models_test_subset_before_train_val_split), "\n")
print("Print people_specialised_models_test_subset_before_train_val_split:\n", people_specialised_models_test_subset_before_train_val_split)
print("type people_specialised_models_test_subset_before_train_val_split:\n", type(people_specialised_models_test_subset_before_train_val_split), "\n")
print("Print animals_specialised_models_test_subset_before_train_val_split:\n", animals_specialised_models_test_subset_before_train_val_split)
print("type animals_specialised_models_test_subset_before_train_val_split:\n", type(animals_specialised_models_test_subset_before_train_val_split), "\n")
print("Print objects_and_vehicles_specialised_models_test_subset_before_train_val_split:\n", objects_and_vehicles_specialised_models_test_subset_before_train_val_split)
print("type objects_and_vehicles_specialised_models_test_subset_before_train_val_split:\n", type(objects_and_vehicles_specialised_models_test_subset_before_train_val_split), "\n")


# Code for splitting the train dataset already processed by class Cifar100Coarse and already divided into "experiences" to serve as the subsets for
# training the specialised models (i.e. the train subsets that are already class 'avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience'
# objects given they are elements within the .streams["train"] returned by class_incremental_benchmark) into train and validation datasets.
# Although, even though the validation dataset comes out of the train dataset only, and not the test one, since benchmark_with_validation_stream
# expects the entire scenario_train_test: CLScenario = class_incremental_benchmark from which the "experiences" to serve as the subsets for
# training the specialised models, scenario_train_val_test = benchmark_with_validation_stream returns a train, validation and test set, with the test
# being identical to the one passed as input and only the train set being edited to produce the validation set / be split into train and validation sets,
# BOTH OF WHICH CONTAINING "experiences" to serve as the subsets for training the specialised models
validation_size = 0.2
foo = lambda data: split_validation_class_balanced(validation_size, data)
scenario_train_val_test = benchmark_with_validation_stream(
    scenario_train_test,
    #validation_size=validation_size,
    #seed=SEED,
    split_strategy=foo
)
# version that lead to "raise ValueError("`datasets` must all have the same classes") ValueError: `datasets` must all have the same classes"
# i.e. the version from last Load_Cifar100.py commit from yesterday
# 31/07/2026 00:06 am --> even with the changes made to Train_ResNet20_v4.py, it still raises the same error as it did before. The error is
# DEFINITELY caused by splitting without split_validation_class_balanced in benchmark_with_validation_stream, and in turn it does NOT seem to be
# what it's causing the new error in Train_ResNet20_v4.py (File "C:\Users\...\avalanche\benchmarks\scenarios\supervised.py", line 133, in class_incremental_benchmark
# curr_data = data_by_class[cls].concat(curr_data) KeyError: 63), so try to debug new error in Train_ResNet20_v4.py using the
# benchmark_with_validation_stream with split_validation_class_balanced; i.e. the one above.
'''scenario_train_val_test = benchmark_with_validation_stream(
    scenario_train_test,
    validation_size=0.2,
    seed=SEED
)'''

print("Print scenario_train_val_test:\n", scenario_train_val_test, "\n")
print("Print scenario_train_val_test.streams:\n", scenario_train_val_test.streams, "\n")
print("Print type(scenario_train_val_test.streams):\n", type(scenario_train_val_test.streams), "\n")
train_stream_class_il = scenario_train_val_test.streams["train"]
val_stream_class_il = scenario_train_val_test.streams["valid"]
test_stream_class_il = scenario_train_val_test.streams["test"]
print("Print train_stream_class_il:\n", train_stream_class_il, "\n")
print("Print val_stream_class_il:\n", val_stream_class_il, "\n")
print("Print test_stream_class_il:\n", test_stream_class_il, "\n")
print("Print type(train_stream_class_il):\n", type(train_stream_class_il), "\n")
print("Print type(val_stream_class_il):\n", type(val_stream_class_il), "\n")
print("Print type(test_stream_class_il):\n", type(test_stream_class_il), "\n")

# Train subsets for each specialised model AFTER train-val split
outdoors_specialised_models_train_subset = train_stream_class_il[0]
nature_specialised_models_train_subset = train_stream_class_il[1]
people_specialised_models_train_subset = train_stream_class_il[2]
animals_specialised_models_train_subset = train_stream_class_il[3]
objects_and_vehicles_specialised_models_train_subset = train_stream_class_il[4]

print("Print outdoors_specialised_models_train_subset:\n", outdoors_specialised_models_train_subset)
print("type outdoors_specialised_models_train_subset:\n", type(outdoors_specialised_models_train_subset), "\n")
print("Print nature_specialised_models_train_subset:\n", nature_specialised_models_train_subset)
print("type nature_specialised_models_train_subset:\n", type(nature_specialised_models_train_subset), "\n")
print("Print people_specialised_models_train_subset:\n", people_specialised_models_train_subset)
print("type people_specialised_models_train_subset:\n", type(people_specialised_models_train_subset), "\n")
print("Print animals_specialised_models_train_subset:\n", animals_specialised_models_train_subset)
print("type animals_specialised_models_train_subset:\n", type(animals_specialised_models_train_subset), "\n")
print("Print objects_and_vehicles_specialised_models_train_subset:\n", objects_and_vehicles_specialised_models_train_subset)
print("type objects_and_vehicles_specialised_models_train_subset:\n", type(objects_and_vehicles_specialised_models_train_subset), "\n")



# Test subsets for each specialised model AFTER train-val split (which are identical to the test subsets before the train-val split since the val
# subset is taken only from the full training subset provided by torchvision, not the test one. The only thing that changes is the DATA TYPE since
# after the split, all three subsets train, test and val will still be CLScenario, but the kind returned by benchmark_with_validation_stream, which
# has three elements/streams to unpack (one per subset), as opposed to only two, like the CLScenario object produced by scenario_train_test =
# class_incremental_benchmark did)
outdoors_specialised_models_test_subset = test_stream_class_il[0]
nature_specialised_models_test_subset = test_stream_class_il[1]
people_specialised_models_test_subset = test_stream_class_il[2]
animals_specialised_models_test_subset = test_stream_class_il[3]
objects_and_vehicles_specialised_models_test_subset = test_stream_class_il[4]

print("Print outdoors_specialised_models_test_subset:\n", outdoors_specialised_models_test_subset)
print("type outdoors_specialised_models_test_subset:\n", type(outdoors_specialised_models_test_subset), "\n")
print("Print nature_specialised_models_test_subset:\n", nature_specialised_models_test_subset)
print("type nature_specialised_models_test_subset:\n", type(nature_specialised_models_test_subset), "\n")
print("Print people_specialised_models_test_subset:\n", people_specialised_models_test_subset)
print("type people_specialised_models_test_subset:\n", type(people_specialised_models_test_subset), "\n")
print("Print animals_specialised_models_test_subset:\n", animals_specialised_models_test_subset)
print("type animals_specialised_models_test_subset:\n", type(animals_specialised_models_test_subset), "\n")
print("Print objects_and_vehicles_specialised_models_test_subset:\n", objects_and_vehicles_specialised_models_test_subset)
print("type objects_and_vehicles_specialised_models_test_subset:\n", type(objects_and_vehicles_specialised_models_test_subset), "\n")



# Validation subsets for each specialised model AFTER train-val split (not that there were any val subsets before the split train-val split)
outdoors_specialised_models_val_subset = val_stream_class_il[0]
nature_specialised_models_val_subset = val_stream_class_il[1]
people_specialised_models_val_subset = val_stream_class_il[2]
animals_specialised_models_val_subset = val_stream_class_il[3]
objects_and_vehicles_specialised_models_val_subset = val_stream_class_il[4]

print("Print outdoors_specialised_models_val_subset:\n", outdoors_specialised_models_val_subset)
print("type outdoors_specialised_models_val_subset:\n", type(outdoors_specialised_models_val_subset), "\n")
print("Print nature_specialised_models_val_subset:\n", nature_specialised_models_val_subset)
print("type nature_specialised_models_val_subset:\n", type(nature_specialised_models_val_subset), "\n")
print("Print people_specialised_models_val_subset:\n", people_specialised_models_val_subset)
print("type people_specialised_models_val_subset:\n", type(people_specialised_models_val_subset), "\n")
print("Print animals_specialised_models_val_subset:\n", animals_specialised_models_val_subset)
print("type animals_specialised_models_val_subset:\n", type(animals_specialised_models_val_subset), "\n")
print("Print objects_and_vehicles_specialised_models_val_subset:\n", objects_and_vehicles_specialised_models_val_subset)
print("type objects_and_vehicles_specialised_models_val_subset:\n", type(objects_and_vehicles_specialised_models_val_subset), "\n")


##################################################################################################################################
###########################################################################################################################################################################################################################

if __name__ == "__main__":
    print("This is the main file of Load_cifar100_v7.")