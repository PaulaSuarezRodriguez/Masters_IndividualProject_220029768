from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.evaluation.metrics.forgetting_bwt import forgetting_to_bwt, bwt_metrics
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from avalanche.models.dynamic_optimizers import update_optimizer
from avalanche.training.plugins.checkpoint import CheckpointPlugin
from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate
from avalanche.training.templates.base import TExperienceType, _experiences_parameter_as_iterable, \
    _group_experiences_by_stream

from Final_implementation_attempt.Config_vFinal import set_seed

set_seed()  # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Final_implementation_attempt.Config_vFinal import set_device

DEVICE = set_device()
from Final_implementation_attempt.Config_vFinal import SEED
from avalanche.training.determinism.rng_manager import RNGManager

RNGManager.set_random_seeds(SEED)
from Final_implementation_attempt.ResNet20_vFinal import ResNet20
from Final_implementation_attempt.ResNet20_DynamicModel_vFinal import ResNet20_DynamicModel
from Final_implementation_attempt.Load_cifar100_vFinal import (objects_and_vehicles_specialised_models_train_subset)
from Final_implementation_attempt.Load_cifar100_vFinal import (outdoors_specialised_models_train_subset)
from Final_implementation_attempt.Load_cifar100_vFinal import (objects_and_vehicles_specialised_models_val_subset)
from Final_implementation_attempt.Load_cifar100_vFinal import (objects_and_vehicles_specialised_models_test_subset)
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.optim import SGD, Adam
from avalanche.training.supervised import Naive, EWC, LwF
from avalanche.training.losses import MaskedCrossEntropy
from avalanche.training.plugins import EarlyStoppingPlugin, EvaluationPlugin, EWCPlugin, LwFPlugin
from avalanche.training import LearningWithoutForgetting, RegularizationMethod
import numpy as np
import copy
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F

from avalanche.models import MultiTaskModule, avalanche_forward
from avalanche._annotations import deprecated

from collections.abc import \
    Iterable
from avalanche.evaluation.metrics.accuracy import accuracy_metrics
from avalanche.core import BasePlugin
from avalanche.evaluation.metrics import loss_metrics, forgetting_metrics
from avalanche.training.plugins.evaluation import default_loggers
import torch
from torch.nn import CrossEntropyLoss
import os.path
from pathlib import Path
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    BinaryIO,
    IO,
    Union,
    Collection,
)
from typing_extensions import TypeAlias

import dill
import torch
from functools import partial
from packaging.version import parse

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from avalanche.models.utils import avalanche_forward

from avalanche.benchmarks import CLExperience, SplitMNIST, AvalancheDataset, OnlineCLExperience, CLStream
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    class_accuracy_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger
from avalanche.models import SimpleMLP, as_multitask
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin
from avalanche.training.supervised import Naive
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from avalanche.training.templates.common_templates import SupervisedTemplate, SGDUpdate
import sys
from typing import Callable, Sequence, Optional, TypeVar, Union, List, Dict, Collection
from collections import defaultdict
import torch
from pathlib import Path
from copy import copy
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.benchmarks import DatasetExperience

from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.templates.strategy_mixin_protocol import (
    CriterionType,
    SupervisedStrategyProtocol,
    TMBOutput,
    TMBInput, SGDStrategyProtocol, TSGDExperienceType,
)
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData, _at_task_boundary
import warnings
import itertools

import operator
from typing import Optional
import warnings
from copy import deepcopy
import pickle
from avalanche.training.plugins import SupervisedPlugin
from torch.optim.adam import Adam
import tracemalloc
import torch
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, _T_co
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_definitions import T_co
from avalanche.training.templates.base_sgd import PeriodicEval
from avalanche.training.plugins.clock import Clock
import hashlib
import json

DEBUG = False
TRAINING_WITH_LOOP = False
TRAINING_WITHOUT_LOOP = False
DYNAMIC_MODELS = True
print(
    "\n\n\n################################ HERE BEGINS Train_ResNet20_Objects_and_Vehicles_vFinal.py ################################ \n\n\n")

if DYNAMIC_MODELS == False:
    resnet20_global_model = ResNet20(num_classes=100)
    outdoors_specialised_model = ResNet20(
        num_classes=100)
    nature_specialised_model = ResNet20(
        num_classes=100)
    people_specialised_model = ResNet20(
        num_classes=100)
    animals_specialised_model = ResNet20(
        num_classes=100)
    objects_and_vehicles_specialised_model = ResNet20(
        num_classes=100)

if DYNAMIC_MODELS == True:
    resnet20_global_model = ResNet20_DynamicModel(num_classes=0)
    outdoors_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    nature_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    people_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    animals_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    objects_and_vehicles_specialised_model = ResNet20_DynamicModel(
        num_classes=0)


from Final_implementation_attempt.Train_ResNet20_Outdoors_LwF_vFinal import avl_train
avl_train_outdoors = avl_train
from Final_implementation_attempt.Train_ResNet20_Nature_LwF_vFinal import avl_train
avl_train_nature = avl_train
from Final_implementation_attempt.Train_ResNet20_People_LwF_vFinal import avl_train
avl_train_people = avl_train
from Final_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vFinal import avl_train
avl_train_objects_and_vehicles = avl_train

print("len(avl_train_outdoors)",len(avl_train_outdoors))
count_el = 0
if DEBUG:
    for i in avl_train_outdoors.targets:
        print("Index:",count_el, "|| for i in avl_train_outdoors.targets:",i)
        count_el += 1

count_el = 0
outdoors_avalanche_dataset_train = AvalancheDataset([avl_train_outdoors],indices=[i for i in range(2400)])
if DEBUG:
    for i in outdoors_avalanche_dataset_train.targets:
        print("Index:",count_el, "|| for i in outdoors_avalanche_dataset_train.targets:",i)
        count_el += 1

print("len(avl_train_nature)",len(avl_train_nature))
count_el = 0
if DEBUG:
    for i in avl_train_nature.targets:
        print("Index:",count_el, "|| for i in avl_train_nature.targets:",i)
        count_el += 1

count_el = 0
nature_avalanche_dataset_train = AvalancheDataset([avl_train_nature],indices=[i for i in range(3600)])
if DEBUG:
    for i in nature_avalanche_dataset_train.targets:
        print("Index:",count_el, "|| for i in nature_avalanche_dataset_train.targets:",i)
        count_el += 1

print("len(avl_train_people)",len(avl_train_people))
count_el = 0
if DEBUG:
    for i in avl_train_people.targets:
        print("Index:",count_el, "|| for i in avl_train_people.targets:",i)
        count_el += 1

count_el = 0
people_avalanche_dataset_train = AvalancheDataset([avl_train_people],indices=[i for i in range(1200)])
if DEBUG:
    for i in people_avalanche_dataset_train.targets:
        print("Index:",count_el, "|| for i in people_avalanche_dataset_train.targets:",i)
        count_el += 1

print("len(avl_train_objects_and_vehicles)",len(avl_train_objects_and_vehicles))
count_el = 0
if DEBUG:
    for i in avl_train_objects_and_vehicles.targets:
        print("Index:",count_el, "|| for i in avl_train_objects_and_vehicles.targets:",i)
        count_el += 1

count_el = 0
objects_and_vehicles_avalanche_dataset_train = AvalancheDataset([avl_train_objects_and_vehicles])
if DEBUG:
    for i in objects_and_vehicles_avalanche_dataset_train.targets:
        print("Index:",count_el, "|| for i in objects_and_vehicles_avalanche_dataset_train.targets:",i)
        count_el += 1

count_el = 0
global_model_naive_baseline_avalanche_dataset_train = AvalancheDataset([outdoors_avalanche_dataset_train ,nature_avalanche_dataset_train, people_avalanche_dataset_train ,objects_and_vehicles_avalanche_dataset_train])
if DEBUG:
    for i in global_model_naive_baseline_avalanche_dataset_train.targets:
        print("Index:",count_el, "|| for i in global_model_naive_baseline_avalanche_dataset_train.targets:",i)
        count_el += 1

count_el = 0
if DEBUG:
    for i in global_model_naive_baseline_avalanche_dataset_train.coarse_label:
        print("Index:",count_el, "|| for i in global_model_naive_baseline_avalanche_dataset_train.coarse_label:",i)
        count_el += 1

avl_train = make_avalanche_dataset(global_model_naive_baseline_avalanche_dataset_train)


from Final_implementation_attempt.Train_ResNet20_Outdoors_LwF_vFinal import avl_val
avl_val_outdoors = avl_val
from Final_implementation_attempt.Train_ResNet20_Nature_LwF_vFinal import avl_val
avl_val_nature = avl_val
from Final_implementation_attempt.Train_ResNet20_People_LwF_vFinal import avl_val
avl_val_people = avl_val
from Final_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vFinal import avl_val
avl_val_objects_and_vehicles = avl_val

print("len(avl_val_outdoors)",len(avl_val_outdoors))
count_el = 0
if DEBUG:
    for i in avl_val_outdoors.targets:
        print("Index:",count_el, "|| for i in avl_val_outdoors.targets:",i)
        count_el += 1

count_el = 0
outdoors_avalanche_dataset_val = AvalancheDataset([avl_val_outdoors],indices=[i for i in range(600)])
if DEBUG:
    for i in outdoors_avalanche_dataset_val.targets:
        print("Index:",count_el, "|| for i in outdoors_avalanche_dataset_val.targets:",i)
        count_el += 1

print("len(avl_val_nature)",len(avl_val_nature))
count_el = 0
if DEBUG:
    for i in avl_val_nature.targets:
        print("Index:",count_el, "|| for i in avl_val_nature.targets:",i)
        count_el += 1

count_el = 0
nature_avalanche_dataset_val = AvalancheDataset([avl_val_nature],indices=[i for i in range(900)])
if DEBUG:
    for i in nature_avalanche_dataset_val.targets:
        print("Index:",count_el, "|| for i in nature_avalanche_dataset_val.targets:",i)
        count_el += 1

print("len(avl_val_people)",len(avl_val_people))
count_el = 0
if DEBUG:
    for i in avl_val_people.targets:
        print("Index:",count_el, "|| for i in avl_val_people.targets:",i)
        count_el += 1

count_el = 0
people_avalanche_dataset_val = AvalancheDataset([avl_val_people],indices=[i for i in range(300)])
if DEBUG:
    for i in people_avalanche_dataset_val.targets:
        print("Index:",count_el, "|| for i in people_avalanche_dataset_val.targets:",i)
        count_el += 1

print("len(avl_val_objects_and_vehicles)",len(avl_val_objects_and_vehicles))
count_el = 0
if DEBUG:
    for i in avl_val_objects_and_vehicles.targets:
        print("Index:",count_el, "|| for i in avl_val_objects_and_vehicles.targets:",i)
        count_el += 1

count_el = 0
objects_and_vehicles_avalanche_dataset_val = AvalancheDataset([avl_val_objects_and_vehicles])
if DEBUG:
    for i in objects_and_vehicles_avalanche_dataset_val.targets:
        print("Index:",count_el, "|| for i in objects_and_vehicles_avalanche_dataset_val.targets:",i)
        count_el += 1

count_el = 0
global_model_naive_baseline_avalanche_dataset_val = AvalancheDataset([outdoors_avalanche_dataset_val ,nature_avalanche_dataset_val, people_avalanche_dataset_val ,objects_and_vehicles_avalanche_dataset_val])
if DEBUG:
    for i in global_model_naive_baseline_avalanche_dataset_val.targets:
        print("Index:",count_el, "|| for i in global_model_naive_baseline_avalanche_dataset_val.targets:",i)
        count_el += 1

count_el = 0
if DEBUG:
    for i in global_model_naive_baseline_avalanche_dataset_val.coarse_label:
        print("Index:",count_el, "|| for i in global_model_naive_baseline_avalanche_dataset_val.coarse_label:",i)
        count_el += 1

avl_val = make_avalanche_dataset(global_model_naive_baseline_avalanche_dataset_val)

from Final_implementation_attempt.Train_ResNet20_Outdoors_LwF_vFinal import avl_test
avl_test_outdoors = avl_test
from Final_implementation_attempt.Train_ResNet20_Nature_LwF_vFinal import avl_test
avl_test_nature = avl_test
from Final_implementation_attempt.Train_ResNet20_People_LwF_vFinal import avl_test
avl_test_people = avl_test
from Final_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vFinal import avl_test
avl_test_objects_and_vehicles = avl_test

from itertools import chain

print("len(avl_test_outdoors)",len(avl_test_outdoors))
count_el = 0
if DEBUG:
    for i in avl_test_outdoors.targets:
        print("Index:",count_el, "|| for i in avl_test_outdoors.targets:",i)
        count_el += 1

count_el = 0
#outdoors: new_to_original: {0: 12, 1: 17, 2: 23, 3: 33, 4: 37, 5: 49, 6: 60, 7: 68, 8: 71, 9: 76}
outdoors_avalanche_dataset_test = AvalancheDataset([avl_test_outdoors], indices=[i for i in chain(range(200, 300), range(800, 900), range(900, 1000), range(400, 500), range(100, 200), range(600, 700))])
if DEBUG:
    for i in outdoors_avalanche_dataset_test.targets:
        print("Index:",count_el, "|| for i in outdoors_avalanche_dataset_test.targets:",i)
        count_el += 1

for i in outdoors_avalanche_dataset_test.targets:
    print("Index:",count_el, "|| for i in outdoors_avalanche_dataset_test.targets:",i)
    count_el += 1

print("len(avl_test_nature)",len(avl_test_nature))
count_el = 0
if DEBUG:
    for i in avl_test_nature.targets:
        print("Index:",count_el, "|| for i in avl_test_nature.targets:",i)
        count_el += 1

for i in avl_test_nature.targets:
    print("Index:",count_el, "|| for i in avl_test_nature.targets:",i)
    count_el += 1
count_el = 0
# nature new_to_original: {0: 0, 1: 47, 2: 51, 3: 52, 4: 53, 5: 54, 6: 56, 7: 57, 8: 59}
#[i for i in chain(range(1400, 1500), range(800, 900), range(600, 700), range(900, 1000), range(400, 500), range(700, 800), range(0, 100), range(1100, 1200))]
nature_avalanche_dataset_test = AvalancheDataset([avl_test_nature],indices=[i for i in chain(range(1400, 1500), range(800, 900), range(600, 700), range(900, 1000),range(100, 200), range(400, 500), range(700, 800), range(0, 100), range(1100, 1200))])#,indices=[i for i in range(900)])
if DEBUG:
    for i in nature_avalanche_dataset_test.targets:
        print("Index:",count_el, "|| for i in nature_avalanche_dataset_test.targets:",i)
        count_el += 1

for i in nature_avalanche_dataset_test.targets:
    print("Index:",count_el, "|| for i in nature_avalanche_dataset_test.targets:",i)
    count_el += 1

print("len(avl_test_people)",len(avl_test_people))
count_el = 0
if DEBUG:
    for i in avl_test_people.targets:
        print("Index:",count_el, "|| for i in avl_test_people.targets:",i)
        count_el += 1

count_el = 0
#people new_to_original: {0: 2, 1: 11, 2: 35, 3: 46, 4: 98}
people_avalanche_dataset_test = AvalancheDataset([avl_test_people],indices=[i for i in chain(range(0, 100), range(400, 500), range(200, 300))])#,indices=[i for i in range(900)])
if DEBUG:
    for i in people_avalanche_dataset_test.targets:
        print("Index:",count_el, "|| for i in people_avalanche_dataset_test.targets:",i)
        count_el += 1


for i in people_avalanche_dataset_test.targets:
    print("Index:",count_el, "|| for i in people_avalanche_dataset_test.targets:",i)
    count_el += 1


print("len(avl_test_objects_and_vehicles)",len(avl_test_objects_and_vehicles))
count_el = 0
#new_to_original: {5, 8, 9, 10, 13, 16, 20, 22, 25, 28, 39, 40, 41, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94}
if DEBUG:
    for i in avl_test_objects_and_vehicles.targets:
        print("Index:",count_el, "|| for i in avl_test_objects_and_vehicles.targets:",i)
        count_el += 1


for i in avl_test_objects_and_vehicles.targets:
    print("Index:",count_el, "|| for i in avl_test_objects_and_vehicles.targets:",i)
    count_el += 1
    #if i == 13: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
     #   print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)

    '''if i == 22:
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 25:
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 28: #28, 39, 40, 41, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 39: #, 40, 41, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 40: #, 41, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 41: #, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 48: #48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 58: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 61: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 69: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 81: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 84: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 85: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 86: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 87: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 89: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 90: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)
        if i == 94: #, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94
            print("Index:", count_el, "|| for i in avl_test_objects_and_vehicles.targets:", i)'''

count_el = 0
'''objects_and_vehicles_avalanche_dataset_test = AvalancheDataset([avl_test_objects_and_vehicles], indices=[i for i in chain(range(200, 300), #5
                                                                                                                        range(2000, 2100), #8
                                                                                                                        range(1400, 1500), #9
                                                                                                                        range(300, 400), #10
                                                                                                                        range(500, 600), #13
                                                                                                                        range(700, 800), #16
                                                                                                                        range(0, 100), #20
                                                                                                                        range(1300, 1400), #22
                                                                                                                        range(1300, 1400), #25
                                                                                                                          #28
                                                                                                                          #39
                                                                                                                          #40
                                                                                                                          #41
                                                                                                                        #48,
                                                                                                                          # 58,
                                                                                                                          #61,
                                                                                                                          # #69,
                                                                                                                        range(1800, 1900), #84
                                                                                                                        range(1900, 2000), #87
                                                                                                                        range(2100, 2200), #39
                                                                                                                        range(2200, 2300), #28
                                                                                                                        range(2300, 2400), #90
                                                                                                                        range(2400, 2500), #86
                                                                                                                        )])'''
objects_and_vehicles_avalanche_dataset_test = AvalancheDataset(
    [avl_test_objects_and_vehicles],
    indices=[
        i for i in chain(
            range(200, 300),    # 5
            range(2000, 2100),  # 8
            range(1400, 1500),  # 9
            range(300, 400),    # 10
            range(500, 600),    # 13
            range(0, 100),      # 16
            range(1300, 1400),  # 20
            range(1700, 1800),  # 22
            range(400, 500),    # 25
            range(2200, 2300),  # 28
            range(2100, 2200),  # 39
            range(900, 1000),   # 40
            range(600, 700),    # 41
            range(800, 900),    # 48
            range(100, 200),    # 58
            range(1100, 1200),  # 61
            range(1200, 1300),  # 69
            range(1600, 1700),  # 81
            range(1800, 1900),  # 84
            range(700, 800),    # 85
            range(2400, 2500),  # 86
            range(1900, 2000),  # 87
            range(1500, 1600),  # 89
            range(2300, 2400),  # 90
            range(1000, 1100),  # 94
        )
    ]
)
if DEBUG:
    for i in objects_and_vehicles_avalanche_dataset_test.targets:
        print("Index:",count_el, "|| for i in objects_and_vehicles_avalanche_dataset_test.targets:",i)
        count_el += 1

count_el = 0
global_model_naive_baseline_avalanche_dataset_test = AvalancheDataset([outdoors_avalanche_dataset_test ,nature_avalanche_dataset_test, people_avalanche_dataset_test ,objects_and_vehicles_avalanche_dataset_test])
if DEBUG:
    for i in global_model_naive_baseline_avalanche_dataset_test.targets:
        print("Index:",count_el, "|| for i in global_model_naive_baseline_avalanche_dataset_test.targets:",i)
        count_el += 1

count_el = 0
if DEBUG:
    for i in global_model_naive_baseline_avalanche_dataset_test.coarse_label:
        print("Index:",count_el, "|| for i in global_model_naive_baseline_avalanche_dataset_test.coarse_label:",i)
        count_el += 1

avl_test = make_avalanche_dataset(global_model_naive_baseline_avalanche_dataset_test)


if DEBUG:
    print(
        "type(objects_and_vehicles_specialised_models_train_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train",
        type(avl_train))
    print(
        "type(objects_and_vehicles_specialised_models_val_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val",
        type(avl_val))

    print("Print avl_train AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train:\n", avl_train)
    print("Print avl_val AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val:\n", avl_val)

avl_train = ClassificationDataset(
    [avl_train],
    data_attributes=[DataAttribute(global_model_naive_baseline_avalanche_dataset_train.coarse_label, "coarse_label"),
                     DataAttribute(global_model_naive_baseline_avalanche_dataset_train.targets, "targets")],
)
avl_val = ClassificationDataset(
    [avl_val],
    data_attributes=[DataAttribute(global_model_naive_baseline_avalanche_dataset_val.coarse_label, "coarse_label"),
                     DataAttribute(global_model_naive_baseline_avalanche_dataset_val.targets, "targets")],
)
avl_test = ClassificationDataset(
    [avl_test],
    data_attributes=[DataAttribute(global_model_naive_baseline_avalanche_dataset_test.coarse_label, "coarse_label"),
                     DataAttribute(global_model_naive_baseline_avalanche_dataset_test.targets, "targets")],
)


if DEBUG:
    print("Type avl_train AFTER being passed to ClassificationDataset", type(avl_train))
    print("Type avl_val AFTER being passed to ClassificationDataset", type(avl_val))

    print("Print avl_train:\n", avl_train)
    print("Print avl_val:\n", avl_val)

print(
    "Print avl_train.targets.uniques (AS .targets IS A DataAttribute AND THE ONLY WAY TO ACCESS AND PRINT ITS ACTUAL ELEMENTS/VALUES IS USING ATTRIBUTE .uniques):",
    avl_train.targets.uniques)
print(
    "Print max(avl_train.targets.uniques) in order to be aware of the value that due to the library bug will be assigned to num_classes no matter what my code does (and in this way better mitigate, follow, identify and debug the consequence of such bug):",
    max(avl_train.targets.uniques))
print(
    "Print len(avl_train.targets.uniques) to know the NUMBER OF FINE CLASSES within the specialised_models_superclasses group of which the specialised subset (e.g. global_model_naive_baseline_train_subset) that is avl_train is composed:",
    len(avl_train.targets.uniques))  # print number of classes
print(
    "IMPORTANT! Print sorted(avl_train.targets.uniques) AS THIS WILL BE THE OBJECT PASSED TO class_order IN class_incremental_benchmark TO AVOID/AS A WORK-AROUND TO THE num_classes BUG IN THE LIBRARY CODE, since by provideing a class_order, 'class_order = torch.randperm(num_classes).tolist()' within 'if class_order is None:' IS AVOIDED!",
    sorted(avl_train.targets.uniques))  # print class_order

class_order = [12, 17, 0, 47, 51, 2, 5, 8, 9, 10, 13, 16, 20, 22, 23, 33, 52, 53, 54, 11, 25, 28, 39, 40, 41, 48, 58, 61, 37, 49, 56, 57, 59, 35, 69, 81, 84, 85, 86, 87, 89, 90, 94]


'''global_model_naive_baseline_benchmark = class_incremental_benchmark({"train": avl_train, "valid": avl_val},
                                                                  class_order=class_order, num_classes_per_exp=[14, 14, 15, 52])'''

global_model_naive_baseline_benchmark = class_incremental_benchmark({"train": avl_train, "valid": avl_val, "test": avl_test},
                                                                  class_order=class_order, num_classes_per_exp=[14, 14, 15, 52])


def map_labels_expanding_classifier_benchmark(ds, class_to_idx_new=None):
    """
    Map labels using provided mapping, or build a new one.
    Allows caller to control state management.
    """
    if class_to_idx_new is None:
        class_to_idx_new = {}

    for fine_label in ds:
        fine_label_item = fine_label
        if fine_label_item not in class_to_idx_new.keys():
            class_to_idx_new[fine_label_item] = len(class_to_idx_new)

    return class_to_idx_new


class_to_idx_new = map_labels_expanding_classifier_benchmark(class_order)
print("class_to_idx_new:", class_to_idx_new)

def new_to_original_creation(new_to_original=None):
    if new_to_original is None:
        new_to_original = {}

    for original_label, new_idx in class_to_idx_new.items():
        new_to_original[new_idx] = original_label
    return new_to_original

new_to_original = new_to_original_creation()
print("new_to_original:", new_to_original)

print("sorted(avl_train.targets.uniques)", sorted(avl_train.targets.uniques))
for i, exp in enumerate(global_model_naive_baseline_benchmark.train_stream):
    print("print(i, exp.classes_in_this_experience -- global_model_naive_baseline_benchmark)", i, exp.classes_in_this_experience)

print("Print global_model_naive_baseline_benchmark:", global_model_naive_baseline_benchmark)
print('Print global_model_naive_baseline_benchmark.streams["train"]',
      global_model_naive_baseline_benchmark.streams["train"])
print('Print global_model_naive_baseline_benchmark.streams["valid"]',
      global_model_naive_baseline_benchmark.streams["valid"])

# IMPORTANT!!!!! This time, we do want to keep these as streams and NOT to further divide each stream into its composing experiences. This time,
# since class_incremental_benchmark is actually being used to create a CLScenario for use in training in a class-IL manner, and NOT as functionality
# for splitting the dataset into specialised model subsets (as we did previously in Load_Cifar100_v7.py, when class_incremental_benchmark was used
# last), we want to keep the stream objects intact without breaking them apart into its composing experience objects, as the experience objects are
# to be managed by the Avalanche training functionality/training loop to ensure the specialised model is trained in a class_IL manner with each
# experience being used for a subsequent incremental training round.
global_model_naive_baseline_train_stream_class_il = global_model_naive_baseline_benchmark.streams["train"]
global_model_naive_baseline_val_stream_class_il = global_model_naive_baseline_benchmark.streams["valid"]
global_model_naive_baseline_test_stream_class_il = global_model_naive_baseline_benchmark.streams["test"]
for i, exp in enumerate(global_model_naive_baseline_train_stream_class_il):
    print("print(i, exp.classes_in_this_experience -- global_model_naive_baseline_train_stream_class_il)", i,
          exp.classes_in_this_experience)

for i, exp in enumerate(global_model_naive_baseline_val_stream_class_il):
    print("print(i, exp.classes_in_this_experience -- global_model_naive_baseline_val_stream_class_il)", i,
          exp.classes_in_this_experience)

for i, exp in enumerate(global_model_naive_baseline_test_stream_class_il):
    print("print(i, exp.classes_in_this_experience -- global_model_naive_baseline_test_stream_class_il)", i,
          exp.classes_in_this_experience)

if DEBUG:
    for i, exp in enumerate(global_model_naive_baseline_train_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- global_model_naive_baseline_train_stream_class_il)", i,
              exp.classes_in_this_experience)

    for i, exp in enumerate(global_model_naive_baseline_val_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- global_model_naive_baseline_val_stream_class_il)", i,
              exp.classes_in_this_experience)

    print(
        "type(global_model_naive_baseline_train_stream_class_il -- OBJECT THROUGH WHICH TRAINING LOOP IS CURRENTLY ITERATING)",
        type(global_model_naive_baseline_train_stream_class_il))
    print(
        "type(global_model_naive_baseline_train_stream_class_il.exps_iter -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS)",
        type(global_model_naive_baseline_train_stream_class_il.exps_iter))
    print(
        'type(global_model_naive_baseline_benchmark.train_stream -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS',
        type(global_model_naive_baseline_benchmark.train_stream),
        'ALSOOOOOO, benchmark.train_stream KNOWN TO BE EQUAL TO benchmark.streams["train"]); CHECK TO VERIFY')
    print("type(global_model_naive_baseline_train_stream_class_il.exps_iter[0])",
          type(global_model_naive_baseline_train_stream_class_il.exps_iter[0]))




class NaiveBaseline(SupervisedTemplate):
    def __init__(
            self,
            model: nn.Module,
            lr=0.001,
            momentum=0.9,
            weight_decay=5e-4,
            lr_decay_factor: float = 0.1,
            lr_decay_every: float = 1.0,
            temperature=2.0,
            alpha=1.0,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = None,
            device: Union[str, torch.device] = "cpu",
            patience: int = 0,
            ft_shared_after: int = 5,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: Union[
                EvaluationPlugin, Callable[[], EvaluationPlugin]
            ] = default_evaluator,
            eval_every=-1,
            peval_mode="epoch"
    ):
        self.results = {}
        self.num_new_classes = 0
        self.patience = patience
        self.original_alpha = alpha
        self.original_lr = lr

        self.plugins: List[BasePlugin] = [] if plugins is None else list(plugins)
        """ List of `SupervisedPlugin`s. """
        self.adapted_dataset = None
        self.optimized_param_id = None
        self.model = model
        self.device = torch.device(device)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay_factor = lr_decay_factor  # 'Multiply lr by this much every step of decay'
        self.lr_decay_every = lr_decay_every  # 'Step decay every this many epochs'

        self.temperature = temperature
        self.alpha = alpha
        self.ft_shared_after = ft_shared_after

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.dataloader: Iterable[Any] = []
        """ Dataloader. """

        self.mbatch: Optional[TMBInput] = None
        """ Current mini-batch. """

        self.mb_output: Optional[TMBOutput] = None
        """ Model's output computed on the current mini-batch. """

        self.train_epochs: int = train_epochs
        """ Number of training epochs. """

        self.train_mb_size: int = train_mb_size
        """ Training mini-batch size. """

        self.eval_mb_size: int = train_mb_size if eval_mb_size is None else eval_mb_size
        """ Eval mini-batch size. """

        self.retain_graph: bool = False
        """ Retain graph when calling loss.backward(). """

        if evaluator is None:
            evaluator = EvaluationPlugin()
        elif callable(evaluator):
            evaluator = evaluator()

        self.plugins.append(evaluator)  # type: ignore
        self.evaluator: EvaluationPlugin = evaluator
        """ EvaluationPlugin used for logging and metric computations. """

        # Configure periodic evaluation.
        assert peval_mode in {"experience", "epoch", "iteration"}
        self.eval_every = eval_every
        peval = PeriodicEval(eval_every, peval_mode)
        self.plugins.append(peval)

        self.clock: Clock = Clock()
        """ Incremental counters for strategy events. """
        # WARNING: Clock needs to be the last plugin, otherwise
        # counters will be wrong for plugins called after it.
        self.plugins.append(self.clock)

        # Teacher from previous experience
        self.teacher = None
        self.teacher_classes = []

        # Classes learned before current experience
        self.old_classes = set()

        # Track accuracy after training each experience
        self.experience_accuracies = {}  # {exp_id: {exp_id: accuracy}}
        self.experience_losses = {}  # {exp_id: {exp_id: loss}}

        # check plugin compatibility
        self._check_plugin_compatibility()

        ###################################################################
        # State variables. These are updated during the train/eval loops. #
        ###################################################################
        self.experience: Optional[TExperienceType] = None
        """ Current experience. """

        self.is_training: bool = False
        """ True if the strategy is in training mode. """

        self.current_eval_stream: Iterable[TExperienceType] = []
        """ Current evaluation stream. """

        self._distributed_check: bool = False
        """
        Internal flag used to verify the support for distributed
        training only once.
        """

        ###################################################################
        # Other variables #
        ###################################################################
        self._eval_streams: Optional[List[List[CLExperience]]] = None

    def distillation_loss(self, student_logits, teacher_logits, scale):

        T = self.temperature

        student_log_probs = F.log_softmax(
            student_logits / T,
            dim=1,
        )

        teacher_probs = F.softmax(
            teacher_logits / T,
            dim=1,
        )

        if MATCH_AUTHOR_DISTILL:  # Match authors' implementation --> F.kl_div(...) * number_of_outputs

            print(f"  scale used in dist_loss: {scale}")
            kl_divergence = F.kl_div(student_log_probs, teacher_probs,
                                     reduction="batchmean") * scale

        if MATCH_AUTHOR_DISTILL == False:
            kl_divergence = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)

        return kl_divergence

    def train(
            self,
            experiences: Union[TExperienceType, Iterable[TExperienceType]],
            eval_streams: Optional[
                Sequence[Union[TExperienceType, Iterable[TExperienceType]]]
            ] = None,
            **kwargs,
    ):
        if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            self._check_distributed_training_compatibility()
            self._distributed_check = True

        self.is_training = True
        self._stop_training = False

        self.model.to(self.device)

        # Normalize training and eval data.
        experiences_list: Iterable[TExperienceType] = (
            _experiences_parameter_as_iterable(experiences)
        )

        if eval_streams is None:
            eval_streams = [experiences_list]

        self._eval_streams = _group_experiences_by_stream(eval_streams)

        for self.experience in experiences_list:
            self._before_training_exp(**kwargs)
            self.expand_classifier(self.model, self.experience)  # IMPORTANT: recreate optimizer after classifier expansion. This is now done from the body of expand_classifier()
            self.train_experience(self.experience, eval_streams, **kwargs)
            # After training each experience, evaluate on ALL previous experiences AND Compute and print backward transfer
            self.evaluate_all_experiences(self.experience, eval_streams)
            self._after_training_exp(**kwargs)
        self._after_training(**kwargs)
        self._train_cleanup()

    def train_experience(self, experience: CLExperience, eval_streams=None, **kwargs):
        c = 0
        loss_c = 0
        prev_epoch_loss = 100
        prev_epoch_acc = 0
        best_acc_so_far = 0
        accuracies = []

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]
        for _ in range(self.train_epochs):

            self.is_training = True
            self.model.train()
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            print(
                f"\n\n***** MY train_experience() IS EXECUTING FOR Experience {experience.current_experience} *****\n\n")

            print(
                f"\n***** EXECUTING Epoch {_} OF Experience {experience.current_experience} --- from inside train_experience() *****\n")
            total_ce = 0.0
            total_dist = 0.0
            total_loss = 0.0
            n = 0
            mini_batch_loss = 0.0
            mini_batch_acc = 0.0
            num_batches = 0

            for self.mbatch in self.dataloader:
                self.is_training = True
                self.model.train()

                self.unpack_minibatch()

                targets = self.map_targets()

                self._before_training_iteration(**kwargs)


                self.optimizer.zero_grad()

                # ------------------------------------------------
                # Current model
                # ------------------------------------------------
                self._before_forward(**kwargs)
                self.mb_output = self.model(self.mb_x)
                self._after_forward(**kwargs)

                # ------------------------------------------------
                # Ordinary CE loss for current experience
                # ------------------------------------------------

                ce_loss = self.criterion(
                    self.mb_output,
                    targets,
                )

                loss = ce_loss

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                loss.backward()
                if VERBOSE:
                    print(f"  Backpropagated loss: {loss.item():.6f}\n Backpropagation is occurring!")

                self.optimizer.step()

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                total_ce += ce_loss.item()
                total_loss += loss.item()

                n += 1

                # Track metrics
                mini_batch_loss += loss.item()
                with torch.no_grad():
                    acc = (self.mb_output.argmax(1) == targets).float().mean()
                    mini_batch_acc += acc.item()
                num_batches += 1

                self._after_training_iteration(**kwargs)

            if VERBOSE:
                print(
                    f"Epoch {_}/{self.train_epochs} | "
                    f"CE: {total_ce / n:.6f} | "
                )

            # Log epoch results
            avg_loss = mini_batch_loss / num_batches
            avg_acc = mini_batch_acc / num_batches
            avg_acc = avg_acc * 100

            print(f"\n{'...' * 60}")
            print(
                f"ALTERNATIVE CODE (in testing) --> EVALUATING Epoch {_} OF Experience {experience.current_experience}")
            print(f"Epoch {_:2d}/{self.train_epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
            print(f"{'...' * 60}")


            epoch_loss = avg_loss
            epoch_acc = avg_acc
            accuracies.append(epoch_acc)
            accuracies.sort(reverse=True)
            print(f"accuracies: {accuracies}")
            self._after_training_epoch(
                **kwargs)
            if EARLY_STOPPING_LOOP_1:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and prev_epoch_acc < 75:
                    # Both metrics improved - reset patience counter
                    c = 0
                    loss_c = 0
                elif epoch_acc <= prev_epoch_acc:
                    # No improvement - increment patience counter
                    c += 1
                    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

                    if c >= self.patience:
                        print(
                            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                        self.lr = self.original_lr
                        c = 0
                        loss_c = 0
                        break
                elif epoch_acc >= 75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_2:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and prev_epoch_acc < 75:
                    # Both metrics improved - reset patience counter
                    c = 0
                    loss_c = 0
                elif epoch_acc <= prev_epoch_acc:
                    # No improvement - increment patience counter
                    c += 1
                    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

                    if c >= self.patience:
                        print(
                            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                        self.lr = self.original_lr
                        c = 0
                        loss_c = 0
                        break
                elif epoch_acc >= 75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_3:
                if epoch_acc > accuracies[0]:
                    best_acc_so_far = epoch_acc
                    print(f"New best accuracy: {best_acc_so_far:.2f}%")
                    checkpoint_dir = Path(
                        "./checkpoints")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': _,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.loss,
                        'experience': experience.current_experience,
                    },
                        f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl")
                    print("Best checkpoint saved.")
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and epoch_acc < 75:
                    # Both metrics improved - reset patience counter
                    c = 0
                    loss_c = 0
                elif epoch_acc <= prev_epoch_acc and epoch_acc < 75:
                    # No improvement - increment patience counter
                    c += 1
                    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

                    if c >= self.patience:
                        print(
                            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                        self.lr = self.original_lr
                        c = 0
                        loss_c = 0
                        break
                elif epoch_acc >= 75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_4:
                if epoch_acc >= accuracies[0]:
                    best_acc_so_far = epoch_acc
                    print(f"New best accuracy: {best_acc_so_far:.2f}%")
                    saved_model = copy.deepcopy(self.model)
                    print("Best checkpoint saved.")
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0
                if epoch_acc > prev_epoch_acc and prev_epoch_acc < 75:
                    # Both metrics improved - reset patience counter
                    c = 0
                    loss_c = 0
                elif epoch_acc <= prev_epoch_acc:
                    # No improvement - increment patience counter
                    c += 1
                    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

                    if c >= self.patience:
                        print(
                            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                        self.lr = self.original_lr
                        c = 0
                        loss_c = 0
                        break
                elif epoch_acc >= 75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_FORMERLY_5_BUT_CHANGED_TO_CONDITIONS_THAT_RESULT_IN_WORSE_EARLY_STOPPING_LOGIC:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and epoch_acc < 75:
                    # Both metrics improved - reset patience counter
                    c = 0
                    loss_c = 0
                elif epoch_acc <= prev_epoch_acc and epoch_acc < 75:
                    # No improvement - increment patience counter
                    c += 1
                    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

                    if c >= self.patience:
                        print(
                            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                        self.lr = self.original_lr
                        c = 0
                        loss_c = 0
                        break
                elif epoch_acc >= 75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_5:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and prev_epoch_acc < acc_limit:
                    # Both metrics improved - reset patience counter
                    c = 0
                    loss_c = 0
                elif epoch_acc <= prev_epoch_acc:
                    # No improvement - increment patience counter
                    c += 1
                    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

                    if c >= self.patience:
                        print(
                            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                        self.lr = self.original_lr
                        c = 0
                        loss_c = 0
                        break
                elif epoch_acc >= acc_limit:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 85%. Training will be stopped.")
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc


        if EARLY_STOPPING_LOOP_4:
            self.model = saved_model
            print(
                f"Loaded saved_model to continue training the next Experience on it --- from inside train_experience()")

        # Add classes from this experience
        experience_classes = set(
            experience.dataset.targets.uniques
        )

        self.old_classes.update(experience_classes)

        print(
            f"Old classes = {self.old_classes}"
        )
        print(f"lr at Experience {experience.current_experience}: {self.lr}")


    def step_lr(self,epoch):
        """Handles step decay of learning rate."""
        factor = np.power(self.lr_decay_factor, np.floor((epoch - 1) / self.lr_decay_every))
        new_lr = self.lr * factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print('Set lr to ', new_lr)
        return self.optimizer

    def expand_classifier(self, model: nn.Module, experience: CLExperience):
        """Expand the classifier AND expand all previous teachers."""

        experience_classes = set(experience.dataset.targets.uniques)
        self.num_new_classes = len(experience_classes - self.old_classes)

        if self.num_new_classes <= 0:
            print(f"No new classes. Classifier remains at {model.num_classes}")
            return False

        old_num_classes = model.num_classes
        new_num_classes = old_num_classes + self.num_new_classes

        print(f"Expanding classifier from {old_num_classes} to {new_num_classes} classes")

        # Expand STUDENT classifier
        device = model.linear.weight.device
        dtype = model.linear.weight.dtype

        new_classifier = nn.Linear(
            in_features=int(model.in_features),
            out_features=int(new_num_classes),
            bias=True
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            new_classifier.weight[:old_num_classes].copy_(model.linear.weight)
            new_classifier.bias[:old_num_classes].copy_(model.linear.bias)
            init.kaiming_normal_(new_classifier.weight[old_num_classes:])
            nn.init.constant_(new_classifier.bias[old_num_classes:], 0.0)

        model.linear = new_classifier
        model.num_classes = new_num_classes

        # Recreate optimizer with expanded student
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        print(f"Classifier expanded successfully. New size: {new_num_classes}\n")
        return True

    @torch.no_grad()
    def eval(
            self,
            experiences: Union[TExperienceType, CLStream[TExperienceType]],
            **kwargs,
    ):
        if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            self._check_distributed_training_compatibility()
            self._distributed_check = True

        # eval can be called inside the train method.
        # Save the shared state here to restore before returning.
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        experiences_list: Iterable[TExperienceType] = (
            _experiences_parameter_as_iterable(experiences)
        )
        self.current_eval_stream = experiences_list

        self._before_eval(**kwargs)
        for self.experience in experiences_list:
            self._before_eval_exp(**kwargs)
            self.eval_experience()
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)
        self._eval_cleanup()

        # restore previous shared state.
        self._load_train_state(prev_train_state)

    @property
    def mb_x(self):
        """Current mini-batch input."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3, "Task label not found."
        return mbatch[-1]

    def unpack_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        mbatch = self.mbatch
        assert mbatch is not None

        if isinstance(mbatch, tuple):
            mbatch = list(mbatch)
            self.mbatch = mbatch

        for i in range(len(mbatch)):
            if isinstance(mbatch[i], dict):
                # Move all tensor values in the dictionary to the specified device
                mbatch[i] = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in mbatch[i].items()
                }
            elif isinstance(mbatch[i], torch.Tensor):
                # Directly move the tensor to the specified device
                mbatch[i] = mbatch[i].to(self.device, non_blocking=True)
            else:
                # Optionally handle other cases (e.g., raise an error if unsupported types are encountered)
                raise TypeError(f"Unsupported type in batch: {type(mbatch[i])}")

    def eval_experience(self, **kwargs):
        self.eval_epoch_mine(**kwargs)

    def eval_epoch_mine(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""

        for self.mbatch in self.dataloader:
            self.unpack_minibatch()
            targets = self.map_targets()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output = self.model(self.mb_x)
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion(self.mb_output,
                                       targets)


            self._after_eval_iteration(**kwargs)

    def evaluate_all_experiences(self, experience: CLExperience, eval_streams: Optional[
        Sequence[Union[TExperienceType, Iterable[TExperienceType]]]] = None):
        """Evaluate on all experiences seen so far."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATING ALL EXPERIENCES UP TO {experience.current_experience}")
        print(f"{'=' * 60}\n")
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        current_exp_id_all_experiences = experience.current_experience

        if current_exp_id_all_experiences not in self.experience_accuracies:
            self.experience_accuracies[current_exp_id_all_experiences] = {}
            self.experience_losses[current_exp_id_all_experiences] = {}

        if eval_streams is None:
            eval_streams = [experience]

        self.is_training = False
        self.model.eval()

        for stream_idx, eval_stream in enumerate(eval_streams):
            self.is_training = False
            self.model.eval()

            if not isinstance(eval_stream, Iterable):
                eval_stream = [eval_stream]

            for exp in eval_stream:
                self.is_training = False
                self.model.eval()

                exp_id_all_experiences = exp.current_experience

                # Don't evaluate future experiences
                if exp_id_all_experiences > current_exp_id_all_experiences:
                    break

                # Prepare eval data
                eval_adapted_dataset = exp.dataset.eval()
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset,
                                                                                                 self.eval_mb_size)

                total_loss_all_experiences = 0.0
                total_correct_all_experiences = 0
                total_samples_all_experiences = 0

                self.is_training = False
                self.model.eval()
                with torch.no_grad():
                    for self.mbatch in eval_dataloader:
                        self.is_training = False
                        self.model.eval()

                        self.unpack_minibatch()
                        targets_all_experiences = self.map_targets()

                        student_logits_all_experiences = self.model(self.mb_x)
                        loss_all_experiences = self.criterion(student_logits_all_experiences, targets_all_experiences)

                        total_loss_all_experiences += loss_all_experiences.item() * targets_all_experiences.size(0)
                        predictions_all_experiences = torch.argmax(student_logits_all_experiences, dim=1)

                        predictions_all_experiences = self.map_targets_back_to_original(predictions_all_experiences)
                        targets_all_experiences = self.map_targets_back_to_original(targets_all_experiences)

                        total_correct_all_experiences += (predictions_all_experiences == targets_all_experiences).sum().item()
                        total_samples_all_experiences += targets_all_experiences.size(0)

                avg_loss_all_experiences = total_loss_all_experiences / total_samples_all_experiences if total_samples_all_experiences > 0 else 0.0
                accuracy_all_experiences = (total_correct_all_experiences / total_samples_all_experiences * 100) if total_samples_all_experiences > 0 else 0.0

                # Store results
                self.experience_accuracies[current_exp_id_all_experiences][
                    exp_id_all_experiences] = accuracy_all_experiences
                self.experience_losses[current_exp_id_all_experiences][
                    exp_id_all_experiences] = avg_loss_all_experiences

                print(
                    f"AFTER TRAINING ON Exp {current_exp_id_all_experiences}, "
                    f"performance on Exp {exp_id_all_experiences}: Loss = {avg_loss_all_experiences:.6f} | Accuracy = {accuracy_all_experiences:.2f}%"
                )

        # Compute backward transfer AFTER all evaluations
        self._compute_backward_transfer_internal()
        self._load_train_state(prev_train_state)
        self.model.train()
        print("print(self.model.training)-- evaluate_all_experiences after self.model.train() before exiting function",
              self.model.training)

    def _compute_backward_transfer_internal(self):
        """Compute and print backward transfer after evaluation."""
        print(f"\n{'=' * 80}")
        print(f"BACKWARD TRANSFER ANALYSIS")
        print(f"{'=' * 80}\n")
        print("print(self.model.training) -- _compute_backward_transfer_internal", self.model.training)

        exp_ids = sorted(self.experience_accuracies.keys())

        # Print accuracy results
        print("Accuracy Results:")
        average_accuracy_history = []  # Stores AA after each training experience SUCH THAT IT CAN BE USED TO COMPUTE AIA
        for trained_exp_id in exp_ids:
            print(f"\nAfter training Experience {trained_exp_id}:")
            average_accuracy_list = []
            accuracy_for_average = 0
            for test_exp_id in sorted(self.experience_accuracies[trained_exp_id].keys()):
                acc = self.experience_accuracies[trained_exp_id][test_exp_id]
                print(f"  - Tested on Experience {test_exp_id}: {acc:.2f}%")
                # Average Accuracy (AA)
                average_accuracy_list.append(acc)
                accuracy_for_average += acc
                average_accuracy_list_len = len(average_accuracy_list)
            average_accuracy = (accuracy_for_average / average_accuracy_list_len)
            print(f"   Average Accuracy (ACC) after training Experience {trained_exp_id}: {average_accuracy:.4f}%")
            # Store this AA for calculating AIA
            average_accuracy_history.append(average_accuracy)

            # Average Incremental Accuracy (AIA)
            average_incremental_accuracy = (sum(average_accuracy_history) / len(average_accuracy_history))
            print("   print(average_accuracy_history)", average_accuracy_history,
                  "print(sum(average_accuracy_history))", sum(average_accuracy_history),
                  "print(average_incremental_accuracy)", average_incremental_accuracy)
            print(
                f"   Average Incremental Accuracy (AIA) after training "
                f"Experience {trained_exp_id}: "
                f"{average_incremental_accuracy:.4f}%"
            )

        print("\n" + "-" * 80 + "\n")

        # Compute backward transfer for each task
        bwt_per_task = {}
        all_bwt_values = []

        for test_exp_id in range(max(exp_ids) + 1 if exp_ids else 0):
            # Find diagonal accuracy (right after training on this task)
            diagonal_acc = None
            for trained_exp_id in exp_ids:
                if trained_exp_id == test_exp_id and test_exp_id in self.experience_accuracies[trained_exp_id]:
                    diagonal_acc = self.experience_accuracies[trained_exp_id][test_exp_id]
                    break

            if diagonal_acc is None:
                continue

            # Find final accuracy (after training all subsequent tasks)
            final_acc = None
            for trained_exp_id in reversed(exp_ids):
                if trained_exp_id >= test_exp_id and test_exp_id in self.experience_accuracies[trained_exp_id]:
                    final_acc = self.experience_accuracies[trained_exp_id][test_exp_id]
                    break

            if final_acc is not None:
                bwt = final_acc - diagonal_acc
                bwt_per_task[test_exp_id] = bwt
                all_bwt_values.append(bwt)

                print(
                    f"Experience {test_exp_id}:\n"
                    f"  Accuracy on Exp {test_exp_id} right after training on Exp {test_exp_id} = {diagonal_acc:.2f}%\n"
                    f"  Accuracy on Exp {test_exp_id} after training on Exp {trained_exp_id} = {final_acc:.2f}%\n"
                    f"  BWT = {bwt:+.2f}%\n"
                )

        if all_bwt_values:
            mean_bwt = sum(all_bwt_values) / len(all_bwt_values)
            print(f"Mean Backward Transfer (BWT): {mean_bwt:+.2f}%")
            print(f"  Positive BWT = improvement on old tasks (good)")
            print(f"  Negative BWT = forgetting on old tasks (bad)")
            print(f"  BWT ≈ 0 = stable performance (ideal)")

        print("=" * 80 + "\n")

        return bwt_per_task

    def evaluate_epoch(self, epoch, experience: CLExperience, eval_streams: Optional[
        Sequence[Union[TExperienceType, Iterable[TExperienceType]]]] = None):
        """Evaluate on all experiences seen so far."""
        print(f"\n{'...' * 60}")
        print(f"EVALUATING Epoch {epoch} OF Experience {experience.current_experience}")
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        current_exp_id_epoch = experience.current_experience

        if eval_streams is None:
            eval_streams = [experience]

        self.is_training = False
        self.model.eval()
        print("print(self.model.training) -- evaluate_epoch after self.model.eval()", self.model.training)

        for stream_idx, eval_stream in enumerate(eval_streams):
            self.is_training = False
            self.model.eval()

            if not isinstance(eval_stream, Iterable):
                eval_stream = [eval_stream]

            for exp in eval_stream:
                self.is_training = False
                self.model.eval()

                exp_id_epoch = exp.current_experience

                # Don't evaluate future experiences
                if exp_id_epoch != current_exp_id_epoch:
                    break

                # Prepare eval data
                eval_adapted_dataset = exp.dataset.eval()
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset,
                                                                                                 self.eval_mb_size)

                total_loss_epoch = 0.0
                total_correct_epoch = 0
                total_samples_epoch = 0

                self.is_training = False
                self.model.eval()
                with torch.no_grad():
                    for self.mbatch in eval_dataloader:
                        self.is_training = False
                        self.model.eval()

                        self.unpack_minibatch()
                        targets_epoch = self.map_targets()

                        student_logits_epoch = self.model(self.mb_x)
                        loss_epoch = self.criterion(student_logits_epoch, targets_epoch)

                        total_loss_epoch += loss_epoch.item() * targets_epoch.size(0)
                        predictions_epoch = torch.argmax(student_logits_epoch, dim=1)
                        total_correct_epoch += (predictions_epoch == targets_epoch).sum().item()
                        total_samples_epoch += targets_epoch.size(0)

                avg_loss_epoch = total_loss_epoch / total_samples_epoch if total_samples_epoch > 0 else 0.0
                accuracy_epoch = (total_correct_epoch / total_samples_epoch * 100) if total_samples_epoch > 0 else 0.0

                print(
                    f"Experience {current_exp_id_epoch}, "
                    f"Epoch {epoch}: Loss = {avg_loss_epoch:.6f} | Accuracy = {accuracy_epoch:.2f}%"
                    f"\n{'...' * 60}\n"
                )

        self._load_train_state(prev_train_state)
        self.model.train()
        return avg_loss_epoch, accuracy_epoch

    def evaluate_single_experience(self, experience: CLExperience, eval_streams: Optional[
        Sequence[Union[TExperienceType, Iterable[TExperienceType]]]] = None):
        """Evaluate on all experiences seen so far."""
        print(f"\n{'...' * 60}")
        print(f"EVALUATING Experience {experience.current_experience}")
        prev_train_state = self._save_train_state()
        self.is_training = False
        self.model.eval()

        current_exp_id_single_exp = experience.current_experience

        if eval_streams is None:
            eval_streams = [experience]

        self.is_training = False
        self.model.eval()
        print("print(self.model.training) -- evaluate_single_experience after self.model.eval()", self.model.training)

        for stream_idx, eval_stream in enumerate(eval_streams):
            self.is_training = False
            self.model.eval()

            if not isinstance(eval_stream, Iterable):
                eval_stream = [eval_stream]

            for exp in eval_stream:
                self.is_training = False
                self.model.eval()

                exp_id_single_exp = exp.current_experience

                # Don't evaluate future experiences
                if exp_id_single_exp != current_exp_id_single_exp:
                    break

                # Prepare eval data
                eval_adapted_dataset = exp.dataset.eval()
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset,
                                                                                                 self.eval_mb_size)

                total_loss_single_exp = 0.0
                total_correct_single_exp = 0
                total_samples_single_exp = 0

                self.is_training = False
                self.model.eval()
                with torch.no_grad():
                    for self.mbatch in eval_dataloader:
                        self.is_training = False
                        self.model.eval()

                        self.unpack_minibatch()
                        targets_single_exp = self.map_targets()

                        student_logits_single_exp = self.model(self.mb_x)
                        loss_single_exp = self.criterion(student_logits_single_exp, targets_single_exp)

                        total_loss_single_exp += loss_single_exp.item() * targets_single_exp.size(0)
                        predictions_single_exp = torch.argmax(student_logits_single_exp, dim=1)
                        total_correct_single_exp += (predictions_single_exp == targets_single_exp).sum().item()
                        total_samples_single_exp += targets_single_exp.size(0)

                avg_loss_single_exp = total_loss_single_exp / total_samples_single_exp if total_samples_single_exp > 0 else 0.0
                accuracy_single_exp = (
                            total_correct_single_exp / total_samples_single_exp * 100) if total_samples_single_exp > 0 else 0.0

                print(
                    f"Experience {current_exp_id_single_exp}, "
                    f"Loss = {avg_loss_single_exp:.6f} | Accuracy = {accuracy_single_exp:.2f}%"
                    f"\n{'...' * 60}\n"
                )
        self._load_train_state(prev_train_state)
        self.model.train()

    def make_eval_dataloader_for_evaluate_all_experiences_and_bwt(self,
                                                                  eval_adapted_dataset: Optional[Dataset],
                                                                  batch_size: int,
                                                                  num_workers=0,
                                                                  shuffle=False,
                                                                  pin_memory=None,
                                                                  persistent_workers=False,
                                                                  drop_last=False,
                                                                  **kwargs
                                                                  ):
        assert eval_adapted_dataset is not None
        eval_dataloader = DataLoader(eval_adapted_dataset,
                                     batch_size=self.eval_mb_size,
                                     num_workers=num_workers,
                                     shuffle=shuffle,
                                     pin_memory=pin_memory,
                                     persistent_workers=persistent_workers,
                                     drop_last=drop_last, )
        return eval_dataloader

    def map_targets(self):
        targets_tensor = torch.tensor([class_to_idx_new[int(fine_label.item())] for fine_label in self.mb_y],
                                      device=self.mb_y.device,
                                      dtype=torch.long
                                      )
        return targets_tensor

    def map_targets_back_to_original(self, targets_tensor):

        original_targets_tensor = torch.tensor(
            [
                new_to_original[int(element.item())] for element in targets_tensor
            ],
            device=self.mb_y.device,
            dtype=torch.long
        )

        return original_targets_tensor

    def evaluate_all_experiences_post_training(self, eval_streams: Optional[
        Sequence[Union[TExperienceType, Iterable[TExperienceType]]]] = None):
        """Evaluate on all experiences seen so far."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATING ALL EXPERIENCES")
        print(f"{'=' * 60}\n")

        self.is_training = False
        self.model.eval()
        print("print(self.model.training) -- evaluate_all_experiences_post_training after self.model.eval()",
              self.model.training)

        for stream_idx, eval_stream in enumerate(eval_streams):
            if not isinstance(eval_stream, Iterable):
                eval_stream = [eval_stream]

            for exp in eval_stream:
                exp_id_all_experiences_post_training = exp.current_experience

                # Prepare eval data
                eval_adapted_dataset = exp.dataset.eval()
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset,
                                                                                                 self.eval_mb_size)

                total_loss_all_experiences_post_training = 0.0
                total_correct_all_experiences_post_training = 0
                total_samples_all_experiences_post_training = 0

                with torch.no_grad():
                    for self.mbatch in eval_dataloader:
                        self.unpack_minibatch()
                        targets_all_experiences_post_training = self.map_targets()

                        student_logits_all_experiences_post_training = self.model(self.mb_x)
                        loss_all_experiences_post_training = self.criterion(
                            student_logits_all_experiences_post_training, targets_all_experiences_post_training)

                        total_loss_all_experiences_post_training += loss_all_experiences_post_training.item() * targets_all_experiences_post_training.size(
                            0)
                        predictions_all_experiences = torch.argmax(student_logits_all_experiences_post_training, dim=1)
                        total_correct_all_experiences_post_training += (
                                    predictions_all_experiences == targets_all_experiences_post_training).sum().item()
                        total_samples_all_experiences_post_training += targets_all_experiences_post_training.size(0)

                avg_loss_all_experiences_post_training = total_loss_all_experiences_post_training / total_samples_all_experiences_post_training if total_samples_all_experiences_post_training > 0 else 0.0
                accuracy_all_experiences_post_training = (
                            total_correct_all_experiences_post_training / total_samples_all_experiences_post_training * 100) if total_samples_all_experiences_post_training > 0 else 0.0

                print(
                    f"Performance on Exp {exp_id_all_experiences_post_training}: Loss = {avg_loss_all_experiences_post_training:.6f} | Accuracy = {accuracy_all_experiences_post_training:.2f}%"
                )


if __name__ == "__main__":

    MATCH_AUTHOR_DISTILL = True
    MATCH_AUTHOR_LOSS = True

    EARLY_STOPPING_LOOP_1 = False
    EARLY_STOPPING_LOOP_2 = False
    EARLY_STOPPING_LOOP_3 = False
    EARLY_STOPPING_LOOP_4 = False
    EARLY_STOPPING_LOOP_FORMERLY_5_BUT_CHANGED_TO_CONDITIONS_THAT_RESULT_IN_WORSE_EARLY_STOPPING_LOGIC = False
    EARLY_STOPPING_LOOP_5 = True

    acc_limit=85
    freeze_onwards=2

    VERBOSE = False

    DIFFERENCE_CE_DIST= False

    TRAINING_WITH_LOOP_LwF =True
    FREEZING_ORIGINAL=True
    FREEZING_ORIGINAL_BUT_INDEPENDENT_CONDITION_FOR_SHARED_LAYERS_AND_OLD_ROWS=False
    FREEZING_ORIGINAL_BUT_FROM_EXPERIENCE_2_ONWARDS=False
    FREEZING_ORIGINAL_BUT_FROM_EXPERIENCE_2_ONWARDS_AND_INDEPENDENT_CONDITION_FOR_SHARED_LAYERS_AND_OLD_ROWS=False
    FREEZING_SHARED_LAYERS_AFTER_EPOCH_SURPASSES_ft_shared_after_AND_INDEPENDENT_OLD_ROWS=False
    FREEZING_SHARED_LAYERS_AND_OLD_ROWS_AFTER_EPOCH_SURPASSES_ft_shared_after=False

    print(f"MATCH_AUTHOR_DISTILL = {MATCH_AUTHOR_DISTILL}, MATCH_AUTHOR_LOSS = {MATCH_AUTHOR_LOSS}, VERBOSE = {VERBOSE}")

    final_specialised_models_baseline_dir = Path("./final_specialised_models_BASELINE")
    final_specialised_models_baseline_dir.mkdir(parents=True, exist_ok=True)

    if TRAINING_WITH_LOOP_LwF:

        naive_baseline = NaiveBaseline(model=resnet20_global_model, device="cpu", lr=0.001, momentum=0.9,
                             weight_decay=5e-4, lr_decay_factor=0.1,
                             temperature=2.0, alpha=1.0,
                             train_mb_size=128, train_epochs=66, eval_mb_size=128,
                             patience=3, ft_shared_after=65,
                             evaluator=EvaluationPlugin(
                                 loss_metrics(experience=True, stream=True),
                                 loggers=InteractiveLogger(), collect_all=True),
                             eval_every=1, )
        seen_experiences = []
        seen_experiences_val = []
        seen_experiences_sizes = 0

        for experience in global_model_naive_baseline_train_stream_class_il:  # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
            i = experience.current_experience  # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of global_model_naive_baseline_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
            if i == 1:
                torch.save(naive_baseline.model.state_dict(),f"./final_specialised_models_BASELINE/Experience_0_TEST2_NAIVE_BASELINE_global_model.pt")
                new_to_original = new_to_original_creation()
                class_to_idx_new = class_to_idx_new
                # Save the mappings used in training
                metadata = {
                    'class_to_idx_new': class_to_idx_new,
                    'new_to_original': new_to_original,
                    'num_subset_classes': len(naive_baseline.old_classes),
                    'feature_dim': naive_baseline.model.linear.weight.shape[1],
                    'subset_name': 'global_model_naive_baseline_train_subset'
                }
                with open('./final_specialised_models_BASELINE/Experience_0_TEST2_NAIVE_BASELINE_global_metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
                print("✓ Saved metadata for BASELINE EVALUATION")
            if i == 2:
                torch.save(naive_baseline.model.state_dict(),f"./final_specialised_models_BASELINE/Experience_1_TEST2_NAIVE_BASELINE_global_model.pt")
                new_to_original = new_to_original_creation()
                class_to_idx_new = class_to_idx_new
                # Save the mappings used in training
                metadata = {
                    'class_to_idx_new': class_to_idx_new,
                    'new_to_original': new_to_original,
                    'num_subset_classes': len(naive_baseline.old_classes),
                    'feature_dim': naive_baseline.model.linear.weight.shape[1],
                    'subset_name': 'global_model_naive_baseline_train_subset'
                }
                with open('./final_specialised_models_BASELINE/Experience_1_TEST2_NAIVE_BASELINE_global_metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
                print("✓ Saved metadata for BASELINE EVALUATION")
            if i == 3:
                torch.save(naive_baseline.model.state_dict(),f"./final_specialised_models_BASELINE/TEST2_NAIVE_BASELINE_global_model.pt")
                new_to_original = new_to_original_creation()
                class_to_idx_new = class_to_idx_new
                # Save the mappings used in training
                metadata = {
                    'class_to_idx_new': class_to_idx_new,
                    'new_to_original': new_to_original,
                    'num_subset_classes': len(naive_baseline.old_classes),
                    'feature_dim': naive_baseline.model.linear.weight.shape[1],
                    'subset_name': 'global_model_naive_baseline_train_subset'
                }
                with open('./final_specialised_models_BASELINE/TEST2_NAIVE_BASELINE_global_metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
                print("✓ Saved metadata for BASELINE EVALUATION")
                print(f"Final value of seen_experiences_sizes = {seen_experiences_sizes}. This is the total number of samples in the training set for this model (that is, across all experiences).")
                break
            print("Start of experience -- current TRAIN experience: ", experience.current_experience)
            print("Start of experience -- current VAL experience: ",
                  global_model_naive_baseline_val_stream_class_il.exps_iter[i].current_experience)
            print("Current Classes -- current TRAIN .classes_in_this_experience: ", experience.classes_in_this_experience)
            print("Current Classes -- current VAL .classes_in_this_experience: ",
                  global_model_naive_baseline_val_stream_class_il.exps_iter[i].classes_in_this_experience)
            print("experience.dataset.coarse_label.uniques (coarse_labels of TRAIN)",
                  experience.dataset.coarse_label.uniques)
            print("experience.dataset.targets (targets of TRAIN)", experience.dataset.targets.uniques)
            print(
                "global_model_naive_baseline_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques (coarse_labels of VAL)",
                global_model_naive_baseline_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques)
            print("global_model_naive_baseline_val_stream_class_il.exps_iter[i].dataset.targets (targets of VAL)",
                  global_model_naive_baseline_val_stream_class_il.exps_iter[i].dataset.targets.uniques)
            print("print(experience.origin_stream.name)", experience.origin_stream.name)
            print("print(global_model_naive_baseline_val_stream_class_il.exps_iter[i].origin_stream.name)",
                  global_model_naive_baseline_val_stream_class_il.exps_iter[i].origin_stream.name)
            print("print(naive_baseline.plugins) BEFORE call to .train()", naive_baseline.plugins)
            seen_experiences.append(experience)
            seen_experiences_sizes += len(experience.dataset)
            print(f"len(experience.dataset) of current TRAIN experience {i} : {len(experience.dataset)}\nAfter experience {i}, seen_experiences_sizes = {seen_experiences_sizes}")
            seen_experiences_val.append(global_model_naive_baseline_val_stream_class_il[i])
            naive_baseline.train(experience, [seen_experiences,seen_experiences_val])
            print("Temperature:", naive_baseline.temperature, "Alpha:", naive_baseline.alpha)
            if EARLY_STOPPING_LOOP_5:
                print("acc_limit:", acc_limit)

        print('Training completed WITH NaiveBaseline')
        print("naive_baseline.results", naive_baseline.results)