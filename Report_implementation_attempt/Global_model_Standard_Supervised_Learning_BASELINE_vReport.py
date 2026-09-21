"""IMPORTANT -- NOTES ON LIBRARY SOURCE CODE:
class CLStream(Generic[TCLExperience]): NOTE: streams should not be used by training strategies since they provide access to past, current, and future data.
class EagerCLStream(SequenceCLStream[TCLExperience]): NOTE: streams should not be used by training strategies since they provide access to past, current, and future data.
class CLScenario(Generic[TCLStream]): NOTE: benchmarks should not be used by training strategies since they provide access to past, current, and future data.

HENCE, THE FOR LOOP FOR ITERATING OVER INDIVIDUAL EXPERIENCES!!! Note, the source code for avalanche.training.templates.base_sgd.BaseSGDTemplate def train reads
"If experiences is a single element trains on it. If it is a sequence, trains the model on each experience in order. Params: experiences – single Experience or sequence",
meaning SEQUENCE OF EXPERIENCE (i.e. DataExperience) OBJECTS; THE TYPE OF THE OBJECTS THAT MAKE UP A STREAM, MEANING MANUAL EXTRACTION OF THE EXPERIENCES FROM THE STREAM
IS REQUIRED!!!!!! ONE CANNOT PASS AN ENTIRE STREAM TO A TRAINING STRATEGY (e.g. Naive) FOR TRAINING THE MODEL!!!! THE EXPERIENCES/DatasetExperience objects CONTAINED WITHIN
THE EagerCLStream OR CLStream (two objects which are in turn contained within a CLScenario object) ARE WHAT MUST BE ITERATED THROUGH AND TRAINED ON (i.e. apply .train once per
experience/DatasetExperience object contained within a EagerCLStream or CLStream object for training a model on all classes in the stream/subset/dataset in an incremental manner
where the CL behaviour (such as mitigation of CF) is handled by the Training Strategy object (note: Training Strategy is not a class itself; each training strategy is its own class!!!
Moreover, each of these classes may be defined as part of a different module!!!! E.g. class Naive is defined in avalanche.training.supervised.strategy_wrappers, but class Culminative
is defined in avalanche.training.supervised.cumulative, despite them both being Training Strategies))

Print outdoors_specialised_models_benchmark: <avalanche.benchmarks.scenarios.generic_scenario.CLScenario object at 0x000001814CF4B490>
Print outdoors_specialised_models_benchmark.train_stream: <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000001F5F391CB10>
Print outdoors_specialised_models_benchmark.streams["train"] <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000001F5F391CB10>
    The fact that outdoors_specialised_models_benchmark.train_stream and outdoors_specialised_models_benchmark.streams["train"] are both EagerCLStream object at 0x000001F5F391CB10,
    proves both commands return the same object. That is, the EagerCLStream that is the training stream/subset comprised of DataExperience object (as opposed to a single Dataset object)
    such that training in a class-IL manner is made possible.
Print outdoors_specialised_models_benchmark.streams["test"] <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000001F5F3951990>
Print outdoors_specialised_models_benchmark.streams["valid"] <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000001F5F3951B50>

######################## REMEMBER!!!! outdoors_specialised_models_train_stream_class_il = outdoors_specialised_models_benchmark.streams["train"] ########################
type(outdoors_specialised_models_train_stream_class_il -- OBJECT THROUGH WHICH TRAINING LOOP IS CURRENTLY ITERATING) <class 'avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream'>
type(outdoors_specialised_models_train_stream_class_il.exps_iter -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS) <class 'avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream'>
type(outdoors_specialised_models_benchmark.train_stream -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS <class 'avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream'> ALSOOOOOO, benchmark.train_stream KNOWN TO BE EQUAL TO benchmark.streams["train"]); CHECK TO VERIFY
type(outdoors_specialised_models_train_stream_class_il.exps_iter[0]) <class 'avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience'>
cl_strategy <avalanche.training.supervised.strategy_wrappers.Naive object at 0x000001F5F4C40990>
type(cl_strategy) <class 'avalanche.training.supervised.strategy_wrappers.Naive'>

type(experience) WITH LOOP for experience in outdoors_specialised_models_train_stream_class_il <class 'avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience'>
    MEANING CURRENT FOR LOOP IS CORRECT!!!! as according to comment on source code for avalanche.training.templates.base_sgd.BaseSGDTemplate def train "If experiences is a single element
    trains on it. If it is a sequence, trains the model on each experience in order. Params: experiences – single Experience or sequence". IN for experience in outdoors_specialised_models_train_stream_class_il
    WE ARE TRAINING ON THE EXPERIENCE/DatasetExperience ONE OBJECT/"single element" AT A TIME, MAKING THIS A CORRECT AND VALID TRAINING LOOP!!! Note, that each of the DataExperience objects
    within the stream can be extracted/references via indexing using outdoors_specialised_models_train_stream_class_il.exps_iter[0]. This is relevant if instead of iterating through all
    DataExperience objects, I need to specifically retrieve one for inspecting it or such, but is not needed for iterating through the DataExperience objects for applying .train to each
    of them/for training -- for experience in outdoors_specialised_models_train_stream_class_il will be used for that.

    The signature for avalanche.training.templates.base_sgd.BaseSGDTemplate def train reads
    def train(self,
          experiences: TDatasetExperience | Iterable[TDatasetExperience],
          eval_streams: Sequence[TDatasetExperience | Iterable[TDatasetExperience]] | None = None,
          **kwargs: Any) -> dict[str, Any]
    , defining TDatasetExperience or an iterable for such as the type for experiences. However, as can be seen in avalanche/avalanche/training/templates/base_sgd.py and in
    avalanche/training/templates/common_templates.py, TDatasetExperience = TypeVar("TDatasetExperience", bound=DatasetExperience). TDatasetExperience is a type variable (TypeVar) used
    solely for Python's type hinting system AND THEREFORE, FOR PRACTICAL PURPOSES, IS ESSENTIALLY JUST DatasetExperience!!!"

    This is also part of avalanche/avalanche/benchmarks/scenarios/deprecated/dataset_scenario.py, but as can be seen from the directory name, it is deprecated. Regardless,
    TDatasetExperience = TypeVar("TDatasetExperience", bound=DatasetExperience) appears in both so there is no doubt that DatasetExperience object is what is meant by TDatasetExperience.


I implemented my own EarlyStoppingPlugin and named it EarlyStoppingPlugin_mine such that I can evaluate on experiences. I did this by overriding the code that prevents a single experience
from being used as the metric to track in earlyStopping. DO NOT FORGET!!!! If ever training using a training loop that iterates through experiences, instead of a .train() call that accepts
the entire train and eval stream (though, unlikely, since the latter seems to be the only one for which EWC / TestEWCStrategy_2 works) THEN REMEMBER TO SET metric_name="Top1_Acc_Stream"
WHEN YOU WANT EarlyStoppingPlugin_mine (just like I would do if I were using the original library source code) TO EVALUATE ON A SINGLE EXPERIENCE, SINCE BY STREAM IT ASSUMES THE WHOLE OF
WHATEVER IS PROVIDED .train() and if only one experience is provided, then it will manipulate this as if it were the entire stream when it comes to tracking metrics for early stopping

"""
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.evaluation.metrics.forgetting_bwt import forgetting_to_bwt, bwt_metrics
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from avalanche.models.dynamic_optimizers import update_optimizer
from avalanche.training.plugins.checkpoint import CheckpointPlugin
from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate
from avalanche.training.templates.base import TExperienceType, _experiences_parameter_as_iterable, \
    _group_experiences_by_stream

from Report_implementation_attempt.Config_vReport import set_seed

set_seed()  # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Report_implementation_attempt.Config_vReport import set_device

DEVICE = set_device()
from Report_implementation_attempt.Config_vReport import SEED
from avalanche.training.determinism.rng_manager import RNGManager

RNGManager.set_random_seeds(SEED)
from Report_implementation_attempt.ResNet20_vReport import ResNet20
from Report_implementation_attempt.ResNet20_DynamicModel_vReport import ResNet20_DynamicModel
#from Report_implementation_attempt.ResNet20_DynamicModel_vReport import ResNet20_Compact_S #ResNet20_Compact
from Report_implementation_attempt.Load_cifar100_vReport import (avl_train)
from Report_implementation_attempt.Load_cifar100_vReport import (objects_and_vehicles_specialised_models_val_subset)
from Report_implementation_attempt.Load_cifar100_vReport import (objects_and_vehicles_specialised_models_test_subset)
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.optim import SGD, Adam
from avalanche.training.supervised import Naive, EWC, LwF  # and many more!
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
    Iterable  # OOOR like in avalanche/training/templates/base_sgd.py , from typing import Any, Callable, Generic, Iterable, Sequence, Optional, TypeVar, Union
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path


####################################################################################################################################################################################
#Imports from Evaluate_BASELINE_vReport
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Dict, Tuple
from collections import OrderedDict
from Report_implementation_attempt.ResNet20_DynamicModel_vReport import ResNet20_DynamicModel
#from Report_implementation_attempt.ResNet20_DynamicModel_Compact_vReport import ResNet20_XL
############ Imports for Evaluating: ##############################################################################################################################################
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, _T_co
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_definitions import T_co

from Report_implementation_attempt.Train_ResNet20_Outdoors_LwF_vReport import outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il, outdoors_specialised_models_val_stream_class_il, outdoors_specialised_models_train_subset
from Report_implementation_attempt.Train_ResNet20_Nature_LwF_vReport import nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il, nature_specialised_models_val_stream_class_il, nature_specialised_models_train_subset
from Report_implementation_attempt.Train_ResNet20_People_LwF_vReport import people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il, people_specialised_models_val_stream_class_il, people_specialised_models_train_subset
from Report_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vReport import objects_and_vehicles_specialised_models_test_stream_class_il, objects_and_vehicles_specialised_models_train_stream_class_il, objects_and_vehicles_specialised_models_val_stream_class_il, objects_and_vehicles_specialised_models_train_subset
from Report_implementation_attempt.Load_cifar100_vReport import avl_test
#TODO: TEMPORARY!!!! Change to that of vReport once I have final Objects_and_Vehicles model!!!!!
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F
import tracemalloc
import torch
####################################################################################################################################################################################


DEBUG = True #TODO: Remember to change DEBUG to False if running experiments.
TRAINING_WITH_LOOP = False  # TODO: Remember to change TRAINING_WITH_LOOP to True if 'Training WITH LOOP VERSION 2'.
TRAINING_WITHOUT_LOOP = False  # True #TODO: Remember to change TRAINING_WITHOUT_LOOP to True if 'Training WITHOUT LOOP'.
# TRAINING_WITH_LOOP_LwF = False#True
# TRAINING_WITH_RUN_SEVERAL_EXPERIMENTS_LwF = True
DYNAMIC_MODELS = True
print(
    "\n\n\n################################ HERE BEGINS Global_model_Standard_Supervised_Learning_BASELINE_vReport.py ################################ \n\n\n")

if DYNAMIC_MODELS == False:
    resnet20_global_model = ResNet20(num_classes=100)
    outdoors_specialised_model = ResNet20(
        num_classes=100)  # No need for num_classes=10 anymore despite 10 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    nature_specialised_model = ResNet20(
        num_classes=100)  # No need for num_classes=15 anymore despite 15 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    people_specialised_model = ResNet20(
        num_classes=100)  # No need for num_classes=5 anymore despite 5 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    animals_specialised_model = ResNet20(
        num_classes=100)  # No need for num_classes=45 anymore despite 45 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    objects_and_vehicles_specialised_model = ResNet20(
        num_classes=100)  # No need for num_classes=25 anymore despite 25 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used

if DYNAMIC_MODELS == True:
    resnet20_global_model = ResNet20_DynamicModel(num_classes=0)#resnet20_global_model = ResNet20_DynamicModel(num_classes=100)
    outdoors_specialised_model = ResNet20_DynamicModel(
        num_classes=0)  # No need for num_classes=10 anymore despite 10 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    nature_specialised_model = ResNet20_DynamicModel(
        num_classes=0)  # No need for num_classes=15 anymore despite 15 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    people_specialised_model = ResNet20_DynamicModel(
        num_classes=0)  # No need for num_classes=5 anymore despite 5 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    animals_specialised_model = ResNet20_DynamicModel(
        num_classes=0)  # No need for num_classes=45 anymore despite 45 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    # Start tracking
    objects_and_vehicles_specialised_model = ResNet20_DynamicModel(
        num_classes=0)  # No need for num_classes=25 anymore despite 25 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used


# TRAIN DATASET all specialised train subsets together
combined_dataset_outdoors_train = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il[0].dataset,
    outdoors_specialised_models_train_stream_class_il[1].dataset,
    outdoors_specialised_models_train_stream_class_il[2].dataset,
    #outdoors_specialised_models_train_stream_class_il[3].dataset,
    #outdoors_specialised_models_train_stream_class_il[4].dataset,
])

combined_dataset_nature_train = ConcatDataset([
    nature_specialised_models_train_stream_class_il[0].dataset,
    nature_specialised_models_train_stream_class_il[1].dataset,
    nature_specialised_models_train_stream_class_il[2].dataset,
    #nature_specialised_models_train_stream_class_il[3].dataset,
    #nature_specialised_models_train_stream_class_il[4].dataset,
])

combined_dataset_people_train = ConcatDataset([
    people_specialised_models_train_stream_class_il[0].dataset,
    people_specialised_models_train_stream_class_il[1].dataset,
    people_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_train = ConcatDataset([
    objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_train = ConcatDataset([
    combined_dataset_outdoors_train,
    combined_dataset_nature_train,
    combined_dataset_people_train,
    combined_dataset_objects_and_vehicles_train
])
train_loader = DataLoader(
    combined_dataset_train,
    batch_size=128,
    shuffle=True
)
print("print train_loader", train_loader)
print("print test_loader.dataset", train_loader.dataset)


# VAL DATASET all specialised val subsets together
combined_dataset_outdoors_val = ConcatDataset([
    outdoors_specialised_models_val_stream_class_il[0].dataset,
    outdoors_specialised_models_val_stream_class_il[1].dataset,
    outdoors_specialised_models_val_stream_class_il[2].dataset,
    #outdoors_specialised_models_val_stream_class_il[3].dataset,
    #outdoors_specialised_models_val_stream_class_il[4].dataset,
])

combined_dataset_nature_val = ConcatDataset([
    nature_specialised_models_val_stream_class_il[0].dataset,
    nature_specialised_models_val_stream_class_il[1].dataset,
    nature_specialised_models_val_stream_class_il[2].dataset,
    #nature_specialised_models_val_stream_class_il[3].dataset,
    #nature_specialised_models_val_stream_class_il[4].dataset,
])

combined_dataset_people_val = ConcatDataset([
    people_specialised_models_val_stream_class_il[0].dataset,
    people_specialised_models_val_stream_class_il[1].dataset,
    people_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_val = ConcatDataset([
    objects_and_vehicles_specialised_models_val_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_val_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_val = ConcatDataset([
    combined_dataset_outdoors_val,
    combined_dataset_nature_val,
    combined_dataset_people_val,
    combined_dataset_objects_and_vehicles_val
])
val_loader = DataLoader(
    combined_dataset_val,
    batch_size=32,
    shuffle=False
)
print("print val_loader", val_loader)
print("print val_loader.dataset", val_loader.dataset)

combined_dataset_outdoors_test = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
    outdoors_specialised_models_test_stream_class_il[1].dataset,
    outdoors_specialised_models_test_stream_class_il[2].dataset,
    #outdoors_specialised_models_test_stream_class_il[3].dataset,
    #outdoors_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_nature_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
    nature_specialised_models_test_stream_class_il[1].dataset,
    nature_specialised_models_test_stream_class_il[2].dataset,
    #nature_specialised_models_test_stream_class_il[3].dataset,
    #nature_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_people_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[0].dataset,
    people_specialised_models_test_stream_class_il[1].dataset,
    people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_test = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_test = ConcatDataset([
    combined_dataset_outdoors_test,
    combined_dataset_nature_test,
    combined_dataset_people_test,
    combined_dataset_objects_and_vehicles_test
])


'''for i in range(5):
    print(f"outdoors_specialised_models_train_stream_class_il[{i}].classes_in_this_experience", outdoors_specialised_models_train_stream_class_il[i].classes_in_this_experience)
    print(f"nature_specialised_models_train_stream_class_il[{i}].classes_in_this_experience",nature_specialised_models_train_stream_class_il[i].classes_in_this_experience)
    print(f"people_specialised_models_train_stream_class_il[{i}].classes_in_this_experience",people_specialised_models_train_stream_class_il[i].classes_in_this_experience)
    print(f"objects_and_vehicles_specialised_models_train_stream_class_il[{i}].classes_in_this_experience",objects_and_vehicles_specialised_models_train_stream_class_il[i].classes_in_this_experience)
'''

test_loader = DataLoader(
    combined_dataset_test,
    batch_size=32,
    shuffle=False
)
print("print test_loader", test_loader)
print("print test_loader.dataset", test_loader.dataset)


def map_targets(batch_labels_tensor):
    targets_tensor = torch.tensor([class_to_idx_new[int(fine_label.item())] for fine_label in batch_labels_tensor],
                                  device=batch_labels_tensor.device,  # fine_label.device,
                                  dtype=torch.long
                                  )
    # print("print(class_to_idx_new)\n", class_to_idx_new)
    ##print("print(targets_tensor)\n",targets_tensor)
    return targets_tensor

# class_to_index={}
def map_labels_expanding_classifier_benchmark(ds_list, class_to_idx_new=None):
    original_to_aggregated_dict = {17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5,  # OUTDOORS
                                   51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14,  # NATURE
                                   2: 15, 11: 16, 35: 17,  # PEOPLE
                                   16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27, 39: 28,
                                   28: 29, 25: 30, 22: 31, 20: 32, 94: 33,  # OBJECTS_AND_VEHICLES
                                   90: 34, 89: 35, 87: 36, 86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42}
    """
    Map labels using provided mapping, or build a new one.
    Allows caller to control state management.
    """
    if class_to_idx_new is None:
        class_to_idx_new = {}

    for ds in ds_list:
        for fine_label in sorted(ds.targets):
            fine_label_item = fine_label
            if fine_label_item not in class_to_idx_new.keys() and fine_label_item in original_to_aggregated_dict.keys():
                class_to_idx_new[fine_label_item] = len(class_to_idx_new)

    return class_to_idx_new


class_to_idx_new = map_labels_expanding_classifier_benchmark([outdoors_specialised_models_train_subset.dataset, nature_specialised_models_train_subset.dataset, people_specialised_models_train_subset.dataset, objects_and_vehicles_specialised_models_train_subset.dataset])
print("class_to_idx_new:", class_to_idx_new)

def new_to_original_creation(new_to_original=None):
    if new_to_original is None:
        new_to_original = {}

    for original_label, new_idx in class_to_idx_new.items():
        new_to_original[new_idx] = original_label #new_to_original[original_label] = new_idx #TEST LATER: new_to_original[new_idx] = original_label #
    return new_to_original

new_to_original = new_to_original_creation()
print("new_to_original:", new_to_original)


def train(model, train_loader, val_loader, num_epochs=200, lr=0.1,weight_decay=1e-4, patience=3, lr_decay_factor=0.1, device='cpu'):#(model, train_loader, optimizer, criterion, device='cpu'):
    """Main training loop"""

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)

    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150],gamma=0.1)

    model.train()


    total_loss = 0
    c = 0
    loss_c = 0
    prev_epoch_loss = 100
    prev_epoch_acc = 0
    best_acc_so_far = 0
    accuracies = []
    original_lr = lr
    acc_limit=85
    best_val_acc = 0


    # Training loop
    for epoch in range(num_epochs):
        print(f'\n=== Epoch [{epoch + 1}/{num_epochs}] ===')
        batch_idx = 0
        total_ce = 0.0
        total_dist = 0.0
        total_loss = 0.0
        n = 0
        mini_batch_loss = 0.0  # epoch_loss = 0.0
        mini_batch_acc = 0.0  # epoch_acc = 0.0
        num_batches = 0
        correct = 0
        total = 0

        for images, labels, batch_task_ids in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            labels = map_targets(labels)

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100 * correct / total:.2f}%')

            # Track metrics
            mini_batch_loss += loss.item()  # epoch_loss += loss.item()
            with torch.no_grad():
                acc = (output.argmax(1) == labels).float().mean()
                mini_batch_acc += acc.item()  # epoch_acc += acc.item()
            num_batches += 1
            
        # Log epoch results
        avg_loss = mini_batch_loss / num_batches  # avg_loss = epoch_loss / num_batches
        avg_acc = mini_batch_acc / num_batches  # avg_acc = epoch_acc / num_batches
        avg_acc = avg_acc * 100

        print(f"\n{'...' * 60}")
        print(
            f"ALTERNATIVE CODE (in testing) --> EVALUATING Epoch {epoch} OF epoch {epoch}")
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              # f"Loss: {avg_loss:.6f} | Acc: {avg_acc:.2f} %")#f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
              f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        print(f"{'...' * 60}")

        epoch_loss = avg_loss
        epoch_acc = avg_acc
        # epoch_loss, epoch_acc = self.evaluate_epoch(_, experience) # TODO: Temporarily commented out to see if alternative calculation of epoch_loss, epoch_acc leads to better results!
        accuracies.append(epoch_acc)
        accuracies.sort(reverse=True)
        print(f"accuracies: {accuracies}")

        if epoch_loss >= prev_epoch_loss:
            loss_c += 1
            if loss_c == patience - 1:  # and epoch_acc <= prev_epoch_acc: # and c == patience-1:
                print(
                    f"lr before being increased at epoch {epoch}, Epoch {epoch}: {lr}")
                lr = lr * lr_decay_factor
                print(
                    f"lr after being increased at epoch {epoch}, Epoch {epoch}: {lr}")
                c = 0

        if epoch_acc > prev_epoch_acc and prev_epoch_acc < acc_limit:  # if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= patience:
            # Both metrics improved - reset patience counter
            # prev_epoch_loss = epoch_loss
            # prev_epoch_acc = epoch_acc
            c = 0
            loss_c = 0
        elif epoch_acc <= prev_epoch_acc:
            # No improvement - increment patience counter
            c += 1
            print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

            if c >= patience:
                print(
                    f"WARNING: The number of epochs equivalent to patience ({patience}) has been reached without improvement. Training will be stopped.")
                lr = original_lr
                c = 0
                loss_c = 0
                break
        elif epoch_acc >= acc_limit:  # elif prev_epoch_acc>=75:
            print(
                f"WARNING: The accuracy of epoch {epoch} has reached 85%. Training will be stopped.")
            # c = patience
            lr = original_lr
            c = 0
            loss_c = 0
            break

        prev_epoch_loss = epoch_loss
        prev_epoch_acc = epoch_acc
        

        train_loss = total_loss / len(train_loader)#avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total #accuracy = 100 * correct / total

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)

    # Learning rate scheduling
    # scheduler.step()

    # elapsed = time.time() - start_time

    # Print results
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    # print(f'Time: {elapsed:.2f}s')

    return model #return avg_loss, accuracy


def validate(model, val_loader, criterion, device='cpu'):
    """Validate on validation set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, batch_task_ids in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            labels = map_targets(labels)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy



'''
def train(model, train_loader, val_loader, num_epochs=200, learning_rate=0.1,
          weight_decay=1e-4, patience=3, lr_decay_factor=0.1):
    """Main training loop"""

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150],gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'\n=== Epoch [{epoch + 1}/{num_epochs}] ===')
        #start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)

        # Learning rate scheduling
        #scheduler.step()

        #elapsed = time.time() - start_time

        # Print results
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        #print(f'Time: {elapsed:.2f}s')

    return trained_model'''


# ============ USAGE EXAMPLE ============

if __name__ == '__main__':
    # Initialize model
    model = ResNet20(num_classes=43)


    # Assuming you have DataLoaders set up
    #train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Start training
    train( #train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,
        lr=000.1,
        weight_decay=1e-4,
    )
    final_specialised_models_baseline_dir = Path("./final_specialised_models_BASELINE")
    final_specialised_models_baseline_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),f"./final_specialised_models_BASELINE/BASELINE_standard_supervised_global_model.pt")
    new_to_original = new_to_original_creation()  # Your existing function
    class_to_idx_new = class_to_idx_new  # Your existing variable
    # Save the mappings used in training
    metadata = {
        'class_to_idx_new': class_to_idx_new,  # YOUR EXISTING VARIABLE
        'new_to_original': new_to_original,  # YOUR EXISTING VARIABLE
        'num_subset_classes':43
    }
    with open('./final_specialised_models_BASELINE/BASELINE_standard_supervised_global_metadata.pkl',
              'wb') as f:
        pickle.dump(metadata, f)
    print("✓ Saved metadata for BASELINE EVALUATION")