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
from Report_implementation_attempt.Load_cifar100_vReport import (outdoors_specialised_models_train_subset)
from Report_implementation_attempt.Load_cifar100_vReport import (outdoors_specialised_models_val_subset)
from Report_implementation_attempt.Load_cifar100_vReport import (outdoors_specialised_models_test_subset)
from Report_implementation_attempt.Load_cifar100_vReport import (nature_specialised_models_train_subset)
from Report_implementation_attempt.Load_cifar100_vReport import (nature_specialised_models_val_subset)
from Report_implementation_attempt.Load_cifar100_vReport import (nature_specialised_models_test_subset)
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

DEBUG = False  # True#False #TODO: Remember to change DEBUG to False if running experiments.
TRAINING_WITH_LOOP = False  # TODO: Remember to change TRAINING_WITH_LOOP to True if 'Training WITH LOOP VERSION 2'.
TRAINING_WITHOUT_LOOP = False  # True #TODO: Remember to change TRAINING_WITHOUT_LOOP to True if 'Training WITHOUT LOOP'.
# TRAINING_WITH_LOOP_LwF = False#True
# TRAINING_WITH_RUN_SEVERAL_EXPERIMENTS_LwF = True
DYNAMIC_MODELS = True
print(
    "\n\n\n################################ HERE BEGINS Train_ResNet20_Nature_LwF_vReport.py ################################ \n\n\n")

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
    resnet20_global_model = ResNet20_DynamicModel(num_classes=100)
    outdoors_specialised_model = ResNet20_DynamicModel(
        num_classes=0)  # No need for num_classes=10 anymore despite 10 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    nature_specialised_model = ResNet20_DynamicModel(
        num_classes=0)  # No need for num_classes=15 anymore despite 15 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    people_specialised_model = ResNet20_DynamicModel(
        num_classes=5)  # No need for num_classes=5 anymore despite 5 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    animals_specialised_model = ResNet20_DynamicModel(
        num_classes=45)  # No need for num_classes=45 anymore despite 45 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    objects_and_vehicles_specialised_model = ResNet20_DynamicModel(
        num_classes=25)  # No need for num_classes=25 anymore despite 25 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used

avl_train = make_avalanche_dataset(nature_specialised_models_train_subset.dataset)
avl_val = make_avalanche_dataset(nature_specialised_models_val_subset.dataset)
avl_test = make_avalanche_dataset(nature_specialised_models_test_subset.dataset)


# class_to_index={}
def map_labels_expanding_classifier_benchmark(ds, class_to_idx_new=None):
    """
    Map labels using provided mapping, or build a new one.
    Allows caller to control state management.
    """
    if class_to_idx_new is None:
        class_to_idx_new = {}

    for fine_label in sorted(ds.targets):
        fine_label_item = fine_label
        if fine_label_item not in class_to_idx_new.keys():
            class_to_idx_new[fine_label_item] = len(class_to_idx_new)

    return class_to_idx_new


class_to_idx_new = map_labels_expanding_classifier_benchmark(nature_specialised_models_train_subset.dataset)
print("class_to_idx_new:", class_to_idx_new)

def new_to_original_creation(new_to_original=None):
    if new_to_original is None:
        new_to_original = {}

    for original_label, new_idx in class_to_idx_new.items():
        new_to_original[new_idx] = original_label #new_to_original[original_label] = new_idx #TEST LATER: new_to_original[new_idx] = original_label #
    return new_to_original

new_to_original = new_to_original_creation()
print("new_to_original:", new_to_original)

if DEBUG:
    print(
        "type(nature_specialised_models_train_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train",
        type(avl_train))
    print(
        "type(nature_specialised_models_val_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val",
        type(avl_val))
    print(
        "type(nature_specialised_models_test_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test",
        type(avl_test))

    print("Print avl_train AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train:\n", avl_train)
    print("Print avl_val AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val:\n", avl_val)
    print("Print avl_test AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test:\n", avl_test)

avl_train = ClassificationDataset(
    [avl_train],  # [nature_specialised_models_train_subset.dataset],
    data_attributes=[DataAttribute(nature_specialised_models_train_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(nature_specialised_models_train_subset.dataset.targets, "targets")],
    # "fine_label")],
)
avl_val = ClassificationDataset(
    [avl_val],  # [nature_specialised_models_val_subset.dataset],
    data_attributes=[DataAttribute(nature_specialised_models_val_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(nature_specialised_models_val_subset.dataset.targets, "targets")],  # "fine_label")],
)
avl_test = ClassificationDataset(
    [avl_test],  # [nature_specialised_models_test_subset.dataset],
    data_attributes=[DataAttribute(nature_specialised_models_test_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(nature_specialised_models_test_subset.dataset.targets, "targets")],
    # "fine_label")],
)

if DEBUG:
    print("Type avl_train AFTER being passed to ClassificationDataset", type(avl_train))
    print("Type avl_val AFTER being passed to ClassificationDataset", type(avl_val))
    print("Type avl_test AFTER being passed to ClassificationDataset", type(avl_test))

    print("Print avl_train:\n", avl_train)
    print("Print avl_val:\n", avl_val)
    print("Print avl_test:\n", avl_test)

print(
    "Print avl_train.targets.uniques (AS .targets IS A DataAttribute AND THE ONLY WAY TO ACCESS AND PRINT ITS ACTUAL ELEMENTS/VALUES IS USING ATTRIBUTE .uniques):",
    avl_train.targets.uniques)
# print("Print avl_train.targets", avl_train.targets) #TO PROVE THE ABOVE, AS WHEN ATTRIBUTE .uniques OF DataAttribute IS NOT USED, THE ACTUAL TARGETS/VALUES CONTAINED WITHIN avl_train.targets ARE NOT ACCESSED. Instead, FlatData (len=4000,subset=True,cat=False,cf=True) DataAttribute (len=4000) is returned)
print(
    "Print max(avl_train.targets.uniques) in order to be aware of the value that due to the library bug will be assigned to num_classes no matter what my code does (and in this way better mitigate, follow, identify and debug the consequence of such bug):",
    max(avl_train.targets.uniques))
print(
    "Print len(avl_train.targets.uniques) to know the NUMBER OF FINE CLASSES within the specialised_models_superclasses group of which the specialised subset (e.g. nature_specialised_models_train_subset) that is avl_train is composed:",
    len(avl_train.targets.uniques))  # print number of classes
print(
    "IMPORTANT! Print sorted(avl_train.targets.uniques) AS THIS WILL BE THE OBJECT PASSED TO class_order IN class_incremental_benchmark TO AVOID/AS A WORK-AROUND TO THE num_classes BUG IN THE LIBRARY CODE, since by provideing a class_order, 'class_order = torch.randperm(num_classes).tolist()' within 'if class_order is None:' IS AVOIDED!",
    sorted(avl_train.targets.uniques))  # print class_order

# Original and best performing benchmark
'''nature_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[3, 3, 3, 3, 85])'''  # which is equivalent to num_experiences=5 due to the reasons explained in previous version)

# Same as original above, but nature_specialised_models_val_stream_class_il[4] now contains the 3 last classes of the specialised model, in case I even wanna use
# nature_specialised_models_val_stream_class_il[4] for the global model
'''nature_specialised_models_benchmark = class_incremental_benchmark(
    {"train": avl_train, "test": avl_test, "valid": avl_val},
    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[3, 3, 3, 3, 3, 82]) # REAL, DEFINITE one used in real nature model, commented-out so that baseline one can be used instead
'''

# one used for NAIVE_BASELINE_BENCHMARK
nature_specialised_models_benchmark = class_incremental_benchmark(
    {"train": avl_train, "test": avl_test, "valid": avl_val},
    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[9, 3, 3, 82])


if DEBUG:
    print("sorted(avl_train.targets.uniques)", sorted(avl_train.targets.uniques))
    for i, exp in enumerate(nature_specialised_models_benchmark.train_stream):
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_benchmark)", i,
              exp.classes_in_this_experience)

    print("Print nature_specialised_models_benchmark:", nature_specialised_models_benchmark)
    # print("Print nature_specialised_models_benchmark.train_stream:", nature_specialised_models_benchmark.train_stream) # it works, but I am not sure that it is not deprecated. I think the current or at least most frequently used way of retrieving a stream is .streams["train"], specially if there is a .streams["valid"] since attribute .val_stream does NOT exist. print(nature_specialised_models_benchmark.train_stream) and print(nature_specialised_models_benchmark.streams["train"]) both return <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000002242B559510>, meaning they return the exact same object so, there is no risk involved in having chosen to use .streams["train"] over .train_stream
    print('Print nature_specialised_models_benchmark.streams["train"]',
          nature_specialised_models_benchmark.streams["train"])
    print('Print nature_specialised_models_benchmark.streams["test"]',
          nature_specialised_models_benchmark.streams["test"])
    print('Print nature_specialised_models_benchmark.streams["valid"]',
          nature_specialised_models_benchmark.streams["valid"])

# IMPORTANT!!!!! This time, we do want to keep these as streams and NOT to further divide each stream into its composing experiences. This time,
# since class_incremental_benchmark is actually being used to create a CLScenario for use in training in a class-IL manner, and NOT as functionality
# for splitting the dataset into specialised model subsets (as we did previously in Load_Cifar100_v7.py, when class_incremental_benchmark was used
# last), we want to keep the stream objects intact without breaking them apart into its composing experience objects, as the experience objects are
# to be managed by the Avalanche training functionality/training loop to ensure the specialised model is trained in a class_IL manner with each
# experience being used for a subsequent incremental training round.
nature_specialised_models_train_stream_class_il = nature_specialised_models_benchmark.streams["train"]
nature_specialised_models_val_stream_class_il = nature_specialised_models_benchmark.streams["valid"]
nature_specialised_models_test_stream_class_il = nature_specialised_models_benchmark.streams["test"]

if DEBUG:
    for i, exp in enumerate(nature_specialised_models_train_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_train_stream_class_il)", i,
              exp.classes_in_this_experience)

    for i, exp in enumerate(nature_specialised_models_val_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_val_stream_class_il)", i,
              exp.classes_in_this_experience)

    for i, exp in enumerate(nature_specialised_models_test_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_test_stream_class_il)", i,
              exp.classes_in_this_experience)

    print(
        "type(nature_specialised_models_train_stream_class_il -- OBJECT THROUGH WHICH TRAINING LOOP IS CURRENTLY ITERATING)",
        type(nature_specialised_models_train_stream_class_il))
    print(
        "type(nature_specialised_models_train_stream_class_il.exps_iter -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS)",
        type(nature_specialised_models_train_stream_class_il.exps_iter))
    print(
        'type(nature_specialised_models_benchmark.train_stream -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS',
        type(nature_specialised_models_benchmark.train_stream),
        'ALSOOOOOO, benchmark.train_stream KNOWN TO BE EQUAL TO benchmark.streams["train"]); CHECK TO VERIFY')
    print("type(nature_specialised_models_train_stream_class_il.exps_iter[0])",
          type(nature_specialised_models_train_stream_class_il.exps_iter[0]))


#####################################################################################################################################################################################################################

class EarlyStoppingPlugin_mine(SupervisedPlugin):
    def __init__(
            self,
            patience: int,
            val_stream_name: str,
            metric_name: str = "Top1_Acc_Stream",
            mode: str = "max",
            peval_mode: str = "epoch",
            margin: float = 0.0,
            verbose=False,
    ):
        """Init.

        :param patience: Number of epochs to wait before stopping the training.
        :param val_stream_name: Name of the validation stream to search in the
            metrics. The corresponding stream will be used to keep track of the
            evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
            reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
            given metric should me maximized (resp. minimized).
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            early stopping should happen after `patience`
            epochs or iterations (Default='epoch').
        :param margin: a minimal margin of improvements required to be
            considered best than a previous one. It should be an float, the
            default value is 0. That means that any improvement is considered
            better.
        :param verbose: If True, prints a message for each update
            (default: False).
        """
        super().__init__()
        self.val_stream_name = val_stream_name
        self.patience = patience
        self.verbose = verbose

        assert peval_mode in {"epoch", "iteration"}
        self.peval_mode = peval_mode

        assert type(margin) == float
        self.margin = margin

        self.metric_name = metric_name
        self.metric_key = f"{self.metric_name}/eval_phase/" f"{self.val_stream_name}"

        if mode not in ("max", "min"):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == "max" else operator.lt

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step: Optional[int] = None

    def before_training_exp(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None

    def before_training_iteration(self, strategy, **kwargs):
        if self.peval_mode == "iteration":
            ub = self._update_best(strategy)
            if ub is None or self.best_step is None:
                return
            curr_step = self._get_strategy_counter(strategy)
            if curr_step - self.best_step >= self.patience:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()

    def before_training_epoch(self, strategy, **kwargs):
        if self.peval_mode == "epoch":
            ub = self._update_best(strategy)
            if ub is None or self.best_step is None:
                return
            curr_step = self._get_strategy_counter(strategy)
            if curr_step - self.best_step >= self.patience:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        if self.metric_name == "Top1_Acc_Exp" or self.metric_name == "Loss_Exp" or "ExperienceBWT" or "ExperienceForgetting":
            self.metric_key = f"{self.metric_name}/eval_phase/" f"{self.val_stream_name}/Exp00{strategy.experience.current_experience}"
        names = [k for k in res.keys() if k.startswith(self.metric_key)]
        if len(names) == 0:
            return None

        full_name = names[-1]
        val_acc = res.get(full_name)
        if self.best_val is None:
            warnings.warn(
                f"Metric {self.metric_name} used by the EarlyStopping plugin "
                f"is not computed yet. EarlyStopping will not be triggered."
            )

        if self.best_val is None:
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            self.best_step = self._get_strategy_counter(strategy)
            return None

        delta_val = float(val_acc - self.best_val)
        if self.operator(delta_val, 0) and abs(delta_val) >= self.margin:
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            self.best_step = self._get_strategy_counter(strategy)
            if self.verbose:
                print("EarlyStopping: new best value:", val_acc)
            # TODO: in testing, REMEMBER TO REMOVE IF IT DOES NOT WORK
            if strategy.clock.train_exp_epochs > 10:
                self.margin += 0.2
                print("self.margin increased to: ", self.margin)
        return self.best_val

    def _get_strategy_counter(self, strategy):
        if self.peval_mode == "epoch":
            return strategy.clock.train_exp_epochs
        elif self.peval_mode == "iteration":
            return strategy.clock.train_exp_iterations
        else:
            raise ValueError("Invalid `peval_mode`:", self.peval_mode)


###################################################################################################################################################################################
# from avalanche.evaluation.metric_definitions import GenericPluginMetric
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
        self.model = model  # .to(device) # TODO: may be the problem
        self.device = torch.device(device)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay_factor = lr_decay_factor  # 0.1  # 'Multiply lr by this much every step of decay'
        self.lr_decay_every = lr_decay_every  # 1 # 'Step decay every this many epochs'

        self.temperature = temperature
        self.alpha = alpha
        self.ft_shared_after = ft_shared_after

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,  # lr,
            momentum=self.momentum,  # momentum,
            weight_decay=self.weight_decay,  # weight_decay,
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
        self.teacher = None  # self.teachers = []
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
                                     reduction="batchmean") * scale  # reduction="mean",

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

        '''self.model.train()'''
        self.model.to(self.device)  # TODO: may be the problem

        # Normalize training and eval data.
        experiences_list: Iterable[TExperienceType] = (
            _experiences_parameter_as_iterable(experiences)
        )

        if eval_streams is None:
            eval_streams = [experiences_list]

        self._eval_streams = _group_experiences_by_stream(eval_streams)
        # self._before_training(**kwargs) #TODO: I do not think it matters if it remains commented-out as it does not trigger any essential callbacks, I believe. But try uncommenting it later on, see if the code works

        for self.experience in experiences_list:
            self._before_training_exp(**kwargs)
            self.expand_classifier(self.model, self.experience)  # DEFINITE POSITION. # IMPORTANT: recreate optimizer after classifier expansion. This is now done from the body of expand_classifier()
            self.train_experience(self.experience, eval_streams, **kwargs)
            # After training each experience, evaluate on ALL previous experiences AND Compute and print backward transfer
            self.evaluate_all_experiences(self.experience, eval_streams)  # self.evaluate_all_experiences_DEBUG(self.experience, eval_streams)
            self._after_training_exp(**kwargs)
        self._after_training(**kwargs)
        self._train_cleanup()

    def train_experience(self, experience: CLExperience, eval_streams=None, **kwargs):
        '''self.is_training = True
        self.model.train()
        #self.model.to(self.device)  # TODO: may be the problem'''
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
            # self.model.to(self.device)  # TODO: may be the problem
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            print(
                f"\n\n***** MY train_experience() IS EXECUTING FOR Experience {experience.current_experience} *****\n\n")
            # self.optimizer = self.step_lr(_) # optimizer = self.step_lr( _, self.args.lr, self.args.lr_decay_every,self.args.lr_decay_factor,self.optimizer.param_groups)

            print(
                f"\n***** EXECUTING Epoch {_} OF Experience {experience.current_experience} --- from inside train_experience() *****\n")
            total_ce = 0.0
            total_dist = 0.0
            total_loss = 0.0
            n = 0
            mini_batch_loss = 0.0  # epoch_loss = 0.0
            mini_batch_acc = 0.0  # epoch_acc = 0.0
            num_batches = 0

            for self.mbatch in self.dataloader:  # self.train_dataloader: # for batch, targets in train_dataloader:
                self.is_training = True
                self.model.train()

                self.unpack_minibatch()

                targets = self.map_targets()  #TODO: Correct definite version that made me choose this as best Nature model Uncomment once testing is done

                self._before_training_iteration(**kwargs)  # TODO: NEW, in testing

                # targets = self.mb_y #targets = self.remap_targets(self.mb_y) #experience.dataset.targets #.uniques #TODO: Removed beecause of re-mapping of targets

                self.optimizer.zero_grad()

                # ------------------------------------------------
                # Current model
                # ------------------------------------------------
                self._before_forward(**kwargs)
                self.mb_output = self.model(self.mb_x)  # student_logits is student output, which in a single-loss set-up would be " self.mb_output = self.forward()". This was previously "student_logits = self.model(self.mb_x)"
                self._after_forward(**kwargs)

                # ------------------------------------------------
                # Ordinary CE loss for current experience
                # ------------------------------------------------

                ce_loss = self.criterion(
                    self.mb_output,  # student_logits,
                    targets,
                )

                loss = ce_loss

                #print("print(targets) BEFORE self.map_targets_back_to_original()", targets)
                #print("print(self.mb_y) BEFORE self.map_targets_back_to_original()", self.mb_y)
                #self.mb_y = self.map_targets_back_to_original() #targets = self.map_targets_back_to_original()
                #print("print(targets) AFTER self.map_targets_back_to_original()", targets)
                #print("print(self.mb_y) AFTER self.map_targets_back_to_original()", self.mb_y)
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

                n += 1  # n = len(self.dataloader)

                # Track metrics
                mini_batch_loss += loss.item()  # epoch_loss += loss.item()
                with torch.no_grad():
                    acc = (self.mb_output.argmax(1) == targets).float().mean()
                    mini_batch_acc += acc.item()  # epoch_acc += acc.item()
                num_batches += 1

                self._after_training_iteration(**kwargs)  # TODO: NEW, in testing

            if VERBOSE:
                print(
                    f"Epoch {_}/{self.train_epochs} | "
                    f"CE: {total_ce / n:.6f} | "
                    f"Distil_loss: {total_dist / n:.6f} | \n"  # f"LwF: {total_dist / n:.6f} | "
                    f"Total (loss = ce_loss + (self.alpha * dist_loss)): {total_loss / n:.6f}\n"
                    f"self.alpha: {self.alpha}\n"
                    f"(self.alpha * dist_loss): {self.alpha * total_dist / n:.6f}\n"
                )

            # Log epoch results
            avg_loss = mini_batch_loss / num_batches  # avg_loss = epoch_loss / num_batches
            avg_acc = mini_batch_acc / num_batches  # avg_acc = epoch_acc / num_batches
            avg_acc = avg_acc * 100

            print(f"\n{'...' * 60}")
            print(
                f"ALTERNATIVE CODE (in testing) --> EVALUATING Epoch {_} OF Experience {experience.current_experience}")
            print(f"Epoch {_:2d}/{self.train_epochs} | "
                  # f"Loss: {avg_loss:.6f} | Acc: {avg_acc:.2f} %")#f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
            print(f"{'...' * 60}")

            # TODO: Commented-out for testing -- remember to uncomment later
            # TODO: uncomment whenever I want to save checkpoint per epoch to see at what point EWC stops mitigating CF for a particular experience
            '''self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl")
            print(f"Saved {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_experience()")
            '''
            # TODO: this method is being tested!!!! Remember to remove if Training WITHOUT LOOP
            '''self.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl")
            print(f"Loaded {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_epoch()")
            '''

            epoch_loss = avg_loss
            epoch_acc = avg_acc
            # epoch_loss, epoch_acc = self.evaluate_epoch(_, experience) # TODO: Temporarily commented out to see if alternative calculation of epoch_loss, epoch_acc leads to better results!
            accuracies.append(epoch_acc)
            accuracies.sort(reverse=True)
            print(f"accuracies: {accuracies}")
            # self.alpha = self.original_alpha
            # print(f"Alpha has been reset to {self.original_alpha}. It is now {self.alpha}")
            self._after_training_epoch(
                **kwargs)  # This is the correct location! DO NOT CHANGE LOCATION OF THIS LINE! If placed after early stopping loop, it does not execute whenever break executes, confusing Avalanche into thinking it is still training the past epoch at the same time as the next experience starts
            if EARLY_STOPPING_LOOP_1:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:  # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and prev_epoch_acc < 75:  # if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
                    # Both metrics improved - reset patience counter
                    # prev_epoch_loss = epoch_loss
                    # prev_epoch_acc = epoch_acc
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
                elif epoch_acc >= 75:  # elif prev_epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    # c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_2:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:  # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and prev_epoch_acc < 75:  # if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
                    # Both metrics improved - reset patience counter
                    # prev_epoch_loss = epoch_loss
                    # prev_epoch_acc = epoch_acc
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
                elif epoch_acc >= 75:  # elif prev_epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    # c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc
                # self.lr = self.original_lr

            if EARLY_STOPPING_LOOP_3:
                # for acc in accuracies:
                if epoch_acc > accuracies[0]:  # and experience.current_experience>0:
                    best_acc_so_far = epoch_acc
                    print(f"New best accuracy: {best_acc_so_far:.2f}%")
                    checkpoint_dir = Path(
                        "./checkpoints")  # checkpoint_dir = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Report_implementation_attempt\checkpoints")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': _,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.loss,
                        'experience': experience.current_experience,
                    },
                        f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl")  # self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl")
                    print("Best checkpoint saved.")
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:  # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and epoch_acc < 75:  # if epoch_acc > prev_epoch_acc:#           #if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
                    # Both metrics improved - reset patience counter
                    # prev_epoch_loss = epoch_loss
                    # prev_epoch_acc = epoch_acc
                    c = 0
                    loss_c = 0
                elif epoch_acc <= prev_epoch_acc and epoch_acc < 75:  # elif epoch_acc <= prev_epoch_acc:
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
                elif epoch_acc >= 75:  # if epoch_acc>=75:#if prev_epoch_acc>=75:#elif epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    # c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc
                # self.lr = self.original_lr

            if EARLY_STOPPING_LOOP_4:
                if epoch_acc >= accuracies[0]:
                    best_acc_so_far = epoch_acc
                    print(f"New best accuracy: {best_acc_so_far:.2f}%")
                    saved_model = copy.deepcopy(self.model)
                    print("Best checkpoint saved.")
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:  # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0
                if epoch_acc > prev_epoch_acc and prev_epoch_acc < 75:  # if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
                    # Both metrics improved - reset patience counter
                    # prev_epoch_loss = epoch_loss
                    # prev_epoch_acc = epoch_acc
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
                elif epoch_acc >= 75:  # elif prev_epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    # c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_FORMERLY_5_BUT_CHANGED_TO_CONDITIONS_THAT_RESULT_IN_WORSE_EARLY_STOPPING_LOGIC:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:  # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and epoch_acc < 75:  # TODO: NEW -- if epoch_acc > prev_epoch_acc and prev_epoch_acc < 80:
                    # Both metrics improved - reset patience counter
                    # prev_epoch_loss = epoch_loss
                    # prev_epoch_acc = epoch_acc
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
                elif epoch_acc >= 75:  # elif prev_epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    # c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

            if EARLY_STOPPING_LOOP_5:
                if epoch_loss >= prev_epoch_loss:
                    loss_c += 1
                    if loss_c == self.patience - 1:  # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
                        print(
                            f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(
                            f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c = 0

                if epoch_acc > prev_epoch_acc and prev_epoch_acc < acc_limit:  # if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
                    # Both metrics improved - reset patience counter
                    # prev_epoch_loss = epoch_loss
                    # prev_epoch_acc = epoch_acc
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
                elif epoch_acc >= acc_limit:  # elif prev_epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 85%. Training will be stopped.")
                    # c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

        

        if EARLY_STOPPING_LOOP_4:  # if EARLY_STOPPING_LOOP_3:
            self.model = saved_model
            print(
                f"Loaded saved_model to continue training the next Experience on it --- from inside train_experience()")

        # Add classes from this experience
        experience_classes = set(
            experience.dataset.targets.uniques
        )

        self.old_classes.update(experience_classes)

        
        print(
            f"Old classes = {sorted(self.old_classes)}"
        )
        # self.alpha += 0.05
        # print(f"Alpha has been increased by 0.05. It is now {self.alpha}")
        print(f"lr at Experience {experience.current_experience}: {self.lr}")
        # print(f"lr before being increased at Experience {experience.current_experience}: {self.lr}")
        # self.lr-=0.01*self.lr
        # print(f"lr has been decreased by {self.lr_decay_factor}. It is now {self.lr}")

        '''# TODO: this method is being tested!!!! Remember to remove if Training WITHOUT LOOP
        self.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl")
        print(f"Loaded {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl'} to continue training the next Experience on it --- from inside train_experience()")
        '''

    def step_lr(self,epoch):  # def step_lr(self, epoch, lr_decay_every, lr_decay_factor):#def step_lr(self, epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
        """Handles step decay of learning rate."""
        factor = np.power(self.lr_decay_factor, np.floor((epoch - 1) / self.lr_decay_every))  # factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
        new_lr = self.lr * factor  # new_lr = base_lr * factor
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
            self.eval_experience()  # self._eval_exp(**kwargs)
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
    def mb_y(self): #, map_targets_bool:False):
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

        for self.mbatch in self.dataloader:  # self.eval_dataloader:
            self.unpack_minibatch()
            targets = self.map_targets()  # targets = self.map_labels_expanding_classifier()
            self._before_eval_iteration(**kwargs)  # formerly commented-out

            self._before_eval_forward(**kwargs)  # formerly commented-out
            self.mb_output = self.model(self.mb_x)  # self.mb_output = self.forward()
            self._after_eval_forward(**kwargs)  # formerly commented-out
            self.loss = self.criterion(self.mb_output,
                                       targets)  # self.loss = self.criterion(self.mb_output, self.mb_y) #loss = self.criterion(self.mb_output, self.mb_y)

            #targets = self.map_targets_back_to_original()

            self._after_eval_iteration(**kwargs)  # formerly commented-out

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

        if current_exp_id_all_experiences not in self.experience_accuracies:  # if current_exp_id_all_experiences not in self.experience_accuracies.keys():
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
                        targets_all_experiences = self.map_targets()  # targets = self.map_labels_expanding_classifier() # targets = self.mb_y

                        student_logits_all_experiences = self.model(self.mb_x)
                        # print("print(self.model.training) -- evaluate all experiences after self.model(self.mb_x)",self.model.training)
                        loss_all_experiences = self.criterion(student_logits_all_experiences, targets_all_experiences)
                        # print("print(self.model.training) -- evaluate all experiences after self.criterion",self.model.training)

                        total_loss_all_experiences += loss_all_experiences.item() * targets_all_experiences.size(0)
                        predictions_all_experiences = torch.argmax(student_logits_all_experiences, dim=1)

                        # TODO: NEW! In testing for re-mapping labels in aggregated model
                        predictions_all_experiences = self.map_targets_back_to_original(predictions_all_experiences)  # TODO: NEW! In testing for re-mapping labels in aggregated model
                        targets_all_experiences = self.map_targets_back_to_original(targets_all_experiences)  # TODO: NEW! In testing for re-mapping labels in aggregated model

                        total_correct_all_experiences += (predictions_all_experiences == targets_all_experiences).sum().item()
                        total_samples_all_experiences += targets_all_experiences.size(0)

                avg_loss_all_experiences = total_loss_all_experiences / total_samples_all_experiences if total_samples_all_experiences > 0 else 0.0
                accuracy_all_experiences = (
                            total_correct_all_experiences / total_samples_all_experiences * 100) if total_samples_all_experiences > 0 else 0.0

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
        # self.model.to(self.device) # TODO: may be the problem

    def _compute_backward_transfer_internal(self):
        """Compute and print backward transfer after evaluation."""
        print(f"\n{'=' * 80}")
        print(f"BACKWARD TRANSFER ANALYSIS")
        print(f"{'=' * 80}\n")
        print("print(self.model.training) -- _compute_backward_transfer_internal", self.model.training)

        exp_ids = sorted(self.experience_accuracies.keys())

        # Print accuracy results
        print("Accuracy Results:")
        for trained_exp_id in exp_ids:
            print(f"\nAfter training Experience {trained_exp_id}:")
            for test_exp_id in sorted(self.experience_accuracies[trained_exp_id].keys()):
                acc = self.experience_accuracies[trained_exp_id][test_exp_id]
                print(f"  - Tested on Experience {test_exp_id}: {acc:.2f}%")

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
                    f"  Accuracy on Exp {test_exp_id} after training on Exp {trained_exp_id} = {final_acc:.2f}%\n"  # f"  Accuracy at the end = {final_acc:.2f}%\n"
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
                        targets_epoch = self.map_targets()  # targets = self.map_labels_expanding_classifier() # targets = self.mb_y

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
        # self.model.to(self.device)  # TODO: may be the problem
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
                        targets_single_exp = self.map_targets()  # targets = self.map_labels_expanding_classifier() # targets = self.mb_y

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
        # self.model.to(self.device)  # TODO: may be the problem

    def make_eval_dataloader_for_evaluate_all_experiences_and_bwt(self,
                                                                  # experiences: Union[TExperienceType, Iterable[TExperienceType]],
                                                                  eval_adapted_dataset: Optional[Dataset],
                                                                  # AvalancheDataset[T_co],
                                                                  batch_size: int,
                                                                  num_workers=0,
                                                                  shuffle=False,
                                                                  pin_memory=None,
                                                                  persistent_workers=False,
                                                                  drop_last=False,
                                                                  **kwargs
                                                                  ):
        assert eval_adapted_dataset is not None
        # CODE FOR "other_dataloader_args = self._obtain_common_dataloader_parameters(..." DELETED, as it does not seem like I needed for the reasons explained above
        # CODE FOR "use task-balanced dataloader for task-aware benchmarks" DELETED since it was causing data type issues and I am not gonna be using task-aware/task-IL benchmark (AT LEAST NOT FOR NOW!)
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
                                      device=self.mb_y.device,  # fine_label.device,
                                      dtype=torch.long
                                      )
        #print("print(class_to_idx_new)\n", class_to_idx_new)
        ##print("print(targets_tensor)\n",targets_tensor)
        return targets_tensor

    def map_targets_back_to_original(self, targets_tensor):#, new_to_original=None):
        #targets_tensor_list = targets_tensor.tolist()
        #print("targets_tensor_list", targets_tensor_list)

        original_targets_tensor = torch.tensor(
            [
                new_to_original[int(element.item())] for element in targets_tensor #new_to_original[int(element)] for element in targets_tensor_list #new_to_original[int(mapped_labels)] for new_label in self.mb_y #new_to_original[int(new_label)] for mapped_labels, new_label in class_to_idx_new.items() #for new_label in self.mb_y
            ],
            device=self.mb_y.device,
            dtype=torch.long
        )
        #print("print(new_to_original)\n", new_to_original)
        ##print("print(original_targets_tensor)\n",original_targets_tensor)

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
                        targets_all_experiences_post_training = self.map_targets()  # targets = self.map_labels_expanding_classifier() # targets = self.mb_y

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

    '''
    def save_entire_checkpoint(self, fname: FILE_LIKE, exclude: Optional[Collection[str]] = None): # FORMERLY #def save_entire_checkpoint(self, path, exclude: Optional[Collection[str]] = None):
        if exclude is None:
            exclude = []
        ended_experience_counter = self.clock.train_exp_counter

        strategy = copy.deepcopy(self) #strategy = copy(self)
        for attr in exclude:
            delattr(strategy, attr)

        checkpoint = {
            "strategy": strategy,
            "rng_manager_state": RNGManager.__getstate__(),#"rng_manager": RNGManager, #.set_random_seeds(SEED),
            #"exp_counter": ended_experience_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            #"loss": self.loss,
            #"ce_loss" : self.ce_loss,
            #"_criterion" : self._criterion,
            #"clock_train_exp_counter": self.clock.train_exp_counter,
            #"clock_train_epoch_counter": self.clock.train_exp_epochs,
            "teacher_state_dict": (
                self.teacher.state_dict()
                if self.teacher is not None
                else None
            ),
        }
        #with open(path, "wb") as f:
         #   pickle.dump(checkpoint, f)
        #torch.save(checkpoint, path)
        checkpoint_dir = Path("./checkpoints") # checkpoint_dir = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Report_implementation_attempt\checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, fname, pickle_module=dill)

    def load_entire_checkpoint_inside_class(self, fname: FILE_LIKE, map_location=None, unique_objects: Any = None):  # , device):
        checkpoint = torch.load(fname, pickle_module=dill, map_location=map_location)
        saved_strategy = checkpoint["strategy"] # NOT self, that would be wrong. Not used within function, but kept in case saved_strategy needs to be retrieved at some point, perhaps
                                                # even through an external call on an instance / object of class EWCStrategy

        #self.model.load_state_dict(checkpoint["model_state_dict"])
        # strategy.model.to(device)
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #self.ce_loss = checkpoint["ce_loss"]
        #self.loss = checkpoint["loss"]
        #self._criterion = checkpoint["_criterion"]
        #self.clock = checkpoint["strategy"].clock
        #self.clock.train_exp_counter = checkpoint["clock_train_exp_counter"]
        #self.clock.train_exp_epochs = checkpoint["clock_train_epoch_counter"]
        #self.train_mb_size = checkpoint["strategy"].train_mb_size
        #self.train_epochs = checkpoint["strategy"].train_epochs
        #self.eval_mb_size = checkpoint["strategy"].eval_mb_size
        #self.device = checkpoint["strategy"].device
        #self.plugins = checkpoint["strategy"].plugins
        #self.evaluator = checkpoint["strategy"].evaluator
        #self.eval_every = checkpoint["strategy"].eval_every
        #self.dataloader = checkpoint["strategy"].dataloader
        #self.experience = checkpoint["strategy"].experience
        #self.teacher=checkpoint["strategy"].teacher
        #(self.teacher.load_state_dict(checkpoint["teacher_state_dict"])
         #if self.teacher is not None
          #  else None),
        #self.old_classes = checkpoint["strategy"].old_classes
        #RNGManager.__setstate__(checkpoint["rng_manager_state"])
        #RNGManager.__setstate__(checkpoint["rng_manager"])
        print(f"Executing load_entire_checkpoint_inside_class from call made from within _train_exp()")
        return checkpoint#return self'''


'''
######################################################################################################################################################
#### Train_ResNet20_Nature_LwF_v13 ###
######################################################################################################################################################
### PERHAPS, NEW BEST Nature model? Even better than that already registered as best in 
    "All_gradient_freezing_strategies_results_NATURE_MODEL.txt"???????????

FREEZING_SHARED_LAYERS_AFTER_EPOCH_SURPASSES_ft_shared_after_AND_INDEPENDENT_OLD_ROWS=True
EARLY_STOPPING_LOOP_5 = True 
acc_limit=80
freeze_onwards=2
DIFFERENCE_CE_DIST = True
MATCH_AUTHOR_DISTILL = True
MATCH_AUTHOR_LOSS = True

standard_supervised_learning_baseline = NaiveBaseline(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                             temperature=2.0, alpha=1.0,
                             train_mb_size=43, train_epochs=10, eval_mb_size=43,
                             patience=5, ft_shared_after=8,
                             evaluator=EvaluationPlugin(loss_metrics(experience=True, stream=True),
                                                        loggers=InteractiveLogger(), collect_all=True),
                             eval_every=1,)

    seen_experiences = []
    seen_experiences_val = []

    for experience in nature_specialised_models_train_stream_class_il:  # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
        i = experience.current_experience  # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of nature_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
        if i == 3:
            break
        print("Start of experience -- current TRAIN experience: ", experience.current_experience)
================================================================================
BACKWARD TRANSFER ANALYSIS
================================================================================

print(self.model.training) -- _compute_backward_transfer_internal False
Accuracy Results:

After training Experience 0:
  - Tested on Experience 0: 81.67%

After training Experience 1:
  - Tested on Experience 0: 50.33%
  - Tested on Experience 1: 59.00%

After training Experience 2:
  - Tested on Experience 0: 44.33%
  - Tested on Experience 1: 44.33%
  - Tested on Experience 2: 37.67%

--------------------------------------------------------------------------------

Experience 0:
  Accuracy on Exp 0 right after training on Exp 0 = 81.67%
  Accuracy on Exp 0 after training on Exp 2 = 44.33%
  BWT = -37.33%

Experience 1:
  Accuracy on Exp 1 right after training on Exp 1 = 59.00%
  Accuracy on Exp 1 after training on Exp 2 = 44.33%
  BWT = -14.67%

Experience 2:
  Accuracy on Exp 2 right after training on Exp 2 = 37.67%
  Accuracy on Exp 2 after training on Exp 2 = 37.67%
  BWT = +0.00%

Mean Backward Transfer (BWT): -17.33%
  Positive BWT = improvement on old tasks (good)
  Negative BWT = forgetting on old tasks (bad)
  BWT ≈ 0 = stable performance (ideal)
================================================================================

print(self.model.training)-- evaluate_all_experiences after self.model.train() before exiting function True
-- >> End of training phase << --
Temperature: 2.0 Alpha: 2.4399648904800415
acc_limit: 80
Training completed WITH NaiveBaseline
standard_supervised_learning_baseline.results {}

Process finished with exit code 0

######################################################################################################################################################

'''

if __name__ == "__main__":

    MATCH_AUTHOR_DISTILL = True
    MATCH_AUTHOR_LOSS = True

    EARLY_STOPPING_LOOP_1 = False  # True #TODO:Change back to EARLY_STOPPING_LOOP_1=True once I am done testing "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch...".
    #                                  EARLY_STOPPING_LOOP_5 is identical to EARLY_STOPPING_LOOP_1, but with 85% limit accuracy instead of 75%, such that it is appropriate for
    #                                  "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch..."
    EARLY_STOPPING_LOOP_2 = False
    EARLY_STOPPING_LOOP_3 = False
    EARLY_STOPPING_LOOP_4 = False
    EARLY_STOPPING_LOOP_FORMERLY_5_BUT_CHANGED_TO_CONDITIONS_THAT_RESULT_IN_WORSE_EARLY_STOPPING_LOGIC = False
    EARLY_STOPPING_LOOP_5 = True  # TODO:Change back to EARLY_STOPPING_LOOP_1=True once I am done testing "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch...".
    #                            EARLY_STOPPING_LOOP_5 is identical to EARLY_STOPPING_LOOP_1, but with 85% limit accuracy instead of 75%, such that it is appropriate for
    #                            "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch..."

    acc_limit=80 # best for binary strategy
    #acc_limit=85  # best for 'delayed_start', {'freeze_start_experience': 2}
    freeze_onwards=2 # for Nature or Outdoors model (I can't remember), {'freeze_start_experience': 2} was best. This is a manual {'freeze_start_experience': 2} with some additional things,
                    # like whether the freezing of the old classifier rows is included in the condition controlled by self.fet_shared_after or not, which is needed, as I am unsure of whether
                    # {'freeze_start_experience': 2} includes freezing of the old classifier rows in the condition controlled by self.fet_shared_after or not. I named the variable freeze_onwrds
                    # to avoid confusion with 'freeze_start_experience' in experiments where GradientFreezeScheduler is used.

    VERBOSE = False  # True
    # DYNAMIC_MODELS = True

    DIFFERENCE_CE_DIST = True
    # DIFFERENCE_CE_DIST= False

    TRAINING_WITH_LOOP_LwF =True# False
    # TODO: Set all of the below to false for # NO FREEZING AT ALL -- as if all freezing code commented out. BASELINE for freezing strategies!!!
    FREEZING_ORIGINAL=False#True #False
    FREEZING_ORIGINAL_BUT_INDEPENDENT_CONDITION_FOR_SHARED_LAYERS_AND_OLD_ROWS=False
    FREEZING_ORIGINAL_BUT_FROM_EXPERIENCE_2_ONWARDS=False #True
    FREEZING_ORIGINAL_BUT_FROM_EXPERIENCE_2_ONWARDS_AND_INDEPENDENT_CONDITION_FOR_SHARED_LAYERS_AND_OLD_ROWS=False #True
    FREEZING_SHARED_LAYERS_AFTER_EPOCH_SURPASSES_ft_shared_after_AND_INDEPENDENT_OLD_ROWS=True
    FREEZING_SHARED_LAYERS_AND_OLD_ROWS_AFTER_EPOCH_SURPASSES_ft_shared_after=False #True

    print(f"MATCH_AUTHOR_DISTILL = {MATCH_AUTHOR_DISTILL}, MATCH_AUTHOR_LOSS = {MATCH_AUTHOR_LOSS}, VERBOSE = {VERBOSE}")

    final_specialised_models_baseline_dir = Path("./final_specialised_models_BASELINE")
    final_specialised_models_baseline_dir.mkdir(parents=True, exist_ok=True)

    if TRAINING_WITH_LOOP_LwF:

        # standard_supervised_learning_baseline = NaiveBaseline(model=nature_specialised_model, device="cuda", lr=0.001, momentum=0.9, weight_decay=5e-4,temperature=2.0, alpha=1.0)
        standard_supervised_learning_baseline = NaiveBaseline(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                                 temperature=2.0, alpha=1.0, #temperature=2.0, alpha=1.0,#temperature=4.0, alpha=5.0,
                                 train_mb_size=43, train_epochs=10, eval_mb_size=43, #eval_mb_size=32, #device=DEVICE,
                                 patience=5, ft_shared_after=8,
                                 #plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream",metric_name="Loss_Exp", mode="min", peval_mode="epoch",margin=0.3, verbose=True)],
                                 evaluator=EvaluationPlugin(#accuracy_metrics(experience=True, stream=True),
                                                            loss_metrics(experience=True, stream=True),
                                                            #forgetting_metrics(experience=True, stream=True), bwt_metrics(experience=True, stream=True),
                                                            loggers=InteractiveLogger(), collect_all=True),
                                 eval_every=1,)

        seen_experiences = []
        seen_experiences_val = []
        seen_experiences_sizes = 0

        for experience in nature_specialised_models_train_stream_class_il:  # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
            i = experience.current_experience  # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of nature_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
            if i == 1:
                torch.save(standard_supervised_learning_baseline.model.state_dict(),f"./final_specialised_models_BASELINE/STANDARD_SUPERVISED_LEARNING_BASELINE_nature_model.pt")
                #torch.save(standard_supervised_learning_baseline.model.linear.state_dict(), f"./final_specialised_models_BASELINE/standard_supervised_learning_baseline_nature_model_CLASSIFIER.pt")
                new_to_original = new_to_original_creation()  # Your existing function
                class_to_idx_new = class_to_idx_new  # Your existing variable
                # Save the mappings used in training
                metadata = {
                    'class_to_idx_new': class_to_idx_new,  # YOUR EXISTING VARIABLE
                    'new_to_original': new_to_original,  # YOUR EXISTING VARIABLE
                    #'num_subset_classes': len(class_to_idx_new),
                    'num_subset_classes': len(standard_supervised_learning_baseline.old_classes),
                    'feature_dim': standard_supervised_learning_baseline.model.linear.weight.shape[1],
                    'subset_name': 'nature_specialised_models_train_subset'  # Change this for different subsets
                }
                with open('./final_specialised_models_BASELINE/Standard_Supervised_Learning_BASELINE_nature_metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
                print("✓ Saved metadata for BASELINE EVALUATION")
                print(f"Final value of seen_experiences_sizes = {seen_experiences_sizes}. This is the total number of samples in the training set for this model (that is, across all experiences).")
                break
            print("Start of experience -- current TRAIN experience: ", experience.current_experience)
            print("Start of experience -- current VAL experience: ",
                  nature_specialised_models_val_stream_class_il.exps_iter[i].current_experience)
            print("Current Classes -- current TRAIN .classes_in_this_experience: ", experience.classes_in_this_experience)
            print("Current Classes -- current VAL .classes_in_this_experience: ",
                  nature_specialised_models_val_stream_class_il.exps_iter[i].classes_in_this_experience)
            print("experience.dataset.coarse_label.uniques (coarse_labels of TRAIN)",
                  experience.dataset.coarse_label.uniques)
            print("experience.dataset.targets (targets of TRAIN)", experience.dataset.targets.uniques)
            print(
                "nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques (coarse_labels of VAL)",
                nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques)
            print("nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets (targets of VAL)",
                  nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques)
            print("print(experience.origin_stream.name)", experience.origin_stream.name)
            print("print(nature_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)",
                  nature_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)
            # print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
            # print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
            # criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
            # print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
            print("print(standard_supervised_learning_baseline.plugins) BEFORE call to .train()", standard_supervised_learning_baseline.plugins)
            seen_experiences.append(experience)
            seen_experiences_sizes += len(experience.dataset)
            print(f"len(experience.dataset) of current TRAIN experience {i} : {len(experience.dataset)}\nAfter experience {i}, seen_experiences_sizes = {seen_experiences_sizes}")
            seen_experiences_val.append(nature_specialised_models_val_stream_class_il[i])
            standard_supervised_learning_baseline.train(experience, [seen_experiences,
                                           seen_experiences_val])  # standard_supervised_learning_baseline.train(experience, [nature_specialised_models_train_stream_class_il,nature_specialised_models_val_stream_class_il])
            print("Temperature:", standard_supervised_learning_baseline.temperature, "Alpha:", standard_supervised_learning_baseline.alpha)
            if EARLY_STOPPING_LOOP_5:
                print("acc_limit:", acc_limit)

        print('Training completed WITH NaiveBaseline')
        print("standard_supervised_learning_baseline.results", standard_supervised_learning_baseline.results)
        # print("standard_supervised_learning_baseline.compare_experiments() AFTER ALL EXPERIMENTS HAVE CONCLUDED:\n", standard_supervised_learning_baseline.compare_experiments())
        '''print("standard_supervised_learning_baseline.results[exp_name]",standard_supervised_learning_baseline.results[exp_name])
        print("standard_supervised_learning_baseline.results[exp_name]['gradient_stats']",standard_supervised_learning_baseline.results[exp_name]['gradient_stats'])
        print("standard_supervised_learning_baseline.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch']",standard_supervised_learning_baseline.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch'])
        print("standard_supervised_learning_baseline.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch'][0]",standard_supervised_learning_baseline.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch'][0])'''
        # print("standard_supervised_learning_baseline.evaluate_all_experiences(nature_specialised_models_val_stream_class_il)")
        # standard_supervised_learning_baseline.evaluate_all_experiences_post_training(nature_specialised_models_val_stream_class_il)#(seen_experiences_val)