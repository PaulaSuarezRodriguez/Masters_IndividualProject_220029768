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

from Twelfth_implementation_attempt.Config_v12 import set_seed
set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Twelfth_implementation_attempt.Config_v12 import set_device
DEVICE=set_device()
from Twelfth_implementation_attempt.Config_v12 import SEED
from avalanche.training.determinism.rng_manager import RNGManager
RNGManager.set_random_seeds(SEED)
from Twelfth_implementation_attempt.ResNet20_v12 import ResNet20
from Twelfth_implementation_attempt.ResNet20_DynamicModel_v12 import ResNet20_DynamicModel
from Twelfth_implementation_attempt.Load_cifar100_v12 import (outdoors_specialised_models_train_subset)
from Twelfth_implementation_attempt.Load_cifar100_v12 import (outdoors_specialised_models_val_subset)
from Twelfth_implementation_attempt.Load_cifar100_v12 import (outdoors_specialised_models_test_subset)
from Twelfth_implementation_attempt.Load_cifar100_v12 import (nature_specialised_models_train_subset)
from Twelfth_implementation_attempt.Load_cifar100_v12 import (nature_specialised_models_val_subset)
from Twelfth_implementation_attempt.Load_cifar100_v12 import (nature_specialised_models_test_subset)
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

from collections.abc import Iterable # OOOR like in avalanche/training/templates/base_sgd.py , from typing import Any, Callable, Generic, Iterable, Sequence, Optional, TypeVar, Union
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
from Twelfth_implementation_attempt.GradientFreezeScheduler_v1.GradientFreezeScheduler import (
    GradientFreezeScheduler,
    apply_gradient_freezing,
    FreezeConfig
)

DEBUG = False#True#False #TODO: Remember to change DEBUG to False if running experiments.
TRAINING_WITH_LOOP = False #TODO: Remember to change TRAINING_WITH_LOOP to True if 'Training WITH LOOP VERSION 2'.
TRAINING_WITHOUT_LOOP = False #True #TODO: Remember to change TRAINING_WITHOUT_LOOP to True if 'Training WITHOUT LOOP'.
TRAINING_WITH_LOOP_LwF = True
DYNAMIC_MODELS = True
print("\n\n\n################################ HERE BEGINS Train_ResNet20_Outdoors_v12.py ################################ \n\n\n")

if DYNAMIC_MODELS == False:
    resnet20_global_model = ResNet20(num_classes=100)
    outdoors_specialised_model = ResNet20(num_classes=100) #No need for num_classes=10 anymore despite 10 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    nature_specialised_model = ResNet20(num_classes=100) #No need for num_classes=15 anymore despite 15 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    people_specialised_model = ResNet20(num_classes=100) #No need for num_classes=5 anymore despite 5 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    animals_specialised_model = ResNet20(num_classes=100) #No need for num_classes=45 anymore despite 45 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    objects_and_vehicles_specialised_model = ResNet20(num_classes=100) #No need for num_classes=25 anymore despite 25 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used

if DYNAMIC_MODELS == True:
    resnet20_global_model = ResNet20_DynamicModel(num_classes=100)
    outdoors_specialised_model = ResNet20_DynamicModel(num_classes=0) #No need for num_classes=10 anymore despite 10 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    nature_specialised_model = ResNet20_DynamicModel(num_classes=0) #No need for num_classes=15 anymore despite 15 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    people_specialised_model = ResNet20_DynamicModel(num_classes=5) #No need for num_classes=5 anymore despite 5 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    animals_specialised_model = ResNet20_DynamicModel(num_classes=45) #No need for num_classes=45 anymore despite 45 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
    objects_and_vehicles_specialised_model = ResNet20_DynamicModel(num_classes=25) #No need for num_classes=25 anymore despite 25 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used

avl_train = make_avalanche_dataset(nature_specialised_models_train_subset.dataset)
avl_val = make_avalanche_dataset(nature_specialised_models_val_subset.dataset)
avl_test = make_avalanche_dataset(nature_specialised_models_test_subset.dataset)

#class_to_index={}
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

if DEBUG:
    print("type(nature_specialised_models_train_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train", type(avl_train))
    print("type(nature_specialised_models_val_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val", type(avl_val))
    print("type(nature_specialised_models_test_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test", type(avl_test))

    print("Print avl_train AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train:\n", avl_train)
    print("Print avl_val AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val:\n", avl_val)
    print("Print avl_test AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test:\n", avl_test)

avl_train = ClassificationDataset(
    [avl_train], #[nature_specialised_models_train_subset.dataset],
    data_attributes=[DataAttribute(nature_specialised_models_train_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(nature_specialised_models_train_subset.dataset.targets,"targets")], # "fine_label")],
)
avl_val = ClassificationDataset(
    [avl_val], #[nature_specialised_models_val_subset.dataset],
    data_attributes=[DataAttribute(nature_specialised_models_val_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(nature_specialised_models_val_subset.dataset.targets, "targets")],# "fine_label")],
)
avl_test = ClassificationDataset(
    [avl_test], #[nature_specialised_models_test_subset.dataset],
    data_attributes=[DataAttribute(nature_specialised_models_test_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(nature_specialised_models_test_subset.dataset.targets, "targets")],# "fine_label")],
)

if DEBUG:
    print("Type avl_train AFTER being passed to ClassificationDataset", type(avl_train))
    print("Type avl_val AFTER being passed to ClassificationDataset", type(avl_val))
    print("Type avl_test AFTER being passed to ClassificationDataset", type(avl_test))

    print("Print avl_train:\n", avl_train)
    print("Print avl_val:\n", avl_val)
    print("Print avl_test:\n", avl_test)

print("Print avl_train.targets.uniques (AS .targets IS A DataAttribute AND THE ONLY WAY TO ACCESS AND PRINT ITS ACTUAL ELEMENTS/VALUES IS USING ATTRIBUTE .uniques):", avl_train.targets.uniques)
#print("Print avl_train.targets", avl_train.targets) #TO PROVE THE ABOVE, AS WHEN ATTRIBUTE .uniques OF DataAttribute IS NOT USED, THE ACTUAL TARGETS/VALUES CONTAINED WITHIN avl_train.targets ARE NOT ACCESSED. Instead, FlatData (len=4000,subset=True,cat=False,cf=True) DataAttribute (len=4000) is returned)
print("Print max(avl_train.targets.uniques) in order to be aware of the value that due to the library bug will be assigned to num_classes no matter what my code does (and in this way better mitigate, follow, identify and debug the consequence of such bug):", max(avl_train.targets.uniques))
print("Print len(avl_train.targets.uniques) to know the NUMBER OF FINE CLASSES within the specialised_models_superclasses group of which the specialised subset (e.g. nature_specialised_models_train_subset) that is avl_train is composed:", len(avl_train.targets.uniques)) # print number of classes
print("IMPORTANT! Print sorted(avl_train.targets.uniques) AS THIS WILL BE THE OBJECT PASSED TO class_order IN class_incremental_benchmark TO AVOID/AS A WORK-AROUND TO THE num_classes BUG IN THE LIBRARY CODE, since by provideing a class_order, 'class_order = torch.randperm(num_classes).tolist()' within 'if class_order is None:' IS AVOIDED!", sorted(avl_train.targets.uniques))# print class_order

# Original and best performing benchmark
'''nature_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[3, 3, 3, 3, 85])''' # which is equivalent to num_experiences=5 due to the reasons explained in previous version)

# Same as original above, but nature_specialised_models_val_stream_class_il[4] now contains the 3 last classes of the specialised model, in case I even wanna use
# nature_specialised_models_val_stream_class_il[4] for the global model
nature_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[3, 3, 3, 3, 3, 82])

#BOTH OF THESE GAVE HORRIBLE RESULTS
'''nature_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[4, 4, 4, 85])
nature_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[4, 4, 4, 4, 81])'''

# TODO: try tomorrow, but this would require modifying the benchmark and stream objects cause otherwise model is still trained on what it thinks are Exp004 and Exp005
'''nature_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[3, 3, 71])'''

if DEBUG:
    print("sorted(avl_train.targets.uniques)",sorted(avl_train.targets.uniques))
    for i, exp in enumerate(nature_specialised_models_benchmark.train_stream):
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_benchmark)", i, exp.classes_in_this_experience)

    print("Print nature_specialised_models_benchmark:", nature_specialised_models_benchmark)
    #print("Print nature_specialised_models_benchmark.train_stream:", nature_specialised_models_benchmark.train_stream) # it works, but I am not sure that it is not deprecated. I think the current or at least most frequently used way of retrieving a stream is .streams["train"], specially if there is a .streams["valid"] since attribute .val_stream does NOT exist. print(nature_specialised_models_benchmark.train_stream) and print(nature_specialised_models_benchmark.streams["train"]) both return <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000002242B559510>, meaning they return the exact same object so, there is no risk involved in having chosen to use .streams["train"] over .train_stream
    print('Print nature_specialised_models_benchmark.streams["train"]',nature_specialised_models_benchmark.streams["train"])
    print('Print nature_specialised_models_benchmark.streams["test"]', nature_specialised_models_benchmark.streams["test"])
    print('Print nature_specialised_models_benchmark.streams["valid"]', nature_specialised_models_benchmark.streams["valid"])

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
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_train_stream_class_il)", i, exp.classes_in_this_experience)

    for i, exp in enumerate(nature_specialised_models_val_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_val_stream_class_il)", i, exp.classes_in_this_experience)

    for i, exp in enumerate(nature_specialised_models_test_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- nature_specialised_models_test_stream_class_il)", i,
              exp.classes_in_this_experience)

    print("type(nature_specialised_models_train_stream_class_il -- OBJECT THROUGH WHICH TRAINING LOOP IS CURRENTLY ITERATING)", type(nature_specialised_models_train_stream_class_il))
    print("type(nature_specialised_models_train_stream_class_il.exps_iter -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS)", type(nature_specialised_models_train_stream_class_il.exps_iter))
    print('type(nature_specialised_models_benchmark.train_stream -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS', type(nature_specialised_models_benchmark.train_stream), 'ALSOOOOOO, benchmark.train_stream KNOWN TO BE EQUAL TO benchmark.streams["train"]); CHECK TO VERIFY')
    print("type(nature_specialised_models_train_stream_class_il.exps_iter[0])", type(nature_specialised_models_train_stream_class_il.exps_iter[0]))

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
            #TODO: in testing, REMEMBER TO REMOVE IF IT DOES NOT WORK
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
#from avalanche.evaluation.metric_definitions import GenericPluginMetric
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

class LwFTrainer(SupervisedTemplate):
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
        freeze_strategy: str = 'binary',  # NEW LINE
        freeze_config: Optional[Dict] = None,  # NEW LINE
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
        self.model = model #.to(device) # TODO: may be the problem
        self.device = torch.device(device)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay_factor = lr_decay_factor #0.1  # 'Multiply lr by this much every step of decay'
        self.lr_decay_every = lr_decay_every #1 # 'Step decay every this many epochs'

        self.temperature = temperature
        self.alpha = alpha
        self.ft_shared_after = ft_shared_after

        # NEW: Gradient freeze strategy configuration
        self.freeze_strategy = freeze_strategy
        self.freeze_config = freeze_config if freeze_config is not None else {}
        self.freeze_scheduler: Optional[GradientFreezeScheduler] = None

        # Tracking for analysis
        self.gradient_freeze_logs = {
            'epoch': [],
            'freeze_config': [],
            'loss': [],
            'accuracy': []
        }

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,#lr,
            momentum=self.momentum,#momentum,
            weight_decay=self.weight_decay,#weight_decay,
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
        self.teacher = None #self.teachers = []
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

        if MATCH_AUTHOR_DISTILL: # Match authors' implementation --> F.kl_div(...) * number_of_outputs

            print(f"  scale used in dist_loss: {scale}")
            kl_divergence = F.kl_div(student_log_probs,teacher_probs,reduction="batchmean") * scale #reduction="mean",

        if MATCH_AUTHOR_DISTILL == False:

            kl_divergence = F.kl_div(student_log_probs,teacher_probs,reduction="batchmean") * (T ** 2)

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
        self.model.to(self.device) # TODO: may be the problem

        # Normalize training and eval data.
        experiences_list: Iterable[TExperienceType] = (
            _experiences_parameter_as_iterable(experiences)
        )

        if eval_streams is None:
            eval_streams = [experiences_list]

        self._eval_streams = _group_experiences_by_stream(eval_streams)
        #self._before_training(**kwargs) #TODO: I do not think it matters if it remains commented-out as it does not trigger any essential callbacks, I believe. But try uncommenting it later on, see if the code works

        for self.experience in experiences_list:
            self._before_training_exp(**kwargs)
            self.expand_classifier(self.model, self.experience)  # DEFINITE POSITION. # IMPORTANT: recreate optimizer after classifier expansion. This is now done from the body of expand_classifier()
            self.train_experience(self.experience, eval_streams, **kwargs)
            # After training each experience, evaluate on ALL previous experiences AND Compute and print backward transfer
            self.evaluate_all_experiences(self.experience, eval_streams)#self.evaluate_all_experiences_DEBUG(self.experience, eval_streams)
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

        # NEW: Initialize freeze scheduler for this experience (first epoch only)
        #if self.freeze_scheduler is None:  # if _ == 0:
        self.freeze_scheduler = GradientFreezeScheduler(
            freeze_strategy=self.freeze_strategy,
            total_epochs=self.train_epochs,
            ft_shared_after=self.ft_shared_after,
            config=self.freeze_config,
            verbose=False  # Set to True for debugging
        )
        # Print schedule (should show decreasing freeze intensity) # Show schedule (for debugging)
        if self.freeze_strategy != 'binary':
            print("print(self.freeze_scheduler.get_schedule_visualization())", self.freeze_scheduler.get_schedule_visualization())

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]
        for _ in range(self.train_epochs):
            '''# NEW: Initialize freeze scheduler for this experience (first epoch only)
            if self.freeze_scheduler is None:#if _ == 0:
                self.freeze_scheduler = GradientFreezeScheduler(
                    freeze_strategy=self.freeze_strategy,
                    total_epochs=self.train_epochs,
                    ft_shared_after=self.ft_shared_after,
                    config=self.freeze_config,
                    verbose=False  # Set to True for debugging
                )
                # Print schedule (should show decreasing freeze intensity)
                print("print(scheduler.get_schedule_visualization())",self.freeze_scheduler.get_schedule_visualization())'''

            self.is_training = True
            self.model.train()
            #self.model.to(self.device)  # TODO: may be the problem
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            print(f"\n\n***** MY train_experience() IS EXECUTING FOR Experience {experience.current_experience} *****\n\n")
            #self.optimizer = self.step_lr(_) # optimizer = self.step_lr( _, self.args.lr, self.args.lr_decay_every,self.args.lr_decay_factor,self.optimizer.param_groups)

            print(f"\n***** EXECUTING Epoch {_} OF Experience {experience.current_experience} --- from inside train_experience() *****\n")
            total_ce = 0.0
            total_dist = 0.0
            total_loss = 0.0
            n=0
            mini_batch_loss = 0.0#epoch_loss = 0.0
            mini_batch_acc = 0.0#epoch_acc = 0.0
            num_batches = 0

            for self.mbatch in self.dataloader:#self.train_dataloader: # for batch, targets in train_dataloader:
                self.is_training = True
                self.model.train()

                self.unpack_minibatch()

                targets = self.map_targets() #targets = self.map_labels_expanding_classifier()

                self._before_training_iteration(**kwargs) #TODO: NEW, in testing

                #targets = self.mb_y #targets = self.remap_targets(self.mb_y) #experience.dataset.targets #.uniques #TODO: Removed beecause of re-mapping of targets

                self.optimizer.zero_grad()

                # ------------------------------------------------
                # Current model
                # ------------------------------------------------
                self._before_forward(**kwargs)
                self.mb_output = self.model(self.mb_x) # student_logits is student output, which in a single-loss set-up would be " self.mb_output = self.forward()". This was previously "student_logits = self.model(self.mb_x)"
                self._after_forward(**kwargs)

                # ------------------------------------------------
                # Ordinary CE loss for current experience
                # ------------------------------------------------

                ce_loss = self.criterion(
                    self.mb_output, #student_logits,
                    targets,
                )

                # ------------------------------------------------
                # LwF -- single teacher
                # ------------------------------------------------
                if self.teacher is not None:
                    #for classes in self.teacher_classes:
                    with torch.no_grad():
                        #print("print(self.teacher.training) -- before teacher_logits = self.teacher(self.mb_x)", self.teacher.training)
                        teacher_logits = self.teacher(self.mb_x)  # self.teacher(self.mbatch)#teacher_logits = self.teacher(batch)
                        #print("print(self.teacher.training) -- after teacher_logits = self.teacher(self.mb_x)",self.teacher.training)

                    # Only old classes
                    for classes in self.teacher_classes:
                        class_indices = torch.tensor(
                            [
                                class_to_idx_new[c]
                                for c in sorted(classes)  # (self.teacher_classes)
                            ],
                            device=self.device,
                            dtype=torch.long
                        )
                    if VERBOSE:
                        print(f"  Classes (original): {sorted(classes)}")#{sorted(self.teacher_classes)}")
                        print(f"  Class indices (mapped): {class_indices.tolist()}")
                        print(f"  Student output shape: {self.mb_output.shape}")
                        print(f"  Teacher output shape: {teacher_logits.shape}")

                    if VERBOSE:
                        print(f"  scale used in dist_loss: {self.mb_output.size(1)}")#{student_old.size(1)}

                    scale = self.mb_output.size(1)

                    dist_loss = self.distillation_loss(
                        self.mb_output,
                        teacher_logits,
                        scale=scale
                    )
                    if VERBOSE:
                        # Debug: check distillation loss value
                        print(f"  CE loss : {ce_loss.item():.6f}")
                        print(f"  Distillation loss (teacher {self.teacher_classes}): {dist_loss.item():.6f}")#print(f"  Distillation loss (teacher {classes}): {dist_loss.item():.6f}")
                        print(f"self.alpha BEFORE INCREASE: {self.alpha}\n")

                    if DIFFERENCE_CE_DIST:
                        difference_ce_dist = (ce_loss.item() - dist_loss.item())
                        if difference_ce_dist > 0:
                            div = (difference_ce_dist / 0.10)#div = (difference_ce_dist / 0.20)
                            self.alpha = self.original_alpha + (0.10 * div)
                            if VERBOSE:
                                print(f"difference_ce_dist = (ce_loss - dist_loss) = {difference_ce_dist},\n "
                                      f"div = (difference_ce_dist/0.10) = {div},\n "
                                      f"Alpha has been increased by {(0.10 * div)} --> self.alpha = self.alpha + (0.10 * div) = {self.alpha},\n"
                                      f"self.alpha AFTER INCREASE: {self.alpha}\n"
                                      f"(self.alpha * dist_loss): {self.alpha * dist_loss}\n"
                                      f"Effective difference CE loss - Distillation loss: {ce_loss.item() - (self.alpha * dist_loss)}\n")

                    if MATCH_AUTHOR_LOSS:
                        loss = ce_loss + (self.alpha * dist_loss)
                        if VERBOSE:
                            print(f"  Total loss AFTER HAVING INCREASED ALPHA, loss that is to be backpropagated: {loss.item():.6f}")
                    if MATCH_AUTHOR_LOSS == False:
                        loss = ((self.alpha * dist_loss) + ((1 - self.alpha) * ce_loss)) #loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss # loss = ((self.alpha * (self.temperature ** 2) * dist_loss) + ((1 - self.alpha) * ce_loss)) #loss = ce_loss + self.alpha * dist_loss

                else:
                    loss = ce_loss

                    dist_loss = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                loss.backward()
                if VERBOSE:
                    print(f"  Backpropagated loss: {loss.item():.6f}\n Backpropagation is occurring!")

                # Apply gradient freezing strategy (after loss.backward())
                if self.teacher is not None and self.freeze_scheduler is not None:

                    freeze_config = self.freeze_scheduler.get_freeze_config(
                        current_epoch=_,
                        loss=loss.item()
                    )

                    apply_gradient_freezing(
                        model=self.model,
                        freeze_config=freeze_config,
                        old_classes=self.old_classes,
                        device=self.device,
                        verbose=False  # Set to True to debug gradient freezing
                    )

                    # Log the freeze configuration for analysis
                    self.gradient_freeze_logs['epoch'].append(_)
                    self.gradient_freeze_logs['freeze_config'].append(freeze_config)
                    self.gradient_freeze_logs['loss'].append(loss.item())
                else:
                    # Fallback to original behavior (binary freeze)
                    self._original_gradient_freezing()

                '''if (self.teacher is not None) and (_ <= self.ft_shared_after):# and (experience.current_experience>=2):#if (self.teacher is not None) and (_ <= self.ft_shared_after) and (experience.current_experience>=2): #if (self.teacher is not None) and (_ <= self.ft_shared_after):#if ((len(self.teachers)>0) and (_ <= self.ft_shared_after)):

                    # Freeze shared layers
                    for module in self.model.modules():

                        if isinstance(module, (nn.Conv2d, nn.Linear)):

                            if module is not self.model.linear:

                                if module.weight.grad is not None:
                                    module.weight.grad.zero_()

                                if module.bias is not None:
                                    if module.bias.grad is not None:
                                        module.bias.grad.zero_()

                # Freeze old classifier rows
                if len(self.old_classes) > 0:

                    old_num_classes = len(self.old_classes)

                    if self.model.linear.weight.grad is not None:
                        self.model.linear.weight.grad[
                            :old_num_classes
                        ].zero_()

                    if self.model.linear.bias.grad is not None:
                        self.model.linear.bias.grad[
                            :old_num_classes
                        ].zero_()'''

                '''if (self.teacher is not None) and len(self.old_classes) > 0 and (_ <= self.ft_shared_after-2):
                    # Freeze old classifier rows
                    old_num_classes = len(self.old_classes)

                    if self.model.linear.weight.grad is not None:
                        self.model.linear.weight.grad[
                            :old_num_classes
                        ].zero_()

                    if self.model.linear.bias.grad is not None:
                        self.model.linear.bias.grad[
                            :old_num_classes
                        ].zero_()'''

                self.optimizer.step()

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                total_ce += ce_loss.item()
                total_dist += dist_loss.item()
                total_loss += loss.item()

                n += 1 #n = len(self.dataloader)

                # Track metrics
                mini_batch_loss += loss.item()#epoch_loss += loss.item()
                with torch.no_grad():
                    acc = (self.mb_output.argmax(1) == targets).float().mean()
                    mini_batch_acc += acc.item()#epoch_acc += acc.item()
                num_batches += 1


                self._after_training_iteration(**kwargs) #TODO: NEW, in testing

            if VERBOSE:
                print(
                    f"Epoch {_}/{self.train_epochs} | "
                    f"CE: {total_ce / n:.6f} | "
                    f"Distil_loss: {total_dist / n:.6f} | \n"#f"LwF: {total_dist / n:.6f} | "
                    f"Total (loss = ce_loss + (self.alpha * dist_loss)): {total_loss / n:.6f}\n"
                    f"self.alpha: {self.alpha}\n"
                    f"(self.alpha * dist_loss): {self.alpha * total_dist/n:.6f}\n"
                )

            # Log epoch results
            avg_loss = mini_batch_loss / num_batches#avg_loss = epoch_loss / num_batches
            avg_acc = mini_batch_acc / num_batches#avg_acc = epoch_acc / num_batches
            avg_acc=avg_acc*100

            print(f"\n{'...' * 60}")
            print(f"ALTERNATIVE CODE (in testing) --> EVALUATING Epoch {_} OF Experience {experience.current_experience}")
            print(f"Epoch {_:2d}/{self.train_epochs} | "
                  #f"Loss: {avg_loss:.6f} | Acc: {avg_acc:.2f} %")#f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
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
            #epoch_loss, epoch_acc = self.evaluate_epoch(_, experience) # TODO: Temporarily commented out to see if alternative calculation of epoch_loss, epoch_acc leads to better results!
            accuracies.append(epoch_acc)
            accuracies.sort(reverse=True)
            print(f"accuracies: {accuracies}")
            #self.alpha = self.original_alpha
            #print(f"Alpha has been reset to {self.original_alpha}. It is now {self.alpha}")
            self._after_training_epoch(**kwargs) #This is the correct location! DO NOT CHANGE LOCATION OF THIS LINE! If placed after early stopping loop, it does not execute whenever break executes, confusing Avalanche into thinking it is still training the past epoch at the same time as the next experience starts
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
                #self.lr = self.original_lr

            if EARLY_STOPPING_LOOP_3:
                # for acc in accuracies:
                if epoch_acc > accuracies[0]: #and experience.current_experience>0:
                    best_acc_so_far = epoch_acc
                    print(f"New best accuracy: {best_acc_so_far:.2f}%")
                    checkpoint_dir = Path("./checkpoints")  # checkpoint_dir = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Twelfth_implementation_attempt\checkpoints")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': _,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.loss,
                        'experience': experience.current_experience,
                    },f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl")#self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl")
                    print("Best checkpoint saved.")
                if epoch_loss >= prev_epoch_loss:
                    loss_c+=1
                    if loss_c == self.patience-1 : # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
                        print(f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        self.lr = self.lr * self.lr_decay_factor
                        print(f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
                        c=0

                if epoch_acc > prev_epoch_acc and epoch_acc<75:#if epoch_acc > prev_epoch_acc:#           #if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
                    # Both metrics improved - reset patience counter
                    #prev_epoch_loss = epoch_loss
                    #prev_epoch_acc = epoch_acc
                    c = 0
                    loss_c=0
                elif epoch_acc <= prev_epoch_acc and epoch_acc < 75: #elif epoch_acc <= prev_epoch_acc:
                    # No improvement - increment patience counter
                    c += 1
                    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

                    if c >= self.patience:
                        print(
                            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                        self.lr = self.original_lr
                        c=0
                        loss_c = 0
                        break
                elif epoch_acc>=75:#if epoch_acc>=75:#if prev_epoch_acc>=75:#elif epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
                    #c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc
                #self.lr = self.original_lr

            if EARLY_STOPPING_LOOP_4:
                if epoch_acc >= accuracies[0]:
                    best_acc_so_far = epoch_acc
                    print(f"New best accuracy: {best_acc_so_far:.2f}%")
                    saved_model=copy.deepcopy(self.model)
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

                if epoch_acc > prev_epoch_acc and prev_epoch_acc < 80:  # if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
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
                elif epoch_acc >= 80:  # elif prev_epoch_acc>=75:
                    print(
                        f"WARNING: The accuracy of experience {experience.current_experience} has reached 80%. Training will be stopped.")
                    # c = self.patience
                    self.lr = self.original_lr
                    c = 0
                    loss_c = 0
                    break

                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc

        # --------------------------------------------------------
        # IMPORTANT:
        # Save teacher AFTER completing the experience
        # --------------------------------------------------------
        #teacher = copy.deepcopy(self.model)
        #teacher.eval()
        #print("self.evaluate_single_experience() AFTER LOADING:")
        #self.evaluate_single_experience(experience, eval_streams)
        '''print("self.evaluate_single_experience(current_experience_val) AFTER TRAINING FOR EXPERIENCE CONCLUDES, before teacher is saved:")
        self.evaluate_single_experience(experience, eval_streams)'''

        self.teacher = copy.deepcopy(self.model)
        print("print(self.teacher.training) -- before self.teacher.eval()", self.teacher.training)
        self.teacher.eval()
        print("print(self.teacher.training) -- after self.teacher.eval()", self.teacher.training)

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        if EARLY_STOPPING_LOOP_4:  # if EARLY_STOPPING_LOOP_3:
            self.model=saved_model
            print(
                f"Loaded saved_model to continue training the next Experience on it --- from inside train_experience()")

        # Add classes from this experience
        experience_classes = set(
            experience.dataset.targets.uniques
        )

        self.old_classes.update(experience_classes)

        #self.teachers.append(teacher)

        self.teacher_classes.append(
            sorted(experience.dataset.targets.uniques)
        )

        print(
            f"Teacher updated. "
            f"Old classes = {sorted(self.old_classes)}"
        )
        #self.alpha += 0.05
        #print(f"Alpha has been increased by 0.05. It is now {self.alpha}")
        print(f"lr at Experience {experience.current_experience}: {self.lr}")
        #print(f"lr before being increased at Experience {experience.current_experience}: {self.lr}")
        #self.lr-=0.01*self.lr
        #print(f"lr has been decreased by {self.lr_decay_factor}. It is now {self.lr}")

        '''# TODO: this method is being tested!!!! Remember to remove if Training WITHOUT LOOP
        self.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl")
        print(f"Loaded {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Best_Epoch.pkl'} to continue training the next Experience on it --- from inside train_experience()")
        '''

    def step_lr(self, epoch):#def step_lr(self, epoch, lr_decay_every, lr_decay_factor):#def step_lr(self, epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
        """Handles step decay of learning rate."""
        factor = np.power(self.lr_decay_factor, np.floor((epoch - 1) / self.lr_decay_every))#factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
        new_lr = self.lr * factor#new_lr = base_lr * factor
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

        # ===== NEW: Expand SINGLE teacher to match student in SINGLE-TEACHER LwF =====
        if self.teacher is None:
            return False
        if self.teacher is not None:
            old_teacher_size = self.teacher.num_classes

            if old_teacher_size >= new_num_classes:
                # Teacher is already large enough, skip
                return False

            print(f"Expanding teacher {i} from {old_teacher_size} to {new_num_classes} classes")

            # Create new classifier for teacher
            teacher_new_classifier = nn.Linear(
                in_features=int(self.teacher.in_features),
                out_features=int(new_num_classes),
                bias=True
            ).to(device=device, dtype=dtype)

            with torch.no_grad():
                # Copy old teacher weights (what it learned)
                teacher_new_classifier.weight[:old_teacher_size].copy_(self.teacher.linear.weight)
                teacher_new_classifier.bias[:old_teacher_size].copy_(self.teacher.linear.bias)

                # Initialize new dimensions to 0 (teacher doesn't know about new classes)
                nn.init.constant_(teacher_new_classifier.weight[old_teacher_size:], 0.0)
                nn.init.constant_(teacher_new_classifier.bias[old_teacher_size:], 0.0)

            # Replace teacher's classifier
            self.teacher.linear = teacher_new_classifier
            self.teacher.num_classes = new_num_classes

            # Keep teacher frozen (no gradients)
            for param in self.teacher.parameters():
                param.requires_grad = False

        # Recreate optimizer with expanded student
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        print(f"Classifier and all teachers expanded successfully. New size: {new_num_classes}\n")
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
            self.eval_experience() # self._eval_exp(**kwargs)
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

        for self.mbatch in self.dataloader:#self.eval_dataloader:
            self.unpack_minibatch()
            targets = self.map_targets() #targets = self.map_labels_expanding_classifier()
            self._before_eval_iteration(**kwargs) # formerly commented-out

            self._before_eval_forward(**kwargs)# formerly commented-out
            self.mb_output = self.model(self.mb_x)#self.mb_output = self.forward()
            self._after_eval_forward(**kwargs)# formerly commented-out
            self.loss = self.criterion(self.mb_output, targets) #self.loss = self.criterion(self.mb_output, self.mb_y) #loss = self.criterion(self.mb_output, self.mb_y)

            self._after_eval_iteration(**kwargs) # formerly commented-out


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

        if current_exp_id_all_experiences not in self.experience_accuracies: #if current_exp_id_all_experiences not in self.experience_accuracies.keys():
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
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset, self.eval_mb_size)

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
                        targets_all_experiences = self.map_targets() #targets = self.map_labels_expanding_classifier() # targets = self.mb_y

                        student_logits_all_experiences = self.model(self.mb_x)
                        #print("print(self.model.training) -- evaluate all experiences after self.model(self.mb_x)",self.model.training)
                        loss_all_experiences = self.criterion(student_logits_all_experiences, targets_all_experiences)
                        #print("print(self.model.training) -- evaluate all experiences after self.criterion",self.model.training)

                        total_loss_all_experiences += loss_all_experiences.item() * targets_all_experiences.size(0)
                        predictions_all_experiences = torch.argmax(student_logits_all_experiences, dim=1)
                        total_correct_all_experiences += (predictions_all_experiences == targets_all_experiences).sum().item()
                        total_samples_all_experiences += targets_all_experiences.size(0)

                avg_loss_all_experiences = total_loss_all_experiences / total_samples_all_experiences if total_samples_all_experiences > 0 else 0.0
                accuracy_all_experiences = (total_correct_all_experiences / total_samples_all_experiences * 100) if total_samples_all_experiences > 0 else 0.0

                # Store results
                self.experience_accuracies[current_exp_id_all_experiences][exp_id_all_experiences] = accuracy_all_experiences
                self.experience_losses[current_exp_id_all_experiences][exp_id_all_experiences] = avg_loss_all_experiences

                print(
                    f"AFTER TRAINING ON Exp {current_exp_id_all_experiences}, "
                    f"performance on Exp {exp_id_all_experiences}: Loss = {avg_loss_all_experiences:.6f} | Accuracy = {accuracy_all_experiences:.2f}%"
                )

        # Compute backward transfer AFTER all evaluations
        self._compute_backward_transfer_internal()
        self._load_train_state(prev_train_state)
        self.model.train()
        print("print(self.model.training)-- evaluate_all_experiences after self.model.train() before exiting function", self.model.training)
        #self.model.to(self.device) # TODO: may be the problem

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
                    f"  Accuracy on Exp {test_exp_id} after training on Exp {trained_exp_id} = {final_acc:.2f}%\n"#f"  Accuracy at the end = {final_acc:.2f}%\n"
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
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset, self.eval_mb_size)

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
                        targets_epoch = self.map_targets() #targets = self.map_labels_expanding_classifier() # targets = self.mb_y

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
        #self.model.to(self.device)  # TODO: may be the problem
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
        print("print(self.model.training) -- evaluate_single_experience after self.model.eval()",self.model.training)

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
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset, self.eval_mb_size)

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
                        targets_single_exp = self.map_targets() #targets = self.map_labels_expanding_classifier() # targets = self.mb_y

                        student_logits_single_exp = self.model(self.mb_x)
                        loss_single_exp = self.criterion(student_logits_single_exp, targets_single_exp)

                        total_loss_single_exp += loss_single_exp.item() * targets_single_exp.size(0)
                        predictions_single_exp = torch.argmax(student_logits_single_exp, dim=1)
                        total_correct_single_exp += (predictions_single_exp == targets_single_exp).sum().item()
                        total_samples_single_exp += targets_single_exp.size(0)

                avg_loss_single_exp = total_loss_single_exp / total_samples_single_exp if total_samples_single_exp > 0 else 0.0
                accuracy_single_exp = (total_correct_single_exp / total_samples_single_exp * 100) if total_samples_single_exp > 0 else 0.0

                print(
                    f"Experience {current_exp_id_single_exp}, "
                    f"Loss = {avg_loss_single_exp:.6f} | Accuracy = {accuracy_single_exp:.2f}%"
                    f"\n{'...' * 60}\n"
                )
        self._load_train_state(prev_train_state)
        self.model.train()
        #self.model.to(self.device)  # TODO: may be the problem

    def make_eval_dataloader_for_evaluate_all_experiences_and_bwt(self,
            # experiences: Union[TExperienceType, Iterable[TExperienceType]],
            eval_adapted_dataset: Optional[Dataset],  # AvalancheDataset[T_co],
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
        targets_tensor = torch.tensor([class_to_idx_new[int(fine_label.item())]for fine_label in self.mb_y],
            device=self.mb_y.device, #fine_label.device,
            dtype=torch.long
        )
        #print("print(targets_tensor)",targets_tensor)
        return targets_tensor

    def map_targets_back_to_original(self):
        original_targets_tensor = torch.tensor([self.mb_y[int(fine_label)]for fine_label in class_to_idx_new],
            device=self.mb_y.device, #fine_label.device,
            dtype=torch.long
        )
        #print("print(original_targets_tensor)",original_targets_tensor)
        return original_targets_tensor

    def evaluate_all_experiences_post_training(self, eval_streams: Optional[
        Sequence[Union[TExperienceType, Iterable[TExperienceType]]]] = None):
        """Evaluate on all experiences seen so far."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATING ALL EXPERIENCES")
        print(f"{'=' * 60}\n")

        self.is_training = False
        self.model.eval()
        print("print(self.model.training) -- evaluate_all_experiences_post_training after self.model.eval()", self.model.training)

        for stream_idx, eval_stream in enumerate(eval_streams):
            if not isinstance(eval_stream, Iterable):
                eval_stream = [eval_stream]

            for exp in eval_stream:
                exp_id_all_experiences_post_training = exp.current_experience

                # Prepare eval data
                eval_adapted_dataset = exp.dataset.eval()
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset, self.eval_mb_size)

                total_loss_all_experiences_post_training = 0.0
                total_correct_all_experiences_post_training = 0
                total_samples_all_experiences_post_training = 0

                with torch.no_grad():
                    for self.mbatch in eval_dataloader:
                        self.unpack_minibatch()
                        targets_all_experiences_post_training = self.map_targets() #targets = self.map_labels_expanding_classifier() # targets = self.mb_y

                        student_logits_all_experiences_post_training = self.model(self.mb_x)
                        loss_all_experiences_post_training = self.criterion(student_logits_all_experiences_post_training, targets_all_experiences_post_training)

                        total_loss_all_experiences_post_training += loss_all_experiences_post_training.item() * targets_all_experiences_post_training.size(0)
                        predictions_all_experiences = torch.argmax(student_logits_all_experiences_post_training, dim=1)
                        total_correct_all_experiences_post_training += (predictions_all_experiences == targets_all_experiences_post_training).sum().item()
                        total_samples_all_experiences_post_training += targets_all_experiences_post_training.size(0)

                avg_loss_all_experiences_post_training = total_loss_all_experiences_post_training / total_samples_all_experiences_post_training if total_samples_all_experiences_post_training > 0 else 0.0
                accuracy_all_experiences_post_training = (total_correct_all_experiences_post_training / total_samples_all_experiences_post_training * 100) if total_samples_all_experiences_post_training > 0 else 0.0

                print(
                    f"Performance on Exp {exp_id_all_experiences_post_training}: Loss = {avg_loss_all_experiences_post_training:.6f} | Accuracy = {accuracy_all_experiences_post_training:.2f}%"
                )

    def _original_gradient_freezing(self):
        """
        Original gradient freezing logic from your LwFTrainer.
        This is the baseline for comparison.
        """
        # This is your original code from lines 828-872
        # (included here for reference)
        pass

    def get_batches(self, experience):
        """Dummy batch generator for example."""
        # This would iterate through actual data
        yield torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))

    def get_gradient_statistics(self) -> Dict:
        """
        Analyze gradient statistics to understand freezing effectiveness.

        Returns:
            Dictionary with statistics about gradient changes
        """
        stats = {
            'freeze_intensity_by_epoch': [],
            'gradient_norms_by_epoch': [],
            'loss_trajectory': []
        }

        # Extract freeze intensities
        for config in self.gradient_freeze_logs['freeze_config']:
            stats['freeze_intensity_by_epoch'].append({
                'shared': config.shared_layer_intensity,
                'classifier': config.old_classifier_intensity
            })

        stats['loss_trajectory'] = self.gradient_freeze_logs['loss']

        return stats

    '''def run_experiment(
            self,
            configs,
            experiences: Union[TExperienceType, Iterable[TExperienceType]],
            #exp_name: str,
            #freeze_strategy: str,
            freeze_config: Optional[Dict] = None,
            eval_streams: Optional[
                Sequence[Union[TExperienceType, Iterable[TExperienceType]]]
            ] = None,
            train_epochs: int = 10,
            ft_shared_after: int = 8,
            **kwargs,
    ) -> Dict:
        """
        Run a single experiment with given configuration.

        Args:
            exp_name: Name for this experiment
            freeze_strategy: Strategy name
            freeze_config: Strategy-specific configuration
            train_epochs: Number of training epochs
            ft_shared_after: Baseline threshold for freezing

        Returns:
            Dictionary with experiment results
            :param eval_streams:
        """
        for (exp_name, freeze_strategy, config) in configs:

            print(f"\n{'=' * 80}")
            print(f"Running Experiment: {exp_name}")
            print(f"Strategy: {freeze_strategy}")
            print(f"Config: {config}")#print(f"Config: {freeze_config}")
            print(f"{'=' * 80}\n")

            exp_name=exp_name
            freeze_strategy=freeze_strategy
            config=config #freeze_config=freeze_config

            # Create fresh model
            # model = self.trainer.model #self.model_factory()

            # Create trainer with specified strategy
            #trainer = LwFTrainer_Experimental(
             #   model=model,
             #   device=self.device,
             #   freeze_strategy=freeze_strategy,
             #   freeze_config=freeze_config,
             #   train_epochs=train_epochs,
             #   ft_shared_after=ft_shared_after
            #)
            #trainer = self.trainer

            # Run training (placeholder - actual implementation would use real data)
            # trainer.train_experience(experience)
            #DOES NOT WORK when .train() is called from here!!!! lwf_trainer.train(experiences, eval_streams)

            # Extract results
            results = {
                'experiment_name': exp_name,
                'strategy': freeze_strategy,
                'config': config,#freeze_config,
                'freeze_logs':self.gradient_freeze_logs,
                'gradient_stats': self.get_gradient_statistics()
            }

            self.results[exp_name] = results
        #return self.results[exp_name]
        #return results'''

    def compare_experiments(self) -> str:
        """
        Generate comparison summary of all experiments.

        Returns:
            String with formatted comparison results
        """
        lines = ["\n" + "=" * 80]
        lines.append("EXPERIMENT COMPARISON SUMMARY")
        lines.append("=" * 80 + "\n")

        lines.append("Experiment Name          | Strategy                | "
                     "Avg Freeze Intensity | Recommendation")
        lines.append("-" * 105)

        for exp_name, result in self.results.items():
            stats = result['gradient_stats']
            freeze_intensities = [
                config['shared'] for config in stats['freeze_intensity_by_epoch']
            ]
            avg_intensity = sum(freeze_intensities) / len(freeze_intensities)

            lines.append(
                f"{exp_name:24s} | {result['freeze_strategy']:23s} | "
                f"{avg_intensity:19.2f} | -"
            )

        lines.append("\n" + "=" * 80 + "\n")
        return "\n".join(lines)


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
        checkpoint_dir = Path("./checkpoints") # checkpoint_dir = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Twelfth_implementation_attempt\checkpoints")
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
#### Train_ResNet20_Nature_LwF_v12_L ###
######################################################################################################################################################
# BEST PERFORMING Nature model SO FAR!!!! 

if epoch_loss < prev_epoch_loss:
    loss_c+=1
    if loss_c >= self.patience:
        print(f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
        self.lr = self.lr * self.lr_decay_factor
        print(f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")

if epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
    # Both metrics improved - reset patience counter
    #prev_epoch_loss = epoch_loss
    #prev_epoch_acc = epoch_acc
    c = 0
elif epoch_acc < prev_epoch_acc:
    # No improvement - increment patience counter
    c += 1
    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

    if c >= self.patience:
        print(
            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
        self.lr = self.original_lr
        break
elif prev_epoch_acc>=75:
    print(
        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
    #c = self.patience
    self.lr = self.original_lr
    break

prev_epoch_loss = epoch_loss
prev_epoch_acc = epoch_acc
self.lr = self.original_lr



difference_ce_dist = (ce_loss.item() - dist_loss.item())
if difference_ce_dist > 0:
    div = (difference_ce_dist / 0.10)
    print(f"self.alpha BEFORE INCREASE: {self.alpha}\n") #NEVER TESTED if VERBOSE else None
    self.alpha = self.original_alpha + (0.10 * div)
    if VERBOSE:
        print(f"difference_ce_dist = (ce_loss - dist_loss) = {difference_ce_dist},\n "
              f"div = (difference_ce_dist/0.10) = {div},\n "
              f"Alpha has been increased by {(0.10 * div)} --> self.alpha = self.alpha + (0.10 * div) = {self.alpha},\n"
              f"self.alpha AFTER INCREASE: {self.alpha}\n"
              f"(self.alpha * dist_loss): {self.alpha * dist_loss}\n")


lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                         temperature=2.0, alpha=1.0, 
                         train_mb_size=43, train_epochs=10, eval_mb_size=43,
                         patience=5, ft_shared_after=10,
                         #plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream",metric_name="Loss_Exp", mode="min", peval_mode="epoch",margin=0.3, verbose=True)],
                         evaluator=EvaluationPlugin(loss_metrics(experience=True, stream=True),loggers=InteractiveLogger(), collect_all=True),
                         eval_every=1)


MATCH_AUTHOR_DISTILL = True, MATCH_AUTHOR_LOSS = True, VERBOSE = True

Accuracy Results:

After training Experience 0:
  - Tested on Experience 0: 79.67%

After training Experience 1:
  - Tested on Experience 0: 42.00%
  - Tested on Experience 1: 40.33%

After training Experience 2:
  - Tested on Experience 0: 39.33%
  - Tested on Experience 1: 32.33%
  - Tested on Experience 2: 7.33%

--------------------------------------------------------------------------------

Experience 0:
  Accuracy on Exp 0 right after training on Exp 0 = 79.67%
  Accuracy on Exp 0 after training on Exp 2 = 39.33%
  BWT = -40.33%

Experience 1:
  Accuracy on Exp 1 right after training on Exp 1 = 40.33%
  Accuracy on Exp 1 after training on Exp 2 = 32.33%
  BWT = -8.00%

Experience 2:
  Accuracy on Exp 2 right after training on Exp 2 = 7.33%
  Accuracy on Exp 2 after training on Exp 2 = 7.33%
  BWT = +0.00%

Mean Backward Transfer (BWT): -16.11%
  Positive BWT = improvement on old tasks (good)
  Negative BWT = forgetting on old tasks (bad)
  BWT ≈ 0 = stable performance (ideal)
================================================================================

-- >> End of training phase << --
Temperature: 2.0 Alpha: 3.4419757425785065
print(lwf_trainer.plugins) AFTER call to .train() [<avalanche.training.plugins.evaluation.EvaluationPlugin object at 0x000001DA7F2DBD90>, <avalanche.training.templates.base_sgd.PeriodicEval object at 0x000001DA09F43890>, <avalanche.training.plugins.clock.Clock object at 0x000001DA39D7F590>]
Training completed WITH LwFTrainer



######################################################################################################################################################################################################################################
# Same set-up as above (the one that gave BEST PERFORMING Nature model SO FAR!!!!), BUT WITH FIXED Early Stopping Loop!!! 
The early stopping logic is fixed, as I tested it in a different set-up (that did not give good results and hence the results were not 
copy-pasted into this list, but it served to test the new early stopping logic) and since this change is to be 
kept across set-ups, take this to be the BEST PERFORMING Nature model SO FAR!!!!

# BEST PERFORMING Nature model SO FAR (same as the one originally labelled best performing Nature model so far, but with updates that do not affect the results or parts of the code 
relevant in producing these results)!!!!

if epoch_loss >= prev_epoch_loss:
    loss_c+=1
    if loss_c == self.patience-1 : # and epoch_acc <= prev_epoch_acc: # and c == self.patience-1:
        print(f"lr before being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
        self.lr = self.lr * self.lr_decay_factor
        print(f"lr after being increased at Experience {experience.current_experience}, Epoch {_}: {self.lr}")
        c=0

if epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#if epoch_loss < prev_epoch_loss and epoch_acc > prev_epoch_acc and prev_epoch_acc<75:#and c >= self.patience:
    # Both metrics improved - reset patience counter
    #prev_epoch_loss = epoch_loss
    #prev_epoch_acc = epoch_acc
    c = 0
    loss_c=0
elif epoch_acc <= prev_epoch_acc:
    # No improvement - increment patience counter
    c += 1
    print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

    if c >= self.patience:
        print(
            f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
        self.lr = self.original_lr
        c=0
        loss_c = 0
        break
elif epoch_acc>=75:#elif prev_epoch_acc>=75:
    print(
        f"WARNING: The accuracy of experience {experience.current_experience} has reached 75%. Training will be stopped.")
    #c = self.patience
    self.lr = self.original_lr
    c = 0
    loss_c = 0
    break

prev_epoch_loss = epoch_loss
prev_epoch_acc = epoch_acc
self.lr = self.original_lr


difference_ce_dist = (ce_loss.item() - dist_loss.item())
if difference_ce_dist > 0:
    div = (difference_ce_dist / 0.10)
    #print(f"self.alpha BEFORE INCREASE: {self.alpha}\n") #NEVER TESTED if VERBOSE else None
    self.alpha = self.original_alpha + (0.10 * div)
    if VERBOSE:
        print(f"difference_ce_dist = (ce_loss - dist_loss) = {difference_ce_dist},\n "
              f"div = (difference_ce_dist/0.10) = {div},\n "
              f"Alpha has been increased by {(0.10 * div)} --> self.alpha = self.alpha + (0.10 * div) = {self.alpha},\n"
              f"self.alpha AFTER INCREASE: {self.alpha}\n"
              f"(self.alpha * dist_loss): {self.alpha * dist_loss}\n")

lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                             temperature=2.0, alpha=1.0, 
                             train_mb_size=43, train_epochs=10, eval_mb_size=43,
                             patience=5, ft_shared_after=10,

Accuracy Results:

After training Experience 0:
  - Tested on Experience 0: 79.67%

After training Experience 1:
  - Tested on Experience 0: 40.33%
  - Tested on Experience 1: 41.00%

After training Experience 2:
  - Tested on Experience 0: 40.00%
  - Tested on Experience 1: 36.33%
  - Tested on Experience 2: 8.33%


######################################################################################################################################################################################################################################
print(f"MATCH_AUTHOR_DISTILL = {MATCH_AUTHOR_DISTILL}, MATCH_AUTHOR_LOSS = {MATCH_AUTHOR_LOSS}, VERBOSE = {VERBOSE}")
if (self.teacher is not None) and (_ <= self.ft_shared_after) and (experience.current_experience>=2)

lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                         temperature=2.0, alpha=1.0, 
                         train_mb_size=43, train_epochs=10, eval_mb_size=43,
                         patience=5, ft_shared_after=8,
                         
Accuracy Results:

After training Experience 0:
  - Tested on Experience 0: 79.67%

After training Experience 1:
  - Tested on Experience 0: 39.67%
  - Tested on Experience 1: 48.00%

After training Experience 2:
  - Tested on Experience 0: 24.67%
  - Tested on Experience 1: 34.67%
  - Tested on Experience 2: 21.67%
  
-- >> End of training phase << --
Temperature: 2.0 Alpha: 3.2631507962942123
######################################################################################################################################################

'''


'''# ==================== EXPERIMENT MANAGER ====================

class ExperimentManager:
    """
    Manages running multiple experiments with different freeze strategies.

    Helps compare performance across configurations.
    """

    def __init__(self, trainer, device: str = "cpu"):  # def __init__(self, model_factory, device: str = "cpu"):
        """
        Initialize experiment manager.

        Args:
            model_factory: Callable that returns a fresh model instance
            device: Device to use for training
        """
        self.trainer: SupervisedTemplate = trainer
        # self.model_factory = model_factory
        self.device = device
        self.results = {}

    def run_experiment(
            self,
            exp_name: str,
            freeze_strategy: str,
            freeze_config: Optional[Dict] = None,
            #train_epochs: int = 10,
            #ft_shared_after: int = 8,
    ) -> Dict:
        """
        Run a single experiment with given configuration.

        Args:
            exp_name: Name for this experiment
            freeze_strategy: Strategy name
            freeze_config: Strategy-specific configuration
            train_epochs: Number of training epochs
            ft_shared_after: Baseline threshold for freezing

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'=' * 80}")
        print(f"Running Experiment: {exp_name}")
        print(f"Strategy: {freeze_strategy}")
        print(f"Config: {freeze_config}")
        print(f"{'=' * 80}\n")

        # Create fresh model
        # model = self.trainer.model #self.model_factory()

        # Create trainer with specified strategy
        ''''''trainer = LwFTrainer_Experimental(
            model=model,
            device=self.device,
            freeze_strategy=freeze_strategy,
            freeze_config=freeze_config,
            train_epochs=train_epochs,
            ft_shared_after=ft_shared_after
        )''''''
        #trainer = self.trainer
        self.trainer.freeze_strategy=freeze_strategy
        self.trainer.freeze_config = freeze_config
        #self.trainer.train_epochs = train_epochs
        #self.trainer.ft_shared_after = ft_shared_after


        # Run training (placeholder - actual implementation would use real data)
        # trainer.train_experience(experience)

        # Extract results
        results = {
            'experiment_name': exp_name,
            'strategy': freeze_strategy,
            'config': freeze_config,
            'freeze_logs': self.trainer.gradient_freeze_logs,
            'gradient_stats': self.trainer.get_gradient_statistics()
        }

        self.results[exp_name] = results
        return results

    def compare_experiments(self) -> str:
        """
        Generate comparison summary of all experiments.

        Returns:
            String with formatted comparison results
        """
        lines = ["\n" + "=" * 80]
        lines.append("EXPERIMENT COMPARISON SUMMARY")
        lines.append("=" * 80 + "\n")

        lines.append("Experiment Name          | Strategy                | "
                     "Avg Freeze Intensity | Recommendation")
        lines.append("-" * 105)

        for exp_name, result in self.results.items():
            stats = result['gradient_stats']
            freeze_intensities = [
                config['shared'] for config in stats['freeze_intensity_by_epoch']
            ]
            avg_intensity = sum(freeze_intensities) / len(freeze_intensities)

            lines.append(
                f"{exp_name:24s} | {result['freeze_strategy']:23s} | "
                f"{avg_intensity:19.2f} | -"
            )

        lines.append("\n" + "=" * 80 + "\n")
        return "\n".join(lines)'''


# ==================== EXAMPLE CONFIGURATIONS ====================

def get_experiment_configs() -> List[tuple]:
    """
    Return list of experiment configurations to test.

    Each tuple is (exp_name, freeze_strategy, config_dict)

    This is where you define different stability-plasticity tradeoffs.
    """
    configs = [
        # Baseline: Original binary freeze freeze_strategy
        ('baseline_binary', 'binary', {
            'freeze_before': 8
        }),

        # Experiment 1: Gradual unfreezing
        ('exp_gradual_early_unfreeze', 'gradual_unfreezing', {
            'start_epoch': 0,
            'end_epoch': 8,
            'unfreeze_shared_first': True
        }),

        ('exp_gradual_late_unfreeze', 'gradual_unfreezing', {
            'start_epoch': 2,
            'end_epoch': 10,
            'unfreeze_shared_first': False
        }),

        # Experiment 2: Two-phase freeze_strategy
        ('exp_two_phase_tight3', 'two_phase', {
            'tight_freeze_epochs': 3,
            'gradual_unfreeze_epochs': 5
        }),

        ('exp_two_phase_tight5', 'two_phase', {
            'tight_freeze_epochs': 5,
            'gradual_unfreeze_epochs': 3
        }),

        # Experiment 3: Exponential decay
        ('exp_exponential_fast', 'exponential_decay', {
            'decay_rate': 0.3,
            'half_life_epochs': 2.0
        }),

        ('exp_exponential_slow', 'exponential_decay', {
            'decay_rate': 0.7,
            'half_life_epochs': 4.0
        }),

        # Experiment 4: Layer-wise curriculum
        ('exp_layer_wise_3stages', 'layer_wise_curriculum', {
            'num_stages': 3,
            'freeze_order': 'deep_to_shallow'
        }),

        # Experiment 5: Classifier-only freezing
        ('exp_classifier_only', 'classifier_only', {
            'freeze_classifier': True,
            'freeze_shared': False
        }),

        # Experiment 6: Cyclic freezing
        ('exp_cyclic_period2', 'cyclic', {
            'cycle_length': 2
        }),

        ('exp_cyclic_period3', 'cyclic', {
            'cycle_length': 3
        }),
    ]
    return configs

def compute_average_accuracy(accuracies_per_task:Dict[int, Dict[int, float]]):#(accuracies_per_task:Dict[Any, Any]):#(accuracies_per_task):
    """
    Average accuracy after all tasks are learned
    """
    final_exp = max(accuracies_per_task.keys())
    all_accuracies = accuracies_per_task[final_exp].values()
    return sum(all_accuracies) / len(all_accuracies)


def compute_bwt(accuracies_per_task:Dict[int, Dict[int, float]]):#(accuracies_per_task:Dict[Any, Any]):#(accuracies_per_task):
    """
    BWT = mean(accuracy drop on old tasks after learning new ones)
    Higher is better (less forgetting)

    accuracies_per_task: Dict[exp_id][exp_id] = accuracy
    """
    print(f"accuracies_per_task = {accuracies_per_task}")
    bwt_values = []
    for exp_new in range(1, len(accuracies_per_task)):
        for exp_old in range(exp_new):
            # Accuracy on old task after seeing new task - accuracy on old task after learning it
            bwt = accuracies_per_task[exp_new][exp_old] - accuracies_per_task[exp_old][exp_old]
            bwt_values.append(bwt)
    return sum(bwt_values) / len(bwt_values)

def compute_fwt(accuracies_per_task:Dict[int, Dict[int, float]]):#(accuracies_per_task:Dict[Any, Any]):
    """
    FWT = mean(accuracy gain on new tasks from seeing old ones)
    """
    fwt_values = []
    for exp_new in range(1, len(accuracies_per_task)):
        # This would need zero-shot or oracle accuracies
        # For now, just new task accuracy
        fwt = accuracies_per_task[exp_new][exp_new]
        fwt_values.append(fwt)
    return sum(fwt_values) / len(fwt_values)




MATCH_AUTHOR_DISTILL = True
MATCH_AUTHOR_LOSS = True
EARLY_STOPPING_LOOP_1=False#True #TODO:Change back to EARLY_STOPPING_LOOP_1=True once I am done testing "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch...".
#                                  EARLY_STOPPING_LOOP_5 is identical to EARLY_STOPPING_LOOP_1, but with 85% limit accuracy instead of 75%, such that it is appropriate for
#                                  "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch..."
EARLY_STOPPING_LOOP_2=False
EARLY_STOPPING_LOOP_3=False
EARLY_STOPPING_LOOP_4=False
EARLY_STOPPING_LOOP_5=True #TODO:Change back to EARLY_STOPPING_LOOP_1=True once I am done testing "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch...".
#                            EARLY_STOPPING_LOOP_5 is identical to EARLY_STOPPING_LOOP_1, but with 85% limit accuracy instead of 75%, such that it is appropriate for
#                            "ALTERNATIVE CODE (in testing) --> EVALUATING Epoch..."
VERBOSE=False#True
#DYNAMIC_MODELS = True

DIFFERENCE_CE_DIST= True
#DIFFERENCE_CE_DIST= False

print(f"MATCH_AUTHOR_DISTILL = {MATCH_AUTHOR_DISTILL}, MATCH_AUTHOR_LOSS = {MATCH_AUTHOR_LOSS}, VERBOSE = {VERBOSE}")

if TRAINING_WITH_LOOP_LwF:

    #lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cuda", lr=0.001, momentum=0.9, weight_decay=5e-4,temperature=2.0, alpha=1.0)
    '''lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                             temperature=2.0, alpha=1.0, #temperature=2.0, alpha=1.0,#temperature=4.0, alpha=5.0,
                             train_mb_size=43, train_epochs=10, eval_mb_size=43, #eval_mb_size=32, #device=DEVICE,
                             patience=5, ft_shared_after=8,
                             #plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream",metric_name="Loss_Exp", mode="min", peval_mode="epoch",margin=0.3, verbose=True)],
                             evaluator=EvaluationPlugin(#accuracy_metrics(experience=True, stream=True),
                                                        loss_metrics(experience=True, stream=True),
                                                        #forgetting_metrics(experience=True, stream=True), bwt_metrics(experience=True, stream=True),
                                                        loggers=InteractiveLogger(), collect_all=True),
                             eval_every=1,)
                             #freeze_strategy='binary',  # NEW parameter
                             #freeze_config={'freeze_before': 8},) # NEW parameter
                             #freeze_strategy='gradual_unfreezing',
                             #freeze_config={'start_epoch': 0, 'end_epoch': 8, 'unfreeze_shared_first': True})
                             #freeze_strategy='two_phase',
                             #freeze_config={'tight_freeze_epochs': 3, 'gradual_unfreeze_epochs': 5},) # First 3 epochs: maximum stability, # Last 5 epochs: increasing plasticity

    print("print(lwf_trainer.plugins) AFTER initialisation, before entering training loop", lwf_trainer.plugins)'''

    #TODO: This one works for 1 experiment
    '''lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9,
                             weight_decay=5e-4, lr_decay_factor=0.1,
                             temperature=2.0, alpha=1.0,
                             train_mb_size=43, train_epochs=10, eval_mb_size=43,
                             patience=5, ft_shared_after=8,
                             # plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream",metric_name="Loss_Exp", mode="min", peval_mode="epoch",margin=0.3, verbose=True)],
                             evaluator=EvaluationPlugin(  # accuracy_metrics(experience=True, stream=True),
                                 loss_metrics(experience=True, stream=True),
                                 loggers=InteractiveLogger(), collect_all=True),
                             eval_every=1,)'''

    # Get experiment configurations
    configs = get_experiment_configs()

    print("\n" + "=" * 80)
    print("LwFTrainer Gradient Freeze Strategy Testing")
    print("=" * 80 + "\n")

    print(f"Running {len(configs)} experiments...\n")

    # Run all experiments
    for (exp_name, freeze_strategy, config) in configs:
        print('Starting experiment WITH LwFTrainer')
        #results = []
        #seen_experiences = [nature_specialised_models_train_stream_class_il[0]]
        #seen_experiences_val = [nature_specialised_models_val_stream_class_il[0]]
        #####lwf_trainer.model = ResNet20_DynamicModel(num_classes=0)
        seen_experiences = []
        seen_experiences_val = []
        checkpoint_dir_path = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Twelfth_implementation_attempt\checkpoints")

        print(f"\n{'=' * 80}")
        print(f"Running Experiment: {exp_name}")
        print(f"Strategy: {freeze_strategy}")
        print(f"Config: {config}")  # print(f"Config: {freeze_config}")
        print(f"{'=' * 80}\n")

        exp_name = exp_name
        freeze_strategy = freeze_strategy
        config = config  # freeze_config=freeze_config
        #lwf_trainer.freeze_strategy = freeze_strategy #TODO: This one works for 1 experiment
        #lwf_trainer.freeze_config = config#TODO: This one works for 1 experiment

        #nature_specialised_model = ResNet20_DynamicModel(num_classes=0) # DOES NOT WORK!!!



        lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                                 temperature=2.0, alpha=1.0,
                                 train_mb_size=43, train_epochs=10, eval_mb_size=43,
                                 patience=5, ft_shared_after=8,
                                 # plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream",metric_name="Loss_Exp", mode="min", peval_mode="epoch",margin=0.3, verbose=True)],
                                 evaluator=EvaluationPlugin(# accuracy_metrics(experience=True, stream=True),
                                     loss_metrics(experience=True, stream=True),
                                     loggers=InteractiveLogger(), collect_all=True),
                                 eval_every=1,
                                 freeze_strategy=freeze_strategy, freeze_config=config)

        print(f"TEST TO MAKE SURE BRAND NEW lwf_trainer IS CREATED FOR EACH EXPERIMENT:"
              #f"lwf_trainer.model = {lwf_trainer.model}"
              f"lwf_trainer.freeze_strategy = {lwf_trainer.freeze_strategy}\n"
              f"lwf_trainer.freeze_config = {lwf_trainer.freeze_config}\n"
              #f"lwf_trainer.train_epochs = {lwf_trainer.train_epochs}"
              #f"lwf_trainer.ft_shared_after = {lwf_trainer.ft_shared_after}"
              #f"lwf_trainer.patience = {lwf_trainer.patience}"
              #f"lwf_trainer.lr_decay_factor = {lwf_trainer.lr_decay_factor}"
              f"lwf_trainer.alpha = {lwf_trainer.alpha}\n"
              #f"lwf_trainer.temperature = {lwf_trainer.temperature}"
              #f"lwf_trainer.train_mb_size = {lwf_trainer.train_mb_size}"
              #f"lwf_trainer.eval_mb_size = {lwf_trainer.eval_mb_size}"
              #f"lwf_trainer.eval_every = {lwf_trainer.eval_every}"
              #f"lwf_trainer.device = {lwf_trainer.device}"
              #f"lwf_trainer.plugins = {lwf_trainer.plugins}"
              #f"lwf_trainer.evaluator = {lwf_trainer.evaluator}"
              f"lwf_trainer.experience_losses = {lwf_trainer.experience_losses}\n"
              f"lwf_trainer.experience_accuracies = {lwf_trainer.experience_accuracies}\n"
              f"lwf_trainer.num_new_classes = {lwf_trainer.num_new_classes}\n"
              f"lwf_trainer.experience = {lwf_trainer.experience}\n"
              f"lwf_trainer.optimizer = {lwf_trainer.optimizer}\n"
              f"lwf_trainer.original_alpha = {lwf_trainer.original_alpha}\n"
              f"lwf_trainer.teacher_classes = {lwf_trainer.teacher_classes}\n"
              f"lwf_trainer.teacher = {lwf_trainer.teacher}\n")


        for experience in nature_specialised_models_train_stream_class_il: # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
            i = experience.current_experience # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of nature_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
            if i == 3:#4:#3:
                break
            print("Start of experience -- current TRAIN experience: ", experience.current_experience)
            print("Start of experience -- current VAL experience: ",nature_specialised_models_val_stream_class_il.exps_iter[i].current_experience)
            print("Current Classes -- current TRAIN .classes_in_this_experience: ", experience.classes_in_this_experience)
            print("Current Classes -- current VAL .classes_in_this_experience: ",nature_specialised_models_val_stream_class_il.exps_iter[i].classes_in_this_experience)
            print("experience.dataset.coarse_label.uniques (coarse_labels of TRAIN)",experience.dataset.coarse_label.uniques)
            print("experience.dataset.targets (targets of TRAIN)", experience.dataset.targets.uniques)
            print("nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques (coarse_labels of VAL)",nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques)
            print("nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets (targets of VAL)",nature_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques)
            print("print(experience.origin_stream.name)", experience.origin_stream.name)
            print("print(nature_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)",nature_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)
            # print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
            # print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
            # criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
            # print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
            print("print(lwf_trainer.plugins) BEFORE call to .train()", lwf_trainer.plugins)
            seen_experiences.append(experience)
            seen_experiences_val.append(nature_specialised_models_val_stream_class_il[i])
            '''if i > 0:
                lwf_trainer.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{i-1}_Best_Epoch.pkl")
                print(f"Loaded {f'./checkpoints/checkpoint_Experience_{i-1}_Best_Epoch.pkl'} to continue training the next Experience on it --- from inside train_experience()")'''
            ####lwf_trainer.train(experience, [seen_experiences])
            lwf_trainer.train(experience, [seen_experiences,seen_experiences_val])#lwf_trainer.train(experience, [nature_specialised_models_train_stream_class_il,nature_specialised_models_val_stream_class_il])
            print("Temperature:" ,lwf_trainer.temperature, "Alpha:", lwf_trainer.alpha)

            '''if i>0:
                print("compute_average_accuracy(lwf_trainer.experience_accuracies)")
                compute_average_accuracy(lwf_trainer.experience_accuracies)
                print("compute_bwt(lwf_trainer.experience_accuracies)")
                compute_bwt(lwf_trainer.experience_accuracies)
                print("compute_fwt(lwf_trainer.experience_accuracies)")
                compute_fwt(lwf_trainer.experience_accuracies)'''
            #lwf_trainer.train(experience, nature_specialised_models_val_stream_class_il)
            #lwf_trainer.train_experience(experience, nature_specialised_models_val_stream_class_il)
            #lwf_trainer.train_experience(experience, [nature_specialised_models_train_stream_class_il,nature_specialised_models_val_stream_class_il])
            #print("print(lwf_trainer.plugins) AFTER call to .train()", lwf_trainer.plugins)
            # print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
            # print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)

        # Extract results
        results = {
            'experiment_name': exp_name,
            'freeze_strategy': freeze_strategy,
            'config': config,  # freeze_config,
            'freeze_logs': lwf_trainer.gradient_freeze_logs,
            'gradient_stats': lwf_trainer.get_gradient_statistics()
        }

        lwf_trainer.results[exp_name] = results
        print("lwf_trainer.results[exp_name]",lwf_trainer.results[exp_name])

        #NONE EXISTING CODE, but follow instructions of how to implement it in QuickReference_LwFTrainer_Modifications.md --> print(f"{freeze_strategy:25s} | BWT: {results['BWT']:.4f} | AvgAcc: {results['AvgAcc']:.4f}")

        '''print("compute_average_accuracy(lwf_trainer.experience_accuracies)")
        compute_average_accuracy(lwf_trainer.experience_accuracies)
        print("compute_bwt(lwf_trainer.experience_accuracies)")
        compute_bwt(lwf_trainer.experience_accuracies)
        print("compute_fwt(lwf_trainer.experience_accuracies)")
        compute_fwt(lwf_trainer.experience_accuracies)'''

        # return self.results[exp_name]
        # return results
        '''lwf_trainer.run_experiment(
            configs=configs,
            experiences=experience,
            #exp_name=exp_name,
            #freeze_strategy=freeze_strategy,
            #freeze_config=config,
            eval_streams=[seen_experiences, seen_experiences_val]
            # train_epochs=10,
            # ft_shared_after=8
        )'''

        '''print("\n" + "=" * 80)
        print("LwFTrainer Gradient Freeze Strategy Testing")
        print("=" * 80 + "\n")

        # Get experiment configurations
        configs = get_experiment_configs()

        print(f"Running {len(configs)} experiments...\n")'''

        # Run all experiments
        '''for exp_name, freeze_strategy, config in configs:
            try:
                lwf_trainer.run_experiment(
                    experiences=experience,
                    exp_name=exp_name,
                    freeze_strategy=freeze_strategy,
                    freeze_config=config,
                    eval_streams=[seen_experiences, seen_experiences_val]
                    # train_epochs=10,
                    # ft_shared_after=8
                )
            except Exception as e:
                print(f"Error in experiment {exp_name}: {e}")'''

        '''lwf_trainer.run_experiment(
            configs=configs,
            experiences=experience,
            #exp_name=exp_name,
            #freeze_strategy=freeze_strategy,
            #freeze_config=config,
            eval_streams=[seen_experiences, seen_experiences_val]
            # train_epochs=10,
            # ft_shared_after=8
        )''' # DOES NOT WORK when placed here

        # Print comparison
        print(lwf_trainer.compare_experiments())
        #DOES NOT WORK! lwf_trainer.model = ResNet20_DynamicModel(num_classes=0)
        nature_specialised_model = ResNet20_DynamicModel(num_classes=0) # WORKS !!!

    print('Training completed WITH LwFTrainer')
    print("lwf_trainer.results",lwf_trainer.results)
    '''print("lwf_trainer.results[exp_name]",lwf_trainer.results[exp_name])
    print("lwf_trainer.results[exp_name]['gradient_stats']",lwf_trainer.results[exp_name]['gradient_stats'])
    print("lwf_trainer.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch']",lwf_trainer.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch'])
    print("lwf_trainer.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch'][0]",lwf_trainer.results[exp_name]['gradient_stats']['freeze_intensity_by_epoch'][0])'''
    #print("lwf_trainer.evaluate_all_experiences(nature_specialised_models_val_stream_class_il)")
    #lwf_trainer.evaluate_all_experiences_post_training(nature_specialised_models_val_stream_class_il)#(seen_experiences_val)

'''
    checkpoint_files = sorted(checkpoint_dir_path.glob("*.pkl"))
    for checkpoint_file in checkpoint_files:
        print(f"Loading {checkpoint_file}")
        cl_strategy_evaluation = load_entire_checkpoint(checkpoint_file)
        print(f"Computing val accuracy on {checkpoint_file.name} AFTER 'Training completed WITH LOOP VERSION 2'")
        cl_strategy_evaluation.eval(nature_specialised_models_val_stream_class_il)

'''
    #print('Computing accuracy on the whole val set OUTSIDE training LOOP')
    #cl_strategy.eval(nature_specialised_models_val_stream_class_il)
    #results.append(cl_strategy.eval(nature_specialised_models_val_stream_class_il))
    #print("Computing accuracy on the whole val set OUTSIDE training LOOP (i.e. print(results)):", results)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    '''print("\n" + "=" * 80)
    print("LwFTrainer Gradient Freeze Strategy Testing")
    print("=" * 80 + "\n")'''

    # Create a simple model factory
    '''def model_factory():
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )'''

    '''# Initialize experiment manager
    manager = ExperimentManager(lwf_trainer, device="cpu")  # manager = ExperimentManager(model_factory, device="cpu")

    # Get experiment configurations
    configs = get_experiment_configs()

    print(f"Running {len(configs)} experiments...\n")

    # Run all experiments
    for exp_name, freeze_strategy, config in configs:
        try:
            manager.run_experiment(
                exp_name=exp_name,
                freeze_strategy=freeze_strategy,
                freeze_config=config,
                #train_epochs=10,
                #ft_shared_after=8
            )
        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")

    # Print comparison
    print(manager.compare_experiments())

    print("\nNext Steps:")
    print("-" * 80)
    print("1. Implement actual data loading in LwFTrainer_Experimental")
    print("2. Run experiments on your Nature dataset")
    print("3. Track BWT (Backward Transfer), FWT (Forward Transfer), and AvgAcc")
    print("4. Identify best freeze_strategy for your problem")
    print("5. Fine-tune hyperparameters of best freeze_strategy")
    print("6. Validate on test set")'''

