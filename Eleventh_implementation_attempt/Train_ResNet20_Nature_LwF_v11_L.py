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

from Eleventh_implementation_attempt.Config_v11 import set_seed
set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Eleventh_implementation_attempt.Config_v11 import set_device
DEVICE=set_device()
from Eleventh_implementation_attempt.Config_v11 import SEED
from avalanche.training.determinism.rng_manager import RNGManager
RNGManager.set_random_seeds(SEED)
from Eleventh_implementation_attempt.ResNet20_v11 import ResNet20
from Eleventh_implementation_attempt.ResNet20_DynamicModel_v11 import ResNet20_DynamicModel
from Eleventh_implementation_attempt.Load_cifar100_v11 import (outdoors_specialised_models_train_subset)
from Eleventh_implementation_attempt.Load_cifar100_v11 import (outdoors_specialised_models_val_subset)
from Eleventh_implementation_attempt.Load_cifar100_v11 import (outdoors_specialised_models_test_subset)
from Eleventh_implementation_attempt.Load_cifar100_v11 import (nature_specialised_models_train_subset)
from Eleventh_implementation_attempt.Load_cifar100_v11 import (nature_specialised_models_val_subset)
from Eleventh_implementation_attempt.Load_cifar100_v11 import (nature_specialised_models_test_subset)
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


def stable_softmax(x):
    z = x - torch.max(x, dim=1, keepdim=True)[0]
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    softmax = numerator / denominator
    return softmax


def cross_entropy_with_oh_targets(outputs, targets, reduction="mean"):
    """Calculates cross-entropy with temperature scaling,
    targets can also be soft targets but they must sum to 1"""
    outputs = stable_softmax(outputs)
    ce = -(targets * outputs.log()).sum(1)
    if reduction == "mean":
        ce = ce.mean()
    elif reduction == "none":
        return ce
    else:
        raise NotImplementedError("reduction must be mean or none")
    return ce


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

DEBUG = False #TODO: Remember to change DEBUG to False if running experiments.
TRAINING_WITH_LOOP = False #TODO: Remember to change TRAINING_WITH_LOOP to True if 'Training WITH LOOP VERSION 2'.
TRAINING_WITHOUT_LOOP = False #True #TODO: Remember to change TRAINING_WITHOUT_LOOP to True if 'Training WITHOUT LOOP'.
TRAINING_WITH_LOOP_LwF = True
DYNAMIC_MODELS = True
print("\n\n\n################################ HERE BEGINS Train_ResNet20_Outdoors_v11.py ################################ \n\n\n")

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

nature_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[3, 3, 3, 3, 85]) # which is equivalent to num_experiences=5 due to the reasons explained in previous version)

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
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch"
    ):
        self.patience = patience
        self.original_alpha = alpha
        self.original_lr = lr
        self.class_to_idx = {}

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

        self.model.train()
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
            self.evaluate_all_experiences(self.experience, eval_streams)
            self._after_training_exp(**kwargs)
        self._after_training(**kwargs)
        self._train_cleanup()

    def train_experience(self, experience: CLExperience, eval_streams=None, **kwargs):
        c = 0
        loss_c = 0
        prev_epoch_loss = 0
        prev_epoch_acc = 0

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]
        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            #prev_epoch_loss, prev_epoch_acc = self.evaluate_epoch(_, experience)

            print(f"\n\n***** MY train_experience() IS EXECUTING FOR Experience {experience.current_experience} *****\n\n")
            #self.optimizer = self.step_lr(_) # optimizer = self.step_lr( _, self.args.lr, self.args.lr_decay_every,self.args.lr_decay_factor,self.optimizer.param_groups)

            print(f"\n***** EXECUTING Epoch {_} OF Experience {experience.current_experience} --- from inside train_experience() *****\n")
            total_ce = 0.0
            total_dist = 0.0
            total_loss = 0.0
            n=0

            for self.mbatch in self.dataloader:#self.train_dataloader: # for batch, targets in train_dataloader:

                self.unpack_minibatch()
                targets = self.map_labels_expanding_classifier() #targets = self.map_targets()

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
                #print("dist_loss = torch.tensor(0.0,device=self.device)")
                '''dist_loss = torch.tensor(
                    0.0,
                    device=self.device
                )'''
                '''print("dist_loss = torch.tensor(total_dist,device=self.device)")
                dist_loss = torch.tensor(
                    total_dist,
                    device=self.device
                )'''
                if self.teacher is not None:
                    #for classes in self.teacher_classes:
                    with torch.no_grad():
                        teacher_logits = self.teacher(self.mb_x)  # self.teacher(self.mbatch)#teacher_logits = self.teacher(batch)

                        # Only old classes
                        #classes = sorted(self.teacher_classes)
                    for classes in self.teacher_classes:
                        class_indices = torch.tensor(
                            [
                                self.class_to_idx[c]
                                for c in sorted(classes)#(self.teacher_classes)
                            ],
                            device=self.device,
                            dtype=torch.long
                        )
                    if VERBOSE:
                        #print(f"  Classes (original): {sorted(classes)}")#{sorted(self.teacher_classes)}")
                        print(f"  Class indices (mapped): {class_indices.tolist()}")
                        print(f"  Student output shape: {self.mb_output.shape}")
                        print(f"  Teacher output shape: {teacher_logits.shape}")

                    #student_old = self.mb_output[:, class_indices]#student_logits[:, old_classes]
                    #teacher_old = teacher_logits[:, class_indices]

                    # ALTERNATE CODE to achieve the same thing -- try if you suspect the above two lines do not work as intended
                    #student_old = self.mb_output.index_select(1,class_indices)  # student_logits.index_select(1, classes)
                    #teacher_old = teacher_logits.index_select(1, class_indices)

                    if VERBOSE:
                        #print(f"  Student old shape: {student_old.shape}")
                        print(f"  scale used in dist_loss: {self.mb_output.size(1)}")#{student_old.size(1)}
                        #print(f"  Teacher old shape: {teacher_old.shape}")

                    scale = self.mb_output.size(1)

                    '''dist_loss = self.distillation_loss(
                        student_old,
                        teacher_old,
                        scale=scale
                    )'''
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

                    difference_ce_dist = (ce_loss.item() - dist_loss.item())
                    if difference_ce_dist > 0:
                        div = (difference_ce_dist / 0.10)#div = (difference_ce_dist / 0.20)
                        print(f"self.alpha BEFORE INCREASE: {self.alpha}\n") #NEVER TESTED if VERBOSE else None
                        self.alpha = self.original_alpha + (0.10 * div)
                        if VERBOSE:
                            print(f"difference_ce_dist = (ce_loss - dist_loss) = {difference_ce_dist},\n "
                                  f"div = (difference_ce_dist/0.10) = {div},\n "
                                  f"Alpha has been increased by {(0.10 * div)} --> self.alpha = self.alpha + (0.10 * div) = {self.alpha},\n"
                                  f"self.alpha AFTER INCREASE: {self.alpha}\n"
                                  f"(self.alpha * dist_loss): {self.alpha * dist_loss}\n")

                    if MATCH_AUTHOR_LOSS:
                        loss = ce_loss + (self.alpha * dist_loss)
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
                print(f"  Backpropagated loss: {loss.item():.6f}\n Backpropagation is occurring!")

                if (self.teacher is not None) and (_ <= self.ft_shared_after):#if ((len(self.teachers)>0) and (_ <= self.ft_shared_after)):

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
                            ].zero_()

                self.optimizer.step()

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                total_ce += ce_loss.item()
                total_dist += dist_loss.item()
                total_loss += loss.item()

                n += 1 #n = len(self.dataloader)

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

            # TODO: Commented-out for testing -- remember to uncomment later
            # TODO: uncomment whenever I want to save checkpoint per epoch to see at what point EWC stops mitigating CF for a particular experience
            '''self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl")
            print(f"Saved {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_experience()")
            '''
            # TODO: this method is being tested!!!! Remember to remove if Training WITHOUT LOOP
            '''self.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl")
            print(f"Loaded {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_epoch()")
            '''
            epoch_loss, epoch_acc = self.evaluate_epoch(_, experience)
            #self.alpha = self.original_alpha
            #print(f"Alpha has been reset to {self.original_alpha}. It is now {self.alpha}")
            self._after_training_epoch(**kwargs)
            '''while prev_epoch_loss > epoch_loss and prev_epoch_acc > epoch_acc:
                print(f"WARNING: Loss and accuracy did not improve during the previous epoch. This may be due to the fact that the loss function is not monotonically decreasing. The current epoch will be skipped.")
                prev_epoch_loss = epoch_loss
                prev_epoch_acc = epoch_acc
                c += 1
                if c >= self.patience:
                    print(f"WARNING: The number of epochs equivalent to self.patience ({self.patience}) has been reached without improvement. Training will be stopped.")
                    break'''
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
        # --------------------------------------------------------
        # IMPORTANT:
        # Save teacher AFTER completing the experience
        # --------------------------------------------------------
        #teacher = copy.deepcopy(self.model)
        #teacher.eval()
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

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
        num_new_classes = len(experience_classes - self.old_classes)

        if num_new_classes <= 0:
            print(f"No new classes. Classifier remains at {model.num_classes}")
            return False

        old_num_classes = model.num_classes
        new_num_classes = old_num_classes + num_new_classes

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
            targets = self.map_labels_expanding_classifier() #targets = self.map_targets()
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

        current_exp_id = experience.current_experience

        if current_exp_id not in self.experience_accuracies:
            self.experience_accuracies[current_exp_id] = {}
            self.experience_losses[current_exp_id] = {}

        if eval_streams is None:
            eval_streams = [experience]

        self.model.eval()

        for stream_idx, eval_stream in enumerate(eval_streams):
            if not isinstance(eval_stream, Iterable):
                eval_stream = [eval_stream]

            for exp in eval_stream:
                exp_id = exp.current_experience

                # Don't evaluate future experiences
                if exp_id > current_exp_id:
                    break

                # Prepare eval data
                eval_adapted_dataset = exp.dataset.eval()
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset, self.eval_mb_size)

                total_loss = 0.0
                total_correct = 0
                total_samples = 0

                with torch.no_grad():
                    for self.mbatch in eval_dataloader:
                        self.unpack_minibatch()
                        targets = self.map_labels_expanding_classifier() #targets = self.map_targets() # targets = self.mb_y

                        student_logits = self.model(self.mb_x)
                        loss = self.criterion(student_logits, targets)

                        total_loss += loss.item() * targets.size(0)
                        predictions = torch.argmax(student_logits, dim=1)
                        total_correct += (predictions == targets).sum().item()
                        total_samples += targets.size(0)

                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0

                # Store results
                self.experience_accuracies[current_exp_id][exp_id] = accuracy
                self.experience_losses[current_exp_id][exp_id] = avg_loss

                print(
                    f"AFTER TRAINING ON Exp {current_exp_id}, "
                    f"performance on Exp {exp_id}: Loss = {avg_loss:.6f} | Accuracy = {accuracy:.2f}%"
                )

        # Compute backward transfer AFTER all evaluations
        self._compute_backward_transfer_internal()

    def _compute_backward_transfer_internal(self):
        """Compute and print backward transfer after evaluation."""
        print(f"\n{'=' * 80}")
        print(f"BACKWARD TRANSFER ANALYSIS")
        print(f"{'=' * 80}\n")

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

        current_exp_id = experience.current_experience

        if current_exp_id not in self.experience_accuracies:
            self.experience_accuracies[current_exp_id] = {}
            self.experience_losses[current_exp_id] = {}

        if eval_streams is None:
            eval_streams = [experience]

        self.model.eval()

        for stream_idx, eval_stream in enumerate(eval_streams):
            if not isinstance(eval_stream, Iterable):
                eval_stream = [eval_stream]

            for exp in eval_stream:
                exp_id = exp.current_experience

                # Don't evaluate future experiences
                if exp_id > current_exp_id:
                    break

                # Prepare eval data
                eval_adapted_dataset = exp.dataset.eval()
                eval_dataloader = self.make_eval_dataloader_for_evaluate_all_experiences_and_bwt(eval_adapted_dataset, self.eval_mb_size)

                total_loss = 0.0
                total_correct = 0
                total_samples = 0

                with torch.no_grad():
                    for self.mbatch in eval_dataloader:
                        self.unpack_minibatch()
                        targets = self.map_labels_expanding_classifier() #targets = self.map_targets() # targets = self.mb_y

                        student_logits = self.model(self.mb_x)
                        loss = self.criterion(student_logits, targets)

                        total_loss += loss.item() * targets.size(0)
                        predictions = torch.argmax(student_logits, dim=1)
                        total_correct += (predictions == targets).sum().item()
                        total_samples += targets.size(0)

                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0

                print(
                    f"Experience {current_exp_id}, "
                    f"Epoch {epoch}: Loss = {avg_loss:.6f} | Accuracy = {accuracy:.2f}%"
                    f"\n{'...' * 60}\n"
                )
                return avg_loss, accuracy

    def make_eval_dataloader_for_evaluate_all_experiences_and_bwt(self,
            # experiences: Union[TExperienceType, Iterable[TExperienceType]],
            eval_adapted_dataset: Optional[Dataset],  # AvalancheDataset[T_co],
            batch_size: int,
            num_workers=0,
            shuffle=True,
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

    def map_labels_expanding_classifier(self):
        for fine_label in sorted(self.mb_y): #self.mb_task_id:#for fine_label in nature_specialised_models_train_stream_class_il.dataset.targets.uniques:
            #self.mb_y.
            fine_label_item = fine_label.item()
            if fine_label_item not in self.class_to_idx.keys():
                self.class_to_idx[fine_label_item] = len(self.class_to_idx)

        return torch.tensor(
            [
                self.class_to_idx[int(fine_label.item())]
                for fine_label in self.mb_y
            ],
            device=fine_label.device,
            dtype=torch.long
        )

        '''for fine_label in sorted(self.experience.targets.uniques):
            if fine_label not in self.class_to_idx.keys():
                self.class_to_idx[fine_label] = len(self.class_to_idx)'''

    '''def map_targets(self):
        return torch.tensor(
            [
                self.class_to_idx[int(fine_label.item())]
                for fine_label in self.mb_y
            ],
            device=self.mb_y.device, #fine_label.device,
            dtype=torch.long
        )'''

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
            "exp_counter": ended_experience_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
            #"ce_loss" : self.ce_loss,
            #"_criterion" : self._criterion,
            "clock_train_exp_counter": self.clock.train_exp_counter,
            "clock_train_epoch_counter": self.clock.train_exp_epochs,
            "teacher_state_dict": (
                self.teacher.state_dict()
                if self.teacher is not None
                else None
            ),
        }
        #with open(path, "wb") as f:
         #   pickle.dump(checkpoint, f)
        #torch.save(checkpoint, path)
        checkpoint_dir = Path("./checkpoints") # checkpoint_dir = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Eleventh_implementation_attempt\checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, fname, pickle_module=dill)

    def load_entire_checkpoint_inside_class(self, fname: FILE_LIKE, map_location=None, unique_objects: Any = None):  # , device):
        checkpoint = torch.load(fname, pickle_module=dill, map_location=map_location)
        saved_strategy = checkpoint["strategy"] # NOT self, that would be wrong. Not used within function, but kept in case saved_strategy needs to be retrieved at some point, perhaps
                                                # even through an external call on an instance / object of class EWCStrategy

        self.model.load_state_dict(checkpoint["model_state_dict"])
        # strategy.model.to(device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #self.ce_loss = checkpoint["ce_loss"]
        #self.loss = checkpoint["loss"]
        #self._criterion = checkpoint["_criterion"]
        self.clock = checkpoint["strategy"].clock
        self.clock.train_exp_counter = checkpoint["clock_train_exp_counter"]
        self.clock.train_exp_epochs = checkpoint["clock_train_epoch_counter"]
        self.train_mb_size = checkpoint["strategy"].train_mb_size
        self.train_epochs = checkpoint["strategy"].train_epochs
        self.eval_mb_size = checkpoint["strategy"].eval_mb_size
        self.device = checkpoint["strategy"].device
        self.plugins = checkpoint["strategy"].plugins
        self.evaluator = checkpoint["strategy"].evaluator
        self.eval_every = checkpoint["strategy"].eval_every
        self.dataloader = checkpoint["strategy"].dataloader
        self.experience = checkpoint["strategy"].experience
        self.teacher=checkpoint["strategy"].teacher
        (self.teacher.load_state_dict(checkpoint["teacher_state_dict"])
         if self.teacher is not None
                else None),
        self.old_classes = checkpoint["strategy"].old_classes
        RNGManager.__setstate__(checkpoint["rng_manager_state"])
        #RNGManager.__setstate__(checkpoint["rng_manager"])
        print(f"Executing load_entire_checkpoint_inside_class from call made from within _train_exp()")
        return self
    '''
'''
######################################################################################################################################################
#### Train_ResNet20_Nature_LwF_v11_L ###
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
######################################################################################################################################################

'''
MATCH_AUTHOR_DISTILL = True#False#True
MATCH_AUTHOR_LOSS = True#False #True
VERBOSE=True #False
#DYNAMIC_MODELS = True
print(f"MATCH_AUTHOR_DISTILL = {MATCH_AUTHOR_DISTILL}, MATCH_AUTHOR_LOSS = {MATCH_AUTHOR_LOSS}, VERBOSE = {VERBOSE}")

if TRAINING_WITH_LOOP_LwF:

    #lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cuda", lr=0.001, momentum=0.9, weight_decay=5e-4,temperature=2.0, alpha=1.0)
    lwf_trainer = LwFTrainer(model=nature_specialised_model, device="cpu", lr=0.01, momentum=0.9, weight_decay=5e-4, lr_decay_factor=0.1,
                             temperature=2.0, alpha=1.0, #temperature=2.0, alpha=1.0,#temperature=4.0, alpha=5.0,
                             train_mb_size=43, train_epochs=10, eval_mb_size=43, #eval_mb_size=32, #device=DEVICE,
                             patience=5, ft_shared_after=10,
                             #plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream",metric_name="Loss_Exp", mode="min", peval_mode="epoch",margin=0.3, verbose=True)],
                             evaluator=EvaluationPlugin(#accuracy_metrics(experience=True, stream=True),
                                                        loss_metrics(experience=True, stream=True),
                                                        #forgetting_metrics(experience=True, stream=True), bwt_metrics(experience=True, stream=True),
                                                        loggers=InteractiveLogger(), collect_all=True),
                             eval_every=1)

    print("print(lwf_trainer.plugins) AFTER initialisation, before entering training loop", lwf_trainer.plugins)

    print('Starting experiment WITH LwFTrainer')
    results = []
    #seen_experiences = [nature_specialised_models_train_stream_class_il[0]]
    #seen_experiences_val = [nature_specialised_models_val_stream_class_il[0]]
    seen_experiences = []
    seen_experiences_val = []
    checkpoint_dir_path = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Eleventh_implementation_attempt\checkpoints")
    for experience in nature_specialised_models_train_stream_class_il: # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
        i = experience.current_experience # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of nature_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
        if i == 3:
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
        lwf_trainer.train(experience, [seen_experiences,seen_experiences_val])#lwf_trainer.train(experience, [nature_specialised_models_train_stream_class_il,nature_specialised_models_val_stream_class_il])
        print("Temperature:" ,lwf_trainer.temperature, "Alpha:", lwf_trainer.alpha)
        #lwf_trainer.train(experience, nature_specialised_models_val_stream_class_il)
        #lwf_trainer.train_experience(experience, nature_specialised_models_val_stream_class_il)
        #lwf_trainer.train_experience(experience, [nature_specialised_models_train_stream_class_il,nature_specialised_models_val_stream_class_il])
        print("print(lwf_trainer.plugins) AFTER call to .train()", lwf_trainer.plugins)
        # print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
        # print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)
    print('Training completed WITH LwFTrainer')
    #lwf_trainer.eval(nature_specialised_models_val_stream_class_il)#(seen_experiences_val)


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




