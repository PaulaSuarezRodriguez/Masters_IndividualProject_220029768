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

from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.training.plugins.checkpoint import CheckpointPlugin
from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate

from Tenth_implementation_attempt.Config_v10 import set_seed
set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Tenth_implementation_attempt.Config_v10 import set_device
DEVICE=set_device()
from avalanche.training.determinism.rng_manager import RNGManager
RNGManager.set_random_seeds(1234)
from Tenth_implementation_attempt.ResNet20_v10 import ResNet20
from Tenth_implementation_attempt.Load_cifar100_v10 import (outdoors_specialised_models_train_subset)
from Tenth_implementation_attempt.Load_cifar100_v10 import (outdoors_specialised_models_val_subset)
from Tenth_implementation_attempt.Load_cifar100_v10 import (outdoors_specialised_models_test_subset)
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.optim import SGD, Adam
from avalanche.training.supervised import Naive, EWC, LwF  # and many more!
from avalanche.training.losses import MaskedCrossEntropy
from avalanche.training.plugins import EarlyStoppingPlugin, EvaluationPlugin, EWCPlugin, LwFPlugin
from avalanche.training import LearningWithoutForgetting, RegularizationMethod

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
from avalanche.evaluation.metrics import loss_metrics
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
from torch.utils.data import DataLoader
from avalanche.models.utils import avalanche_forward

from avalanche.benchmarks import CLExperience, SplitMNIST
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
from avalanche.training.templates.common_templates import SupervisedTemplate
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
    TMBInput,
)
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData
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
TRAINING_WITHOUT_LOOP = True #TODO: Remember to change TRAINING_WITHOUT_LOOP to True if 'Training WITHOUT LOOP'.

print("\n\n\n################################ HERE BEGINS Train_ResNet20_Outdoors_v10.py ################################ \n\n\n")

resnet20_global_model = ResNet20(num_classes=100)
outdoors_specialised_model = ResNet20(num_classes=100) #No need for num_classes=10 anymore despite 10 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
nature_specialised_model = ResNet20(num_classes=100) #No need for num_classes=15 anymore despite 15 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
people_specialised_model = ResNet20(num_classes=100) #No need for num_classes=5 anymore despite 5 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
animals_specialised_model = ResNet20(num_classes=100) #No need for num_classes=45 anymore despite 45 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
objects_and_vehicles_specialised_model = ResNet20(num_classes=100) #No need for num_classes=25 anymore despite 25 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used

avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset)
avl_val = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset)
avl_test = make_avalanche_dataset(outdoors_specialised_models_test_subset.dataset)

if DEBUG:
    print("type(outdoors_specialised_models_train_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train", type(avl_train))
    print("type(outdoors_specialised_models_val_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val", type(avl_val))
    print("type(outdoors_specialised_models_test_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test", type(avl_test))

    print("Print avl_train AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train:\n", avl_train)
    print("Print avl_val AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val:\n", avl_val)
    print("Print avl_test AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test:\n", avl_test)

avl_train = ClassificationDataset(
    [avl_train], #[outdoors_specialised_models_train_subset.dataset],
    data_attributes=[DataAttribute(outdoors_specialised_models_train_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(outdoors_specialised_models_train_subset.dataset.targets,"targets")], # "fine_label")],
)
avl_val = ClassificationDataset(
    [avl_val], #[outdoors_specialised_models_val_subset.dataset],
    data_attributes=[DataAttribute(outdoors_specialised_models_val_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(outdoors_specialised_models_val_subset.dataset.targets, "targets")],# "fine_label")],
)
avl_test = ClassificationDataset(
    [avl_test], #[outdoors_specialised_models_test_subset.dataset],
    data_attributes=[DataAttribute(outdoors_specialised_models_test_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(outdoors_specialised_models_test_subset.dataset.targets, "targets")],# "fine_label")],
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
print("Print len(avl_train.targets.uniques) to know the NUMBER OF FINE CLASSES within the specialised_models_superclasses group of which the specialised subset (e.g. outdoors_specialised_models_train_subset) that is avl_train is composed:", len(avl_train.targets.uniques)) # print number of classes
print("IMPORTANT! Print sorted(avl_train.targets.uniques) AS THIS WILL BE THE OBJECT PASSED TO class_order IN class_incremental_benchmark TO AVOID/AS A WORK-AROUND TO THE num_classes BUG IN THE LIBRARY CODE, since by provideing a class_order, 'class_order = torch.randperm(num_classes).tolist()' within 'if class_order is None:' IS AVOIDED!", sorted(avl_train.targets.uniques))# print class_order

outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[2, 2, 2, 2, 69]) # which is equivalent to num_experiences=5 due to the reasons explained in previous version)

if DEBUG:
    print("sorted(avl_train.targets.uniques)",sorted(avl_train.targets.uniques))
    for i, exp in enumerate(outdoors_specialised_models_benchmark.train_stream):
        print("print(i, exp.classes_in_this_experience -- outdoors_specialised_models_benchmark)", i, exp.classes_in_this_experience)

    print("Print outdoors_specialised_models_benchmark:", outdoors_specialised_models_benchmark)
    #print("Print outdoors_specialised_models_benchmark.train_stream:", outdoors_specialised_models_benchmark.train_stream) # it works, but I am not sure that it is not deprecated. I think the current or at least most frequently used way of retrieving a stream is .streams["train"], specially if there is a .streams["valid"] since attribute .val_stream does NOT exist. print(outdoors_specialised_models_benchmark.train_stream) and print(outdoors_specialised_models_benchmark.streams["train"]) both return <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000002242B559510>, meaning they return the exact same object so, there is no risk involved in having chosen to use .streams["train"] over .train_stream
    print('Print outdoors_specialised_models_benchmark.streams["train"]',outdoors_specialised_models_benchmark.streams["train"])
    print('Print outdoors_specialised_models_benchmark.streams["test"]', outdoors_specialised_models_benchmark.streams["test"])
    print('Print outdoors_specialised_models_benchmark.streams["valid"]', outdoors_specialised_models_benchmark.streams["valid"])

# IMPORTANT!!!!! This time, we do want to keep these as streams and NOT to further divide each stream into its composing experiences. This time,
# since class_incremental_benchmark is actually being used to create a CLScenario for use in training in a class-IL manner, and NOT as functionality
# for splitting the dataset into specialised model subsets (as we did previously in Load_Cifar100_v7.py, when class_incremental_benchmark was used
# last), we want to keep the stream objects intact without breaking them apart into its composing experience objects, as the experience objects are
# to be managed by the Avalanche training functionality/training loop to ensure the specialised model is trained in a class_IL manner with each
# experience being used for a subsequent incremental training round.
outdoors_specialised_models_train_stream_class_il = outdoors_specialised_models_benchmark.streams["train"]
outdoors_specialised_models_val_stream_class_il = outdoors_specialised_models_benchmark.streams["valid"]
outdoors_specialised_models_test_stream_class_il = outdoors_specialised_models_benchmark.streams["test"]

if DEBUG:
    for i, exp in enumerate(outdoors_specialised_models_train_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- outdoors_specialised_models_train_stream_class_il)", i, exp.classes_in_this_experience)

    for i, exp in enumerate(outdoors_specialised_models_val_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- outdoors_specialised_models_val_stream_class_il)", i, exp.classes_in_this_experience)

    for i, exp in enumerate(outdoors_specialised_models_test_stream_class_il):
        print("print(i, exp.classes_in_this_experience -- outdoors_specialised_models_test_stream_class_il)", i,
              exp.classes_in_this_experience)

    print("type(outdoors_specialised_models_train_stream_class_il -- OBJECT THROUGH WHICH TRAINING LOOP IS CURRENTLY ITERATING)", type(outdoors_specialised_models_train_stream_class_il))
    print("type(outdoors_specialised_models_train_stream_class_il.exps_iter -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS)", type(outdoors_specialised_models_train_stream_class_il.exps_iter))
    print('type(outdoors_specialised_models_benchmark.train_stream -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS', type(outdoors_specialised_models_benchmark.train_stream), 'ALSOOOOOO, benchmark.train_stream KNOWN TO BE EQUAL TO benchmark.streams["train"]); CHECK TO VERIFY')
    print("type(outdoors_specialised_models_train_stream_class_il.exps_iter[0])", type(outdoors_specialised_models_train_stream_class_il.exps_iter[0]))

#####################################################################################################################################################################################################################
'''class LearningWithoutForgetting_mine(RegularizationMethod):
    """Learning Without Forgetting.

    The method applies knowledge distillation to mitigate forgetting.
    The teacher is the model checkpoint after the last experience.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """
        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None
        self.expcount = 0
        # count number of experiences (used to increase alpha)
        self.prev_classes = set()
        """ In Avalanche, targets of different experiences are not ordered.
        As a result, some units may be allocated even though their
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding
        to old classes (tracked here by target/class id, not by task).
        """

    def _distillation_loss(self, out, prev_out, active_units):
        """Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)

        # some people use the crossentropy instead of the KL
        # They are equivalent. We compute
        # kl_div(log_p_curr, p_prev) = p_prev * (log (p_prev / p_curr)) =
        #   p_prev * log(p_prev) - p_prev * log(p_curr).
        # Now, the first term is constant (we don't optimize the teacher),
        # so optimizing the crossentropy and kl-div are equivalent.
        log_p = torch.log_softmax(out[:, au] / self.temperature, dim=1)
        q = torch.softmax(prev_out[:, au] / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res

    def _lwf_penalty(self, out, x, curr_model):
        """
        Compute weighted distillation loss.
        """
        if self.prev_model is None or len(self.prev_classes) == 0:
            return 0
        else:
            with torch.no_grad():
                y_prev = self.prev_model(x)
            y_curr = out

            # compute kd only for the targets/classes seen so far.
            au = self.prev_classes
            dist_loss = self._distillation_loss(y_curr, y_prev, au)
            return dist_loss

    def __call__(self, mb_x, mb_pred, model):
        """
        Add distillation loss
        """
        alpha = (
            self.alpha[self.expcount]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )
        return alpha * self._lwf_penalty(mb_pred, mb_x, model)

    def post_adapt(self, agent, exp):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned targets.

        :param agent: agent state
        :param exp: current experience
        """
        self.expcount += 1
        self.prev_model = copy.deepcopy(agent.model)

        pc = set(exp.dataset.targets.uniques)
        self.prev_classes = self.prev_classes.union(pc)

    #@deprecated(0.7, "switch to post_udpate method")
    def update(self, experience, model):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned targets.

        :param experience: current experience
        :param model: current model
        """
        self.expcount += 1
        self.prev_model = copy.deepcopy(model)

        pc = set(experience.dataset.targets.uniques)
        self.prev_classes = self.prev_classes.union(pc)
'''
class EWCStrategy(SupervisedTemplate): #formerly TestEWCStrategy_2, but renamed to EWCStrategy now that it is the definite version/implementation of the class to be used
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        ewc_lambda: float,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.prev_classes = set()
        self.alpha = 1
        self.temperature = 2
        self.prev_classes_by_task = defaultdict(set)
        self.expcount = 0
        self.prev_model = None
        #self.lwf_mine = LearningWithoutForgetting_mine(alpha=1, temperature=2)

        if mode not in ("separate", "online"):
            raise ValueError(
                "mode must be either 'separate' or 'online'."
            )

        if mode == "online" and decay_factor is None:
            raise ValueError(
                "decay_factor must be specified when mode='online'."
            )

        if mode != "online" and decay_factor is not None:
            raise ValueError(
                "decay_factor is only used when mode='online'."
            )

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.ce_loss = self._make_empty_loss()  # self.ce_loss = torch.tensor(0).float().to(self.device)

    def training_epoch(self, **kwargs):
        print("***** MY training_epoch() WAS CALLED *****")

        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Ordinary loss
            self.loss += self.criterion()
            self.ce_loss = self.loss.detach().item()  # ce_loss = self.loss

            # print("CrossEntropyLoss (i.e. normal loss before EWC loss):", ce_loss.item())

            if self.experience.current_experience > 0:
                self.loss += self.final_weighted_loss(self.mb_x, self.mb_output, self.model)

            exp_counter = self.clock.train_exp_counter # code changed to avoid training_epoch from being exited before backward pass if Experience 0. Now, backward pass still takes place
            # for Experience 0, but with normal loss (i.e. ce_loss/CrossEntropyLoss or MaskedCrossEntropy) WITHOUT penalty or lambda (EWC affects the loss starting at Experience 1;
            # i.e. if exp_counter > 0)
            '''if exp_counter == 0:
                return'''
            penalty = torch.tensor(0).float().to(self.device)
            if exp_counter > 0:
                if self.mode == "separate":
                    for experience in range(exp_counter):
                        for k, cur_param in self.model.named_parameters():
                            # new parameters do not count
                            if k not in self.saved_params[experience]:
                                continue
                            saved_param = self.saved_params[experience][k]
                            imp = self.importances[experience][k]
                            new_shape = cur_param.shape
                            penalty += (
                                    imp.expand(new_shape)
                                    * (cur_param - saved_param.expand(new_shape)).pow(2)
                            ).sum()
                elif self.mode == "online":  # may need importance and param expansion
                    prev_exp = exp_counter - 1
                    for k, cur_param in self.model.named_parameters():
                        # new parameters do not count
                        if k not in self.saved_params[prev_exp]:
                            continue
                        saved_param = self.saved_params[prev_exp][k]
                        imp = self.importances[prev_exp][k]
                        new_shape = cur_param.shape
                        penalty += (
                                imp.expand(new_shape)
                                * (cur_param - saved_param.expand(new_shape)).pow(2)
                        ).sum()
                else:
                    raise ValueError("Wrong EWC mode.")

            self.loss += self.ewc_lambda * penalty

            # Plugins
            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

        # TODO: uncomment whenever I want to save checkpoint per epoch to see at what point EWC stops mitigating CF for a particular experience
        self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{self.clock.train_exp_counter}_Epoch_{self.clock.train_exp_epochs}.pkl")
        print(f"Saved {f'./checkpoints/checkpoint_Experience_{self.clock.train_exp_counter}_Epoch_{self.clock.train_exp_epochs}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_epoch()")

        if TRAINING_WITH_LOOP:
            # TODO: this method is being tested!!!! Remember to remove if Training WITHOUT LOOP
            self.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{self.clock.train_exp_counter}_Epoch_{self.clock.train_exp_epochs}.pkl")
            print(f"Loaded {f'./checkpoints/checkpoint_Experience_{self.clock.train_exp_counter}_Epoch_{self.clock.train_exp_epochs}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_epoch()")


    def _train_exp(self, experience: CLExperience, eval_streams=None, **kwargs):
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

            print(f"\n\n***** MY _train_exp() IS EXECUTING FOR Experience {experience.current_experience} *****\n\n")
            self.training_epoch(**kwargs)
            '''# TODO: uncomment whenever I want to save checkpoint per epoch to see at what point EWC stops mitigating CF for a particular experience
            # It works, but it seems that when self.load_entire_checkpoint_inside_class is placed on training_epoch, it works better.
            # Keep in case something changes and it is worth testing what would happen if self.load_entire_checkpoint_inside_class was placed on _train_exp
            self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl")
            print(f"Saved {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl'} --- from inside _training_exp()")

            #TODO: MUST BE BEFORE self._after_training_epoch(**kwargs). It works, but it seems that when self.load_entire_checkpoint_inside_class is placed on training_epoch, it works better.
            # Keep in case something changes and it is worth testing what would happen if self.load_entire_checkpoint_inside_class was placed on _train_exp
            if TRAINING_WITH_LOOP: # MUST BE BEFORE self._after_training_epoch(**kwargs)
                # TODO: this method is being tested!!!! Remember to remove if Training WITHOUT LOOP
                self.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl")
                print(f"Loaded {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{_}.pkl'} to continue training the next experience on it --- from inside _training_exp()")'''

            self._after_training_epoch(**kwargs)

        # LwF code
        #self.lwf_mine.post_adapt(self, self.experience)
        #if experience.current_experience > 0:
        self.post_adapt(self, self.experience)

        exp_counter = self.clock.train_exp_counter
        importances = self.compute_importances(
            self.model,
            self._criterion,
            self.optimizer,
            self.experience.dataset,
            self.device,
            self.train_mb_size,
            num_workers=kwargs.get("num_workers", 0),
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(self.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

        # to save checkpoint per entire experience (the strategy and model as they are at the end of training on that experience will be saved -- i.e. as they are once the last epoch for
        # that experience concludes, hence this checkpoint for the entire experience will be equivalent to the checkpoint for the last epoch of that experience saved above)
        #TODO: Apparently it does NOT WORK AS INTENDED! For some reason it saves checkpoint of Epoch_2 instead of checkpoint of last epoch as you would expect given it is called once
        # _train_exp is about to exit and so, you'd expect the .self on which it is called to be the state of the model / the state of the strategy as it was during the last epoch
        # TODO: Find out why it fails OR make the decision of NOT USING THIS and instead perform aggregation from the saved checkpoints of the last epoch of corresponding experiences
        #  of the different specialised models (i.e. to aggregate experience 1 of all specialised models as I am supposed to, aggregate the last epoch checkpoint of experience 1, instead
        #  of the one that is supposed to be saved at the end of _train_exp in the line of code below). The fact that the checkpoint of Epoch_2 is the one saved tells me nothing about what
        #  is causing this issue, as if it was that the code in the body of _train_exp that comes after call to training_epoch was actually executed after every epoch and not once per experience,
        #  and then the checkpoint for Epoch 0 would be the one being saved as the experience epoch, so, since I have such little idea of what could be causing this bug, perhaps just use the
        #  appropriate epoch checkpoints instead of wasting time trying to fix this bug!!!!

        #self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{experience.current_experience}.pkl")
        #print(f"Saved {f'./checkpoints/checkpoint_Experience_{experience.current_experience}.pkl'} --- from inside _training_exp()")
        #DOES NOT SAVE model_state_dict !!! save_checkpoint(self.model, f"./checkpoint_Experience_{experience.current_experience}.pth")
        # DO NOT call self.load_entire_checkpoint_inside_class !!! IT LEADS TO AN ERROR. I tried this to see if it would re-load the last experience checkpoint
        # (checkpoint_Experience_{experience.current_experience}.pkl; NOT epoch checkpoint) saved and start the next .train() call on the checkpoint saved for the last experience/last
        # .train() call that concluded, but it leads to an error. This seemed like the only logical place where to place such call in order to avoid differences in numbers that occur
        # across .train() calls / experiences when training with loop when compared to when training without loop, but it did not work. There are no more alternatives to try, so accept
        # that the numbers will be slightly different when training with loop --- THIS IS AS ACCURATE A SET-UP FOR WHEN TRAINING WITH LOOP AS I WILL GET, since its not that important and
        # not worth spending more time on it !!!



    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size, num_workers=0
    ) -> Dict[str, ParamData]:
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t: int):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1].items(),
                importances.items(),
                fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[t][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                # manage expansion of existing layers
                self.importances[t][k1] = ParamData(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                    + curr_imp.data,
                    device=curr_imp.device,
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


    def _distillation_loss(self, out, prev_out, active_units):
        """Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)

        # some people use the crossentropy instead of the KL
        # They are equivalent. We compute
        # kl_div(log_p_curr, p_prev) = p_prev * (log (p_prev / p_curr)) =
        #   p_prev * log(p_prev) - p_prev * log(p_curr).
        # Now, the first term is constant (we don't optimize the teacher),
        # so optimizing the crossentropy and kl-div are equivalent.
        log_p = torch.log_softmax(out[:, au] / self.temperature, dim=1)
        q = torch.softmax(prev_out[:, au] / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res

    def _lwf_penalty(self, out, x, curr_model):
        """
        Compute weighted distillation loss.
        """
        '''
        if self.prev_model is None or len(self.prev_classes) == 0:
            return 0
        else:
            with torch.no_grad():
                y_prev = self.prev_model(x)
            y_curr = out

            # compute kd only for the targets/classes seen so far.
            au = self.prev_classes
            dist_loss = self._distillation_loss(y_curr, y_prev, au)
            return dist_loss
        '''
        if self.prev_model is None:
            return 0
        else:
            if isinstance(self.prev_model, MultiTaskModule):
                # output from previous output heads.
                with torch.no_grad():
                    y_prev = avalanche_forward(self.prev_model, x, None)
                y_prev = {k: v for k, v in y_prev.items()}
                # in a multitask scenario we need to compute the output
                # from all the heads, so we need to call forward again.
                # TODO: can we avoid this?
                y_curr = avalanche_forward(curr_model, x, None)
                y_curr = {k: v for k, v in y_curr.items()}
            else:  # no task labels. Single task LwF
                with torch.no_grad():
                    y_prev = {0: self.prev_model(x)}
                y_curr = {0: out}

            #dist_loss = 0
            # compute kd only for the targets/classes seen so far.
            au = self.prev_classes
            dist_loss = self._distillation_loss(y_curr[0], y_prev[0], au)
            return dist_loss


    def final_weighted_loss(self, mb_x, mb_pred, model):
        """
        Add distillation loss
        """
        alpha = (
            self.alpha[self.expcount]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )
        return alpha * self._lwf_penalty(mb_pred, mb_x, model)

    def post_adapt(self, agent, exp):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned targets.

        :param agent: agent state
        :param exp: current experience
        """
        self.expcount += 1
        #self.prev_model = copy.deepcopy(agent.model)
        self.prev_model = deepcopy(agent.model)

        pc = set(exp.dataset.targets.uniques)
        self.prev_classes = self.prev_classes.union(pc)

    #@deprecated(0.7, "switch to post_udpate method")
    def update(self, experience, model):
        """Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned targets.

        :param experience: current experience
        :param model: current model
        """
        self.expcount += 1
        #self.prev_model = copy.deepcopy(model)
        self.prev_model = deepcopy(model)

        pc = set(experience.dataset.targets.uniques)
        self.prev_classes = self.prev_classes.union(pc)


    def save_entire_checkpoint(self, fname: FILE_LIKE, exclude: Optional[Collection[str]] = None): # FORMERLY #def save_entire_checkpoint(self, path, exclude: Optional[Collection[str]] = None):
        if exclude is None:
            exclude = []
        ended_experience_counter = self.clock.train_exp_counter

        strategy = copy(self)
        for attr in exclude:
            delattr(strategy, attr)

        checkpoint = {
            "strategy": strategy,
            "rng_manager_state": RNGManager.__getstate__(),#"rng_manager": RNGManager, #.set_random_seeds(SEED),
            "exp_counter": ended_experience_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
            "ce_loss" : self.ce_loss,
            "_criterion" : self._criterion,
            "saved_params": self.saved_params,  # EWC anchors
            "importances": self.importances,  # EWC Fisher values
            "clock_train_exp_counter": self.clock.train_exp_counter,
            "clock_train_epoch_counter": self.clock.train_exp_epochs
        }
        #with open(path, "wb") as f:
         #   pickle.dump(checkpoint, f)
        #torch.save(checkpoint, path)
        checkpoint_dir = Path("./checkpoints") # checkpoint_dir = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Tenth_implementation_attempt\checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, fname, pickle_module=dill)

    def load_entire_checkpoint_inside_class(self, fname: FILE_LIKE, map_location=None, unique_objects: Any = None):  # , device):
        checkpoint = torch.load(fname, pickle_module=dill, map_location=map_location)
        saved_strategy = checkpoint["strategy"] # NOT self, that would be wrong. Not used within function, but kept in case saved_strategy needs to be retrieved at some point, perhaps
                                                # even through an external call on an instance / object of class EWCStrategy

        self.model.load_state_dict(checkpoint["model_state_dict"])
        # strategy.model.to(device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.saved_params = checkpoint["saved_params"]
        self.importances = checkpoint["importances"]
        self.ce_loss = checkpoint["ce_loss"]
        self.loss = checkpoint["loss"]
        self._criterion = checkpoint["_criterion"]
        self.clock = checkpoint["strategy"].clock
        self.clock.train_exp_counter = checkpoint["clock_train_exp_counter"]
        self.ewc_lambda = checkpoint["strategy"].ewc_lambda
        self.mode = checkpoint["strategy"].mode
        self.decay_factor = checkpoint["strategy"].decay_factor
        self.keep_importance_data = checkpoint["strategy"].keep_importance_data
        self.train_mb_size = checkpoint["strategy"].train_mb_size
        self.train_epochs = checkpoint["strategy"].train_epochs
        self.eval_mb_size = checkpoint["strategy"].eval_mb_size
        self.device = checkpoint["strategy"].device
        self.plugins = checkpoint["strategy"].plugins
        self.evaluator = checkpoint["strategy"].evaluator
        self.eval_every = checkpoint["strategy"].eval_every
        self.dataloader = checkpoint["strategy"].dataloader
        self.experience = checkpoint["strategy"].experience
        RNGManager.__setstate__(checkpoint["rng_manager_state"])
        #RNGManager.__setstate__(checkpoint["rng_manager"])

        print(f"Executing load_entire_checkpoint_inside_class from call made from within _train_exp()")
        return self


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
        if self.metric_name == "Top1_Acc_Exp" or self.metric_name == "Loss_Exp":
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



'''
# function for re-loading checkpoints for evaluation and aggregation
def load_entire_checkpoint(fname: FILE_LIKE, map_location=None, unique_objects: Any = None):  # , device):
    checkpoint = torch.load(fname,pickle_module=dill,map_location=map_location)
    strategy = checkpoint["strategy"]

    strategy.model.load_state_dict(checkpoint["model_state_dict"])
    # strategy.model.to(device)
    strategy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    strategy.saved_params = checkpoint["saved_params"]
    strategy.importances = checkpoint["importances"]
    strategy.ce_loss = checkpoint["ce_loss"] #new
    strategy.loss = checkpoint["loss"] #new
    strategy._criterion = checkpoint["_criterion"] # new
    strategy.clock = checkpoint["strategy"].clock
    strategy.clock.train_exp_counter = checkpoint["clock_train_exp_counter"]
    strategy.ewc_lambda = checkpoint["strategy"].ewc_lambda
    strategy.mode = checkpoint["strategy"].mode
    strategy.decay_factor = checkpoint["strategy"].decay_factor
    strategy.keep_importance_data = checkpoint["strategy"].keep_importance_data
    strategy.train_mb_size = checkpoint["strategy"].train_mb_size
    strategy.train_epochs = checkpoint["strategy"].train_epochs
    strategy.eval_mb_size = checkpoint["strategy"].eval_mb_size
    strategy.device = checkpoint["strategy"].device
    strategy.plugins = checkpoint["strategy"].plugins
    strategy.evaluator = checkpoint["strategy"].evaluator
    strategy.eval_every = checkpoint["strategy"].eval_every
    strategy.dataloader = checkpoint["strategy"].dataloader
    strategy.experience = checkpoint["strategy"].experience
    RNGManager.__setstate__(checkpoint["rng_manager_state"])

    #RNGManager.__setstate__(checkpoint["rng_manager"])
    return strategy
'''

# TRAINING LOOP
# original optimizer:
# optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9)

#optimizer = Adam(outdoors_specialised_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
#criterion_masked = MaskedCrossEntropy(mask="seen") # Using MaskedCrossEntropy for masking all but the classes that appear in the specialised model's subset (https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/losses.py , https://avalanche-api.continualai.org/en/latest/generated/avalanche.training.MaskedCrossEntropy.html#avalanche.training.MaskedCrossEntropy)

#ewc_lambda=1e2
#todo: Changed metric_name to "Top1_Acc_Exp" since now we are training on the entire train stream!!!!
# DEFINITE:
'''cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Exp", mode="max", peval_mode="epoch", margin=0.3, verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)'''
# tested and worked pretty good --- individual acc for subsequent experiences was never as high, but knowledge of past experiences was retained better, although the knowledge that was retained was with less accuracy
optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.0001, momentum=0.9)
cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=12, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream", metric_name="Loss_Exp", mode="min", peval_mode="epoch", margin=0.3, verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)
'''optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.0001, momentum=0.9)
cl_strategy = EWC(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=12, eval_mb_size=100, #device=DEVICE,
                    plugins=[EWCPlugin( ewc_lambda=0.9, mode="separate")],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)'''
'''optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.0001, momentum=0.9)
cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=10.1, mode="online", decay_factor=5.1, keep_importance_data=True,
                          train_mb_size=100, train_epochs=12, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream", metric_name="Loss_Exp", mode="min", peval_mode="epoch", margin=0.3, verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)'''
#TODO: NEXT uncomment the below cl_strategy and test
'''cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin_mine(patience=1, val_stream_name="valid_stream", metric_name="Loss_Exp", mode="min", peval_mode="epoch", margin=0.8, verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)'''
'''cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=2, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Exp", mode="max", peval_mode="epoch", margin=0.3, verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)'''
'''cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=2, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin_mine(patience=1, val_stream_name="valid_stream", metric_name="Loss_Exp", mode="min", peval_mode="epoch", margin=0.8, verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)'''


'''cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="online", decay_factor=0.9, train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Exp", mode="max", peval_mode="epoch", margin=0.3, verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)''' # Also maybe worth trying --> ewc_lambda=0.09, mode="online", decay_factor=0.1 AND criterion=criterion_masked, ewc_lambda=00.9, mode="online", decay_factor=0.5, keep_importance_data=True,

if TRAINING_WITHOUT_LOOP:
    print('Training WITHOUT LOOP Starting...')
    cl_strategy.train(outdoors_specialised_models_train_stream_class_il, [outdoors_specialised_models_train_stream_class_il, outdoors_specialised_models_val_stream_class_il])
    print("print(cl_strategy.evaluator.get_last_metrics())", cl_strategy.evaluator.get_last_metrics())
    print("print(cl_strategy.evaluator.get_all_metrics())", cl_strategy.evaluator.get_all_metrics())
    print("print(cl_strategy.plugins)", cl_strategy.plugins)
    print('Training completed WITHOUT LOOP')
    #print('Computing accuracy on the whole val set WITHOUT training LOOP')
    #cl_strategy.eval(outdoors_specialised_models_val_stream_class_il)

#TODO: Uncomment later
'''
    # loop for re-loading every saved checkpoint and evaluating on val stream
    checkpoint_dir_path = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Tenth_implementation_attempt\checkpoints")
    checkpoint_files = sorted(checkpoint_dir_path.glob("*.pkl"))

    for checkpoint_file in checkpoint_files:
        print(f"Loading {checkpoint_file}")
        cl_strategy_evaluation = load_entire_checkpoint(checkpoint_file)
        print(f"Computing val accuracy on {checkpoint_file.name} AFTER 'Training completed WITHOUT LOOP'")
        cl_strategy_evaluation.eval(outdoors_specialised_models_val_stream_class_il)
'''












# CORRECTED 'Starting experiment WITH LOOP...' VERSION 2 (with loading from within loop to ensure training continues on the adequate, definitely uncorrupted, srategy object)
# sort of works, but not as well as NO TRAINING LOOP / WITHOUT TRAINING LOOP --- EDITED given new implementations of my own saving functionality
if TRAINING_WITH_LOOP:
    print('Starting experiment WITH LOOP... VERSION 2')
    results = []
    checkpoint_dir_path = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Tenth_implementation_attempt\checkpoints")
    for experience in outdoors_specialised_models_train_stream_class_il: # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
        i = experience.current_experience # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of outdoors_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
        print("Start of experience -- current TRAIN experience: ", experience.current_experience)
        print("Start of experience -- current VAL experience: ", outdoors_specialised_models_val_stream_class_il.exps_iter[i].current_experience)
        print("Current Classes -- current TRAIN .classes_in_this_experience: ", experience.classes_in_this_experience)
        print("Current Classes -- current VAL .classes_in_this_experience: ", outdoors_specialised_models_val_stream_class_il.exps_iter[i].classes_in_this_experience)
        print("experience.dataset.coarse_label.uniques (coarse_labels of TRAIN)", experience.dataset.coarse_label.uniques)
        print("experience.dataset.targets (targets of TRAIN)", experience.dataset.targets.uniques)
        print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques (coarse_labels of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques)
        print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets (targets of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques)
        print("print(experience.origin_stream.name)", experience.origin_stream.name)
        print("print(outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)
        #print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
        #print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
        #criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
        #print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
        print(f"Computing cl_strategy.eval() FROM INSIDE .train() --> cl_strategy.train(experience, [outdoors_specialised_models_train_stream_class_il, outdoors_specialised_models_val_stream_class_il.exps_iter[i]])")
        cl_strategy.train(experience, [outdoors_specialised_models_train_stream_class_il, outdoors_specialised_models_val_stream_class_il])
        #cl_strategy.train(experience, [experience, outdoors_specialised_models_val_stream_class_il.exps_iter[i]])
        print("print(cl_strategy.evaluator.get_last_metrics())", cl_strategy.evaluator.get_last_metrics())
        print("print(cl_strategy.evaluator.get_all_metrics())", cl_strategy.evaluator.get_all_metrics())
        print("print(cl_strategy.plugins)", cl_strategy.plugins)
        #print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
        #print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)
    print('Training completed WITH LOOP')



    checkpoint_files = sorted(checkpoint_dir_path.glob("*.pkl"))
    for checkpoint_file in checkpoint_files:
        print(f"Loading {checkpoint_file}")
        cl_strategy_evaluation = load_entire_checkpoint(checkpoint_file)
        print(f"Computing val accuracy on {checkpoint_file.name} AFTER 'Training completed WITH LOOP VERSION 2'")
        cl_strategy_evaluation.eval(outdoors_specialised_models_val_stream_class_il)


    #print('Computing accuracy on the whole val set OUTSIDE training LOOP')
    #cl_strategy.eval(outdoors_specialised_models_val_stream_class_il)
    #results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))
    #print("Computing accuracy on the whole val set OUTSIDE training LOOP (i.e. print(results)):", results)