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
#from Eleventh_implementation_attempt.ResNet20_DynamicModel_v11 import ResNet20_DynamicModel
from Eleventh_implementation_attempt.Load_cifar100_v11 import (outdoors_specialised_models_train_subset)
from Eleventh_implementation_attempt.Load_cifar100_v11 import (outdoors_specialised_models_val_subset)
from Eleventh_implementation_attempt.Load_cifar100_v11 import (outdoors_specialised_models_test_subset)
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
TRAINING_WITH_LOOP_LwF = True #False

print("\n\n\n################################ HERE BEGINS Train_ResNet20_Outdoors_v11.py ################################ \n\n\n")

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

# TODO: try tomorrow, but this would require modifying the benchmark and stream objects cause otherwise model is still trained on what it thinks are Exp004 and Exp005
'''outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[3, 3, 71])'''

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

class LwFStrategy(SupervisedTemplate): #formerly TestEWCStrategy_2, but renamed to EWCStrategy now that it is the definite version/implementation of the class to be used
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
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

        self.ce_loss = self._make_empty_loss()  # self.ce_loss = torch.tensor(0).float().to(self.device)
        #self.retain_graph = True

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
        #self.update(self.experience, self.model)


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
            "clock_train_exp_counter": self.clock.train_exp_counter,
            "clock_train_epoch_counter": self.clock.train_exp_epochs
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
        self.ce_loss = checkpoint["ce_loss"]
        self.loss = checkpoint["loss"]
        self._criterion = checkpoint["_criterion"]
        self.clock = checkpoint["strategy"].clock
        self.clock.train_exp_counter = checkpoint["clock_train_exp_counter"]
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
cl_strategy = LwFStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=12, eval_mb_size=100, #device=DEVICE,
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
    checkpoint_dir_path = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Eleventh_implementation_attempt\checkpoints")
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
    checkpoint_dir_path = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Eleventh_implementation_attempt\checkpoints")
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


'''
    checkpoint_files = sorted(checkpoint_dir_path.glob("*.pkl"))
    for checkpoint_file in checkpoint_files:
        print(f"Loading {checkpoint_file}")
        cl_strategy_evaluation = load_entire_checkpoint(checkpoint_file)
        print(f"Computing val accuracy on {checkpoint_file.name} AFTER 'Training completed WITH LOOP VERSION 2'")
        cl_strategy_evaluation.eval(outdoors_specialised_models_val_stream_class_il)

'''
    #print('Computing accuracy on the whole val set OUTSIDE training LOOP')
    #cl_strategy.eval(outdoors_specialised_models_val_stream_class_il)
    #results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))
    #print("Computing accuracy on the whole val set OUTSIDE training LOOP (i.e. print(results)):", results)




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
# class LwFTrainer(SGDStrategyProtocol):
#class LwFTrainer(SGDStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]):
#class LwFTrainer():
class LwFTrainer(SupervisedTemplate):
    def __init__(
        self,
        model: nn.Module,
        #train_adapted_dataset,
        #train_dataloader: DataLoader,
        #val_adapted_dataset,
        #val_dataloader,
        #device="cuda",
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        temperature=2.0,
        alpha=1.0,
        #batch_size=32,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch"
    ):
        '''
        super().__init__(
            model=model,
            optimizer=optimizer,
            #criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        '''
        #self.genericPluginMetric :  GenericPluginMetric[TResult_co, TMetric]= GenericPluginMetric() #: GenericPluginMetric[PluginMetric[TResult_co], Generic[TResult_co, TMetric]]  #GenericPluginMetric #GenericPluginMetric(PluginMetric[TResult_co], Generic[TResult_co, TMetric])
        self.plugins = plugins
        self.adapted_dataset = None
        self.optimized_param_id = None
        self.model = model.to(device)
        self.device = torch.device(device)

        self.temperature = temperature
        self.alpha = alpha
        #self.batch_size = batch_size

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.dataloader: Iterable[Any] = []
        """ Dataloader. """
        '''
        #self.train_dataloader: Iterable[Any] = train_dataloader
        self.train_dataloader = None#Optional[Dataset] # None
        self.eval_dataloader = None #val_dataloader
        self.train_adapted_dataset = None #Dataset[_T_co] # None #Optional[Dataset] # None
        self.eval_adapted_dataset = None #Dataset[_T_co] #None #val_adapted_dataset
        '''
        self.mbatch: Optional[TMBInput] = None
        """ Current mini-batch. """

        self.mb_output: Optional[TMBOutput] = None
        """ Model's output computed on the current mini-batch. """

        ####self._stop_training = False

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

        # Classes learned before current experience
        self.old_classes = set()


        #self.plugins: List[BasePlugin] = [] if plugins is None else list(plugins)
        #""" List of `SupervisedPlugin`s. """

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

    '''
    def distillation_loss(
        self,
        student_logits,
        teacher_logits,
    ):

        T = self.temperature

        student_log_probs = F.log_softmax(
            student_logits / T,
            dim=1,
        )

        teacher_probs = F.softmax(
            teacher_logits / T,
            dim=1,
        )

        # Match authors' implementation:
        #
        # F.kl_div(...) * number_of_outputs

        scale = student_logits.size(1)

        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",#reduction="mean",
        ) * scale
    '''

    def distillation_loss(self, student_logits, teacher_logits):

        T = self.temperature

        student_log_probs = F.log_softmax(
            student_logits / T,
            dim=1,
        )

        teacher_probs = F.softmax(
            teacher_logits / T,
            dim=1,
        )

        return (
                F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                )
                * (T ** 2)
        )
    #def train_experience(self, experience: CLExperience, eval_streams: Union[TExperienceType, CLStream[TExperienceType]],**kwargs, ):#def train_experience(self, experience: CLExperience, eval_streams: Optional[Sequence[Union[TExperienceType, Iterable[TExperienceType]]]] = None,**kwargs,): #def train_experience(self, experience: CLExperience, experiences: Union[TExperienceType, CLStream[TExperienceType]]): #def train_experience(self, experience: CLExperience, eval_streams=None, epochs=1, ** kwargs):
    # definition of train, not of train_exp in source code --> def train_experience(self, experiences: Union[TExperienceType, Iterable[TExperienceType]], eval_streams: Optional[Sequence[Union[TExperienceType, Iterable[TExperienceType]]]] = None,**kwargs,):
    def train_experience(self, experience: CLExperience, eval_streams: Union[CLExperience, CLStream],**kwargs, ):
        self.is_training = True
        self._stop_training = False
        self.experience = experience
        self.model.train()
        self.model.to(self.device)

        '''
        # Normalize training and eval data.
        experiences_list: Iterable[TExperienceType] = (
            _experiences_parameter_as_iterable(experiences)
        )

        if eval_streams is None:
            eval_streams = [experiences_list]
        '''
        #self._eval_streams = _group_experiences_by_stream(eval_streams)

        print(f"\n\n***** MY train_experience() IS EXECUTING FOR Experience {experience.current_experience} *****\n\n")
        '''    
        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]
        '''
        self._before_training_exp(**kwargs)# mine which is currently commented-out --> self.before_training_exp(experience)

        #self.model.train()
        #self.is_training = True

        for epoch in range(self.train_epochs):
            print(f"\n***** EXECUTING Epoch {epoch} OF Experience {experience.current_experience} --- from inside train_experience() *****\n")
            total_ce = 0.0
            total_dist = 0.0
            total_loss = 0.0
            n=0

            for self.mbatch in self.dataloader:#self.train_dataloader: # for batch, targets in train_dataloader:

                self.unpack_minibatch()

                #batch = batch.to(self.device)
                targets = self.mb_y #experience.dataset.targets #.uniques

                self.optimizer.zero_grad()

                # ------------------------------------------------
                # Current model
                # ------------------------------------------------
                #I think I added this myself -- self.mb_output = self.model(self.mbatch)

                #student_logits is student output, which in a single-loss set-up would be " self.mb_output = self.forward() "
                student_logits = self.model(self.mb_x) #student_logits = self.model(self.mbatch) #student_logits = self.model(batch)

                # ------------------------------------------------
                # Ordinary CE loss for current experience
                # ------------------------------------------------

                ce_loss = self.criterion(
                    student_logits,
                    targets,
                )

                # ------------------------------------------------
                # LwF
                # ------------------------------------------------

                if self.teacher is not None:

                    with torch.no_grad():
                        teacher_logits =self.teacher(self.mb_x) #self.teacher(self.mbatch)#teacher_logits = self.teacher(batch)

                    # Only old classes
                    old_classes = sorted(self.old_classes)

                    student_old = student_logits[:, old_classes]
                    teacher_old = teacher_logits[:, old_classes]

                    dist_loss = self.distillation_loss(
                        student_old,
                        teacher_old,
                    )

                    loss = (
                        ce_loss
                        + self.alpha * dist_loss
                    )

                else:

                    dist_loss = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                    loss = ce_loss

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                loss.backward()

                self.optimizer.step()

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                total_ce += ce_loss.item()
                total_dist += dist_loss.item()
                total_loss += loss.item()

                n += 1 #n = len(self.train_dataloader)

            print(
                f"Epoch {epoch + 1}/{self.train_epochs} | "
                f"CE: {total_ce / n:.6f} | "
                f"LwF: {total_dist / n:.6f} | "
                f"Total: {total_loss / n:.6f}"
            )

            self.eval(eval_streams)

            # TODO: uncomment whenever I want to save checkpoint per epoch to see at what point EWC stops mitigating CF for a particular experience
            self.save_entire_checkpoint(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{epoch}.pkl")
            print(f"Saved {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{epoch}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_experience()")

            '''
            if TRAINING_WITH_LOOP:
                # TODO: this method is being tested!!!! Remember to remove if Training WITHOUT LOOP
                self.load_entire_checkpoint_inside_class(f"./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{epoch}.pkl")
                print(f"Loaded {f'./checkpoints/checkpoint_Experience_{experience.current_experience}_Epoch_{epoch}.pkl'} to continue training the next epoch of the experience on it --- from inside _training_epoch()")
            '''
            self._eval_streams=eval_streams
            self._after_training_epoch(**kwargs)
        #self.eval(eval_streams, experience)
        #self.eval(eval_streams)
        # --------------------------------------------------------
        # IMPORTANT:
        # Save teacher AFTER completing the experience
        # --------------------------------------------------------

        self.teacher = copy.deepcopy(self.model)

        self.teacher.eval()

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        # Add classes from this experience
        experience_classes = set(
            experience.dataset.targets.uniques
        )

        self.old_classes.update(experience_classes)

        print(
            f"Teacher updated. "
            f"Old classes = {sorted(self.old_classes)}"
        )

        self.eval(eval_streams)

    @torch.no_grad()
    def eval(
        self,
        experiences: Union[TExperienceType, CLStream[TExperienceType]],
        **kwargs,
    ):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        #if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            #self._check_distributed_training_compatibility()
            #self._distributed_check = True

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
            self._eval_exp(**kwargs)
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


    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:#self.eval_dataloader:
            self.unpack_minibatch()
            self._before_eval_iteration(**kwargs) # formerly commented-out

            self._before_eval_forward(**kwargs)# formerly commented-out
            self.mb_output = self.model(self.mb_x)#self.mb_output = self.forward()
            self._after_eval_forward(**kwargs)# formerly commented-out
            self.loss = self.criterion(self.mb_output, self.mb_y) #loss = self.criterion(self.mb_output, self.mb_y)

            self._after_eval_iteration(**kwargs) # formerly commented-out


    '''
    def train_dataset_adaptation(self, experience: CLExperience):#experiences: Union[TExperienceType, Iterable[TExperienceType]]):  # -> AvalancheDataset[T_co]: #(experience: Optional[TExperienceType]):
        """Initialize `self.adapted_dataset`."""
        assert experience is not None
        self.adapted_dataset = experience.dataset
        assert self.adapted_dataset is not None
        self.adapted_dataset = self.adapted_dataset.train()
        #return self.adapted_dataset

    def make_train_dataloader(self,
            # experiences: Union[TExperienceType, Iterable[TExperienceType]],
            #adapted_dataset: Optional[Dataset],  # AvalancheDataset[T_co],
            #batch_size: int,
            num_workers=0,
            shuffle=True,
            pin_memory=None,
            persistent_workers=False,
            drop_last=False,
            **kwargs
    ):
        assert self.adapted_dataset is not None
        # CODE FOR "other_dataloader_args = self._obtain_common_dataloader_parameters(..." DELETED, as it does not seem like I needed for the reasons explained above
        # CODE FOR "use task-balanced dataloader for task-aware benchmarks" DELETED since it was causing data type issues and I am not gonna be using task-aware/task-IL benchmark (AT LEAST NOT FOR NOW!)
        self.train_dataloader = DataLoader(self.adapted_dataset,
                                batch_size=self.batch_size,
                                num_workers=num_workers,
                                shuffle=shuffle,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers,
                                drop_last=drop_last, )

        #return dataloader
    '''
    '''
    def eval_dataset_adaptation(self,experience: CLExperience):  # experiences: Union[TExperienceType, Iterable[TExperienceType]]):  # -> AvalancheDataset[T_co]: #(experience: Optional[TExperienceType]):
        """Initialize `self.adapted_dataset`."""
        assert experience is not None
        self.eval_adapted_dataset = experience.dataset
        assert self.eval_adapted_dataset is not None
        self.eval_adapted_dataset = self.eval_adapted_dataset.eval()
        # return self.adapted_dataset
        #assert experience is not None
        #self.adapted_dataset = experience.dataset
        #assert self.adapted_dataset is not None
        #self.adapted_dataset = self.adapted_dataset.eval()
        # return self.adapted_dataset

    def make_eval_dataloader(self,
            # experiences: Union[TExperienceType, Iterable[TExperienceType]],
            #adapted_dataset: Optional[Dataset],  # AvalancheDataset[T_co],
            #batch_size: int,
            num_workers=0,
            shuffle=True,
            pin_memory=None,
            persistent_workers=False,
            drop_last=False,
            **kwargs
    ):
        assert self.eval_adapted_dataset is not None
        # CODE FOR "other_dataloader_args = self._obtain_common_dataloader_parameters(..." DELETED, as it does not seem like I needed for the reasons explained above
        # CODE FOR "use task-balanced dataloader for task-aware benchmarks" DELETED since it was causing data type issues and I am not gonna be using task-aware/task-IL benchmark (AT LEAST NOT FOR NOW!)
        self.eval_dataloader = DataLoader(self.eval_adapted_dataset,
                                batch_size=self.batch_size,
                                num_workers=num_workers,
                                shuffle=shuffle,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers,
                                drop_last=drop_last, )

        #return dataloader
    '''
    '''
    def check_model_and_optimizer(self, experience: CLExperience, reset_optimizer_state=False, **kwargs):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        assert experience is not None

        if self.optimized_param_id is None:
            self.make_optimizer(reset_optimizer_state=True, **kwargs)

        if _at_task_boundary(experience, before=True):
            self.model = self.model_adaptation(experience)
            self.make_optimizer(reset_optimizer_state=reset_optimizer_state, **kwargs)

    def model_adaptation(self, experience: CLExperience, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        assert experience is not None

        # For training:
        if isinstance(experience, OnlineCLExperience) and self.is_training:
            if experience.access_task_boundaries:
                avalanche_model_adaptation(model, experience.origin_experience)
            else:
                avalanche_model_adaptation(model, experience)
        else:
            avalanche_model_adaptation(model, experience)

        return model.to(self.device)

    def make_optimizer(
            self,
            reset_optimizer_state=False,
            remove_params=False,
            verbose_optimizer=False,
            **kwargs
    ):
        self.optimized_param_id = update_optimizer(
            self.optimizer,
            dict(self.model.named_parameters()),
            self.optimized_param_id,
            reset_state=reset_optimizer_state,
            remove_params=remove_params,
            verbose=verbose_optimizer,
        )

    def before_training_exp(self, experience: CLExperience):
        """Setup to train on a single experience."""
        # Data Adaptation (e.g. add new samples/data augmentation)
        self.train_dataset_adaptation(experience)
        self.make_train_dataloader()#(self.train_adapted_dataset) #, self.batch_size)
        # Model Adaptation (e.g. freeze/add new units)
        self.check_model_and_optimizer(experience)

    # self._after_training_epoch(**kwargs) CALL eval !!!!
    '''
    '''
    def _save_train_state(self, experience: CLExperience, **kwargs):
        """Save the training state, which may be modified by the eval loop.

        TODO: we probably need a better way to do this.
        """
        # save each layer's training mode, to restore it later
        _prev_model_training_modes = {}
        for name, layer in self.model.named_modules():
            _prev_model_training_modes[name] = layer.training

        _prev_state = {
            "experience": experience,
            "is_training": self.is_training,
            "model_training_mode": _prev_model_training_modes,
        }
        return _prev_state

    def _load_train_state(self, prev_state):
        # restore train-state variables and training mode.
        self.experience = prev_state["experience"]
        self.is_training = prev_state["is_training"]

        # restore each layer's training mode to original
        prev_training_modes = prev_state["model_training_mode"]
        for name, layer in self.model.named_modules():
            try:
                prev_mode = prev_training_modes[name]
                layer.train(mode=prev_mode)
            except KeyError:
                # Unknown parameter, probably added during the eval
                # model's adaptation. We set it to train mode.
                layer.train()
    '''
    '''
    @torch.no_grad()
    def eval(
        self,
        experiences: Union[TExperienceType, CLStream[TExperienceType]],
        experience: CLExperience,
        **kwargs,
    ):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        #if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            #self._check_distributed_training_compatibility()
            #self._distributed_check = True

        # eval can be called inside the train method.
        # Save the shared state here to restore before returning.
        prev_train_state = self.save_train_state(experience)
        self.is_training = False
        self.model.eval()

        experiences_list: Iterable[TExperienceType] = (
            _experiences_parameter_as_iterable(experiences)
        )
        self.current_eval_stream = experiences_list

        #self._before_eval(**kwargs)
        for self.experience in experiences_list:
            self.before_eval_exp(self.experience)#(experiences)
            self.eval_exp()
            #self._after_eval_exp(**kwargs)

        #from avalanche.evaluation.metric_definitions import GenericPluginMetric
        #GenericPluginMetric.after_eval_exp(self, )
        #GenericPluginMetric.after_eval(**kwargs)
        self._after_eval(**kwargs)
        #self.eval_cleanup()

        # restore previous shared state.
        self.load_train_state(prev_train_state)

    def eval_exp(self):
        self.eval_epoch()

    def eval_epoch(self):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.eval_dataloader:
            self.unpack_minibatch()
            #self._before_eval_iteration(**kwargs)

            #self._before_eval_forward(**kwargs)
            self.mb_output = self.model(self.mb_x)#self.mb_output = self.forward()
            #self._after_eval_forward(**kwargs)
            self.criterion(self.mb_output, self.mb_y) #loss = self.criterion(self.mb_output, self.mb_y)

            #self._after_eval_iteration(**kwargs)


    def before_eval_exp(self, experience: CLExperience):
        # Data Adaptation
        #self._before_eval_dataset_adaptation(**kwargs)
        self.eval_dataset_adaptation(experience)
        #self._after_eval_dataset_adaptation(**kwargs)
        self.make_eval_dataloader() #(self.train_adapted_dataset) #, self.batch_size)
        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation(experience)
    
    #def eval_cleanup had been commented out due to errors anyway
    #def eval_cleanup(self):
        #super()._eval_cleanup()
        # reset for faster serialization
        self.adapted_dataset = None
        self.dataloader = []
        self.mbatch = None
        self.mb_output = None
        self.loss = self.make_empty_loss()

    def make_empty_loss(self) -> torch.Tensor: #-> Tensor:
        return torch.zeros(1, device=self.device)

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
            "clock_train_epoch_counter": self.clock.train_exp_epochs
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
        self.ce_loss = checkpoint["ce_loss"]
        self.loss = checkpoint["loss"]
        self._criterion = checkpoint["_criterion"]
        self.clock = checkpoint["strategy"].clock
        self.clock.train_exp_counter = checkpoint["clock_train_exp_counter"]
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




if TRAINING_WITH_LOOP_LwF:

    #lwf_trainer = LwFTrainer(model=outdoors_specialised_model, device="cuda", lr=0.001, momentum=0.9, weight_decay=5e-4,temperature=2.0, alpha=1.0)
    lwf_trainer = LwFTrainer(model=outdoors_specialised_model, device="cuda", lr=0.0001, momentum=0.9, weight_decay=5e-4,
                             temperature=2.0, alpha=5.0,#1.0,
                             train_mb_size=100, train_epochs=5, eval_mb_size=100, #device=DEVICE,
                             plugins=[EarlyStoppingPlugin_mine(patience=3, val_stream_name="valid_stream",
                                                               metric_name="Loss_Exp", mode="min", peval_mode="epoch",
                                                               margin=0.3, verbose=True)],
                             evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True),
                                                        loss_metrics(experience=True, stream=True),
                                                        loggers=InteractiveLogger(), collect_all=True),
                             eval_every=1)

    print('Starting experiment WITH LwFTrainer')
    results = []
    checkpoint_dir_path = Path("C:\\Users\lamus\PycharmProjects\Masters_IndividualProject_220029768\\Eleventh_implementation_attempt\checkpoints")
    for experience in outdoors_specialised_models_train_stream_class_il: # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
        i = experience.current_experience # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of outdoors_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
        print("Start of experience -- current TRAIN experience: ", experience.current_experience)
        print("Start of experience -- current VAL experience: ",outdoors_specialised_models_val_stream_class_il.exps_iter[i].current_experience)
        print("Current Classes -- current TRAIN .classes_in_this_experience: ", experience.classes_in_this_experience)
        print("Current Classes -- current VAL .classes_in_this_experience: ",outdoors_specialised_models_val_stream_class_il.exps_iter[i].classes_in_this_experience)
        print("experience.dataset.coarse_label.uniques (coarse_labels of TRAIN)",experience.dataset.coarse_label.uniques)
        print("experience.dataset.targets (targets of TRAIN)", experience.dataset.targets.uniques)
        print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques (coarse_labels of VAL)",outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques)
        print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets (targets of VAL)",outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques)
        print("print(experience.origin_stream.name)", experience.origin_stream.name)
        print("print(outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)",outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)
        # print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
        # print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
        # criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
        # print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
        lwf_trainer.train_experience(experience, outdoors_specialised_models_val_stream_class_il)
        # print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
        # print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)
print('Training completed WITH LwFTrainer')


'''
    checkpoint_files = sorted(checkpoint_dir_path.glob("*.pkl"))
    for checkpoint_file in checkpoint_files:
        print(f"Loading {checkpoint_file}")
        cl_strategy_evaluation = load_entire_checkpoint(checkpoint_file)
        print(f"Computing val accuracy on {checkpoint_file.name} AFTER 'Training completed WITH LOOP VERSION 2'")
        cl_strategy_evaluation.eval(outdoors_specialised_models_val_stream_class_il)

'''
    #print('Computing accuracy on the whole val set OUTSIDE training LOOP')
    #cl_strategy.eval(outdoors_specialised_models_val_stream_class_il)
    #results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))
    #print("Computing accuracy on the whole val set OUTSIDE training LOOP (i.e. print(results)):", results)












######### Copy-paste of Avalanche eval classes and functionality so I can edit them NOT to use strategy objects: #################
################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta, Andrea Cossu                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from abc import ABC, abstractmethod
from typing import (
    Generic,
    TypeVar,
    Optional,
    TYPE_CHECKING,
    List,
    Union,
    overload,
    Literal,
    Protocol,
)
from avalanche.evaluation.metric_results import MetricValue, MetricType, AlternativeValues
#from avalanche.evaluation.metric_utils import (get_metric_name,generic_get_metric_name,default_metric_name_template,)

if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricResult
    #from avalanche.training.templates import SupervisedTemplate

TResult_co = TypeVar("TResult_co", covariant=True)
TMetric = TypeVar("TMetric", bound="Metric")


class Metric(Protocol[TResult_co]):
    """Standalone metric.

    A standalone metric exposes methods to reset its internal state and
    to emit a result. Emitting a result does not automatically cause
    a reset in the internal state.

    The specific metric implementation exposes ways to update the internal
    state. Usually, standalone metrics like :class:`Sum`, :class:`Mean`,
    :class:`Accuracy`, ... expose an `update` method.

    The `Metric` class can be used as a standalone metric by directly calling
    its methods.
    In order to automatically integrate the metric with the training and
    evaluation flows, you can use :class:`PluginMetric` class. The class
    receives events directly from the :class:`EvaluationPlugin` and can
    emits values on each callback. Usually, an instance of `Metric` is
    created within `PluginMetric`, which is then responsible for its
    update and results. See :class:`PluginMetric` for more details.
    """

    def result(self) -> Optional[TResult_co]:
        """
        Obtains the value of the metric.

        :return: The value of the metric.
        """
        pass

    def reset(self) -> None:
        """
        Resets the metric internal state.

        :return: None.
        """
        pass


class PluginMetric(Metric[TResult_co], ABC):
    """A metric that can be used together with :class:`EvaluationPlugin`.

    This class leaves the implementation of the `result` and `reset` methods
    to child classes while providing an empty implementation of the callbacks
    invoked by the :class:`EvaluationPlugin`. Subclasses should implement
    the `result`, `reset` and the desired callbacks to compute the specific
    metric.

    Remember to call the `super()` method when overriding
    `after_train_iteration` or `after_eval_iteration`.

    An instance of this class usually leverages a `Metric` instance to update,
    reset and emit metric results at appropriate times
    (during specific callbacks).
    """

    def __init__(self):
        """
        Creates an instance of a plugin metric.

        Child classes can safely invoke this (super) constructor as the first
        experience.
        """
        pass

    @abstractmethod
    def result(self) -> Optional[TResult_co]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def before_training(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_training_exp(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_train_dataset_adaptation(
        self, strategy: "LwFTrainer"
    ) -> "MetricResult":
        pass

    def after_train_dataset_adaptation(
        self, strategy: "LwFTrainer"
    ) -> "MetricResult":
        pass

    def before_training_epoch(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_training_iteration(
        self, strategy: "LwFTrainer"
    ) -> "MetricResult":
        pass

    def before_forward(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_forward(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_backward(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_backward(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_training_iteration(
        self, strategy: "LwFTrainer"
    ) -> "MetricResult":
        pass

    def before_update(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_update(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_training_epoch(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_training_exp(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_training(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_eval(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_eval_dataset_adaptation(
        self, strategy: "LwFTrainer"
    ) -> "MetricResult":
        pass

    def after_eval_dataset_adaptation(
        self, strategy: "LwFTrainer"
    ) -> "MetricResult":
        pass

    def before_eval_exp(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_eval_exp(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_eval(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_eval_iteration(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def before_eval_forward(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_eval_forward(self, strategy: "LwFTrainer") -> "MetricResult":
        pass

    def after_eval_iteration(self, strategy: "LwFTrainer") -> "MetricResult":
        pass


class GenericPluginMetric(PluginMetric[TResult_co], Generic[TResult_co, TMetric]):
    """
    This class provides a generic implementation of a Plugin Metric.
    The user can subclass this class to easily implement custom plugin
    metrics.
    """

    @overload
    def __init__(
        self,
        metric: TMetric,
        reset_at: Literal[
            "iteration", "epoch", "experience", "stream", "never"
        ] = "experience",
        emit_at: Literal["iteration", "epoch", "experience", "stream"] = "experience",
        mode: Literal["train"] = "train",
    ): ...

    @overload
    def __init__(
        self,
        metric: TMetric,
        reset_at: Literal["iteration", "experience", "stream", "never"] = "experience",
        emit_at: Literal["iteration", "experience", "stream"] = "experience",
        mode: Literal["eval"] = "eval",
    ): ...

    def __init__(
        self, metric: TMetric, reset_at="experience", emit_at="experience", mode="eval"
    ):
        super(GenericPluginMetric, self).__init__()
        assert mode in {"train", "eval"}
        if mode == "train":
            assert reset_at in {
                "iteration",
                "epoch",
                "experience",
                "stream",
                "never",
            }
            assert emit_at in {"iteration", "epoch", "experience", "stream"}
        else:
            assert reset_at in {"iteration", "experience", "stream", "never"}
            assert emit_at in {"iteration", "experience", "stream"}
        self._metric: TMetric = metric
        self._reset_at = reset_at
        self._emit_at = emit_at
        self._mode = mode

    def reset(self) -> None:
        self._metric.reset()

    def result(self):
        return self._metric.result()

    def update(self, strategy: "LwFTrainer"):
        pass

    def _package_result(self, strategy: "LwFTrainer") -> "MetricResult":
        metric_value = self.result()
        add_exp = self._emit_at == "experience"
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def before_training(self, strategy: "LwFTrainer"):
        super().before_training(strategy)
        if self._reset_at == "stream" and self._mode == "train":
            self.reset()

    def before_training_exp(self, strategy: "LwFTrainer"):
        super().before_training_exp(strategy)
        if self._reset_at == "experience" and self._mode == "train":
            self.reset()

    def before_training_epoch(self, strategy: "LwFTrainer"):
        super().before_training_epoch(strategy)
        if self._reset_at == "epoch" and self._mode == "train":
            self.reset()

    def before_training_iteration(self, strategy: "LwFTrainer"):
        super().before_training_iteration(strategy)
        if self._reset_at == "iteration" and self._mode == "train":
            self.reset()

    def after_training_iteration(self, strategy: "LwFTrainer"):
        super().after_training_iteration(strategy)
        if self._mode == "train":
            self.update(strategy)
        if self._emit_at == "iteration" and self._mode == "train":
            return self._package_result(strategy)

    def after_training_epoch(self, strategy: "LwFTrainer"):
        super().after_training_epoch(strategy)
        if self._emit_at == "epoch" and self._mode == "train":
            return self._package_result(strategy)

    def after_training_exp(self, strategy: "LwFTrainer"):
        super().after_training_exp(strategy)
        if self._emit_at == "experience" and self._mode == "train":
            return self._package_result(strategy)

    def after_training(self, strategy: "LwFTrainer"):
        super().after_training(strategy)
        if self._emit_at == "stream" and self._mode == "train":
            return self._package_result(strategy)

    def before_eval(self, strategy: "LwFTrainer"):
        super().before_eval(strategy)
        if self._reset_at == "stream" and self._mode == "eval":
            self.reset()

    def before_eval_exp(self, strategy: "LwFTrainer"):
        super().before_eval_exp(strategy)
        if self._reset_at == "experience" and self._mode == "eval":
            self.reset()

    def after_eval_exp(self, strategy: "LwFTrainer"):
        super().after_eval_exp(strategy)
        if self._emit_at == "experience" and self._mode == "eval":
            return self._package_result(strategy)

    def after_eval(self, strategy: "LwFTrainer"):
        #super().after_eval(strategy)
        if self._emit_at == "stream" and self._mode == "eval":
            return self._package_result(strategy)

    def after_eval_iteration(self, strategy: "LwFTrainer"):
        super().after_eval_iteration(strategy)
        if self._mode == "eval":
            self.update(strategy)
        if self._emit_at == "iteration" and self._mode == "eval":
            return self._package_result(strategy)

    def before_eval_iteration(self, strategy: "LwFTrainer"):
        super().before_eval_iteration(strategy)
        if self._reset_at == "iteration" and self._mode == "eval":
            self.reset()


class _ExtendedPluginMetricValue:
    """
    A data structure used to describe a metric value.

    Mainly used to compose the final "name" or "path" of a metric.

    For the moment, this class should be considered an internal utility. Use it
    at your own risk!
    """

    def __init__(
        self,
        *,
        metric_name: str,
        metric_value: Union[MetricType, AlternativeValues],
        phase_name: str,
        stream_name: Optional[str],
        experience_id: Optional[int],
        task_label: Optional[int],
        plot_position: Optional[int] = None,
        **other_info
    ):
        super().__init__()
        self.metric_name = metric_name
        """
        The name of metric, as a string (cannot be None).
        """

        self.metric_value = metric_value
        """
        The metric value name (cannot be None).
        """

        self.phase_name = phase_name
        """
        The phase name, as a str (cannot be None).
        """

        self.stream_name = stream_name
        """
        The stream name, as a str (can be None if stream-agnostic).
        """

        self.experience_id = experience_id
        """
        The experience id, as an int (can be None if experience-agnostic).
        """

        self.task_label = task_label
        """
        The task label, as an int (can be None if task-agnostic).
        """

        self.plot_position = plot_position
        """
        The x position of the value, as an int (cannot be None).
        """
        self.other_info = other_info
        """
        Additional info for this value, as a dictionary (may be empty).
        """


class _ExtendedGenericPluginMetric(
    GenericPluginMetric[List[_ExtendedPluginMetricValue], TMetric]
):
    """
    A generified version of :class:`GenericPluginMetric` which supports emitting
    multiple metrics from a single metric instance.
    Child classes need to emit metrics via `result()` as a list of
    :class:`ExtendedPluginMetricValue`.
    This is in contrast with :class:`GenericPluginMetric`, that expects a
    simpler dictionary "task_label -> value".

    The resulting metric name will be given by the implementation of the
    :meth:`metric_value_name` method.

    For the moment, this class should be considered an internal utility. Use it
    at your own risk!
    """

    def __init__(self, *args, **kwargs):
        """
        Creates an instance of an extended :class:`GenericPluginMetric`.

        :param args: The positional arguments to be passed to the
            :class:`GenericPluginMetric` constructor.

        :param kwargs: The named arguments to be passed to the
            :class:`GenericPluginMetric` constructor.
        """
        super().__init__(*args, **kwargs)

    def _package_result(self, strategy: "LwFTrainer") -> "MetricResult":
        emitted_values = self.result()
        default_plot_x_position = strategy.clock.train_iterations

        metrics = []
        for m_value in emitted_values:
            if not isinstance(m_value, _ExtendedPluginMetricValue):
                raise RuntimeError(
                    "Emitted a value that is not of type " "ExtendedPluginMetricValue"
                )

            m_name = self.metric_value_name(m_value)
            x_pos = m_value.plot_position
            if x_pos is None:
                x_pos = default_plot_x_position
            metrics.append(MetricValue(self, m_name, m_value.metric_value, x_pos))

        return metrics

    def metric_value_name(self, m_value: _ExtendedPluginMetricValue) -> str:
        return generic_get_metric_name(default_metric_name_template, vars(m_value))


__all__ = [
    "Metric",
    "PluginMetric",
    "GenericPluginMetric",
    "_ExtendedPluginMetricValue",
    "_ExtendedGenericPluginMetric",
]








################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import (
    Dict,
    Optional,
    Union,
    Iterable,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    List,
    Callable,
    Any,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray, arange
from torch import Tensor

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
    from avalanche.benchmarks.scenarios import ClassificationExperience
    from avalanche.evaluation import PluginMetric


EVAL = "eval"
TRAIN = "train"


def default_cm_image_creator(
    confusion_matrix_tensor: Tensor,
    display_labels: Optional[Iterable[Any]] = None,
    include_values=False,
    xticks_rotation=0,
    yticks_rotation=0,
    values_format=None,
    cmap="viridis",
    image_title="",
):
    """
    The default Confusion Matrix image creator.
    Code adapted from
    `Scikit learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html>`_ # noqa

    :param confusion_matrix_tensor: The tensor describing the confusion matrix.
        This can be easily obtained through Scikit-learn `confusion_matrix`
        utility.
    :param display_labels: Target names used for plotting. By default, `labels`
        will be used if it is defined, otherwise the values will be inferred by
        the matrix tensor.
    :param include_values: Includes values in confusion matrix. Defaults to
        `False`.
    :param xticks_rotation: Rotation of xtick labels. Valid values are
        float point value. Defaults to 0.
    :param yticks_rotation: Rotation of ytick labels. Valid values are
        float point value. Defaults to 0.
    :param values_format: Format specification for values in confusion matrix.
        Defaults to `None`, which means that the format specification is
        'd' or '.2g', whichever is shorter.
    :param cmap: Must be a str or a Colormap recognized by matplotlib.
        Defaults to 'viridis'.
    :param image_title: The title of the image. Defaults to an empty string.
    :return: The Confusion Matrix as a PIL Image.
    """

    fig, ax = plt.subplots()

    cm = confusion_matrix_tensor.numpy()
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i in range(n_classes):
            for j in range(n_classes):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

    if display_labels is None:
        display_labels = np.arange(n_classes)

    fig.colorbar(im_, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    if image_title != "":
        ax.set_title(image_title)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.setp(ax.get_yticklabels(), rotation=yticks_rotation)

    fig.tight_layout()
    return fig


SEABORN_COLORS = (
    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
)


def repartition_pie_chart_image_creator(
    label2counts: Dict[int, List[int]],
    counters: List[int],
    colors: Union[ndarray, Iterable, int, float] = SEABORN_COLORS,
    fmt: str = "%1.1f%%",
):
    """
    Create a pie chart representing the labels repartition.

    :param label2counts: A dict holding the counts for each label, of the form
        {label: [count_at_step_0, count_at_step_1, ...]}. Only the last count of
        each label is used here.
    :param counters: (unused) The steps the counts were taken at.
    :param colors: The colors to use in the chart.
    :param fmt: Formatting used to display the text values in the chart.
    """
    ax: Axes
    fig, ax = plt.subplots()

    labels, counts = zip(*((label, c[-1]) for label, c in label2counts.items()))

    ax.pie(counts, labels=labels, autopct=fmt, colors=colors)

    fig.tight_layout()
    return fig


def repartition_bar_chart_image_creator(
    label2counts: Dict[int, List[int]],
    counters: List[int],
    colors: Union[ndarray, Iterable, int, float] = SEABORN_COLORS,
):
    """
    Create a bar chart representing the labels repartition.

    :param label2counts: A dict holding the counts for each label, of the form
        {label: [count_at_step_0, count_at_step_1, ...]}. Only the last count of
        each label is used here.
    :param counters: (unused) The steps the counts were taken at.
    :param colors: The colors to use in the chart.
    """
    ax: Axes
    fig, ax = plt.subplots()

    y = -arange(len(label2counts))
    labels, counts = zip(*((label, c[-1]) for label, c in label2counts.items()))
    total = sum(counts)

    ax.barh(y, width=counts, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Number of exemplars")
    ax.set_ylabel("Class")

    for i, count in enumerate(counts):
        ax.text(count / 2, -i, f"{count/total:.1%}", va="center", ha="center")

    fig.tight_layout()
    return fig


def default_history_repartition_image_creator(
    label2counts: Dict[int, List[int]],
    counters: List[int],
    colors: Union[ndarray, Iterable, int, float] = SEABORN_COLORS,
):
    """
    Create a stack plot representing the labels repartition with their history.

    :param label2counts: A dict holding the counts for each label, of the form
        {label: [count_at_step_0, count_at_step_1, ...]}.
    :param counters: The steps the counts were taken at.
    :param colors: The colors to use in the chart.
    """
    ax: Axes
    fig, ax = plt.subplots()

    ax.stackplot(
        counters,
        label2counts.values(),
        labels=label2counts.keys(),
        colors=colors,
    )
    ax.legend(loc="upper left")
    ax.set_ylabel("Number of examples")
    ax.set_xlabel("step")

    fig.tight_layout()
    return fig


def stream_type(experience: "ClassificationExperience") -> str:
    """
    Returns the stream name from which the experience belongs to.
    e.g. the experience can be part of train or test stream.

    :param experience: the instance of the experience
    """

    return experience.origin_stream.name


def phase_and_task(strategy: "LwFTrainer") -> Tuple[str, int]:
    """
    Returns the current phase name and the associated task label.

    The current task label depends on the phase. During the training
    phase, the task label is the one defined in the "train_task_label"
    field. On the contrary, during the eval phase the task label is the one
    defined in the "eval_task_label" field.

    :param strategy: The strategy instance to get the task label from.
    :return: The current phase name as either "Train" or "Task" and the
        associated task label.
    """
    task_labels = getattr(strategy.experience, "task_labels", None)
    if task_labels is not None:
        task = task_labels
        if len(task) > 1:
            task = None  # task labels per patterns
        else:
            task = task[0]
    else:
        task = None

    if strategy.is_eval:
        return EVAL, task
    else:
        return TRAIN, task


def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ("K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return "%.1f%s" % (value, s)
    return "%sB" % n


def default_metric_name_template(metric_info: Dict[str, Any]):
    add_task = metric_info.get("task_label", None) is not None
    add_phase = metric_info.get("phase_name", None) is not None
    add_experience = metric_info.get("experience_id", None) is not None
    add_stream = metric_info.get("stream_name", None) is not None

    if "metric_name" not in metric_info:
        raise RuntimeError("Key metric_name missing from value dictionary.")

    result_template = "{metric_name}/"
    if add_phase:
        result_template += "{phase_name}_phase/"

    if add_stream:
        result_template += "{stream_name}_stream/"

    if add_task:
        result_template += "Task{task_label:03}/"

    if add_experience:
        result_template += "Exp{experience_id:03}/"

    return result_template[:-1]


def generic_get_metric_name(
    value_name_template: Union[str, Callable[[Dict[str, Any]], str]],
    metric_info: Dict[str, Any],
):
    if isinstance(value_name_template, str):
        name_template = value_name_template
    else:
        name_template = value_name_template(metric_info)

    # https://stackoverflow.com/a/17895844
    return name_template.format(**metric_info)


def get_metric_name(
    metric: Union["PluginMetric", str],
    strategy: "LwFTrainer",
    add_experience=False,
    add_task=True,
):
    """
    Return the complete metric name used to report its current value.
    The name is composed by:
    metric string representation /phase type/stream type/task id
    where metric string representation is a synthetic string
    describing the metric, phase type describe if the user
    is training (train) or evaluating (eval), stream type describes
    the type of stream the current experience belongs to (e.g. train, test)
    and task id is the current task label.

    :param metric: the metric object for which return the complete name
    :param strategy: the current strategy object
    :param add_experience: if True, add eval_exp_id to the main metric name.
            Default to False.
    :param add_task: if True the main metric name will include the task
        information. If False, no task label will be displayed.
        If an int, that value will be used as task label for the metric name.
    """
    task_label: Optional[int]
    phase_name, task_label = phase_and_task(strategy)
    assert strategy.experience is not None
    stream = stream_type(strategy.experience)
    experience_id = strategy.experience.current_experience
    if type(add_task) == bool and add_task is False:
        task_label = None
    elif type(add_task) == int:
        task_label = add_task

    if not add_experience:
        experience_id = None

    return generic_get_metric_name(
        default_metric_name_template,
        {
            "metric_name": str(metric),
            "task_label": task_label,
            "phase_name": phase_name,
            "experience_id": experience_id,
            "stream_name": stream,
        },
    )


__all__ = [
    "default_cm_image_creator",
    "phase_and_task",
    "default_metric_name_template",
    "generic_get_metric_name",
    "get_metric_name",
    "stream_type",
    "bytes2human",
    "default_history_repartition_image_creator",
    "repartition_pie_chart_image_creator",
    "repartition_bar_chart_image_creator",
]






'''
################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24/07/2021                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
#from avalanche.training.plugins import SupervisedPlugin


class Clock():#(SupervisedPlugin, supports_distributed=True):
    """Counter for strategy events.

    WARNING: Clock needs to be the last plugin, otherwise counters will be
    wrong for plugins called after it.
    """

    def __init__(self):
        """Init."""
        #super().__init__()
        # train
        self.train_iterations = 0
        """ Total number of training iterations. """

        self.train_exp_counter = 0
        """ Number of past training experiences. """

        self.train_exp_epochs = 0
        """ Number of training epochs for the current experience. """

        self.train_exp_iterations = 0
        """ Number of training iterations for the current experience. """

        self.train_epoch_iterations = 0
        """ Number of iterations for the current epoch. """

        self.total_iterations = 0
        """ Total number of iterations in training and eval mode. """

    def before_training_exp(self, strategy, **kwargs):
        self.train_exp_iterations = 0
        self.train_exp_epochs = 0

    def before_training_epoch(self, strategy, **kwargs):
        self.train_epoch_iterations = 0

    def after_training_iteration(self, strategy, **kwargs):
        self.train_epoch_iterations += 1
        self.train_exp_iterations += 1
        self.train_iterations += 1
        self.total_iterations += 1

    def after_training_epoch(self, strategy, **kwargs):
        self.train_exp_epochs += 1

    def after_training_exp(self, strategy, **kwargs):
        self.train_exp_counter += 1

    def after_eval_iteration(self, strategy, **kwargs):
        self.total_iterations += 1


__all__ = ["Clock"]
'''