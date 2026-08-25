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
from avalanche.training.templates.base import TExperienceType

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

from avalanche.benchmarks import CLExperience, SplitMNIST, AvalancheDataset, OnlineCLExperience
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
'''
"""Implements Learning without Forgetting (LwF)."""
from __future__ import division, print_function

import argparse
import copy
import json
import warnings

import dataset
import networks as net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
import utils as utils
from prune import SparsePruner
from torch.autograd import Variable
from tqdm import tqdm

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode',
                   choices=['finetune', 'eval'],
                   help='Run mode')
FLAGS.add_argument('--finetune_layers',
                   choices=['all', 'fc', 'classifier'], default='all',
                   help='Which layers to finetune')
FLAGS.add_argument('--num_outputs', type=int, default=-1,
                   help='Num outputs for dataset')
# Optimization options.
FLAGS.add_argument('--lr', type=float,
                   help='Learning rate')
FLAGS.add_argument('--lr_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--lr_decay_factor', type=float,
                   help='Multiply lr by this much every step of decay')
FLAGS.add_argument('--finetune_epochs', type=int,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--batch_size', type=int, default=32,
                   help='Batch size')
FLAGS.add_argument('--dropout', type=float, default=0.5,
                   help='Dropout ratio')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4,
                   help='Weight decay')
FLAGS.add_argument('--temperature', type=float, default=2.0,
                   help='LwF logit temperature')
FLAGS.add_argument('--ft_shared_after', type=int, default=5,
                   help='Finetune shared layers after this epoch')
# Paths.
FLAGS.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/',
                   help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='',
                   help='Location to save model')
# Other.
FLAGS.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')


def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * scale


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, original_model, model, dataset2idx):
        self.args = args
        self.cuda = args.cuda
        self.dataset2idx = dataset2idx
        self.original_model = original_model
        self.model = model

        if args.mode != 'check':
            # Set up data loader, criterion, and pruner.
            if 'cropped' in args.train_path:
                train_loader = dataset.train_loader_cropped
                test_loader = dataset.test_loader_cropped
            else:
                train_loader = dataset.train_loader
                test_loader = dataset.test_loader
            self.train_data_loader = train_loader(
                args.train_path, args.batch_size, pin_memory=args.cuda)
            self.test_data_loader = test_loader(
                args.test_path, args.batch_size, pin_memory=args.cuda)
            self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None

        print('Performing eval...')
        for batch, label in tqdm(self.test_data_loader, desc='Eval'):
            if self.cuda:
                batch = batch.cuda()
            batch = Variable(batch, volatile=True)

            output = self.model(batch)

            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))
        self.model.train()
        return errors

    def do_batch(self, optimizer, batch, label, epoch_idx):
        """Runs model for one batch."""
        batch_original = batch.clone()
        if self.cuda:
            batch_original = batch_original.cuda(1)
            batch = batch.cuda()
            label = label.cuda()
        batch_original = Variable(batch_original, requires_grad=False)
        batch = Variable(batch)
        label = Variable(label)

        # Get targets using original model.
        self.original_model.eval()
        x = self.original_model.shared(batch_original)
        target_logits = [classifier(x).data.cpu()
                         for classifier in self.original_model.classifiers]
        # Move to same GPU as current model.
        target_logits = [Variable(item.cuda(), requires_grad=False)
                         for item in target_logits]
        scale = [item.size(-1) for item in target_logits]

        # Work with current model.
        # Set grads to 0.
        self.model.zero_grad()

        # Do forward.
        x = self.model.shared(batch)
        pred_logits = [classifier(x) for classifier in self.model.classifiers]

        # Compute loss.
        dist_loss = 0
        # Apply distillation loss to all old tasks.
        for idx in range(len(target_logits)):
            dist_loss += distillation_loss(
                pred_logits[idx], target_logits[idx], self.args.temperature, scale[idx])
        # Apply cross entropy for current task.
        output = pred_logits[-1]
        new_loss = self.criterion(output, label)
        loss = dist_loss + new_loss

        # Do backward.
        loss.backward()

        if epoch_idx <= self.args.ft_shared_after:
            # Set shared layer gradients to 0 if early epochs.
            for module in self.model.shared.modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)
            # Set old classifier gradients to 0 if early epochs.
            for idx in range(len(self.model.classifiers) - 1):
                module = self.model.classifiers[idx]
                module.weight.grad.data.fill_(0)
                module.bias.grad.data.fill_(0)

        # Update params.
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label, epoch_idx)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        base_model = self.model
        if isinstance(self.model, nn.DataParallel):
            base_model = self.model.module

        # Prepare the ckpt.
        ckpt = {
            'epoch': epoch,
            'dataset2idx': self.dataset2idx,
            'accuracy': best_accuracy,
            'errors': errors,
            'args': self.args,
            'model': base_model,
        }

        # Save to file.
        torch.save(ckpt, savename + '.pt')

    def train(self, epochs, optimizer, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            optimizer = utils.step_lr(epoch_idx, self.args.lr, self.args.lr_decay_every,
                                      self.args.lr_decay_factor, optimizer)
            self.model.train()
            self.do_epoch(epoch_idx, optimizer)
            errors = self.eval()
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)

    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))


def main():
    """Do stuff."""
    args = FLAGS.parse_args()

    # Set default train and test path if not provided as input.
    if not args.train_path:
        args.train_path = '../data/%s/train' % (args.dataset)
    if not args.test_path:
        if args.dataset == 'imagenet' or args.dataset == 'places':
            args.test_path = '../data/%s/val' % (args.dataset)
        else:
            args.test_path = '../data/%s/test' % (args.dataset)

    # Load the required model.
    if 'finetune' in args.mode and not args.loadname:
        model = net.ModifiedVGG16()
    else:
        ckpt = torch.load(args.loadname)
        model = ckpt['model']
        dataset2idx = ckpt['dataset2idx']

    original_model = copy.deepcopy(model)

    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_outputs)
    model.set_dataset(args.dataset)
    if args.dataset not in dataset2idx:
        idxs = [dataset2idx[key] for key in dataset2idx]
        dataset2idx[args.dataset] = max(idxs) + 1
    if args.cuda:
        model = model.cuda(0)
        if args.mode == 'finetune':
            original_model = original_model.cuda(1)

    # Create the manager object.
    manager = Manager(args, original_model, model, dataset2idx)

    # Perform necessary mode operations.
    if args.mode == 'finetune':
        # Get optimizer with correct params.
        if args.finetune_layers == 'all':
            params_to_optimize = model.parameters()
        elif args.finetune_layers == 'classifier':
            for param in model.shared.parameters():
                param.requires_grad = False
            params_to_optimize = model.classifier.parameters()
        elif args.finetune_layers == 'fc':
            params_to_optimize = []
            # Add fc params.
            for param in model.shared.parameters():
                if param.size(0) == 4096:
                    param.requires_grad = True
                    params_to_optimize.append(param)
                else:
                    param.requires_grad = False
            # Add classifier params.
            for param in model.classifier.parameters():
                params_to_optimize.append(param)
            params_to_optimize = iter(params_to_optimize)
        optimizer = optim.SGD(params_to_optimize, lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
        # Perform finetuning.
        manager.train(args.finetune_epochs, optimizer,
                      save=True, savename=args.save_prefix)
    elif args.mode == 'check':
        # Load model and make sure everything is fine.
        manager.check(verbose=True)
    elif args.mode == 'eval':
        # Just run the model on the eval set.
        manager.eval()


if __name__ == '__main__':
    main()
'''






import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

'''
def train_dataset_adaptation(self, **kwargs):
    """Initialize `self.adapted_dataset`."""
    assert self.experience is not None
    self.adapted_dataset = self.experience.dataset
    assert self.adapted_dataset is not None
    self.adapted_dataset = self.adapted_dataset.train()


    def eval_dataset_adaptation(self, **kwargs):
        """Initialize `self.adapted_dataset`."""
        assert self.experience is not None
        self.adapted_dataset = self.experience.dataset
        assert self.adapted_dataset is not None
        self.adapted_dataset = self.adapted_dataset.eval()

def make_train_dataloader(
        self,
        num_workers=0,
        shuffle=True,
        pin_memory=None,
        persistent_workers=False,
        drop_last=False,
        **kwargs
    ):
    assert self.adapted_dataset is not None
    self.dataloader = DataLoader(self.adapted_dataset, num_workers=0, shuffle=True,pin_memory=None,persistent_workers=False,drop_last=False,)

def make_eval_dataloader(
        self,
        num_workers=0,
        shuffle=False,
        pin_memory=None,
        persistent_workers=False,
        **kwargs
    ):
    assert self.adapted_dataset is not None
    self.dataloader = SingleDatasetDataLoader(self.adapted_dataset,num_workers=0,shuffle=False,pin_memory=None,persistent_workers=False,) # BUT I WILL BE REPLACING THIS SingleDatasetDataLoader
    # WITH normal PyTorch DataLoader!!!! I do not think it will interfere, as SingleDatasetDataLoader is Avalanche functionality that allows you to use functionality of its parent class
    # MultiDatasetDataLoader, which is a Custom data loader for multiple datasets -- for combining minibatches of several datasets into a single dataset, which I do not think I need so,
    # I believe I can savely switch to using PyTorch Dataloader

    # THE IMPORTANT THING TO REMEBER FROM THIS IS --> Dataloader for train has shuffle=True, and that of eval has shuffle=False. The dataloaders for train and eval do take experience.dataset
    # as the dataset, BUT they take the "adapted_dataset" version, which as can be seen from the Avalanche source code I have pasted, is simply self.adapted_dataset.train(), a class
    # AvalancheDataset function that returns the train transformation of the dataset, which in my case, given the previous data manipulation, should be the entire experience being passed to
    # train. Same thing with eval. THIS IS STILL ALL DATA PREPARATION, FOR WHICH I STILL INTEND TO USE THE AVvalanche FUNCTIONALITY, hence why this and knowing what data I should be passing
    # to the Dataloaders according to Avalanche, IS RELEVANT!!!!!!!!!!
'''

'''
from avalanche.models import avalanche_forward
from avalanche.training.templates.strategy_mixin_protocol import (
    SupervisedStrategyProtocol,
    TSGDExperienceType,
    TMBInput,
    TMBOutput,
)
import torch

# Types are perfectly ok for MyPy
# Also confirmed here: https://stackoverflow.com/a/70907644
# PyLance just does not understand it
class SupervisedProblem(
    SupervisedStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def criterion(self):
        """Loss function for supervised problems."""
        return self._criterion(self.mb_output, self.mb_y)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        # use task-aware forward only for task-aware benchmarks
        if hasattr(self.experience, "task_labels"):
            return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        else:
            return self.model(self.mb_x)

    def _unpack_minibatch(self):
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


'''


#class LwFTrainer(Generic[TSGDExperienceType, TMBInput, TMBOutput], BaseStrategyProtocol[TSGDExperienceType],Protocol[TSGDExperienceType, TMBInput, TMBOutput],):
'''    
    mbatch: Optional[TMBInput]

    mb_output: Optional[TMBOutput]

    dataloader: Iterable[TMBInput]

    _stop_training: bool

    optimizer: Optimizer

    loss: Tensor

    _criterion: CriterionType
'''

'''class Data_Preparation(BaseSGDTemplate):
    self._before_training_exp(self, **kwargs):'''

'''data_preparation_object = BaseSGDTemplate
data_preparation_object.train_dataset_adaptation(experience=outdoors_specialised_models_test_stream_class_il, )
data_preparation_object.eval_dataset_adaptation()
data_preparation_object.make_train_dataloader()
data_preparation_object.make_eval_dataloader()'''

from torch.utils.data.dataset import Dataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_definitions import T_co


def model_adaptation(self, model=None):
    """Adapts the model to the current data.

    Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
    """
    if model is None:
        model = self.model

    assert self.experience is not None

    # For training:
    if isinstance(self.experience, OnlineCLExperience) and self.is_training:
        if self.experience.access_task_boundaries:
            avalanche_model_adaptation(model, self.experience.origin_experience)
        else:
            avalanche_model_adaptation(model, self.experience)
    else:
        avalanche_model_adaptation(model, self.experience)

    return model.to(self.device)


def make_optimizer(
        self,
        reset_optimizer_state=False,
        remove_params=False,
        verbose_optimizer=False,
        **kwargs
):
    """Optimizer initialization.

    Called before each training experience to configure the optimizer.

    :param reset_optimizer_state: bool, whether to reset the
        state of the optimizer, defaults to False
    :param remove_params: bool, whether to remove parameters that
                          are in the optimizer but not found in the current model
    :param verbose_optimizer: bool, print optimized parameters
                              along with their parameter group

    Warnings:
        - The first time this function is called
          for a given strategy it will reset the
          optimizer to gather the (name, param)
          correspondance of the optimized parameters
          all of the model parameters will be put in the
          optimizer, regardless of what parameters are
          initially put in the optimizer.

    """
    self.optimized_param_id = update_optimizer(
        self.optimizer,
        dict(self.model.named_parameters()),
        self.optimized_param_id,
        reset_state=reset_optimizer_state,
        remove_params=remove_params,
        verbose=verbose_optimizer,
    )


def check_model_and_optimizer(self, reset_optimizer_state=False, **kwargs):
    # If strategy has access to the task boundaries, and the current
    # sub-experience is the first sub-experience in the online stream,
    # then adapt the model with the full origin experience:
    assert self.experience is not None

    if self.optimized_param_id is None:
        self.make_optimizer(reset_optimizer_state=True, **kwargs)

    if _at_task_boundary(self.experience, before=True):
        self.model = self.model_adaptation()
        self.make_optimizer(reset_optimizer_state=reset_optimizer_state, **kwargs)


def train_dataset_adaptation(experiences: Union[TExperienceType, Iterable[TExperienceType]]):# -> AvalancheDataset[T_co]: #(experience: Optional[TExperienceType]):
    """Initialize `self.adapted_dataset`."""
    assert experiences is not None
    adapted_dataset = experiences.dataset
    assert adapted_dataset is not None
    adapted_dataset = adapted_dataset.train()
    return adapted_dataset


#outdoors_train_adapted_dataset = train_dataset_adaptation(experiences=outdoors_specialised_models_train_stream_class_il)
outdoors_train_adapted_dataset = train_dataset_adaptation(experiences=outdoors_specialised_models_train_stream_class_il.exps_iter[0])


def make_train_dataloader(
        #experiences: Union[TExperienceType, Iterable[TExperienceType]],
        adapted_dataset: Optional[Dataset], #AvalancheDataset[T_co],
        batch_size: int,
        num_workers=0,
        shuffle=True,
        pin_memory=None,
        persistent_workers=False,
        drop_last=False,
        **kwargs
):
    """Data loader initialization.

    Called at the start of each learning experience after the dataset
    adaptation.

    :param num_workers: number of thread workers for the data loading.
    :param shuffle: True if the data should be shuffled, False otherwise.
    :param pin_memory: If True, the data loader will copy Tensors into CUDA
        pinned memory before returning them. Defaults to True.
    """

    assert adapted_dataset is not None

    # CODE FOR "other_dataloader_args = self._obtain_common_dataloader_parameters(..." DELETED, as it does not seem like I needed for the reasons explained above

    # CODE FOR "use task-balanced dataloader for task-aware benchmarks" DELETED since it was causing data type issues and I am not gonna be using task-aware/task-IL benchmark (AT LEAST NOT FOR NOW!)
    dataloader = DataLoader(adapted_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            drop_last=drop_last,)

    return dataloader

outdoors_train_dataloader = make_train_dataloader(outdoors_train_adapted_dataset, batch_size=32)


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

'''
def criterion(self):
    """Loss function for supervised problems."""
    return self._criterion(self.mb_output, self.mb_y)


def forward(self):
    """Compute the model's output given the current mini-batch."""
    # use task-aware forward only for task-aware benchmarks
    if hasattr(self.experience, "task_labels"):
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
    else:
        return self.model(self.mb_x)

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

'''
# class LwFTrainer(SGDStrategyProtocol):
#class LwFTrainer(SGDStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]):
class LwFTrainer():
    def __init__(
        self,
        model: nn.Module,
        train_adapted_dataset,
        train_dataloader: DataLoader,
        #val_adapted_dataset,
        #val_dataloader,
        device="cuda",
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        temperature=2.0,
        alpha=1.0,
        #batch_size=32,
    ):

        self.adapted_dataset = None
        self.optimized_param_id = None
        self.model = model.to(device)
        self.device = device

        self.temperature = temperature
        self.alpha = alpha
        self.batch_size = train_dataloader.batch_size

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()
        #self.dataloader: Iterable[Any] = []
        #""" Dataloader. """
        #self.train_dataloader: Iterable[Any] = train_dataloader
        self.train_dataloader= train_dataloader
        #self.val_dataloader = val_dataloader
        self.train_adapted_dataset = train_adapted_dataset
        #self.val_adapted_dataset = val_adapted_dataset

        self.mbatch: Optional[TMBInput] = None
        """ Current mini-batch. """

        self.mb_output: Optional[TMBOutput] = None
        """ Model's output computed on the current mini-batch. """

        ####self._stop_training = False

        # Teacher from previous experience
        self.teacher = None

        # Classes learned before current experience
        self.old_classes = set()
        '''
        self.dataloader = DataLoader(
            CLExperience.dataset.train(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False
        )
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
            reduction="mean",
        ) * scale

    def train_experience(self, experience: CLExperience, epochs=1): #def train_experience(self, experience: CLExperience, eval_streams=None, epochs=1, ** kwargs):

        '''train_dataloader = DataLoader(
            experience.dataset.train(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False
        )'''
        self.before_training_exp(experience)

        self.model.train()

        for epoch in range(epochs):

            total_ce = 0.0
            total_dist = 0.0
            total_loss = 0.0
            n=0

            for self.mbatch in self.train_dataloader: # for batch, targets in train_dataloader:

                self.unpack_minibatch()

                #batch = batch.to(self.device)
                targets = experience.dataset.targets #.uniques

                self.optimizer.zero_grad()

                # ------------------------------------------------
                # Current model
                # ------------------------------------------------

                student_logits = self.model(self.mbatch)#student_logits = self.model(batch)

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
                        teacher_logits = self.teacher(self.mbatch)#teacher_logits = self.teacher(batch)

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

                n += 1 # n = len(train_dataloader)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"CE: {total_ce / n:.6f} | "
                f"LwF: {total_dist / n:.6f} | "
                f"Total: {total_loss / n:.6f}"
            )

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
        """Optimizer initialization.

        Called before each training experience to configure the optimizer.

        :param reset_optimizer_state: bool, whether to reset the
            state of the optimizer, defaults to False
        :param remove_params: bool, whether to remove parameters that
                              are in the optimizer but not found in the current model
        :param verbose_optimizer: bool, print optimized parameters
                                  along with their parameter group

        Warnings:
            - The first time this function is called
              for a given strategy it will reset the
              optimizer to gather the (name, param)
              correspondance of the optimized parameters
              all of the model parameters will be put in the
              optimizer, regardless of what parameters are
              initially put in the optimizer.

        """
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
        self.make_train_dataloader(self.train_adapted_dataset, self.batch_size)
        # Model Adaptation (e.g. freeze/add new units)
        self.check_model_and_optimizer(experience)

    def train_dataset_adaptation(self, experience: CLExperience):#experiences: Union[TExperienceType, Iterable[TExperienceType]]):  # -> AvalancheDataset[T_co]: #(experience: Optional[TExperienceType]):
        """Initialize `self.adapted_dataset`."""
        assert experience is not None
        self.adapted_dataset = experience.dataset
        assert self.adapted_dataset is not None
        self.adapted_dataset = self.adapted_dataset.train()
        #return self.adapted_dataset

    def make_train_dataloader(self,
            # experiences: Union[TExperienceType, Iterable[TExperienceType]],
            adapted_dataset: Optional[Dataset],  # AvalancheDataset[T_co],
            batch_size: int,
            num_workers=0,
            shuffle=True,
            pin_memory=None,
            persistent_workers=False,
            drop_last=False,
            **kwargs
    ):
        assert adapted_dataset is not None
        # CODE FOR "other_dataloader_args = self._obtain_common_dataloader_parameters(..." DELETED, as it does not seem like I needed for the reasons explained above
        # CODE FOR "use task-balanced dataloader for task-aware benchmarks" DELETED since it was causing data type issues and I am not gonna be using task-aware/task-IL benchmark (AT LEAST NOT FOR NOW!)
        dataloader = DataLoader(adapted_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=shuffle,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers,
                                drop_last=drop_last, )

        #return dataloader



if TRAINING_WITH_LOOP_LwF:

    lwf_trainer = LwFTrainer(model=outdoors_specialised_model, train_adapted_dataset =outdoors_train_adapted_dataset, train_dataloader=outdoors_train_dataloader, device="cuda", lr=0.001, momentum=0.9, weight_decay=5e-4,temperature=2.0, alpha=1.0)

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
        lwf_trainer.train_experience(experience, 100)
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

