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
"""
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.training.plugins.checkpoint import CheckpointPlugin
from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate

from Ninth_implementation_attempt.Config_v9 import set_seed


set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Ninth_implementation_attempt.Config_v9 import set_device
DEVICE=set_device()
from avalanche.training.determinism.rng_manager import RNGManager
RNGManager.set_random_seeds(1234)
from Ninth_implementation_attempt.ResNet20_v9 import ResNet20
from Ninth_implementation_attempt.Load_cifar100_v9 import (outdoors_specialised_models_train_subset)
from Ninth_implementation_attempt.Load_cifar100_v9 import (outdoors_specialised_models_val_subset)
from Ninth_implementation_attempt.Load_cifar100_v9 import (outdoors_specialised_models_test_subset)
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.optim import SGD
from avalanche.training.supervised import Naive, EWC  # and many more!
from avalanche.training.losses import MaskedCrossEntropy
from avalanche.training.plugins import EarlyStoppingPlugin, EvaluationPlugin, EWCPlugin
from collections.abc import Iterable # OOOR like in avalanche/training/templates/base_sgd.py , from typing import Any, Callable, Generic, Iterable, Sequence, Optional, TypeVar, Union
from avalanche.evaluation.metrics.accuracy import accuracy_metrics
from avalanche.core import BasePlugin
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins.evaluation import default_loggers
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import CLExperience, SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    class_accuracy_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger
from avalanche.models import SimpleMLP, as_multitask
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin
from avalanche.training.supervised import Naive

DEBUG = False #TODO: Remember to change DEBUG to False if running experiments.

print("\n\n\n################################ HERE BEGINS Train_ResNet20_Outdoors_v9.py ################################ \n\n\n")

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
from avalanche.training.templates.common_templates import SupervisedTemplate
import sys
from typing import Callable, Sequence, Optional, TypeVar, Union, List
import torch

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



class TestEWCStrategy(SupervisedTemplate):
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

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        self.saved_params = {}
        self.importances = {}

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
            ce_loss = self.loss

            print(
                "Normal loss / CE loss (before EWC loss):",
                ce_loss.item()
            )

            if len(self.saved_params) > 0:
                penalty = self.ewc_penalty()

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

    def ewc_penalty(self):

        penalty = self._make_empty_loss()

        for experience in self.saved_params:

            for name, parameter in self.model.named_parameters():

                if name not in self.saved_params[experience]:
                    continue

                saved_parameter = self.saved_params[experience][name]
                importance = self.importances[experience][name]

                penalty += (
                        importance *
                        (
                                parameter -
                                saved_parameter
                        ).pow(2)
                ).sum()

        return penalty

    def _calculate_fisher(self):

        fisher = {}

        # Initialize Fisher tensors
        for name, parameter in self.model.named_parameters():

            if parameter.requires_grad:
                fisher[name] = torch.zeros_like(parameter)

        # Save current training mode
        was_training = self.model.training

        self.model.eval()

        # ---------------------------------------------------------
        # Iterate over the data from the current experience
        # ---------------------------------------------------------

        for mbatch in self.dataloader:

            self.mbatch = mbatch

            self._unpack_minibatch()

            self.optimizer.zero_grad()

            # Forward
            self.mb_output = self.forward()

            # Compute loss
            loss = self.criterion()

            # Compute gradients
            loss.backward()

            # Accumulate squared gradients
            for name, parameter in self.model.named_parameters():

                if parameter.grad is None:
                    continue

                fisher[name] += parameter.grad.detach().pow(2)

        # ---------------------------------------------------------
        # Average over minibatches
        # ---------------------------------------------------------

        num_batches = len(self.dataloader)

        if num_batches > 0:

            for name in fisher:
                fisher[name] /= num_batches

        # Restore original training mode
        self.model.train(was_training)

        return fisher

    def _after_training_exp(self, **kwargs):
        """
        Called after all epochs of the current experience have completed.

        At this point:

            1. The current experience has been trained.
            2. self.model contains the trained parameters theta*.
            3. We calculate Fisher using the current experience.
            4. We save theta*.
            5. Future experiences can use these values for EWC.
        """

        if self.experience is None:
            raise RuntimeError(
                "[EWC] self.experience is None in _after_training_exp."
            )

        experience_id = self.experience.current_experience

        print(
            f"[EWC] ===== End of experience {experience_id} ====="
        )

        # ---------------------------------------------------------
        # IMPORTANT ORDER:
        #
        # The model is currently at theta*.
        #
        # Calculate Fisher around this trained model and then save
        # theta*.
        # ---------------------------------------------------------

        self._calculate_fisher()

        self._save_parameters()

        print(
            f"[EWC] saved_params experiences: "
            f"{list(self.saved_params.keys())}"
        )

        print(
            f"[EWC] importances experiences: "
            f"{list(self.importances.keys())}"
        )

        # ---------------------------------------------------------
        # Allow Avalanche's normal after_training_exp processing
        # to happen.
        # ---------------------------------------------------------

        super()._after_training_exp(**kwargs)

    def _save_parameters(self):
        """
        Save the model parameters at the end of the current experience.

        These parameters represent theta* in the EWC objective:

            sum_i F_i * (theta_i - theta_i*)^2
        """

        experience_id = self.experience.current_experience

        self.saved_params[experience_id] = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

        print(
            f"[EWC] Saved parameters for experience {experience_id}: "
            f"{len(self.saved_params[experience_id])} tensors"
        )











from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.strategy_mixin_protocol import CriterionType


'''class EWCStrategy(SupervisedTemplate):
    """
    EWC implemented as a Strategy rather than a Plugin.

    The penalty is added to self.loss inside training_epoch, BEFORE
    _before_backward(**kwargs) is dispatched to plugins. This guarantees
    the penalty always survives into the backward pass regardless of what
    other plugins (e.g. EarlyStoppingPlugin) do in their own callbacks —
    which was the failure mode with EWCPlugin.
    """

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
        evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = default_evaluator,
        eval_every=-1,
        **base_kwargs,
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
            **base_kwargs,
        )

        assert mode in ("separate", "online"), "mode must be 'separate' or 'online'"
        if mode == "online":
            assert decay_factor is not None, "decay_factor is required in 'online' mode"

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor
        self.keep_importance_data = True if mode == "separate" else keep_importance_data

        self.saved_params: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            self.loss += self.criterion()

            # Added BEFORE any plugin callback touches self.loss.
            if len(self.saved_params) > 0:
                self.loss = self.loss + self.ewc_lambda * self.ewc_penalty()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def ewc_penalty(self) -> torch.Tensor:
        penalty = self._make_empty_loss()

        for exp_id, params in self.saved_params.items():
            for name, parameter in self.model.named_parameters():
                if name not in params:
                    continue
                saved_parameter = params[name]
                importance = self.importances[exp_id][name]

                # Handles a classifier head that grew between experiences.
                if importance.shape != parameter.shape:
                    importance = self._expand(importance, parameter.shape)
                    saved_parameter = self._expand(saved_parameter, parameter.shape)

                penalty = penalty + (
                    importance * (parameter - saved_parameter).pow(2)
                ).sum()

        return penalty

    # ------------------------------------------------------------------ #
    # Fisher / importance computation — runs once per experience
    # ------------------------------------------------------------------ #
    def _after_training_exp(self, **kwargs):
        # Read the counter BEFORE calling super(): the Clock plugin
        # (which increments train_exp_counter) only fires once we call
        # super()._after_training_exp(), so right now this value still
        # refers to the experience we just finished training on.
        exp_id = self.clock.train_exp_counter

        fisher = self._calculate_fisher()
        self._update_importances(fisher, exp_id)

        with torch.no_grad():
            self.saved_params[exp_id] = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
            }

        if exp_id > 0 and not self.keep_importance_data:
            del self.saved_params[exp_id - 1]
            del self.importances[exp_id - 1]

        super()._after_training_exp(**kwargs)

    def _calculate_fisher(self) -> Dict[str, torch.Tensor]:
        fisher = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

        was_training = self.model.training
        self.model.eval()

        num_batches = 0
        for mbatch in self.dataloader:
            self.mbatch = mbatch
            self._unpack_minibatch()

            self.optimizer.zero_grad()
            self.mb_output = self.forward()
            loss = self.criterion()
            loss.backward()

            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().pow(2)

            num_batches += 1

        if num_batches > 0:
            for name in fisher:
                fisher[name] /= num_batches

        self.optimizer.zero_grad()
        self.model.train(was_training)
        return fisher

    def _update_importances(self, fisher: Dict[str, torch.Tensor], exp_id: int):
        if self.mode == "separate" or exp_id == 0:
            self.importances[exp_id] = fisher
            return

        prev = self.importances[exp_id - 1]
        merged = {}
        for name, curr_imp in fisher.items():
            if name in prev:
                old_imp = prev[name]
                if old_imp.shape != curr_imp.shape:
                    old_imp = self._expand(old_imp, curr_imp.shape)
                merged[name] = self.decay_factor * old_imp + curr_imp
            else:
                merged[name] = curr_imp
        self.importances[exp_id] = merged

    @staticmethod
    def _expand(tensor: torch.Tensor, new_shape: torch.Size) -> torch.Tensor:
        """Zero-pad tensor up to new_shape. Mirrors what ParamData.expand
        does, without depending on that class."""
        if tensor.shape == new_shape:
            return tensor
        expanded = torch.zeros(new_shape, device=tensor.device, dtype=tensor.dtype)
        expanded[tuple(slice(0, s) for s in tensor.shape)] = tensor
        return expanded'''

# TRAINING LOOP
optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9)
criterion_masked = MaskedCrossEntropy(mask="seen") # Using MaskedCrossEntropy for masking all but the classes that appear in the specialised model's subset (https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/losses.py , https://avalanche-api.continualai.org/en/latest/generated/avalanche.training.MaskedCrossEntropy.html#avalanche.training.MaskedCrossEntropy)

'''print('WHERE Naive as strategy AND criterion=CrossEntropyLoss() AND loggers=InteractiveLogger() AND NOOOO trained_experience=True AND ewc_lambda=0.9 AND EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)')
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=0.9, mode="separate")],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)# Yes callbaks!!! # FAIL --> EWCPlugin(ewc_lambda=0.9, mode="online", decay_factor=1, keep_importance_data=False)
'''
#TODO: Change train_epochs back to 100
cl_strategy = TestEWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False)], # EWCPlugin(ewc_lambda=0.9, mode="separate")],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)

'''cl_strategy = EWCStrategy(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=100, eval_mb_size=100,
                          plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False)],
                          evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                          eval_every=1)'''
# This code that I took from https://avalanche.continualai.org/avalanche/from-zero-to-hero-tutorial/04_training#training-and-evaluation
# is not the one and only way to build a training loop -- IT IS THE WAY TO BUILD A TRAINING LOOP IF optimizer = SGD !!!  When optimizer = SGD, .train
# is the .train of class BaseSGDTemplate, where the training loop IS DESIGNED TO BE EXECUTED ONCE PER EXPERIENCE, and regardless of what the comments
# in .train of class BaseSGDTemplate, it is clear from the comments in CLStream and EagerCLStream that only single DataExperience objects must be
# provided as input via a for loop like the one in this example AND THE ONE I IMPLEMENTED TAKING INSPIRATION FROM THIS EXAMPLE. You should NOT pass
# the entire stream as input to .train(), and let the training loop handle the experience-by-experience, regardless of the data types supported
# by .train of class BaseSGDTemplate (more info on the comments in CLStream and EagerCLStream above).
print('Starting experiment WITH LOOP...')
results = []
#cl_strategy, initial_exp = maybe_load_checkpoint(cl_strategy, fname) #, map_location=DEVICE)
#for experience in outdoors_specialised_models_benchmark.train_stream[initial_exp:]:
# print("outdoors_specialised_models_benchmark.train_stream[initial_exp:]", outdoors_specialised_models_benchmark.train_stream[initial_exp:]) # EagerCLStream

for experience in outdoors_specialised_models_train_stream_class_il: # only acceptable correct loop (iterating through experiences of train subset ONLY) -- iterating through experiences of train and val subsets results in undesired outcomes, and for the reasons explained above one should never provide .train with the entire CLStream or EagerCLStream.
    i = experience.current_experience # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of outdoors_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
    fname = f"./checkpoint_{i}.pkl"
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
    print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
    criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
    print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    cl_strategy.train(experience, [experience, outdoors_specialised_models_val_stream_class_il.exps_iter[i]])
    print("print(cl_strategy.evaluator.get_last_metrics())", cl_strategy.evaluator.get_last_metrics())
    print("print(cl_strategy.evaluator.get_all_metrics())", cl_strategy.evaluator.get_all_metrics())
    print("print(cl_strategy.plugins)", cl_strategy.plugins)
    print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
    print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)
    save_checkpoint(cl_strategy, fname)
    print("save_checkpoint(strategy, fname)")
    print('Training completed WITH LOOP')

print('Computing accuracy on the whole val set OUTSIDE training LOOP')
cl_strategy.eval(outdoors_specialised_models_val_stream_class_il)
#results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))
#print("Computing accuracy on the whole val set OUTSIDE training LOOP (i.e. print(results)):", results)

'''print('Computing accuracy on one val experience at a time OUTSIDE training LOOP')
print('Computing accuracy on on val experience 0 OUTSIDE training LOOP')
cl_strategy.eval(outdoors_specialised_models_val_stream_class_il[0])
print('Computing accuracy on on val experience 1 OUTSIDE training LOOP')
cl_strategy.eval(outdoors_specialised_models_val_stream_class_il[1])
print('Computing accuracy on on val experience 2 OUTSIDE training LOOP')
cl_strategy.eval(outdoors_specialised_models_val_stream_class_il[2])
print('Computing accuracy on on val experience 3 OUTSIDE training LOOP')
cl_strategy.eval(outdoors_specialised_models_val_stream_class_il[3])
print('Computing accuracy on on val experience 4 OUTSIDE training LOOP')
cl_strategy.eval(outdoors_specialised_models_val_stream_class_il[4])'''


'''print('Computing val accuracy on checkpoint_4.pkl (outside training loop)')
cl_strat, initial_exp = maybe_load_checkpoint(cl_strategy, f"./checkpoint_4.pkl") #, map_location=DEVICE)
cl_strat.eval(outdoors_specialised_models_val_stream_class_il)'''





























