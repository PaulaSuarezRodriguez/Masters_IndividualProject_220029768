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
from collections import defaultdict
from typing import Dict, Tuple, Union
import warnings
import itertools

import torch
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData

from typing import Callable, Optional, Sequence, List, Union
import torch
from torch.nn.parameter import Parameter

from torch import sigmoid
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.models.packnet import PackNetModel, PackNetModule, PackNetPlugin

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import (
    default_evaluator,
    default_loggers,
)
from avalanche.training.storage_policy import ExemplarsBuffer
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation.metrics import loss_metrics
from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.models.expert_gate import AE_loss
from avalanche.logging import InteractiveLogger
from avalanche.training.templates.strategy_mixin_protocol import CriterionType

'''ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
if plugins is None:
    plugins = [ewc]
else:
    plugins.append(ewc)'''

'''def __init__( # EWCPlugin
    self,
    ewc_lambda,
    mode="separate",
    decay_factor=None,
    keep_importance_data=False,
):'''

'''class EWC_MyOwn(SupervisedTemplate):
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
        #ewc_lambda=float,
        #mode="separate",
        #decay_factor=None,
        #keep_importance_data=False,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device=device,
        plugins=plugins,
        evaluator=evaluator,
        eval_every=eval_every,
        **base_kwargs
    )

        #super().__init__()
        assert (decay_factor is None) or (
            mode == "online"
        ), "You need to set `online` mode to use `decay_factor`."
        assert (decay_factor is not None) or (
            mode != "online"
        ), "You need to set `decay_factor` to use the `online` mode."
        assert (
            mode == "separate" or mode == "online"
        ), "Mode must be separate or online."

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, ParamData]] = defaultdict(dict)

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == "separate":
            for experience in range(exp_counter):
                for k, cur_param in strategy.model.named_parameters():
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
            for k, cur_param in strategy.model.named_parameters():
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

        strategy.loss += self.ewc_lambda * penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
            num_workers=kwargs.get("num_workers", 0),
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

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


ParamDict = Dict[str, Union[ParamData]]
EwcDataType = Tuple[ParamDict, ParamDict]'''





# TRAINING LOOP
optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9)
#######criterion_masked = MaskedCrossEntropy(mask="seen") # Using MaskedCrossEntropy for masking all but the classes that appear in the specialised model's subset (https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/losses.py , https://avalanche-api.continualai.org/en/latest/generated/avalanche.training.MaskedCrossEntropy.html#avalanche.training.MaskedCrossEntropy)
'''print('WHERE ewc_lambda=1, mode="separate", train_mb_size=100, train_epochs=20, eval_mb_size=100')
cl_strategy = EWC(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, ewc_lambda=1, mode="separate", train_mb_size=100, train_epochs=20, eval_mb_size=100, device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True, trained_experience=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)''' # Loss_Stream/eval_phase/valid_stream = 6.4713 # Top1_Acc_Stream/eval_phase/valid_stream = 0.1820

'''print('WHERE ewc_lambda=1, mode="separate", keep_importance_data=True, train_mb_size=100, train_epochs=100, eval_mb_size=100, AND EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)')
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, train_mb_size=100, train_epochs=100, eval_mb_size=100, device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)''' # Loss_Stream/eval_phase/valid_stream = 6.5255 # Top1_Acc_Stream/eval_phase/valid_stream = 0.1860

'''print('WHERE ewc_lambda=1, mode="separate", keep_importance_data=True, train_mb_size=100, train_epochs=100, eval_mb_size=100,')
cl_strategy = EWC(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, ewc_lambda=1, mode="separate", train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True, trained_experience=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)'''

'''print('WHERE SupervisedTemplate')
cl_strategy = SupervisedTemplate(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True, trained_experience=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)'''

'''print('WHERE SupervisedTemplate AND criterion=CrossEntropyLoss()')
cl_strategy = SupervisedTemplate(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True, trained_experience=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)''' # no improvement

'''print('WHERE EWC as strategy AND criterion=CrossEntropyLoss() AND EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)')
cl_strategy = EWC(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion, ewc_lambda=1, mode="separate", keep_importance_data=True, train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True, trained_experience=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)''' # no improvement

'''print('WHERE EWC as strategy AND criterion=CrossEntropyLoss() AND NOOOO EWCPlugin')
cl_strategy = EWC(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", keep_importance_data=True, train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True, trained_experience=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)''' # no improvement

'''print('WHERE EWC as strategy AND criterion=CrossEntropyLoss() AND NOOOO EWCPlugin AND loggers=InteractiveLogger() AND NOOOO trained_experience=True')
cl_strategy = EWC(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", keep_importance_data=True, train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)''' # no improvement

'''print('WHERE Naive as strategy AND criterion=CrossEntropyLoss() AND EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True) AND NOOOO trained_experience=True')
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1)'''

print('WHERE SupervisedTemplate as strategy AND criterion=CrossEntropyLoss() AND loggers=InteractiveLogger() AND NOOOO trained_experience=True')
'''cl_strategy = SupervisedTemplate(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False),
                             EWCPlugin(ewc_lambda=1, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)''' # no penalty is stored even though is computed it seems

'''print('WHERE EWC as strategy AND criterion=CrossEntropyLoss() AND loggers=InteractiveLogger() AND NOOOO trained_experience=True AND ewc_lambda=0.9 AND EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)')
cl_strategy = EWC(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", keep_importance_data=True, train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False),
                             EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)''' # does not seem to have any callbacks

'''print('WHERE Naive as strategy AND criterion=CrossEntropyLoss() AND loggers=InteractiveLogger() AND NOOOO trained_experience=True AND ewc_lambda=0.9 AND EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)')
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False),
                             EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)# Yes callbaks!!!'''

'''print('WHERE SupervisedTemplate as strategy AND criterion=CrossEntropyLoss() AND loggers=InteractiveLogger() AND NOOOO trained_experience=True AND ewc_lambda=0.9 AND EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)')
cl_strategy = SupervisedTemplate(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=0.9, mode="separate")],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)# Yes callbaks!!!'''

'''print('WHERE EWC MY OWN as strategy AND criterion=CrossEntropyLoss() AND loggers=InteractiveLogger() AND NOOOO trained_experience=True AND ewc_lambda=0.9 AND EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)')
cl_strategy = EWC_MyOwn(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), ewc_lambda=0.9, mode="separate", train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False)],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1) '''
# does not seem to have any callbacks AND LEARNED SOMETHING NEW: EWCPlugin IS NEEDED!!!!!!

print('WHERE Naive as strategy AND criterion=CrossEntropyLoss() AND loggers=InteractiveLogger() AND NOOOO trained_experience=True AND ewc_lambda=0.9 AND EWCPlugin(ewc_lambda=0.9, mode="separate", keep_importance_data=True)')
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=CrossEntropyLoss(), train_mb_size=100, train_epochs=100, eval_mb_size=100, #device=DEVICE,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name="valid_stream", metric_name="Top1_Acc_Stream", mode="max", peval_mode="epoch", verbose=False), EWCPlugin(ewc_lambda=0.9, mode="separate")],
                    evaluator=EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loss_metrics(experience=True, stream=True), loggers=InteractiveLogger(), collect_all=True),
                    eval_every=1)# Yes callbaks!!! # FAIL --> EWCPlugin(ewc_lambda=0.9, mode="online", decay_factor=1, keep_importance_data=False)


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
    #######print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    #######print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
    #######criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
    #######print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    cl_strategy.train(experience, [experience, outdoors_specialised_models_val_stream_class_il.exps_iter[i]])
    print("print(cl_strategy.evaluator.get_last_metrics())", cl_strategy.evaluator.get_last_metrics())
    print("print(cl_strategy.evaluator.get_all_metrics())", cl_strategy.evaluator.get_all_metrics())
    print("print(cl_strategy.plugins)", cl_strategy.plugins)
    #######print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
    #######print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)
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





























