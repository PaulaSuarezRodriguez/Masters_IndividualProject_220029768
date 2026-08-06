from avalanche.core import BasePlugin
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins.evaluation import default_loggers, default_evaluator

from Seventh_implementation_attempt.Config_v7 import set_seed
set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Seventh_implementation_attempt.ResNet20_v7 import ResNet20
from Seventh_implementation_attempt.Load_cifar100_v7 import (outdoors_specialised_models_train_subset, nature_specialised_models_train_subset,
                                                            people_specialised_models_train_subset, animals_specialised_models_train_subset, objects_and_vehicles_specialised_models_train_subset)
from Seventh_implementation_attempt.Load_cifar100_v7 import (outdoors_specialised_models_val_subset, nature_specialised_models_val_subset,
                                                            people_specialised_models_val_subset, animals_specialised_models_val_subset, objects_and_vehicles_specialised_models_val_subset)
from Seventh_implementation_attempt.Load_cifar100_v7 import (outdoors_specialised_models_test_subset, nature_specialised_models_test_subset,
                                                            people_specialised_models_test_subset, animals_specialised_models_test_subset, objects_and_vehicles_specialised_models_test_subset)
#from avalanche.benchmarks import class_incremental_benchmark
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.benchmarks import benchmark_with_validation_stream
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!
from avalanche.training.losses import MaskedCrossEntropy
from avalanche.training.storage_policy import ExemplarsSelectionStrategy
from avalanche.training.plugins import EarlyStoppingPlugin, EvaluationPlugin
from collections.abc import Iterable # OOOR like in avalanche/training/templates/base_sgd.py , from typing import Any, Callable, Generic, Iterable, Sequence, Optional, TypeVar, Union
from avalanche.evaluation.metrics.accuracy import accuracy_metrics, EpochAccuracy, ExperienceAccuracy, StreamAccuracy, TrainedExperienceAccuracy
from avalanche.training.templates.base_sgd import PeriodicEval

# NAIVE / BASELINE IMPLEMENTATION OF MY STRATEGY --> train each specialised model in a Class-IL manner without attempting to mitigate CF.
# Additionally, pre-train the global model on the EXPERIENCE 1 TRAIN SUBSETS and attempt to naively update it (using a hard-coded version of FedAvg
# and no techniques to mitigate ST-CF):

# class Naive(SupervisedTemplate) DESCRIPTION PROVIDED IN LIBRARY SOURCE CODE: "Naive finetuning.The simplest (and least effective) Continual Learning strategy. Naive just incrementally
# fine tunes a single model WITHOUT EMPLOYING ANY METHOD to contrast the catastrophic forgetting of previous knowledge. This strategy does not use task identities. Naive is easy to set up
# and its results are COMMONLY USED TO SHOW THE WORST PERFORMING BASELINE."

print("\n\n\n################################ HERE BEGINS Train_ResNet20_v7.py ################################ \n\n\n")

resnet20_global_model = ResNet20(num_classes=100)
outdoors_specialised_model = ResNet20(num_classes=100) #No need for num_classes=10 anymore despite 10 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
nature_specialised_model = ResNet20(num_classes=100) #No need for num_classes=15 anymore despite 15 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
people_specialised_model = ResNet20(num_classes=100) #No need for num_classes=5 anymore despite 5 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
animals_specialised_model = ResNet20(num_classes=100) #No need for num_classes=45 anymore despite 45 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used
objects_and_vehicles_specialised_model = ResNet20(num_classes=100) #No need for num_classes=25 anymore despite 25 being the number of classes in the specialised subset for this specialised model, as MaskedCrossEntropy is being used

avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset)
avl_val = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset)
avl_test = make_avalanche_dataset(outdoors_specialised_models_test_subset.dataset)

'''print("type(outdoors_specialised_models_train_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train", type(avl_train))
print("type(outdoors_specialised_models_val_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val", type(avl_val))
print("type(outdoors_specialised_models_test_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test", type(avl_test))

print("Print avl_train AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train:\n", avl_train)
print("Print avl_val AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val:\n", avl_val)
print("Print avl_test AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_test:\n", avl_test)'''

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

'''print("Type avl_train AFTER being passed to ClassificationDataset", type(avl_train))
print("Type avl_val AFTER being passed to ClassificationDataset", type(avl_val))
print("Type avl_test AFTER being passed to ClassificationDataset", type(avl_test))

print("Print avl_train:\n", avl_train)
print("Print avl_val:\n", avl_val)
print("Print avl_test:\n", avl_test)'''

print("Print avl_train.targets.uniques (AS .targets IS A DataAttribute AND THE ONLY WAY TO ACCESS AND PRINT ITS ACTUAL ELEMENTS/VALUES IS USING ATTRIBUTE .uniques):", avl_train.targets.uniques)
#print("Print avl_train.targets TO PROVE THE ABOVE, AS WHEN ATTRIBUTE .uniques OF DataAttribute IS NOT USED, THE ACTUAL TARGETS/VALUES CONTAINED WITHIN avl_train.targets ARE NOT ACCESSED:, ", avl_train.targets)
print("Print max(avl_train.targets.uniques) in order to be aware of the value that due to the library bug will be assigned to num_classes no matter what my code does (and in this way better mitigate, follow, identify and debug the consequence of such bug):", max(avl_train.targets.uniques))
print("Print len(avl_train.targets.uniques) to know the NUMBER OF FINE CLASSES within the specialised_models_superclasses group of which the specialised subset (e.g. outdoors_specialised_models_train_subset) that is avl_train is composed:", len(avl_train.targets.uniques)) # print number of classes
print("IMPORTANT! Print sorted(avl_train.targets.uniques) AS THIS WILL BE THE OBJECT PASSED TO class_order IN class_incremental_benchmark TO AVOID/AS A WORK-AROUND TO THE num_classes BUG IN THE LIBRARY CODE, since by provideing a class_order, 'class_order = torch.randperm(num_classes).tolist()' within 'if class_order is None:' IS AVOIDED!", sorted(avl_train.targets.uniques))# print class_order

outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[2, 2, 2, 2, 69]) # which is euivalent to num_experiences=5 due to the reasons explained in previous version)

print("sorted(avl_train.targets.uniques)",sorted(avl_train.targets.uniques))
for i, exp in enumerate(outdoors_specialised_models_benchmark.train_stream):
    print("print(i, exp.classes_in_this_experience)", i, exp.classes_in_this_experience)

print("Print outdoors_specialised_models_benchmark:", outdoors_specialised_models_benchmark)
print("Print outdoors_specialised_models_benchmark.train_stream:", outdoors_specialised_models_benchmark.train_stream) # it works, but I am not sure that it is not deprecated. I think the current or at least most frequently used way of retrieving a stream is .streams["train"], specially if there is a .streams["valid"] since attribute .val_stream does NOT exist. print(outdoors_specialised_models_benchmark.train_stream) and print(outdoors_specialised_models_benchmark.streams["train"]) both return <avalanche.benchmarks.scenarios.generic_scenario.EagerCLStream object at 0x000002242B559510>, meaning they return the exact same object so, there is no risk involved in having chosen to use .streams["train"] over .train_stream
print('Print outdoors_specialised_models_benchmark.streams["train"]',outdoors_specialised_models_benchmark.streams["train"])
print('Print outdoors_specialised_models_benchmark.streams["test"]', outdoors_specialised_models_benchmark.streams["test"])
print('Print outdoors_specialised_models_benchmark.streams["valid"]', outdoors_specialised_models_benchmark.streams["valid"])

''' IMPORTANT -- NOTES ON LIBRARY SOURCE CODE:
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
'''

# IMPORTANT!!!!! This time, we do want to keep these as streams and NOT to further divide each stream into its composing experiences. This time,
# since class_incremental_benchmark is actually being used to create a CLScenario for use in training in a class-IL manner, and NOT as functionality
# for splitting the dataset into specialised model subsets (as we did previously in Load_Cifar100_v7.py, when class_incremental_benchmark was used
# last), we want to keep the stream objects intact without breaking them apart into its composing experience objects, as the experience objects are
# to be managed by the Avalanche training functionality/training loop to ensure the specialised model is trained in a class_IL manner with each
# experience being used for a subsequent incremental training round.
outdoors_specialised_models_train_stream_class_il = outdoors_specialised_models_benchmark.streams["train"]
outdoors_specialised_models_val_stream_class_il = outdoors_specialised_models_benchmark.streams["valid"]
outdoors_specialised_models_test_stream_class_il = outdoors_specialised_models_benchmark.streams["test"]

print("type(outdoors_specialised_models_train_stream_class_il -- OBJECT THROUGH WHICH TRAINING LOOP IS CURRENTLY ITERATING)", type(outdoors_specialised_models_train_stream_class_il))
print("type(outdoors_specialised_models_train_stream_class_il.exps_iter -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS)", type(outdoors_specialised_models_train_stream_class_il.exps_iter))
print('type(outdoors_specialised_models_benchmark.train_stream -- OBJECT THROUGH WHICH TRAINING LOOP HAS BEEN SET TO ITERATE THROUGH IN THE PAST FOR TESTING EFFECTS', type(outdoors_specialised_models_benchmark.train_stream), 'ALSOOOOOO, benchmark.train_stream KNOWN TO BE EQUAL TO benchmark.streams["train"]); CHECK TO VERIFY')
print("type(outdoors_specialised_models_train_stream_class_il.exps_iter[0])", type(outdoors_specialised_models_train_stream_class_il.exps_iter[0]))

#####################################################################################################################################################################################################################
'''# TRAINING LOOP
criterion_masked = MaskedCrossEntropy(mask="seen")
optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9)
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, train_mb_size=100, train_epochs=8, eval_mb_size=100,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name='valid', verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(epoch=True, experience=True, stream=True), loggers=default_loggers()), eval_every=1)

print("WHICH OF THE FOLLOWING WOULD BE ADEQUATE FOR eval_streams PARAMETER OF .train() METHOD OF avalanche.training.templates.base_sgd.BaseSGDTemplate?\nExpected type: Sequence[TDatasetExperience | Iterable[TDatasetExperience]], WICH MEANS Sequence[TDatasetExperience] OR Sequence[Iterable[TDatasetExperience]]")
print("type([outdoors_specialised_models_val_stream_class_il])", type([outdoors_specialised_models_val_stream_class_il]))
print("type([outdoors_specialised_models_val_stream_class_il.exps_iter])", type([outdoors_specialised_models_val_stream_class_il.exps_iter]))
print("print(isinstance(outdoors_specialised_models_val_stream_class_il, Iterable))", isinstance(outdoors_specialised_models_val_stream_class_il, Iterable))
print("print(isinstance(outdoors_specialised_models_val_stream_class_il.exps_iter, Iterable))", isinstance(outdoors_specialised_models_val_stream_class_il.exps_iter, Iterable))

print('Starting experiment WITH LOOP...') # the correct training loop that iterates through DataExperiences, NOT CLStreams or Benchmarks or CLScenario objects (full explanation in the comments above)
results = []
for experience in outdoors_specialised_models_train_stream_class_il: # NOTE: DO NOT DO THE FOLLOWING, it results in the loop only iterating through the item behind the and operator since for loops are not supposed to have and operators in python causing the for loop logic to fail ---- for experience in outdoors_specialised_models_train_stream_class_il and outdoors_specialised_models_val_stream_class_il: # DO NOT DO THE FOLLOWING, it performed validation on every DataExperience object of the val subset per DataExperience object of the train subset being iterated over ---- for val_experience in outdoors_specialised_models_val_stream_class_il:
    i = experience.current_experience # to be used in .exps_iter[i] for the validation stream to iterate through the DataExperiences. Equivalent of outdoors_specialised_models_val_stream_class_il.exps_iter[experience] if it were possible to iterate using the DataExperience object as an index to itself
    print("Start of experience -- current TRAIN experience: ", experience.current_experience)
    print("Start of experience -- current VAL experience: ", outdoors_specialised_models_val_stream_class_il.exps_iter[i].current_experience)
    print("Current Classes -- current TRAIN experience: ", experience.classes_in_this_experience)
    print("Current Classes -- current VAL experience: ", outdoors_specialised_models_val_stream_class_il.exps_iter[i].classes_in_this_experience)
    #print("experience.dataset (of TRAIN)", experience.dataset)
    #print(".exps_iter[i].dataset (of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset)
    print("type(experience) WITH LOOP for experience in outdoors_specialised_models_train_stream_class_il", type(experience))
    print("type(outdoors_specialised_models_val_stream_class_il.exps_iter[i])", type(outdoors_specialised_models_val_stream_class_il.exps_iter[i]))
    print("experience.dataset.coarse_label.uniques (coarse_labels of TRAIN)", experience.dataset.coarse_label.uniques)
    print("experience.dataset.targets (targets of TRAIN)", experience.dataset.targets.uniques)
    print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques (coarse_labels of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques)
    print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques (targets of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques)
    # print("experience", experience)
    # x, y, z = experience.dataset[0] # returns images, targets, coarse_labels
    # print(f"x: {x}, \ny: {y}, \nz: {z}")
    print("print(experience.origin_stream.name)", experience.origin_stream.name)
    print("print(outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)
    print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
    print("print(criterion_masked.mask) BEFORE CALL TO cl_strategy.train", criterion_masked.mask)
    print("print(criterion_reduction) BEFORE CALL TO cl_strategy.train", criterion_masked.reduction)
    print("type(criterion_masked)", type(criterion_masked))
    print("isinstance(criterion_masked, BasePlugin)", isinstance(criterion_masked, BasePlugin))
    criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED
    print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    cl_strategy.train(experience, [experience, outdoors_specialised_models_val_stream_class_il.exps_iter[i]])
    #cl_strategy.train(experience, [[experience], [outdoors_specialised_models_val_stream_class_il.exps_iter[i]]])
                                                # ,[outdoors_specialised_models_val_stream_class_il[0],
                                                # outdoors_specialised_models_val_stream_class_il[1],
                                                # outdoors_specialised_models_val_stream_class_il[2],
                                                # outdoors_specialised_models_val_stream_class_il[3],
                                                # outdoors_specialised_models_val_stream_class_il[4]])
    print("print(cl_strategy.plugins)", cl_strategy.plugins)
    print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
    print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)
    print("print(criterion_masked.mask) AFTER CALL TO cl_strategy.train", criterion_masked.mask)
    print("print(criterion_reduction) AFTER CALL TO cl_strategy.train", criterion_masked.reduction)
    print('Training completed WITH LOOP')

    print('Computing accuracy on the whole test set WITH LOOP')
    results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))
    print("Computing accuracy on the whole test set WITH LOOP (i.e. print(results)):", results)'''
################################################################################################################################################################################################################################################


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
############### CLEANED CODE (no comments for easier interpretation) AND CODE FOR TO BE EDITED FOR EXPERIMENTING !!!!!!!!!!!!!! ###############
print("\n\n\n################################ CLEANED CODE (no comments for easier interpretation) AND CODE FOR TO BE EDITED FOR EXPERIMENTING !!!!!!!!!!!!!! ################################ \n\n\n")
# TRAINING LOOP
criterion_masked = MaskedCrossEntropy(mask="seen")
optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9)
print("WHERE metric_name=RunningEpochAccuracy")
'''cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, train_mb_size=100, train_epochs=8, eval_mb_size=100,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name='valid', metric_name="EpochAccuracy" , peval_mode="epoch", verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(epoch=True, experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=0, peval_mode="experience")'''

'''cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, train_mb_size=100, train_epochs=8, eval_mb_size=100,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name='valid', metric_name="RunningEpochAccuracy" , peval_mode="epoch", verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(epoch=True, epoch_running=True, experience=True, trained_experience=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1, peval_mode="experience")'''

cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked, train_mb_size=100, train_epochs=8, eval_mb_size=100,
                    plugins=[EarlyStoppingPlugin(patience=3, val_stream_name='valid', metric_name="RunningEpochAccuracy", mode="max", peval_mode="epoch", verbose=True)],
                    evaluator=EvaluationPlugin(accuracy_metrics(epoch=True, epoch_running=True, experience=True, trained_experience=True), loss_metrics(minibatch=False, epoch=True, experience=True, stream=True), loggers=default_loggers(), collect_all=True),
                    eval_every=1, peval_mode="experience")

print("type(cl_strategy)",type(cl_strategy))
print("print(cl_strategy.loss)", cl_strategy.loss)
print("print(cl_strategy.experience)", cl_strategy.experience)
print("print(cl_strategy.loss)", cl_strategy.loss)
print("WHICH OF THE FOLLOWING WOULD BE ADEQUATE FOR eval_streams PARAMETER OF .train() METHOD OF avalanche.training.templates.base_sgd.BaseSGDTemplate?\nExpected type: Sequence[TDatasetExperience | Iterable[TDatasetExperience]], WICH MEANS Sequence[TDatasetExperience] OR Sequence[Iterable[TDatasetExperience]]")
print("type([outdoors_specialised_models_val_stream_class_il])", type([outdoors_specialised_models_val_stream_class_il]))
print("type([outdoors_specialised_models_val_stream_class_il.exps_iter])", type([outdoors_specialised_models_val_stream_class_il.exps_iter]))
print("print(isinstance(outdoors_specialised_models_val_stream_class_il, Iterable))", isinstance(outdoors_specialised_models_val_stream_class_il, Iterable))
print("print(isinstance(outdoors_specialised_models_val_stream_class_il.exps_iter, Iterable))", isinstance(outdoors_specialised_models_val_stream_class_il.exps_iter, Iterable))

print('Starting experiment WITH LOOP...')
results = []
for experience in outdoors_specialised_models_train_stream_class_il:
    i = experience.current_experience
    print("Start of experience -- current TRAIN experience: ", experience.current_experience)
    print("Start of experience -- current VAL experience: ", outdoors_specialised_models_val_stream_class_il.exps_iter[i].current_experience)
    print("Current Classes -- current TRAIN experience: ", experience.classes_in_this_experience)
    print("Current Classes -- current VAL experience: ", outdoors_specialised_models_val_stream_class_il.exps_iter[i].classes_in_this_experience)
    #print("experience.dataset (of TRAIN)", experience.dataset)
    #print(".exps_iter[i].dataset (of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset)
    print("type(experience) WITH LOOP for experience in outdoors_specialised_models_train_stream_class_il", type(experience))
    print("type(outdoors_specialised_models_val_stream_class_il.exps_iter[i])", type(outdoors_specialised_models_val_stream_class_il.exps_iter[i]))
    print("experience.dataset.coarse_label.uniques (coarse_labels of TRAIN)", experience.dataset.coarse_label.uniques)
    print("experience.dataset.targets (targets of TRAIN)", experience.dataset.targets.uniques)
    print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques (coarse_labels of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.coarse_label.uniques)
    print("outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques (targets of VAL)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].dataset.targets.uniques)

    print("print(experience.origin_stream.name)", experience.origin_stream.name)
    print("print(outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)", outdoors_specialised_models_val_stream_class_il.exps_iter[i].origin_stream.name)
    print("print(criterion_masked.current_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    print("print(criterion_masked.old_classes) BEFORE CALL TO cl_strategy.train", criterion_masked.old_classes)
    print("print(criterion_masked.mask) BEFORE CALL TO cl_strategy.train", criterion_masked.mask)
    print("print(criterion_reduction) BEFORE CALL TO cl_strategy.train", criterion_masked.reduction)
    print("type(criterion_masked)", type(criterion_masked))
    print("isinstance(criterion_masked, BasePlugin)", isinstance(criterion_masked, BasePlugin))
    # TEST OF REMOVING criterion_masked.pre_adapt(cl_strategy, experience) TO SEE IF IT IS STILL NEEDED EVEN WITH UPDATED CODE/cl_strategy INITIALISATION FAILED !!!! criterion_masked.pre_adapt(cl_strategy, experience) is most definitely needed
    criterion_masked.pre_adapt(cl_strategy, experience) #### .pre_adapt LIBRARY FUNCTION HARD-CODED --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
    #criterion_masked.pre_adapt(criterion_masked, experience)  #### .pre_adapt TEST --> DeprecationWarning: Call to deprecated function before_training_exp (removal in version 0.7: Please switch to the `pre_adapt` method.)getattr(p, event)(strategy, **kwargs)
    #criterion_masked.pre_adapt(experience)  #### .pre_adapt TEST 2 causes ERROR THAT HALTS EXECUTION --> TypeError: MaskedCrossEntropy.pre_adapt() missing 1 required positional argument: 'exp'
    print("type(cl_strategy)", type(cl_strategy))
    print("print(criterion_masked.current_classes) AFTER CALL TO .pre_adapt BUT BEFORE CALL TO cl_strategy.train", criterion_masked.current_classes)
    cl_strategy.train(experience, [experience, outdoors_specialised_models_val_stream_class_il.exps_iter[i]])
    print("type(cl_strategy)", type(cl_strategy))
    print("print(cl_strategy.experience)", cl_strategy.experience)
    print("print(cl_strategy.loss)", cl_strategy.loss)
    #cl_strategy.train(experience, [[experience], [outdoors_specialised_models_val_stream_class_il.exps_iter[i]]])
                                                # ,[outdoors_specialised_models_val_stream_class_il[0],
                                                # outdoors_specialised_models_val_stream_class_il[1],
                                                # outdoors_specialised_models_val_stream_class_il[2],
                                                # outdoors_specialised_models_val_stream_class_il[3],
                                                # outdoors_specialised_models_val_stream_class_il[4]])
    print("print(cl_strategy.plugins)", cl_strategy.plugins)
    print("print(criterion_masked.current_classes) AFTER CALL TO cl_strategy.train", criterion_masked.current_classes)
    print("print(criterion_masked.old_classes) AFTER CALL TO cl_strategy.train", criterion_masked.old_classes)
    print("print(criterion_masked.mask) AFTER CALL TO cl_strategy.train", criterion_masked.mask)
    print("print(criterion_reduction) AFTER CALL TO cl_strategy.train", criterion_masked.reduction)
    print('Training completed WITH LOOP')

    print('Computing accuracy on the whole test set WITH LOOP')
    #############results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))
    print("Computing accuracy on the whole test set WITH LOOP (i.e. print(results)):", results)
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################








if __name__ == "__main__":
    #print("results", results)

    # Print outdoors_specialised_models_train_subset: <avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience object at 0x000001CA7B613150>
    # type outdoors_specialised_models_train_subset: <class 'avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience'>
    '''print("outdoors_specialised_models_train_subset.dataset", outdoors_specialised_models_train_subset.dataset)
    print("type(outdoors_specialised_models_train_subset.dataset)",type(outdoors_specialised_models_train_subset.dataset))
    print("outdoors_specialised_models_train_subset.dataset.targets",outdoors_specialised_models_train_subset.dataset.targets)
    print("type(outdoors_specialised_models_train_subset.dataset.targets)",type(outdoors_specialised_models_train_subset.dataset.targets))
    print("outdoors_specialised_models_train_subset.dataset.coarse_label)",outdoors_specialised_models_train_subset.dataset.coarse_label)
    print("type(outdoors_specialised_models_train_subset.dataset.coarse_label))",type(outdoors_specialised_models_train_subset.dataset.coarse_label))

    print("outdoors_specialised_models_val_subset.dataset", outdoors_specialised_models_val_subset.dataset)
    print("type(outdoors_specialised_models_val_subset.dataset)", type(outdoors_specialised_models_val_subset.dataset))
    print("outdoors_specialised_models_val_subset.dataset.targets",outdoors_specialised_models_val_subset.dataset.targets)
    print("type(outdoors_specialised_models_val_subset.dataset.targets)",type(outdoors_specialised_models_val_subset.dataset.targets))
    print("outdoors_specialised_models_val_subset.dataset.coarse_label)",outdoors_specialised_models_val_subset.dataset.coarse_label)
    print("type(outdoors_specialised_models_val_subset.dataset.coarse_label))",type(outdoors_specialised_models_val_subset.dataset.coarse_label))'''

    # x, y, t = outdoors_specialised_models_val_subset.dataset[0] # returns images, targets, coarse_labels
    # print("x", x, "\ny", y, "\nt", t) # returns images, targets, coarse_labels

    # NOTE: .name for the DataAttribute objects (.coarse_label and .targets) of either subset/ClassificationDataset (.dataset) within the DataExperience
    # object (e.g. outdoors_specialised_models_train_subset) LITERALLY RETURNS THE VARIABLE NAME OF THE DataAttribute. That is, .dataset.coarse_label.name
    # returns "coarse_label" and .dataset.targets.name returns "targets". The class type for .name is "string". Since neither of these things provide any
    # useful information, nor can be used in any useful manner in the current implementation, the code for inspecting .name below has been commented out.
    '''print("outdoors_specialised_models_train_subset.dataset.coarse_label.name", outdoors_specialised_models_train_subset.dataset.coarse_label.name)
    print("outdoors_specialised_models_train_subset.dataset.targets.name", outdoors_specialised_models_train_subset.dataset.targets.name)
    print("outdoors_specialised_models_val_subset.dataset.coarse_label.name", outdoors_specialised_models_val_subset.dataset.coarse_label.name)
    print("outdoors_specialised_models_val_subset.dataset.targets.name", outdoors_specialised_models_val_subset.dataset.targets.name)
    print("type(outdoors_specialised_models_train_subset.dataset.coarse_label.name)", type(outdoors_specialised_models_train_subset.dataset.coarse_label.name))
    print("type(outdoors_specialised_models_train_subset.dataset.targets.name)", type(outdoors_specialised_models_train_subset.dataset.targets.name))
    print("type(outdoors_specialised_models_val_subset.dataset.coarse_label.name)", type(outdoors_specialised_models_val_subset.dataset.coarse_label.name))
    print("type(outdoors_specialised_models_val_subset.dataset.targets.name)", type(outdoors_specialised_models_val_subset.dataset.targets.name))'''