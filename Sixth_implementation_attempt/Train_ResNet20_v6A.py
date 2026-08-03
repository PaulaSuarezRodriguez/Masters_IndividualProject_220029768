from Sixth_implementation_attempt.Config_v6 import set_seed
set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Sixth_implementation_attempt.ResNet20_v6 import ResNet20
from Sixth_implementation_attempt.Load_cifar100_v6 import (outdoors_specialised_models_train_subset, nature_specialised_models_train_subset,
                                                            people_specialised_models_train_subset, animals_specialised_models_train_subset, objects_and_vehicles_specialised_models_train_subset)
from Sixth_implementation_attempt.Load_cifar100_v6 import (outdoors_specialised_models_val_subset, nature_specialised_models_val_subset,
                                                            people_specialised_models_val_subset, animals_specialised_models_val_subset, objects_and_vehicles_specialised_models_val_subset)
from Sixth_implementation_attempt.Load_cifar100_v6 import (outdoors_specialised_models_test_subset, nature_specialised_models_test_subset,
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

# NAIVE / BASELINE IMPLEMENTATION OF MY STRATEGY --> train each specialised model in a Class-IL manner without attempting to mitigate CF.
# Additionally, pre-train the global model on the EXPERIENCE 1 TRAIN SUBSETS and attempt to naively update it (using a hard-coded version of FedAvg
# and no techniques to mitigate ST-CF):
print("\n\n\n################################ HERE BEGINS Train_ResNet20_v6.py ################################ \n\n\n")

resnet20_global_model = ResNet20(num_classes=100)
outdoors_specialised_model = ResNet20(num_classes=100)#TODO: (num_classes=10)
nature_specialised_model = ResNet20(num_classes=15)
people_specialised_model = ResNet20(num_classes=5)
animals_specialised_model = ResNet20(num_classes=45)
objects_and_vehicles_specialised_model = ResNet20(num_classes=25)

#Print outdoors_specialised_models_train_subset: <avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience object at 0x000001CA7B613150>
#type outdoors_specialised_models_train_subset: <class 'avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience'>
print("outdoors_specialised_models_train_subset.dataset", outdoors_specialised_models_train_subset.dataset)
print("type(outdoors_specialised_models_train_subset.dataset)", type(outdoors_specialised_models_train_subset.dataset))
print("outdoors_specialised_models_train_subset.dataset.targets", outdoors_specialised_models_train_subset.dataset.targets)
print("type(outdoors_specialised_models_train_subset.dataset.targets)", type(outdoors_specialised_models_train_subset.dataset.targets))
print("outdoors_specialised_models_train_subset.dataset.coarse_label)", outdoors_specialised_models_train_subset.dataset.coarse_label)
print("type(outdoors_specialised_models_train_subset.dataset.coarse_label))", type(outdoors_specialised_models_train_subset.dataset.coarse_label))

print("outdoors_specialised_models_val_subset.dataset", outdoors_specialised_models_val_subset.dataset)
print("type(outdoors_specialised_models_val_subset.dataset)", type(outdoors_specialised_models_val_subset.dataset))
print("outdoors_specialised_models_val_subset.dataset.targets", outdoors_specialised_models_val_subset.dataset.targets)
print("type(outdoors_specialised_models_val_subset.dataset.targets)", type(outdoors_specialised_models_val_subset.dataset.targets))
print("outdoors_specialised_models_val_subset.dataset.coarse_label)", outdoors_specialised_models_val_subset.dataset.coarse_label)
print("type(outdoors_specialised_models_val_subset.dataset.coarse_label))", type(outdoors_specialised_models_val_subset.dataset.coarse_label))

#x, y, t = outdoors_specialised_models_val_subset.dataset[0]
#print("x", x, "\ny", y, "\nt", t)

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

avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset)
avl_val = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset)
avl_test = make_avalanche_dataset(outdoors_specialised_models_test_subset.dataset)

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

print("Type avl_train AFTER being passed to ClassificationDataset", type(avl_train))
print("Type avl_val AFTER being passed to ClassificationDataset", type(avl_val))
print("Type avl_test AFTER being passed to ClassificationDataset", type(avl_test))

print("Print avl_train:\n", avl_train)
print("Print avl_val:\n", avl_val)
print("Print avl_test:\n", avl_test)

print("Print avl_train.targets.uniques (AS .targets IS A DataAttribute AND THE ONLY WAY TO ACCESS AND PRINT ITS ACTUAL ELEMENTS/VALUES IS USING ATTRIBUTE .uniques):", avl_train.targets.uniques)
#print("Print avl_train.targets TO PROVE THE ABOVE, AS WHEN ATTRIBUTE .uniques OF DataAttribute IS NOT USED, THE ACTUAL TARGETS/VALUES CONTAINED WITHIN avl_train.targets ARE NOT ACCESSED:, ", avl_train.targets)
print("Print max(avl_train.targets.uniques) in order to be aware of the value that due to the library bug will be assigned to num_classes no matter what my code does (and in this way better mitigate, follow, identify and debug the consequence of such bug):", max(avl_train.targets.uniques))
print("Print len(avl_train.targets.uniques) to know the NUMBER OF FINE CLASSES within the specialised_models_superclasses group of which the specialised subset (e.g. outdoors_specialised_models_train_subset) that is avl_train is composed:", len(avl_train.targets.uniques)) # print number of classes
print("IMPORTANT! Print sorted(avl_train.targets.uniques) AS THIS WILL BE THE OBJECT PASSED TO class_order IN class_incremental_benchmark TO AVOID/AS A WORK-AROUND TO THE num_classes BUG IN THE LIBRARY CODE, since by provideing a class_order, 'class_order = torch.randperm(num_classes).tolist()' within 'if class_order is None:' IS AVOIDED!", sorted(avl_train.targets.uniques))# print class_order

outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_classes_per_exp=[2, 2, 2, 2, 69])
                                                                    #num_experiences=5)

print("sorted(avl_train.targets.uniques)",sorted(avl_train.targets.uniques))
for i, exp in enumerate(outdoors_specialised_models_benchmark.train_stream):
    print("print(i, exp.classes_in_this_experience)", i, exp.classes_in_this_experience)

print("Print outdoors_specialised_models_benchmark:", outdoors_specialised_models_benchmark)
#print("Print outdoors_specialised_models_benchmark.train_stream:", outdoors_specialised_models_benchmark.train_stream) # it works, but I am not sure that it is not deprecated. I think the current or at least most frequently used way of retrieving a stream is .streams["train"], specially if there is a .streams["valid"] since attribute .val_stream does NOT exist
print('Print outdoors_specialised_models_benchmark.streams["train"]',outdoors_specialised_models_benchmark.streams["train"])
print('Print outdoors_specialised_models_benchmark.streams["test"]', outdoors_specialised_models_benchmark.streams["test"])
print('Print outdoors_specialised_models_benchmark.streams["valid"]', outdoors_specialised_models_benchmark.streams["valid"])

# IMPORTANT!!!!! This time, we do want to keep these as streams and NOT to further divide each stream into its composing experiences. This time,
# since class_incremental_benchmark is actually being used to create a CLScenario for use in training in a class-IL manner, and NOT as functionality
# for splitting the dataset into specialised model subsets (as we did previously in Load_Cifar100_v6.py, when class_incremental_benchmark was used
# last), we want to keep the stream objects intact without breaking them apart into its composing experience objects, as the experience objects are
# to be managed by the Avalanche training functionality/training loop to ensure the specialised model is trained in a class_IL manner with each
# experience being used for a subsequent incremental training round.
outdoors_specialised_models_train_stream_class_il = outdoors_specialised_models_benchmark.streams["train"]
outdoors_specialised_models_val_stream_class_il = outdoors_specialised_models_benchmark.streams["valid"]
outdoors_specialised_models_test_stream_class_il = outdoors_specialised_models_benchmark.streams["test"]

#####################################################################################################################################################################################################################
#class ExemplarSelection(ExemplarsSelectionStrategy):



# Attempting to use MaskedCrossEntropy for masking all but the classes that appear in the specialised model's subset / the classes for which a particular specialised model is meant to be specialised for
#criterion_masked = MaskedCrossEntropy(classes=[33, 68, 37, 71, 76, 12, 49, 17, 23, 60], mask="seen") #sorted(avl_train.targets.uniques)
criterion_masked = MaskedCrossEntropy(mask="seen") #sorted(avl_train.targets.uniques)

# TRAINING LOOP
optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9) # example from Avalanche docs: model = SimpleMLP(num_classes=10) and since num_classes=10 in outdoors_specialised_model, I have kept the lr=0.001, momentum=0.9 as provided in the model = SimpleMLP(num_classes=10) example for the outdoors_specialised_model
criterion = CrossEntropyLoss()
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion_masked) # criterion=criterion) #, train_mb_size=100, train_epochs=4, eval_mb_size=100)

#the block to uncomment
# REALISATION! This code that I took from https://avalanche.continualai.org/avalanche/from-zero-to-hero-tutorial/04_training#training-and-evaluation
# is not the one and only way to build a training loop -- IT IS THE WAY TO BUILD A TRAINING LOOP IF optimizer = SGD !!!  When optimizer = SGD, .train
# is the .train of class BaseSGDTemplate, where the training loop IS DESIGNED TO BE EXECUTED ONCE PER EXPERIENCE, whether you provide the entire
# stream or a single experience as input (i.e. whether you call .train() on each experience within the stream one experience at a time, perhaps via
# a for loop like the one in this example, or you pass the entire stream as input to .train(), and let the training loop handle the
# experience-by-experience, given that the function supports both options).
'''print('Starting experiment...')
results = []

cl_strategy.train(outdoors_specialised_models_train_stream_class_il.exps_iter)
print('Training completed')

print('Computing accuracy on the whole test set')
results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))'''

print('Starting experiment WITH LOOP...')
results = []
for experience in outdoors_specialised_models_benchmark.train_stream:
#for experience in outdoors_specialised_models_train_stream_class_il:
    print("Start of experience WITH LOOP: ", experience.current_experience)
    print("Current Classes WITH LOOP: ", experience.classes_in_this_experience)
    ###################active_classes = experience.classes_in_this_experience.tolist()
    print("experience.dataset WITH LOOP", experience.dataset)
    print("experience.dataset.targets WITH LOOP", experience.dataset.targets.uniques)
    #print("experience.dataset.coarse_label.uniques", experience.dataset.coarse_label.uniques)
    print("experience", experience)
    #x, y, z = experience.dataset[0] # returns images, targets, coarse_labels
    #print("x", x, "\ny", y, "\nz", z)

    cl_strategy.train(experience)#,[outdoors_specialised_models_train_stream_class_il[0],
                                  #outdoors_specialised_models_train_stream_class_il[1],
                                  #outdoors_specialised_models_train_stream_class_il[2],
                                  #outdoors_specialised_models_train_stream_class_il[3],
                                  #outdoors_specialised_models_train_stream_class_il[4]]) #.train(outdoors_specialised_models_train_stream_class_il)
    print('Training completed WITH LOOP')

    print('Computing accuracy on the whole test set WITH LOOP')
    results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))

######## WHERE TO START NEXT:
'''MaskedCrossEntropy is a cross-entropy loss wrapper for class-incremental learning. Its main purpose is to compute the loss only over a selected subset of output classes, instead of over every output neuron.

This is useful in continual learning because, at a given experience, you may only want the model to compete among:

the classes seen so far,
only the newly introduced classes,
only the previously learned classes,
or all classes.

https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/losses.py

https://avalanche-api.continualai.org/en/latest/generated/avalanche.training.MaskedCrossEntropy.html#avalanche.training.MaskedCrossEntropy'''



##### DOES NOOOOT WORK -- not using avl_train = ClassificationDataset and passing the .dataset attribute of the DataExperience object that is outdoors_specialised_models_train_subset #####
'''outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": outdoors_specialised_models_train_subset.dataset, "test": outdoors_specialised_models_test_subset.dataset, "valid": outdoors_specialised_models_val_subset.dataset},
                                                                    class_order=sorted(outdoors_specialised_models_train_subset.dataset.targets.uniques), num_experiences=5)'''


##### DOES NOOOOT WORK -- using avl_train = ClassificationDataset BUT NOT avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset)
# and instead just passing the .dataset attribute of the DataExperience object that is outdoors_specialised_models_train_subset to the avl_train = ClassificationDataset
# which is then the object passed to outdoors_specialised_models_benchmark = class_incremental_benchmark ####
'''avl_train = ClassificationDataset(
    [outdoors_specialised_models_train_subset.dataset],
    data_attributes=[
        DataAttribute(outdoors_specialised_models_train_subset.dataset.coarse_label, "coarse_label"),
        DataAttribute(outdoors_specialised_models_train_subset.dataset.targets, "targets")], # "fine_label")],
)
avl_val = ClassificationDataset(
    [outdoors_specialised_models_val_subset.dataset],
    data_attributes=[
        DataAttribute(outdoors_specialised_models_val_subset.dataset.coarse_label, "coarse_label"),
        DataAttribute(outdoors_specialised_models_val_subset.dataset.targets, "targets")],# "fine_label")],
)
avl_test = ClassificationDataset(
    [outdoors_specialised_models_test_subset.dataset],
    data_attributes=[
        DataAttribute(outdoors_specialised_models_test_subset.dataset.coarse_label, "coarse_label"),
        DataAttribute(outdoors_specialised_models_test_subset.dataset.targets, "targets")],# "fine_label")],
)
outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_experiences=5)'''

##### DOES NOOOOT WORK -- same as above but instead of passing a string to the DataAttribute object for the "name field" in ClassificationDataset's
# data_attributes parameter, I passed .dataset.coarse_label.name and .dataset.targets.name)
'''avl_train = ClassificationDataset(
    [outdoors_specialised_models_train_subset.dataset],
    data_attributes=[
        DataAttribute(outdoors_specialised_models_train_subset.dataset.coarse_label, outdoors_specialised_models_train_subset.dataset.coarse_label.name),
        DataAttribute(outdoors_specialised_models_train_subset.dataset.targets, outdoors_specialised_models_train_subset.dataset.targets.name)], # "fine_label")],
)
avl_val = ClassificationDataset(
    [outdoors_specialised_models_val_subset.dataset],
    data_attributes=[
        DataAttribute(outdoors_specialised_models_val_subset.dataset.coarse_label, outdoors_specialised_models_val_subset.dataset.coarse_label.name),
        DataAttribute(outdoors_specialised_models_val_subset.dataset.targets, outdoors_specialised_models_val_subset.dataset.targets.name)],# "fine_label")],
)
avl_test = ClassificationDataset(
    [outdoors_specialised_models_test_subset.dataset],
    data_attributes=[
        DataAttribute(outdoors_specialised_models_test_subset.dataset.coarse_label, outdoors_specialised_models_test_subset.dataset.coarse_label.name),
        DataAttribute(outdoors_specialised_models_test_subset.dataset.targets, outdoors_specialised_models_test_subset.dataset.targets.name)],# "fine_label")],
)
outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_experiences=5)'''



##### DOES NOOOOT WORK -- using avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset) AND avl_train = ClassificationDataset
# but where avl_train.coarse_label.uniques and avl_train.targets.uniques were passed to the DataAttribute objects passed to ClassificationDataset's data_attributes parameter.
# That is, the .uniques was used leading to an error that would not even let execution terminate (raise ValueError(ValueError: Data attribute targets has length 10 but the dataset has length 4000),
# but additionally also NOT SOLVE THE ISSUE THIS CODE WAS MEANT TO SOLVE
'''
avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset)
avl_val = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset)
avl_test = make_avalanche_dataset(outdoors_specialised_models_test_subset.dataset)

avl_train = ClassificationDataset(
    [avl_train], #[outdoors_specialised_models_train_subset.dataset],
    data_attributes=[
        DataAttribute(avl_train.coarse_label.uniques, "coarse_label"),
        DataAttribute(avl_train.targets.uniques, "targets")], # "fine_label")],
)
avl_val = ClassificationDataset(
    [avl_val], #[outdoors_specialised_models_val_subset.dataset],
    data_attributes=[
        DataAttribute(avl_val.coarse_label.uniques, "coarse_label"),
        DataAttribute(avl_val.targets.uniques, "targets")],# "fine_label")],
)
avl_test = ClassificationDataset(
    [avl_test], #[outdoors_specialised_models_test_subset.dataset],
    data_attributes=[
        DataAttribute(avl_test.coarse_label.uniques, "coarse_label"),
        DataAttribute(avl_test.targets.uniques, "targets")],# "fine_label")],
)
outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_experiences=5)
'''

#### DOES NOOOOT WORK -- using avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset) AND avl_train = ClassificationDataset
# but where data_attributes are passed TO BOTH avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset) AND avl_train = ClassificationDataset
'''avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset,
                                   data_attributes=[DataAttribute(outdoors_specialised_models_train_subset.dataset.coarse_label, "coarse_label"),
                                                    DataAttribute(outdoors_specialised_models_train_subset.dataset.targets, "targets")]) # "fine_label")],)

avl_val = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset,
                                 data_attributes=[DataAttribute(outdoors_specialised_models_val_subset.dataset.coarse_label, "coarse_label"),
                                                  DataAttribute(outdoors_specialised_models_val_subset.dataset.targets, "targets")]) # "fine_label")],))

avl_test = make_avalanche_dataset(outdoors_specialised_models_test_subset.dataset,
                                  data_attributes=[DataAttribute(outdoors_specialised_models_test_subset.dataset.coarse_label, "coarse_label"),
                                                   DataAttribute(outdoors_specialised_models_test_subset.dataset.targets, "targets")]) # "fine_label")],))
                                                   
avl_train = ClassificationDataset(
    [avl_train], #[outdoors_specialised_models_train_subset.dataset],
    data_attributes=[
        DataAttribute(avl_train.coarse_label, "coarse_label"),
        DataAttribute(avl_train.targets, "targets")], # "fine_label")],
)
avl_val = ClassificationDataset(
    [avl_val], #[outdoors_specialised_models_val_subset.dataset],
    data_attributes=[
        DataAttribute(avl_val.coarse_label, "coarse_label"),
        DataAttribute(avl_val.targets, "targets")],# "fine_label")],
)
avl_test = ClassificationDataset(
    [avl_test], #[outdoors_specialised_models_test_subset.dataset],
    data_attributes=[
        DataAttribute(avl_test.coarse_label, "coarse_label"),
        DataAttribute(avl_test.targets, "targets")],# "fine_label")],
)                                                
'''

#### DOES NOOOOT WORK -- using avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset) AND avl_train = ClassificationDataset
# but where data_attributes are passed ONLY TO avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset)
# and NOT to avl_train = ClassificationDataset, where they are in all previous versions
'''avl_train = make_avalanche_dataset(outdoors_specialised_models_train_subset.dataset,
                                   data_attributes=[DataAttribute(outdoors_specialised_models_train_subset.dataset.coarse_label, "coarse_label"),
                                                    DataAttribute(outdoors_specialised_models_train_subset.dataset.targets, "targets")]) # "fine_label")],)

avl_val = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset,
                                 data_attributes=[DataAttribute(outdoors_specialised_models_val_subset.dataset.coarse_label, "coarse_label"),
                                                  DataAttribute(outdoors_specialised_models_val_subset.dataset.targets, "targets")]) # "fine_label")],))

avl_test = make_avalanche_dataset(outdoors_specialised_models_test_subset.dataset,
                                  data_attributes=[DataAttribute(outdoors_specialised_models_test_subset.dataset.coarse_label, "coarse_label"),
                                                   DataAttribute(outdoors_specialised_models_test_subset.dataset.targets, "targets")]) # "fine_label")],))
                                                   
avl_train = ClassificationDataset([avl_train])
avl_val = ClassificationDataset([avl_val])
avl_test = ClassificationDataset([avl_test])
'''