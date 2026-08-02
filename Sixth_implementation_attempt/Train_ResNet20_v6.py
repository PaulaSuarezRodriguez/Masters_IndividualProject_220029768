from Sixth_implementation_attempt.Config_v6 import set_seed
set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Sixth_implementation_attempt.ResNet20_v6 import ResNet20
from Sixth_implementation_attempt.Load_cifar100_v6 import (outdoors_specialised_models_train_subset, nature_specialised_models_train_subset,
                                                            people_specialised_models_train_subset, animals_specialised_models_train_subset, objects_and_vehicles_specialised_models_train_subset)
from Sixth_implementation_attempt.Load_cifar100_v6 import (outdoors_specialised_models_val_subset, nature_specialised_models_val_subset,
                                                            people_specialised_models_val_subset, animals_specialised_models_val_subset, objects_and_vehicles_specialised_models_val_subset)
from Sixth_implementation_attempt.Load_cifar100_v6 import (outdoors_specialised_models_test_subset, nature_specialised_models_test_subset,
                                                            people_specialised_models_test_subset, animals_specialised_models_test_subset, objects_and_vehicles_specialised_models_test_subset)
from avalanche.benchmarks import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.benchmarks import benchmark_with_validation_stream
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!

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

print("Type avl_train AFTER being passed to ClassificationDataset", type(avl_train))
print("Type avl_val AFTER being passed to ClassificationDataset", type(avl_val))
print("Type avl_test AFTER being passed to ClassificationDataset", type(avl_test))

print("Print avl_train.targets.uniques (AS .targets IS A DataAttribute AND THE ONLY WAY TO ACCESS AND PRINT ITS ACTUAL ELEMENTS/VALUES IS USING ATTRIBUTE .uniques):", avl_train.targets.uniques)
#print("Print avl_train.targets TO PROVE THE ABOVE, AS WHEN ATTRIBUTE .uniques OF DataAttribute IS NOT USED, THE ACTUAL TARGETS/VALUES CONTAINED WITHIN avl_train.targets ARE NOT ACCESSED:, ", avl_train.targets)
print("Print max(avl_train.targets.uniques) in order to be aware of the value that due to the library bug will be assigned to num_classes no matter what my code does (and in this way better mitigate, follow, identify and debug the consequence of such bug):", max(avl_train.targets.uniques))
print("Print len(avl_train.targets.uniques) to know the NUMBER OF FINE CLASSES within the specialised_models_superclasses group of which the specialised subset (e.g. outdoors_specialised_models_train_subset) that is avl_train is composed:", len(avl_train.targets.uniques)) # print number of classes
print("IMPORTANT! Print sorted(avl_train.targets.uniques) AS THIS WILL BE THE OBJECT PASSED TO class_order IN class_incremental_benchmark TO AVOID/AS A WORK-AROUND TO THE num_classes BUG IN THE LIBRARY CODE, since by provideing a class_order, 'class_order = torch.randperm(num_classes).tolist()' within 'if class_order is None:' IS AVOIDED!", sorted(avl_train.targets.uniques))# print class_order


outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val},
                                                                    class_order=sorted(avl_train.targets.uniques), num_experiences=5)
#, num_classes_per_exp=[2, 2, 2, 2, 2])
# NAIVE IMPLEMENTATION: same num_experiences=5 for every specialised model, and since in order to avoid bug, num_experiences needs to be set
# using num_classes_per_exp, for every specialised model, the values within num_classes_per_exp will be set to the length of class_order
# (as returned by class_order_len = len(avl_train.targets.uniques)) divided by 5, which will be the num_experiences intended for every specialised
# model, hence why num_classes_per_exp list for the class_incremental_benchmark of every specialised model subset will always have 5 elements (all
# of them set to the value equal to the length of class_order divided by 5). CHANGE OF PLANS!!! It raises error given that num_classes is still set
# according to the bug/assumption made by the library:
# num_classes: int = 1 + max(list(datasets_dict.values())[0].targets.uniques)
#     if (num_classes_per_exp is not None) and (num_classes != sum(num_classes_per_exp)):
#         raise ValueError(
#             "`sum(num_classes_per_exp)` must be equal to the total number of classes."
#         )

print("Print outdoors_specialised_models_benchmark:", outdoors_specialised_models_benchmark)
#print("Print outdoors_specialised_models_benchmark.train_stream:", outdoors_specialised_models_benchmark.train_stream)
print('Print outdoors_specialised_models_benchmark.streams["train"]',outdoors_specialised_models_benchmark.streams["train"])
print('Print outdoors_specialised_models_benchmark.streams["test"]', outdoors_specialised_models_benchmark.streams["test"])
print('Print outdoors_specialised_models_benchmark.streams["valid"]', outdoors_specialised_models_benchmark.streams["valid"]) # NOT benchmark_with_validation_stream, hence KeyError: 'valid'

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
# TRAINING LOOP
optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9) # example from Avalanche docs: model = SimpleMLP(num_classes=10) and since num_classes=10 in outdoors_specialised_model, I have kept the lr=0.001, momentum=0.9 as provided in the model = SimpleMLP(num_classes=10) example for the outdoors_specialised_model
criterion = CrossEntropyLoss()
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion) #, train_mb_size=100, train_epochs=4, eval_mb_size=100)

print("outdoors_specialised_models_train_stream_class_il[0]",outdoors_specialised_models_train_stream_class_il[0].dataset[0])
print("outdoors_specialised_models_val_stream_class_il[0]",outdoors_specialised_models_val_stream_class_il[0].dataset[0])
print("outdoors_specialised_models_test_stream_class_il[0]",outdoors_specialised_models_test_stream_class_il[0].dataset[0])
#the block to uncomment
# REALISATION! This code that I took from https://avalanche.continualai.org/avalanche/from-zero-to-hero-tutorial/04_training#training-and-evaluation
# is not the one and only way to build a training loop -- IT IS THE WAY TO BUILD A TRAINING LOOP IF optimizer = SGD !!!  When optimizer = SGD, .train
# is the .train of class BaseSGDTemplate, where the training loop IS DESIGNED TO BE EXECUTED ONCE PER EXPERIENCE, whether you provide the entire
# stream or a single experience as input (i.e. whether you call .train() on each experience within the stream one experience at a time, perhaps via
# a for loop like the one in this example, or you pass the entire stream as input to .train(), and let the training loop handle the
# experience-by-experience, given that the function supports both options). MORE IMPORTANTLY, HOWEVER, EVEN FOR optimizer = SGD, THIS TRAINING LOOP
# IS WRONG!!! The code example provided by https://avalanche.continualai.org/avalanche/from-zero-to-hero-tutorial/04_training#training-and-evaluation
# and copy-pasted here is wrong!!! One of the ways in which is wrong is that as indicate by source code super().train(experiences, eval_streams, **kwargs)
# .train() expects eval_streams to be passed, which is something the code example copy-pasted here does not do and in debugging the call cl_strategy.train(experience),
# it can be seen that eval_streams = None and the execution path followed is that for if eval_streams = None.
# PLAN: Instead of wasting time trying to figure out how the .train for when optimizer = SGD should work to make this code example work, learn how the
# training functionality of Avalanche works from scratch
# .train of class BaseSGDTemplate. M
print('Starting experiment...')
results = []
for experience in outdoors_specialised_models_train_stream_class_il:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    #active_classes = experience.classes_in_this_experience.tolist()
    print("experience.dataset", experience.dataset)
    print("experience.dataset.targets", experience.dataset.targets.uniques)
    print("experience.dataset.coarse_label", experience.dataset.coarse_label.uniques)
    print("experience", experience)
    #x, y, z = experience.dataset[0]
    #print("x", x, "\ny", y, "\nz", z)

    cl_strategy.train(experience)#,[outdoors_specialised_models_train_stream_class_il[0],
                                 #outdoors_specialised_models_train_stream_class_il[1],
                                  #outdoors_specialised_models_train_stream_class_il[2],
                                  #outdoors_specialised_models_train_stream_class_il[3],
                                  #outdoors_specialised_models_train_stream_class_il[4]]) #.train(outdoors_specialised_models_train_stream_class_il)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))



def label_remapping_specialised_model(outdoors_specialised_models_train_stream_class_il_param):
    #for exp, expp in enumerate(outdoors_specialised_models_train_stream_class_il.exps_iter):
    for exp in outdoors_specialised_models_train_stream_class_il_param:
        i, t, c_l = exp.dataset[0]
        print("targets:", t, "coarse_label:", c_l)
        '''print("len(exp.dataset.targets)",len(exp[0].dataset.targets.uniques))
        print("len(exp.dataset.coarse_label)",len(exp[0].dataset.coarse_label))'''
        return t, c_l



import copy
import torch


def remap_subset_labels(dataset):
    """
    Returns a copy of the dataset with labels remapped to
    consecutive integers 0...(K-1).

    Also returns the forward and inverse mappings.
    """

    dataset = copy.deepcopy(dataset)

    unique_labels = sorted(dataset.targets.uniques.tolist())

    label_map = {
        old_label: new_label
        for new_label, old_label in enumerate(unique_labels)
    }

    inverse_label_map = {
        new_label: old_label
        for old_label, new_label in label_map.items()
    }

    dataset.targets = torch.tensor(
        [label_map[int(label)] for label in dataset.targets],
        dtype=dataset.targets.dtype,
    )

    return dataset, label_map, inverse_label_map




label_remapping_specialised_model(outdoors_specialised_models_train_stream_class_il)











#####################################################################################################################################################################################################################
'''#the block to uncomment
print('Starting experiment...')
results = []
for experience in outdoors_specialised_models_train_stream_class_il:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(outdoors_specialised_models_val_stream_class_il))'''

'''for group in groups:
    for sub_experience in group:
        strategy.train(sub_experience)'''

'''print('Starting experiment...')
results = []
for experience in outdoors_specialised_models_train_subset.dataset.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(outdoors_specialised_models_val_subset.dataset))'''
'''results.append(cl_strategy.eval(outdoors_specialised_models_val_subset[experience.current_experience]))
    print('Test accuracy:', results[-1].metrics['accuracy'])'''


