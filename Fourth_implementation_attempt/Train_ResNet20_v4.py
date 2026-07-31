from Fourth_implementation_attempt.Config_v4 import set_seed
set_seed() # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Fourth_implementation_attempt.ResNet20_v4 import ResNet20
from Fourth_implementation_attempt.Load_cifar100_v4 import (outdoors_specialised_models_train_subset, nature_specialised_models_train_subset,
                                                            people_specialised_models_train_subset, animals_specialised_models_train_subset, objects_and_vehicles_specialised_models_train_subset)
from Fourth_implementation_attempt.Load_cifar100_v4 import (outdoors_specialised_models_val_subset, nature_specialised_models_val_subset,
                                                            people_specialised_models_val_subset, animals_specialised_models_val_subset, objects_and_vehicles_specialised_models_val_subset)
from Fourth_implementation_attempt.Load_cifar100_v4 import (outdoors_specialised_models_test_subset, nature_specialised_models_test_subset,
                                                            people_specialised_models_test_subset, animals_specialised_models_test_subset, objects_and_vehicles_specialised_models_test_subset)
from avalanche.benchmarks import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from avalanche.benchmarks import benchmark_with_validation_stream

#resnet20_global_model = ResNet20()

# NAIVE / BASELINE IMPLEMENTATION OF MY STRATEGY --> train each specialised model in a Class-IL manner without attempting to mitigate CF.
# Additionally, pre-train the global model on the EXPERIENCE 1 TRAIN SUBSETS and attempt to naively update it (using a hard-coded version of FedAvg
# and no techniques to mitigate ST-CF):

resnet20_global_model = ResNet20(num_classes=100)
outdoors_specialised_model = ResNet20(num_classes=10)
nature_specialised_model = ResNet20(num_classes=15)
people_specialised_model = ResNet20(num_classes=5)
animals_specialised_model = ResNet20(num_classes=45)
objects_and_vehicles_specialised_model = ResNet20(num_classes=25)

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!

optimizer = SGD(outdoors_specialised_model.parameters(), lr=0.001, momentum=0.9) # example from Avalanche docs: model = SimpleMLP(num_classes=10) and since num_classes=10 in outdoors_specialised_model, I have kept the lr=0.001, momentum=0.9 as provided in the model = SimpleMLP(num_classes=10) example for the outdoors_specialised_model
criterion = CrossEntropyLoss()
cl_strategy = Naive(model=outdoors_specialised_model, optimizer=optimizer, criterion=criterion, train_mb_size=100, train_epochs=4, eval_mb_size=100)

#print("outdoors_specialised_models_train_subset.dataset", outdoors_specialised_models_train_subset.dataset)
#print("outdoors_specialised_models_val_subset.dataset", outdoors_specialised_models_val_subset.dataset)
print("type(outdoors_specialised_models_train_subset.dataset)", type(outdoors_specialised_models_train_subset.dataset))
print("type(outdoors_specialised_models_val_subset.dataset)", type(outdoors_specialised_models_val_subset.dataset))

#print("outdoors_specialised_models_train_subset.dataset.targets", outdoors_specialised_models_train_subset.dataset.targets)
#print("outdoors_specialised_models_val_subset.dataset.targets", outdoors_specialised_models_val_subset.dataset.targets)
print("type(outdoors_specialised_models_train_subset.dataset.targets)", type(outdoors_specialised_models_train_subset.dataset.targets))
print("type(outdoors_specialised_models_val_subset.dataset.targets)", type(outdoors_specialised_models_val_subset.dataset.targets))

#print("outdoors_specialised_models_train_subset.dataset.coarse_label)", outdoors_specialised_models_train_subset.dataset.coarse_label)
#print("outdoors_specialised_models_val_subset.dataset.coarse_label)", outdoors_specialised_models_val_subset.dataset.coarse_label)
print("type(outdoors_specialised_models_train_subset.dataset.coarse_label))", type(outdoors_specialised_models_train_subset.dataset.coarse_label))
print("type(outdoors_specialised_models_val_subset.dataset.coarse_label))", type(outdoors_specialised_models_val_subset.dataset.coarse_label))

OPTION_1 = True
OPTION_1A = False #True
OPTION_1B = False #True
OPTION_1C = False #True
OPTION_1D = False #True
OPTION_1E = False #True
OPTION_1F = True
if OPTION_1:
    if OPTION_1 and OPTION_1A:
        avl_train = ClassificationDataset(
            [outdoors_specialised_models_train_subset.dataset],
            data_attributes=[DataAttribute(outdoors_specialised_models_train_subset.dataset.targets, "targets")], #"fine_label")],
        )
        avl_val = ClassificationDataset(
            [outdoors_specialised_models_val_subset.dataset],
            data_attributes=[DataAttribute(outdoors_specialised_models_val_subset.dataset.targets, "targets")], #"fine_label")],
        )

    if OPTION_1 and OPTION_1B:
        avl_train = ClassificationDataset([outdoors_specialised_models_train_subset.dataset],
            data_attributes = [DataAttribute(outdoors_specialised_models_train_subset.dataset.coarse_label, "coarse_label"),
                               DataAttribute(outdoors_specialised_models_train_subset.dataset.targets, "targets")],  # "fine_label")],
            )
        avl_val = ClassificationDataset(
        [outdoors_specialised_models_val_subset.dataset],
        data_attributes = [DataAttribute(outdoors_specialised_models_val_subset.dataset.coarse_label, "coarse_label"),
                           DataAttribute(outdoors_specialised_models_val_subset.dataset.targets, "targets")],  # "fine_label")],
        )

    if OPTION_1 and OPTION_1C:
        avl_train = ClassificationDataset(
            [outdoors_specialised_models_train_subset.dataset],
            data_attributes=[outdoors_specialised_models_train_subset.dataset.coarse_label, outdoors_specialised_models_train_subset.dataset.targets],  # "fine_label")],
            )
        avl_val = ClassificationDataset(
            [outdoors_specialised_models_val_subset.dataset],
            data_attributes=[outdoors_specialised_models_val_subset.dataset.coarse_label, outdoors_specialised_models_val_subset.dataset.targets],  # "fine_label")],
            )

    '''if OPTION_1 and OPTION_1D: # This version of OPTION 1 leads to a completely new error THAT THE OTHER VERSIONS DO NOT GIVE; DISCARD THE BELOW:
        avl_train = ClassificationDataset(
            [outdoors_specialised_models_train_subset.dataset],
            data_attributes=[(outdoors_specialised_models_train_subset.dataset.coarse_label, "coarse_label"), (outdoors_specialised_models_train_subset.dataset.targets, "targets")],  # "fine_label")],
            )
        avl_val = ClassificationDataset(
            [outdoors_specialised_models_val_subset.dataset],
            data_attributes=[(outdoors_specialised_models_val_subset.dataset.coarse_label, "coarse_label"), (outdoors_specialised_models_val_subset.dataset.targets, "targets")],  # "fine_label")],
            )'''

    if OPTION_1 and OPTION_1E: # ValueError: Data attribute coarse_label has length 12 but the dataset has length 4000
        print("type(outdoors_specialised_models_train_subset.dataset.coarse_label.name)", type(outdoors_specialised_models_train_subset.dataset.coarse_label.name))
        #print("type(outdoors_specialised_models_train_subset.dataset.coarse_label.name.tolist())", type(outdoors_specialised_models_train_subset.dataset.coarse_label.name.tolist()))
        #print("type(outdoors_specialised_models_train_subset.dataset.coarse_label.tolist())",type(outdoors_specialised_models_train_subset.dataset.coarse_label.tolist()))

        avl_train = ClassificationDataset(
            [outdoors_specialised_models_train_subset.dataset],
            data_attributes=[DataAttribute(outdoors_specialised_models_train_subset.dataset.coarse_label.name, "coarse_label"),
                             DataAttribute(outdoors_specialised_models_train_subset.dataset.targets.name, "targets")],  # "fine_label")],
            )
        avl_val = ClassificationDataset(
            [outdoors_specialised_models_val_subset.dataset],
            data_attributes=[DataAttribute(outdoors_specialised_models_val_subset.dataset.coarse_label.name, "coarse_label"),
                             DataAttribute(outdoors_specialised_models_val_subset.dataset.targets.name, "targets")],  # "fine_label")],
            )

    if OPTION_1 and OPTION_1F:  # ValueError: Data attribute coarse_label has length 12 but the dataset has length 4000
        print("\n\n\n\n\n OPTION_1F:")
        print("type(outdoors_specialised_models_train_subset.dataset.coarse_label.name)", type(outdoors_specialised_models_train_subset.dataset.coarse_label.name))
        print("type(outdoors_specialised_models_train_subset.dataset.targets.name)", type(outdoors_specialised_models_train_subset.dataset.targets.name))
        print("type(outdoors_specialised_models_train_subset.dataset)", type(outdoors_specialised_models_train_subset.dataset))

        print("outdoors_specialised_models_train_subset.dataset.coarse_label.name", outdoors_specialised_models_train_subset.dataset.coarse_label.name)
        print("outdoors_specialised_models_train_subset.dataset.targets.name", outdoors_specialised_models_train_subset.dataset.targets.name)
        print("outdoors_specialised_models_train_subset.dataset", outdoors_specialised_models_train_subset.dataset)


        avl_train = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset)
        avl_val = make_avalanche_dataset(outdoors_specialised_models_val_subset.dataset)

        print("type(outdoors_specialised_models_train_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_train", type(avl_train))
        print("type(outdoors_specialised_models_val_subset.dataset) AFTER BEING PASSED TO make_avalanche_dataset and re-named avl_val", type(avl_val))

        avl_train = ClassificationDataset(
            [avl_train], #[outdoors_specialised_models_train_subset.dataset],
            data_attributes=[
                DataAttribute(avl_train.coarse_label, "coarse_label"),
                DataAttribute(avl_train.targets, "targets")],
            # "fine_label")],
        )
        avl_val = ClassificationDataset(
            [avl_val], #[outdoors_specialised_models_val_subset.dataset],
            data_attributes=[
                DataAttribute(avl_val.coarse_label, "coarse_label"),
                DataAttribute(avl_val.targets, "targets")],
            # "fine_label")],
        )

        '''print("type(avl_train)", type(avl_train))
        print("type(avl_val)", type(avl_val))'''
        print("Type avl_train AFTER being passed to ClassificationDataset", type(avl_train))
        print("Type avl_val AFTER being passed to ClassificationDataset", type(avl_val))

        outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_val},
                                                                            num_experiences=5)

OPTION_2 = False
if OPTION_2:
    outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": outdoors_specialised_models_train_subset.dataset, "test": outdoors_specialised_models_val_subset.dataset},
                                                                    num_experiences=5)




# TRAINING LOOP
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

#the block to uncomment
'''print('Starting experiment...')
results = []
for experience in outdoors_specialised_models_benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(outdoors_specialised_models_benchmark.test_stream))'''


'''for group in groups:
    for sub_experience in group:
        strategy.train(sub_experience)'''