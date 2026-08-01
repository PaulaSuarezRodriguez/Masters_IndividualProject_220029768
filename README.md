# Masters_IndividualProject_220029768

### Difference between First_implementation_attempt and Second_implementation_attempt: 

split_by_specialised_models_superclasses_using_avalanche gives me a "ValueError: Data attribute fine_label has length 45000
but the dataset has length 50000" even when the data_attributes is in the format required for ClassificationDataset to avoid
error "assert "targets" in self._data_attributes, AssertionError: The supervised version of the ClassificationDataset requires 
the targets field". That is, even when data_attributes=[DataAttribute(train_dataset_param.dataset.coarse_labels.tolist(), "coarse_label"), DataAttribute(train_dataset_param.indices, "fine_label")],
meaning even when ClassificationDataset and class_incremental_benchmark are being used correctly, the code still does not work, 
this time not due to incorrect formatting, BUT DUE TO INCOMPATIBILITY BETWEEN HOW THE DATASETS HAVE BEEN MANUALLY PRE-PROCESSED 
(e.g. how train and val where split, among other pre-processing steps) AND THE DATASET FORMAT EXPECTED BY Avalanche. I suspect 
Avalanche expects all pre-rpocessing to be performed using its own functionality and so, does not accept the dataset already 
pre-processed by me. avalanche.benchmarks.benchmark_with_validation_stream (v0.6.0) can be used to split the train and val sets.
I will be splitting the train and val sets using this Avalanche functionality and all other Avalanche functionality pertinent to
ensure compatibility in the Load_cifar100.py and whatever other .py file appropriate of Second_implementation_attempt, abandoning
First_implementation_attempt.

### Difference between Second_implementation_attempt and Third_implementation_attempt:

In Load_Cifar100_v2.py of Second_implementation_attempt, I implemented 5 different versions for trying to split the dataset 
into train and val subsets and/or trying to split anyone of these by "specialised_models_superclasses". 
None of them work. Versions 1 to 3 I had already discarded in the past. The novelty attempts were versions 4 and 5,
and these do not work either. Possible reasons for VERSION_4 and VERSION_5 not working: 

- thanks to the errors from VERSION_4, I now know this is due to the dataset_dict object(s) expecting a target and that 
it does not consider the coarse nor fine labels I passed as DataAttributes to the avl_train_dataset of classil_dict passed
as the dataset_dict in class_incremental_benchmark to be the "targets" it is referring to


- in VERSION_5, I was supposed to change from using make_avalanche_dataset to ClassificationDatasets because class_incremental_benchmark takes 
ClassificationDatasets, which are a class inheriting from AvalancheDataset, and since I cannot  manage to fix the error that the 
make_avalanche_dataset and class_incremental_benchmark combination of VERSION_4 gives me (namely, what it means by targets, not only in terms 
of value as in does it mean coarse or fine labels, but also as in does it mean targets is a particular attribute the dataset passed to 
class_incremental_benchmark should have and that AvalancheDataset is missing and only ClassificationDatasets has?. Hence, why I was gonna try 
using ClassificationDatasets and class_incremental_benchmark in VERSION_5. However, I diverted from my objective and instead edited several of the
functions splitting the datasets as I became concerned that I was passing the wrong elements to parameter class_order of class_incremental_benchmark,
and thought for a moment that instead of a list of all 50000 indices in order, it expected 100 indices in order for each fine-grained class.    
I could not manage to make function split_by_classID_specialised_models_superclasses_VERSION_5 that I 
devised for this purpose and so, I was unable to verify what parameter class_order of class_incremental_benchmark expects and moreover, I did not
test the change from using make_avalanche_dataset to ClassificationDatasets and therefore did not figure out if that would fix the 
AttributeError: 'AvalancheDataset' object has no attribute 'targets' of VERSION_4 

Hence, Third_implementation_attempt will start from VERSION_4 of Second_implementation_attempt and test the change from using 
make_avalanche_dataset to ClassificationDatasets, for now ignoring the possibility that wrong elements are being passed to parameter class_order 
of class_incremental_benchmark

The difference between Second_implementation_attempt and Third_implementation_attempt is I will first focus on solving the 
AttributeError: 'AvalancheDataset' object has no attribute 'targets' of VERSION_4 and if after that is fixed I suspect wrong 
elements are being passed to parameter class_order of class_incremental_benchmark, I will fix that, and it will be then that 
I will have to change the functions for splitting tha dataset that are already there, but changing these before fixing the 
AttributeError of VERSION_4 makes no sense, as unless I fix the AttributeError of VERSION_4, I cannot be certain of what 
the requirements for the parameters of certain class_incremental_benchmark after the dataset_dict throwing this error are,
including parameter class_order.

### Difference between Third_implementation_attempt and Fourth_implementation_attempt:

In Load_Cifar100_v4.py of Fourth_implementation_attempt, I am gonna clean up the code and remove the functions that are no longer needed. I am doing that in 
Fourth_implementation_attempt to make sure Load_cifar100_v3.py of Third_implementation_attempt can serve as a SAFECOPY of
the version of this file THAT DOES WORK (in case the code breaks later on when cleaning it or something)!!!!!!!!!!
It will also be in Fourth_implementation_attempt that I try to load the avalanche.benchmarks.scenarios.dataset_scenario.DatasetExperience'objects 
that are the different subsets each corresponding to (i.e. for training a) different specialised model into the adequate
.py files for use in training of the specialised models. 

Finally, it will also be in Fourth_implementation_attempt that I attempt to further divide each of the experiences/DataExperience objects
(i.e. specialised_models_subset) into further experiences, this time meaning actual experiences since until now 
(Load_cifar100_v3.py of Third_implementation_attempt) I used class_incremental_benchmark to split the dataset into the 
different specialised subsets. But since I also plan on training each of the specialised models in a class incremental manner / potentially 
using class_incremental_benchmark, each of the experiences/DataExperience objects/ specialised_models_subset needs 
to be further divided into experiences / incremental chunks for Class-IL, which is what I am trying to do in Fourth_implementation_attempt.

### Difference between Fourth_implementation_attempt and Fifth_implementation_attempt:
TASK AT HAND: How to further divide each of the experiences/DataExperience objects (i.e. specialised_models_subset imported into Train_ResNet20_v4.py)
into further experiences this time meaning actual experiences since in this case I used class_incremental_benchmark to split the dataset into the 
different specialised subsets, but I also plan on training each of the specialised models in a class incremental manner/potentially using 
class_incremental_benchmark, hence each of the experiences/DataExperience objects/ specialised models_subset needs to be further divided into 
experiences / incremental chunks for Class-IL?

WHAT I LEARNED SO FAR: After fixing the "raise ValueError("`datasets` must all have the same classes") ValueError: `datasets` must all have the
same classes" caused by the consequences of NOT using the benchmark_with_validation_stream with split_validation_class_balanced, which was fixed by
modifying the scenario_train_val_test = benchmark_with_validation_stream in Load_cifar100_v4.py to use split_validation_class_balanced, the NEXT
ERROR THAT I AM STILL CURRENTLY WORKING ON IN ORDER TO ACHIEVE THE FURTHER SPLIT OF EACH DataExperience can be seen by running Train_ResNet20_v4.py
OPTION_1 none-commented out versions and OPTION_2 ; they both lead to the error in question: "curr_data = data_by_class[cls].concat(curr_data)" of
class_incremental_benchmark source code due to KeyError: 63. This error seems to be caused by a "bug"/assumption made by the library code that the
number of classes is a ordered sequence of integers starting from 0. Hence, it sets num_classes variable to the value of the highest label (despite
labels not carrying numberic value, just being the class name/identificator) using max(targets) as part of line 
"num_classes = 1 + max(list(datasets_dict.values())[0].targets.uniques)" of class_incremental_benchmark source code. Hence, IN ORDER TO MAKE 
class_incremental_benchmark WORK WITH A SUBSET (i.e. a fraction of the original torchvision.datasets.cifar100 dataset, not necessarily an object of
class Subset. That is, a subset of ANY TYPE, whether DataExperience or ClassificationDataset or whatever other appropriate data type), THERE 
SEEMS TO BE NO OPTION BUT TO OVERRIDE THE LIBRARY SOURCE CODE FOR class_incremental_benchmark. THAT IS, THERE NO OPTION BUT TO OVERRIDE THE LIBRARY
SOURCE CODE FOR class_incremental_benchmark IF I INSIST ON USING class_incremental_benchmark ON SUBSETS OF THE ORIGINAL DATASET (in this case, meaning
the subsets that are the specialised_models_subsets for training each of the specialised models such that these can be divided further into 
incremental chunks for Class-IL). 

As such, what I plan on doing in Fifth_implementation_attempt before resorting to overriding the library code for class_incremental_benchmark is to 
pass class_order=sorted(avl_train.targets.uniques) into class_incremental_benchmark call for creating class_incremental_benchmark for each 
specialised models subset. This seems to provide a work-around to the consequences of the "bug"/assumption made by the library code that the number 
of classes is an ordered sequence of integers starting from 0. Whilst it still sets num_classes: int = 1 + max(list(datasets_dict.values())[0].targets.uniques), 
because "if class_order is None" IS NOW FALSE (i.e. is made false by passing class_order=sorted(avl_train.targets.uniques) into class_incremental_benchmark),
"class_order = torch.randperm(num_classes).tolist()" IS SKIPPED !!!!!! This seems to work so far in 4th_implementation_attempt so far, and I will 
formalise it and test it further in 5th_implementation_attempt.

###### POTENTIAL ISSUES WITH THIS WORK-AROUND:
num_classes is still set to the problematic num_classes: int = 1 + max(list(datasets_dict.values())[0].targets.uniques) and then used to set current 
classes per experience in curr_classess_per_exp: int = num_classes // num_experiences. Won't this cause problems with the code "# split classes by experience"
and "# create the streams using class_order to split the data" in class_incremental_benchmark? FIND OUT FOR 5th_implementation_attempt WHEN EXPERIMENTING 
WITH THIS WORK-AROUND !!!

###### IMPORTANT THINGS TO FIND OUT FOR 5th_implementation_attempt VIA EXPERIMENTING WITH THE IMPLEMENTATION OF THIS WORK-AROUND:
MOST IMPORTANT!!!!! Does providing avl_train, avl_val, and avl_test to class_incremental_benchmark mean that the class_incremental_benchmark
for each specialised_models_subset (e.g. outdoors_specialised_models_benchmark = class_incremental_benchmark({"train": avl_train, "test": avl_test, "valid": avl_val}) 
is already divided into train, test, and validation splits as I originally intended when deciding to do scenario_train_val_test = benchmark_with_validation_stream(scenario_train_test
where scenario_train_test is a class_incremental_benchmark created from train_dataset and test_dataset (i.e. the torchvision.datasets.cifar100 
dataset that had already been processed by class CoarseCifar100) and hence, ready for training after being extracted from outdoors_specialised_models_benchmark
using .stream objects .stream["train"], .stream["val"], and .stream["test"] ?????? FIND OUT FOR 5th_implementation_attempt WHEN EXPERIMENTING WITH THE WORK-AROUND !!!
If not, would benchmark_with_validation_stream need to be applied after the class_incremental_benchmark for each specialised_models_subset to the 
class_incremental_benchmark for each specialised_models_subset???? Is that even possible or would I have to find another function? Regardless of the
function, I would have to make sure that the samples are divided into train, test and val splits fairly as they otherwise would when using the 
benchmark_with_validation_stream on scenario_train_test, both to avoid library-related errors in the future and to ensure fair experientation (e.g.
that the model is not overfitting because it has already seen part of the validation or test set in one of the experiences as a result of unfair splitting).

THINGS TO KEEP IN MIND FOR 5th_implementation_attempt: 
1. streams / CLStreams / amy of benchmark_with_validation_stream's streams / .stream objects such as .stream["train"], .stream["val"], and .stream["test"]
contain a SEQUENCE / COLLECTION of DataExperience objects divided by class / specialised_models_superclasses in my case AND these DataExperience 
objects for which type() returns class DataExperience IN TURN CONTAIN:
   - an object of type ClassificationDataset returned by name_of_stream_variable.dataset (e.g.outdoors_specialised_models_train_subset.dataset)
   - an object of type DataAttribute returned by name_of_stream_variable.dataset.targets (e.g.outdoors_specialised_models_train_subset.dataset.targets)
   - an object of type DataAttribute returned by name_of_stream_variable.dataset.coarse_label (e.g.outdoors_specialised_models_train_subset.dataset.coarse_label)
And, as any ClassificationDataset object, it contains a set of samples, which can be reached via indexing (e.g. x, y, t = outdoors_specialised_models_train_subset.dataset[0]) 
would return one element of the dataset (that at index 0) where x is the tensor, and y and t are the coarse_label and fine_label of that element.
