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