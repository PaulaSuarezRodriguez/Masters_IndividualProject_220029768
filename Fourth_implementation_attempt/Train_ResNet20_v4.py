from Fourth_implementation_attempt.Config_v4 import set_seed
set_seed()
from Fourth_implementation_attempt.ResNet20_v4 import ResNet20
from Fourth_implementation_attempt.Load_cifar100_v4 import (outdoors_specialised_models_train_subset, nature_specialised_models_train_subset,
                                                            people_specialised_models_train_subset, animals_specialised_models_train_subset, objects_and_vehicles_specialised_models_train_subset)
from Fourth_implementation_attempt.Load_cifar100_v4 import (outdoors_specialised_models_val_subset, nature_specialised_models_val_subset,
                                                            people_specialised_models_val_subset, animals_specialised_models_val_subset, objects_and_vehicles_specialised_models_val_subset)
from Fourth_implementation_attempt.Load_cifar100_v4 import (outdoors_specialised_models_test_subset, nature_specialised_models_test_subset,
                                                            people_specialised_models_test_subset, animals_specialised_models_test_subset, objects_and_vehicles_specialised_models_test_subset)


#resnet20_global_model = ResNet20()

# NAIVE / BASELINE IMPLEMENTATIION OF MY STRATEGY --> train each specialised model in a Class-IL manner without attempting to mitigate CF.
# Additionally, pre-train the global model on the EXPERIENCE 1 TRAIN SUBSETS and attempt to naively update it (using a hard-coded version of FedAvg
# and no techniques to mitigate ST-CF):

