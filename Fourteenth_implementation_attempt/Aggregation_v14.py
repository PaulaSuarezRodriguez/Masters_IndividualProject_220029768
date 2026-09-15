"""for round in range(max_rounds) :
    local_weights = []
    client_sizes = []

# 1. Broadcast global model to a random subset of clients
#selected_clients = sample_clients(all_clients, fraction=0.1)
for client in selected_clients:
    # 2. Local training (Standard ML Loop)
    weights, data_size = client.train(globalmodel)
    local_weights. append(weights)
    client_sizes. append data_size)
# 3. Aggregate (Federated Averaging)
total_data - sum(client_sizes)
global_model = sum( (w * (size / total_data)) for w, size in zip(local_weights)"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Fourteenth_implementation_attempt.ResNet20_v14 import ResNet20
from Fourteenth_implementation_attempt.FedAvg_code_v3_Fourteenth_implementation_attempt.fedavg_aggregator_backbone_only_Fourteenth import FedAvgAggregatorBackboneOnly
from Fourteenth_implementation_attempt.ResNet20_DynamicModel_v14 import ResNet20_DynamicModel
############ Imports for Evaluating: ##############################################################################################################################################
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, _T_co
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_definitions import T_co
from Fourteenth_implementation_attempt.Train_ResNet20_Outdoors_LwF_v14 import outdoors_specialised_models_test_stream_class_il
from Fourteenth_implementation_attempt.Train_ResNet20_Nature_LwF_v14 import nature_specialised_models_test_stream_class_il
from Fourteenth_implementation_attempt.Train_ResNet20_People_LwF_v14 import people_specialised_models_test_stream_class_il
from Fourteenth_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_v14 import objects_and_vehicles_specialised_models_test_stream_class_il
#TODO: TEMPORARY!!!! Change to that of v14 once I have final Objects_and_Vehicles model!!!!!
from torch.utils.data import ConcatDataset, DataLoader
####################################################################################################################################################################################

VERBOSE = False

'''original_to_aggregated_dict = {0: 12, 1: 17, 2: 23, 3: 33, 4: 37, 5: 49, 6: 60, 7: 68, 8: 71, 9: 76, #OUTDOORS
                                10: 0, 11: 47, 12: 51, 13: 52, 14: 53, 15: 54, 16: 56, 17: 57, 18: 59, 19: 62, 20: 70, 21: 82, 22: 83, 23: 92, 24: 96, #NATURE
                                25: 2, 26: 11, 27: 35, 28: 46, 29: 98, #PEOPLE
                                30: 5, 31: 8, 32: 9, 33: 10, 34: 13, 35: 16, 36: 20, 37: 22, 38: 25, 39: 28, 40: 39, 41: 40, 42: 41, 43: 48, 44: 58, 45: 61, 46: 69, 47: 81, 48: 84, 49: 85, 50: 86, 51: 87, 52: 89, 53: 90, 54: 94
                          }'''
original_to_aggregated_dict = {17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5, #OUTDOORS
                               51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14, #NATURE
                               2: 15, 11: 16, 35: 17, #PEOPLE
                               16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27, 39: 28, 28: 29, 25: 30, 22: 31, 20: 32, 94: 33, #OBJECTS_AND_VEHICLES
                               90: 34, 89: 35, 87: 36, 86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42} #OBJECTS_AND_VEHICLES

def map_original_to_aggregated(targets_tensor, original_to_aggregated=original_to_aggregated_dict):
    ###print("print(targets_tensor)",targets_tensor)
    aggregated_targets_tensor = torch.tensor(
        [
            original_to_aggregated[int(element.item())] for element in targets_tensor #if element in original_to_aggregated.keys()
        ],
        device=targets_tensor.device,
        dtype=torch.long
    )
    ###print("print(aggregated_targets_tensor)\n",aggregated_targets_tensor)

    return aggregated_targets_tensor

def map_original_to_aggregated_predictions(targets_tensor, original_to_aggregated=original_to_aggregated_dict):
    aggregated_predictions_targets_tensor = torch.tensor(
        [
            original_to_aggregated[int(element.item())] for element in targets_tensor #if element not in original_to_aggregated.keys()
        ],
        device=targets_tensor.device,
        dtype=torch.long
    )
    ###print("print(aggregated_predictions_targets_tensor)\n",aggregated_predictions_targets_tensor)

    return aggregated_predictions_targets_tensor

def aggregate_into_model_from_checkpoint(checkpoint,
                         model: nn.Module,) -> nn.Module:
    """
    Load aggregated backbone weights into a model.
    Args:
        model (nn.Module): Target model to load weights
    Returns:
        nn.Module: Model with aggregated backbone loaded
    """
    # Get target model's state dict
    target_state = model.state_dict()

    # Create new state dict with aggregated backbone + original classifier
    new_state = {}

    for key, value in target_state.items():
        # Load aggregated backbone weights
        if key in checkpoint:
            new_state[key] = checkpoint[key]
        else:
            new_state[key] = value

    model.load_state_dict(new_state)
    target_state_after = model.state_dict()
    return model

weights_outdoors = torch.load(f"../Fourteenth_implementation_attempt/final_specialised_models/final_outdoors_model.pt", map_location="cpu")
weights_nature = torch.load(f"../Fourteenth_implementation_attempt/final_specialised_models/final_nature_model.pt", map_location="cpu")
weights_people = torch.load(f"../Fourteenth_implementation_attempt/final_specialised_models/final_people_model.pt", map_location="cpu")
weights_objects_and_vehicles = torch.load(f"../Fourteenth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model.pt", map_location="cpu")

print(f"print(type(weights_objects_and_vehicles)):{type(weights_objects_and_vehicles)}\n"
      f"print(weights_objects_and_vehicles.keys()):{print(weights_objects_and_vehicles.keys())}")
print(f"print(type(weights_people)):{type(weights_people)}\n"
      f"print(weights_people.keys()):{print(weights_people.keys())}")

#print(weights.keys())
#odict_keys(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'linear.weight', 'linear.bias'])
odict_keys=['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
            'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var',
            'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias',
            'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight',
            'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked',
            'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var',
            'layer1.1.bn2.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean',
            'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias',
            'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight',
            'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight',
            'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked',
            'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var',
            'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean',
            'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias',
            'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight',
            'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer3.0.conv1.weight',
            'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked',
            'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var',
            'layer3.0.bn2.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean',
            'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias',
            'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight',
            'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight',
            'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked',
            'linear.weight', 'linear.bias']

#TODO: OPTION 1
#list_load={"weights_objects_and_vehicles":weights_objects_and_vehicles, "weights_people":weights_people}
#list_load=[weights_objects_and_vehicles, weights_people]
#for specialised_state_dict in list_load:
'''def create_weight_list(specialised_state_dict):
    weights = {}
    for ok, tensor in specialised_state_dict.items():
        if "weight" in ok:
            layer_name = ok.rsplit(".", 1)[0]

            if layer_name not in weights:
                weights[layer_name] = []

            weights[layer_name].append(tensor)
    return weights

weights_objects_and_vehicles = create_weight_list(weights_objects_and_vehicles)
weights_people = create_weight_list(weights_people)
weights_nature = create_weight_list(weights_nature)
weights_outdoors = create_weight_list(weights_outdoors)'''

#TODO: OPTION 2
weights = {}
list_load=[weights_outdoors, weights_nature, weights_people, weights_objects_and_vehicles]
for key in list_load[0]:#state_dicts[0]:
    if "weight" in key:
        weights[key] = []

        for state_dict in list_load:#state_dicts:
            weights[key].append(state_dict[key])

if VERBOSE:
    print("print(weights.keys())",weights.keys())
    print("print(weights)",weights)

if VERBOSE:
    for i in range(len(list_load)):
        sizes={}
        for key in weights.keys():
            print("key:",key)
            print("i:",i)
            print(f"weights[key][i] is therefore --> weights[{key}][{i}]")
            #print(f"print(weights[key][i]). That is, print(weights[{key}][{i}])\n",weights[key][i],"\n")
            print(f"print(weights[{key}][{i}])\n", weights[key][i],"\n")
            sizes[f"weights[{key}][{i}]"]=weights[key][i].size()
            print(f"\nprint(sizes):\n{sizes}")

model_paths=[f"../Fourteenth_implementation_attempt/final_specialised_models/final_outdoors_model.pt",
             f"../Fourteenth_implementation_attempt/final_specialised_models/final_nature_model.pt",
             f"../Fourteenth_implementation_attempt/final_specialised_models/final_people_model.pt",
             f"../Fourteenth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model.pt"]

#training_samples_per_model=[3600, 1200, 10000]
training_samples_per_model=[2400, 3600, 1200, 10000]

# Create aggregator
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',#'uniform',#'data_driven',
    exclude_layers=['linear']#exclude_layers=['linear', 'num_batches_tracked']#exclude_layers=['classifier']  # ← Skip classifier
)

# Load models
aggregator.load_model_weights_from_files(
    model_paths, #['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=training_samples_per_model
)


# Load into model
print("\nLoading aggregated backbone into target model...")
target_model = ResNet20_DynamicModel(num_classes=100)#TODO:Temporary ###########(num_classes=43)  # For new classes

target_model = aggregator.aggregate_into_model(target_model, keep_classifier=True)#keep_classifier NEVER USED IN FUNCTION CODE, so, set to true as reminder that classifier from target_model IS PRESENT. target_model = aggregator.aggregate_into_model(target_model, keep_classifier=False) #target_model = aggregator.aggregate_into_model(target_model, keep_classifier=True)#
print("  ✓ Aggregated backbone loaded")
print("  ✓ Classifier from target model preserved")
###print("SAME AS THOSE FOR global_model PRINTED BELOW?????? THEY SHOULD BE!!!!! ANSWER:THEY ARE THE SAME! :")#TODO:THEY ARE THE SAME!
###print("print(target_model.linear.weight)",target_model.linear.weight) #TODO: IMPORTANT! PROVES keep_classifier ALWAYS TRUE regardless of whether True or False is passed as argument!!!!
###print("print(target_model.state_dict())",target_model.state_dict()) #TODO: IMPORTANT! PROVES keep_classifier ALWAYS TRUE regardless of whether True or False is passed as argument!!!!

torch.save(target_model.state_dict(), 'aggregated_backbone_state_dict.pth')
# See aggregation details
aggregator.print_report()
# Aggregate
aggregator.save_aggregated_weights('aggregated_backbone.pth')

# Verify
print("\nVerifying aggregation...")
agg_state = torch.load('aggregated_backbone.pth')
print(f"  Aggregated parameters:")
for key in agg_state.keys():
    print(f"    - {key}: {agg_state[key].shape}")






import torch
import torch.nn as nn
import pickle


def remap_classifier_to_cifar100(model, new_to_original, num_subset_classes, num_classes=100):
    """
    Remap a specialized model's classifier from local space to CIFAR-100 space.

    new_to_original = {0→12, 1→17, 2→23, 3→33, 4→37, 5→49} for outdoors model

    Maps model's local classifier [6, feature_dim] to global [100, feature_dim]
    placing each row at its correct CIFAR-100 position.
    """
    model.eval()
    classifier = model.linear
    feature_dim = classifier.weight.shape[1]

    with torch.no_grad():
        # Initialize 100×feature_dim with zeros (untrained classes)
        remapped_weights = torch.zeros(num_classes, feature_dim)
        remapped_bias = torch.zeros(num_classes)

        local_idx_count = 0
        # Place each trained class at its CIFAR-100 position
        for local_idx, cifar_class in new_to_original.items():
            print("new_to_original", new_to_original)
            print("print(local_idx, cifar_class)",local_idx, cifar_class)
            local_idx_count +=1
            print("print(local_idx_count)",local_idx_count)
            print("print(num_subset_classes)",num_subset_classes)
            if local_idx_count >= num_subset_classes:
                print("print(num_subset_classes) THAT CAUSED BREAK",num_subset_classes)
                break
            # local_idx: 0-5 for outdoors (position in model's classifier)
            # cifar_class: 12, 17, 23, ... (actual CIFAR class)
            remapped_weights[cifar_class] = classifier.weight[local_idx]
            remapped_bias[cifar_class] = classifier.bias[local_idx]

        if VERBOSE:
            print("remapped_weights", remapped_weights)
            print("remapped_bias", remapped_bias)

    return remapped_weights, remapped_bias


'''#with torch.no_grad():
global_weights = torch.zeros(100, 64)# Feature dimension of first model
global_bias = torch.zeros(100)
# Load model
model = ResNet20_DynamicModel(num_classes=6)#model = ResNet20_DynamicModel(num_classes=num_subset_classes) #model = ResNet20(num_classes=num_subset_classes) #model = ResNet20_DynamicModel(num_classes=0) # ORIGINALLY: model = ResNet20(num_classes=None)
model.eval()
with torch.no_grad():
    model.load_state_dict(torch.load(f'./final_specialised_models/final_outdoors_model.pt'))
    print(f"LOADED MODEL WEIGHTS (final_outdoors_model):\n{model.state_dict()}")
# Remap this model's classifier
w, b = remap_classifier_to_cifar100(model, {0: 12, 1: 17, 2: 23, 3: 33, 4: 37, 5: 49, 6: 60, 7: 68, 8: 71, 9: 76}, 6, num_classes=100)#w, b = remap_classifier_to_cifar100(model, mapping, num_subset_classes, num_classes=100)
print(f"w to see what it looks like after additions (final_outdoors_model):\n", w)
print(f"b to see what it looks like after additions (final_outdoors_model):\n", b)
# Add to global classifier (no averaging - classes don't overlap!)
global_weights += w
global_bias += b
print(f"global_weights += w to see what it looks like after additions (final_outdoors_model):\n", global_weights)
print(f"global_bias += b to see what it looks like after additions (final_outdoors_model):\n", global_bias)

print(f"✓ Aggregated final_outdoors_model")'''


# Load all trained models
models = {}
mappings = {}
num_subset_classes = {}
feature_dim = {}
for model_name in ['outdoors', 'nature', 'people', 'objects_and_vehicles']:
    # Load mapping
    with open(f'./final_specialised_models/{model_name}_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    mappings[model_name] = metadata['new_to_original']
    num_subset_classes[model_name] = metadata['num_subset_classes']
    feature_dim[model_name] = metadata['feature_dim']
    print(f"print(feature_dim) for {model_name}:\n{feature_dim}")


    # Load model
    model = ResNet20_DynamicModel(num_classes=num_subset_classes[model_name])#model = ResNet20_DynamicModel(num_classes=num_subset_classes) #model = ResNet20(num_classes=num_subset_classes) #model = ResNet20_DynamicModel(num_classes=0) # ORIGINALLY: model = ResNet20(num_classes=None)
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load(f'./final_specialised_models/final_{model_name}_model.pt'))
        print(f"LOADED MODEL WEIGHTS ({model_name}):\n{model.state_dict()}")
    models[model_name] = model

    #feature_dim = model.linear.weight.shape[1] # Feature dimension of first model

# Remap and aggregate
with torch.no_grad():
    global_weights = torch.zeros(100, feature_dim['outdoors'])# Feature dimension of first model
    print(f"print(feature_dim) passed to global_weights = torch.zeros(100, feature_dim):\n{feature_dim}")
    global_bias = torch.zeros(100)

for model_name in ['outdoors', 'nature', 'people', 'objects_and_vehicles']:
    model = models[model_name]
    print("model = models[model_name]", model) if VERBOSE else None
    mapping = mappings[model_name]
    ####num_subset_classes = num_subset_classes[model_name]

    # Remap this model's classifier
    w, b = remap_classifier_to_cifar100(model, mapping, num_subset_classes[model_name], num_classes=100)#w, b = remap_classifier_to_cifar100(model, mapping, num_subset_classes, num_classes=100)
    print(f"w to see what it looks like after additions ({model_name}):\n", w)
    print(f"b to see what it looks like after additions ({model_name}):\n", b)
    # Add to global classifier (no averaging - classes don't overlap!)
    global_weights += w
    global_bias += b
    print(f"global_weights += w to see what it looks like after additions ({model_name}):\n", global_weights)
    print(f"global_bias += b to see what it looks like after additions ({model_name}):\n", global_bias)

    print(f"✓ Aggregated {model_name}")


# Create global model
#global_model = ResNet20(num_classes=100) #global_model = ResNet20_DynamicModel(num_classes=100) #ORIGINALLY: global_model = ResNet20(num_classes=100)
#TODO: CHANGE: Since feature extractor layers were aggregated first, the global_model already exists. It is the target_model!
global_model=target_model
###print("SAME AS THOSE FOR target_model PRINTED ABOVE?????? THEY SHOULD BE!!!!! ANSWER:THEY ARE THE SAME! :")#TODO:THEY ARE THE SAME!
###print("print(global_model.linear.weights)",global_model.linear.weight)
###print("print(global_model.state_dict())",global_model.state_dict())
###print("global_weights", global_weights)
#TODO: If this fails, try setting keep_classifier=False instead of true and build a new classifier layer that is instantiated with the two lines below, as if current code does not work,
# it is probably due to the overwriting that is supposed to occur if linear layer is "kept" (i.e. if keep_classifier=True) DID NOT WORK AS INTENDED and not keeping the linear layer,
# creating a new one, and instantiating the new one with the two lines of code below, should be attempted.
# Code for creating new linear layer: target_model.linear = torch.nn.Linear(64, 100)
global_model.linear = torch.nn.Linear(64, 100)
with torch.no_grad():
    global_model.linear.weight = nn.Parameter(global_weights)
    global_model.linear.bias = nn.Parameter(global_bias)

print(f"✓ Created global model")
print(f"  - Classifier shape: {global_model.linear.weight.shape}")
print(f"  - Output: 100 classes\n")

# Save
torch.save(global_model.state_dict(), './final_specialised_models/global_model.pt')































# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================
# NEW NEW NEW
test_stream_dict ={}
test_stream_dict["outdoors_specialised_models_test_stream_class_il_variable"]= outdoors_specialised_models_test_stream_class_il
test_stream_dict["nature_specialised_models_test_stream_class_il_variable"]= nature_specialised_models_test_stream_class_il
test_stream_dict["people_specialised_models_test_stream_class_il_variable"]= people_specialised_models_test_stream_class_il
test_stream_dict["objects_and_vehicles_specialised_models_test_stream_class_il_variable"]=objects_and_vehicles_specialised_models_test_stream_class_il

for test_stream_name, test_stream in test_stream_dict.items():
    for test_exp in test_stream:
        if len(test_exp.dataset) != 0:
            print(
                f"{test_stream_name}.Exp{test_exp.current_experience}.dataset.coarse_label.uniques (coarse_labels of TEST)",
                test_exp.dataset.coarse_label.uniques
            )

            print(
                f"{test_stream_name}.Exp{test_exp.current_experience}.dataset.targets.uniques (targets of TEST)",
                test_exp.dataset.targets.uniques
            )

print("type", type(outdoors_specialised_models_test_stream_class_il[0].dataset))

combined_dataset_outdoors = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
    outdoors_specialised_models_test_stream_class_il[1].dataset,
    outdoors_specialised_models_test_stream_class_il[2].dataset,
    #outdoors_specialised_models_test_stream_class_il[3].dataset,
    #outdoors_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_nature = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
    nature_specialised_models_test_stream_class_il[1].dataset,
    nature_specialised_models_test_stream_class_il[2].dataset,
    #nature_specialised_models_test_stream_class_il[3].dataset,
    #nature_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_people = ConcatDataset([
    people_specialised_models_test_stream_class_il[0].dataset,
    people_specialised_models_test_stream_class_il[1].dataset,
    people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset = ConcatDataset([
    combined_dataset_outdoors,
    combined_dataset_nature,
    combined_dataset_people, combined_dataset_objects_and_vehicles
])

test_loader = DataLoader(
    combined_dataset,
    batch_size=128,#32,
    shuffle=False
)
print("print test_loader", test_loader)
print("print test_loader.dataset", test_loader.dataset)


from collections import Counter

all_labels = []

for i in range(len(combined_dataset)):
    sample = combined_dataset[i]

    # Adjust depending on your dataset output
    label = sample[1]

    all_labels.append(int(label))

print(Counter(all_labels))
print("Unique labels:", sorted(set(all_labels)))
print("Number of unique labels:", len(set(all_labels)))


for batch in test_loader:
    print(type(batch))
    print(len(batch))
    for i, item in enumerate(batch):
        print(i, type(item), getattr(item, "shape", None))
        print(type(batch))
    break


#print("test_loader.batch_sampler",test_loader.batch_sampler)

def unpack_minibatch(mbatch):
    """Check if the current mini-batch has 3 components."""
    mbatch = mbatch
    assert mbatch is not None

    if isinstance(mbatch, tuple):
        mbatch = list(mbatch)
        mbatch = mbatch

    for i in range(len(mbatch)):
        if isinstance(mbatch[i], dict):
            # Move all tensor values in the dictionary to the specified device
            mbatch[i] = {
                key: value.to(non_blocking=True)
                for key, value in mbatch[i].items()
            }
        elif isinstance(mbatch[i], torch.Tensor):
            # Directly move the tensor to the specified device
            mbatch[i] = mbatch[i].to(non_blocking=True)
        else:
            # Optionally handle other cases (e.g., raise an error if unsupported types are encountered)
            raise TypeError(f"Unsupported type in batch: {type(mbatch[i])}")


def evaluate_aggregated_model_FedAvg_code_v3(target_model: nn.Module):
    correct = 0
    total = 0

    target_model.eval() #TODO: IN TESTING! If it leads to problems, comment-out again
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in test_loader:
            #print(type(batch_data))
            #print(type(batch_labels))
            #print(type(batch_task_ids))

            ######print("\nprint(batch_labels)\n",batch_labels)
            ##print("print(batch_task_ids)",batch_task_ids)

            #original_to_aggregated = mapping_original_to_aggregated(batch_labels)
            #print("original_to_aggregated:", original_to_aggregated)

            outputs = target_model(batch_data)
            targets_all_experiences = batch_labels

            #outputs = map_original_to_aggregated(outputs)
            #batch_labels = map_original_to_aggregated(batch_labels)
            #####print("\nprint(batch_labels) AFFTER map_original_to_aggregated(batch_labels)\n", batch_labels)

            #predictions = outputs
            #####print("predictions before argmax:",predictions)
            #####print("predictions before argmax SHAPE:", predictions.shape)
            #####print("predictions before argmax LENGTH:", len(predictions))
            #####print("batch_labels before argmax SHAPE:", batch_labels.shape)
            #####print("batch_labels before argmax LENGTH:", len(batch_labels))

            predictions = outputs.argmax(dim=1)

            #####print("\npredictions", predictions)
            #predictions = map_original_to_aggregated(predictions)
            #batch_labels = map_original_to_aggregated(batch_labels)#batch_labels = map_targets_to_aggregated(original_to_aggregated, batch_labels)


            correct += (predictions == batch_labels).sum().item() #correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    #accuracy = correct / total
    accuracy = (correct / total ) if total > 0 else 0.0
    print(f"Fine-tuned model accuracy: {accuracy:.2%}")#print(f"Fine-tuned model accuracy: {accuracy:.2%}")


from torch.nn import Module, CrossEntropyLoss
def evaluate_aggregated_model(model: nn.Module):
    total_loss_all_experiences = 0.0
    total_correct_all_experiences = 0
    total_samples_all_experiences = 0

    model.eval() #TODO: IN TESTING! If it leads to problems, comment-out again
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in test_loader:
            unpack_minibatch(batch_data)

            #original_to_aggregated = mapping_original_to_aggregated(batch_labels)
            #print("original_to_aggregated:", original_to_aggregated)

            logits_all_experiences = model(batch_data)

            targets_all_experiences = batch_labels

            #####targets_all_experiences = map_original_to_aggregated(batch_labels) #targets_all_experiences = map_targets_to_aggregated(original_to_aggregated, batch_labels) #TODO: Old location, test later if it works or not
            #####logits_all_experiences = map_original_to_aggregated(logits_all_experiences) #TODO: COMPLETELY NEW, BEING TESTED IF NEEDED/CORRECT

            #loss_all_experiences = CrossEntropyLoss(logits_all_experiences, targets_all_experiences, reduction='mean')
            loss_all_experiences = F.cross_entropy(
                logits_all_experiences,
                targets_all_experiences,
                reduction='mean'
            )

            total_loss_all_experiences += loss_all_experiences.item() * targets_all_experiences.size(0)
            predictions_all_experiences = torch.argmax(logits_all_experiences, dim=1)

            #predictions_all_experiences = map_original_to_aggregated(predictions_all_experiences)
            #targets_all_experiences = map_original_to_aggregated(batch_labels)

            total_correct_all_experiences += (predictions_all_experiences == targets_all_experiences).sum().item()
            total_samples_all_experiences += targets_all_experiences.size(0)

    avg_loss_all_experiences = total_loss_all_experiences / total_samples_all_experiences if total_samples_all_experiences > 0 else 0.0
    accuracy_all_experiences = (total_correct_all_experiences / total_samples_all_experiences * 100) if total_samples_all_experiences > 0 else 0.0

    print(
        f"Loss = {avg_loss_all_experiences:.6f} | Accuracy = {accuracy_all_experiences:.2f}%"
    )

def evaluate_aggregated_model_v3(model: nn.Module):
    total_loss_all_experiences = 0.0
    total_correct_all_experiences = 0
    total_samples_all_experiences = 0

    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in test_loader:

            #original_to_aggregated = mapping_original_to_aggregated(batch_labels)
            #print("original_to_aggregated:", original_to_aggregated)

            logits_all_experiences = model(batch_data)
            targets_all_experiences = batch_labels

            #print("print(type(logits_all_experiences))", type(logits_all_experiences))
            #print("print(type(batch_data))", type(batch_data))
            #print("print(logits_all_experiences.shape)", logits_all_experiences.shape)
            #print("print(batch_data.shape)", batch_data.shape)
            #print("print(batch_labels.shape)", logits_all_experiences.shape)
            #print("print(batch_labels.shape)", batch_data.shape)

            #print("batch_labels", batch_labels)
            #print("logits_all_experiences", logits_all_experiences)

            ####targets_all_experiences = map_original_to_aggregated(batch_labels) #TODO: Location being tested.  #targets_all_experiences = map_targets_to_aggregated(original_to_aggregated, batch_labels)
            ####logits_all_experiences = map_original_to_aggregated_predictions(logits_all_experiences)  # TODO: COMPLETELY NEW, BEING TESTED IF NEEDED/CORRECT

            loss_all_experiences = criterion(
                logits_all_experiences,
                targets_all_experiences
            )

            batch_size = targets_all_experiences.size(0)

            total_loss_all_experiences += (
                loss_all_experiences.item() * batch_size
            )

            predictions_all_experiences = torch.argmax(
                logits_all_experiences,
                dim=1
            )

            #predictions_all_experiences = map_original_to_aggregated(predictions_all_experiences)
            #targets_all_experiences = map_original_to_aggregated(batch_labels)

            total_correct_all_experiences += (
                predictions_all_experiences == targets_all_experiences
            ).sum().item()

            total_samples_all_experiences += batch_size

    if total_samples_all_experiences > 0:
        avg_loss_all_experiences = (
            total_loss_all_experiences / total_samples_all_experiences
        )

        accuracy_all_experiences = (
            total_correct_all_experiences
            / total_samples_all_experiences
            * 100
        )
    else:
        avg_loss_all_experiences = 0.0
        accuracy_all_experiences = 0.0

    print(
        f"Loss = {avg_loss_all_experiences:.6f} | "
        f"Accuracy = {accuracy_all_experiences:.2f}%"
    )

    return avg_loss_all_experiences, accuracy_all_experiences


# ============================================================================
# TEST BEHAVIOUR
# ============================================================================
# Test on CIFAR-100 with original labels (0-99)
'''correct=0
for image, label, task_id in test_loader:
    prediction = global_model(image)  # Outputs logits for classes 0-99
    predicted_class = torch.argmax(prediction)

    # Now predicted_class and label are in the same space!
    # ✓ This comparison makes sense
    if predicted_class == label:
        correct += 1'''



if __name__ == "__main__":
    evaluate_aggregated_model_FedAvg_code_v3(target_model)
    evaluate_aggregated_model(target_model)
    evaluate_aggregated_model_v3(target_model)