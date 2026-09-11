import torch
from Thirteenth_implementation_attempt.FedAvg_code_v1.fedavg_aggregator import FedAvgAggregator
from Thirteenth_implementation_attempt.FedAvg_code_v2.fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier
from Thirteenth_implementation_attempt.ResNet20_v13 import ResNet20
from Thirteenth_implementation_attempt.ResNet20_DynamicModel_v13 import ResNet20_DynamicModel
'''for round in range(max_rounds) :
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
global_model = sum( (w * (size / total_data)) for w, size in zip(local_weights)'''

#for client in ["objects_and_vehicles"]:#["outdoors", "nature", "people", "objects_and_vehicles"]:#selected_clients:
 #   weights = torch.load(f"../Twelfth_implementation_attempt/final_specialised_models/final_{client}_model.pt", map_location="cpu")

weights_objects_and_vehicles = torch.load(f"../Twelfth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model.pt", map_location="cpu")
weights_people = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model.pt", map_location="cpu")
weights_nature = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_nature_model.pt", map_location="cpu")
weights_outdoors = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_outdoors_model.pt", map_location="cpu")


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
list_load=[weights_objects_and_vehicles, weights_people, weights_nature, weights_outdoors]
for key in list_load[0]:#state_dicts[0]:
    if "weight" in key:
        weights[key] = []

        for state_dict in list_load:#state_dicts:
            weights[key].append(state_dict[key])

print("print(weights.keys())",weights.keys())
print("print(weights)",weights)

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

'''for layer_name in weights:
    if'''


'''if weights.keys() == {"objects_and_vehicles"}:
only_weights=[]
only_weights.append(weights)
# dict_keys(['layer1.weight', 'layer1.bias', ...])

for key in weights_objects_and_vehicles.keys():
    if weights_objects_and_vehicles[key]=='''

model_paths=[f"../Twelfth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model.pt",
             f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model.pt",
             f"../Thirteenth_implementation_attempt/final_specialised_models/final_nature_model.pt",
             f"../Thirteenth_implementation_attempt/final_specialised_models/final_outdoors_model.pt"]

training_samples_per_model=[]


'''def scenario_uniform_aggregation_mine(model_paths):#, training_samples_per_model):
    """
    Scenario: Multiple models trained on different tasks with LwF.
    All models should contribute equally to the final aggregated model.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Uniform FedAvg Aggregation")
    print("=" * 70)
    print("""
    Use case: You have trained 3 models using Learning without Forgetting
    on different domains/tasks. Each model is equally important.

    We aggregate using: w_final = (1/3)*w_1 + (1/3)*w_2 + (1/3)*w_3
    """)

    # Initialize aggregator with uniform weighting
    aggregator = FedAvgAggregator(weighting_strategy='uniform')

    # Load models into aggregator
    print("\nLoading model weights...")
    aggregator.load_model_weights_from_files(model_paths)#, training_samples_per_model)

    # Get aggregation report
    report = aggregator.get_aggregation_report()
    print(f"✓ Loaded {report['num_models']} models")
    print(f"  Aggregation strategy: {report['weighting_strategy']}")
    print(f"  Each model weight: {report['model_weights'][0]:.4f}")

    # Aggregate and load into model
    print("\nPerforming FedAvg aggregation...")
    aggregated_model = ResNet20_DynamicModel(num_classes=43)
    aggregated_model = aggregator.aggregate_into_model(aggregated_model)
    print("✓ Weights aggregated successfully")

    # Save aggregated model
    save_path = 'aggregated_model_uniform.pth'
    aggregator.save_aggregated_weights(save_path)
    print(f"✓ Aggregated model saved to {save_path}")

    # Test inference
    print("\nTesting inference on aggregated model...")
    aggregated_model.eval()
    test_input = torch.randn(5, 28 * 28)
    with torch.no_grad():
        output = aggregated_model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ Inference successful")

    return aggregated_model'''

'''def scenario_uniform_aggregation_mine(model_paths):#, training_samples_per_model):
    """
    Scenario: Multiple models trained on different tasks with LwF.
    All models should contribute equally to the final aggregated model.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Uniform FedAvg Aggregation")
    print("=" * 70)
    print("""
    Use case: You have trained 3 models using Learning without Forgetting
    on different domains/tasks. Each model is equally important.

    We aggregate using: w_final = (1/3)*w_1 + (1/3)*w_2 + (1/3)*w_3
    """)

    # Initialize aggregator with uniform weighting
    aggregator = FedAvgAggregator(weighting_strategy='uniform')

    # Load models into aggregator
    #print("\nLoading model weights...")
    #aggregator.load_model_weights_from_files(model_paths)#, training_samples_per_model)
    aggregator.add_model_weights(weights)

    # Get aggregation report
    report = aggregator.get_aggregation_report()
    print(f"✓ Loaded {report['num_models']} models")
    print(f"  Aggregation strategy: {report['weighting_strategy']}")
    print(f"  Each model weight: {report['model_weights'][0]:.4f}")

    # Aggregate and load into model
    print("\nPerforming FedAvg aggregation...")
    aggregated_model = ResNet20_DynamicModel(num_classes=43)
    aggregated_model = aggregator.aggregate_into_model(aggregated_model)
    print("✓ Weights aggregated successfully")

    # Save aggregated model
    save_path = 'aggregated_model_uniform.pth'
    aggregator.save_aggregated_weights(save_path)
    print(f"✓ Aggregated model saved to {save_path}")

    # Test inference
    print("\nTesting inference on aggregated model...")
    aggregated_model.eval()
    test_input = torch.randn(5, 28 * 28)
    with torch.no_grad():
        output = aggregated_model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ Inference successful")

    return aggregated_model'''

'''
def scenario_1_explicit_expanding_layers():
    """
    Explicit approach: Manually specify which layers have expanding dimensions.

    Recommended when you know exactly which layers expand (e.g., classifier).
    """
    print("\n" + "=" * 80)
    print("SCENARIO 1: Explicit Expanding Layer Specification")
    print("=" * 80)
    print("""
    Use case: You trained models with expanding classifiers.
    - Task 1: 10 classes
    - Task 2: 10 + 5 = 15 classes
    - Task 3: 10 + 5 + 7 = 22 classes

    Only the classifier weights/bias expand.
    Backbone (feature_extractor) stays the same.

    We want to:
    1. Aggregate only the shared 10 classes
    2. Preserve task-specific new classes
    """)

    # Create and save example models with different number of classes
    print("\nCreating models with expanding classifiers...")

    model_configs = [
        ('model_task1_lwf.pth', 10, 1000),  # Task 1: 10 classes, 1000 samples
        ('model_task2_lwf.pth', 15, 1200),  # Task 2: 15 classes, 1200 samples
        ('model_task3_lwf.pth', 22, 900),  # Task 3: 22 classes, 900 samples
    ]

    model_paths = []
    dataset_sizes = []

    for path, num_classes, dataset_size in model_configs:
        model = ContinualLearningModel(num_classes=num_classes)
        torch.save(model.state_dict(), path)
        model_paths.append(path)
        dataset_sizes.append(dataset_size)
        print(f"  ✓ {path} ({num_classes} classes, {dataset_size} samples)")

    print("\nInitializing aggregator with explicit expanding layers...")
    aggregator = FedAvgAggregatorExpandingClassifier(
        weighting_strategy='data_driven',
        expanding_layers=['classifier.weight', 'classifier.bias']  # ← Specify which layers expand
    )

    print("Loading model weights...")
    aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

    print("\nAggregation details:")
    aggregator.print_aggregation_details()

    print("\nPerforming aggregation (shared portion only)...")
    aggregated_weights = aggregator.aggregate(expanding_layers=['classifier.weight', 'classifier.bias'])

    print(f"  ✓ Aggregated weights keys: {list(aggregated_weights.keys())}")
    print(f"  ✓ Aggregated classifier.weight shape: {aggregated_weights['classifier.weight'].shape}")
    print(f"    → This is the shared portion (10 classes)")

    print("\nSaving aggregated model...")
    aggregator.save_aggregated_weights('aggregated_explicit.pth')

    # Load into target model
    print("\nLoading aggregated weights into target model...")
    target_model = ContinualLearningModel(num_classes=22)  # Max classes from all tasks
    target_model = aggregator.aggregate_into_model(target_model,
                                                   expanding_layers=['classifier.weight', 'classifier.bias'])

    print("  ✓ Aggregated weights loaded")
    print(f"  ✓ Target model classifier shape: {target_model.classifier.weight.shape}")
    print("    → Shared 10 classes: aggregated from all models")
    print("    → Classes 11-22: from original task-specific models")

    # Cleanup
    for path, _, _ in model_configs:
        import os
        os.remove(path)

    return aggregator, target_model'''


from Thirteenth_implementation_attempt.FedAvg_code_v3.fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

# Create aggregator
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier']  # ← Skip classifier
)

# Load models
aggregator.load_model_weights_from_files(
    model_paths, #['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=[7200, 1200, 3600, 2400]#dataset_sizes=[1000, 1200, 900]
)

# See aggregation details
aggregator.print_report()
# Aggregate
aggregator.save_aggregated_weights('aggregated_backbone.pth')

# Load into model
print("\nLoading aggregated backbone into target model...")
target_model = ResNet20_DynamicModel(num_classes=43)  # For new classes
target_model = aggregator.aggregate_into_model(target_model)
print("  ✓ Aggregated backbone loaded")
print("  ✓ Classifier from target model preserved")

# Verify
print("\nVerifying aggregation...")
agg_state = torch.load('aggregated_backbone.pth')
print(f"  Aggregated parameters:")
for key in agg_state.keys():
    print(f"    - {key}: {agg_state[key].shape}")

# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, _T_co
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_definitions import T_co
from Thirteenth_implementation_attempt.Train_ResNet20_Outdoors_LwF_v13 import outdoors_specialised_models_test_stream_class_il
from Thirteenth_implementation_attempt.Train_ResNet20_Nature_LwF_v13 import nature_specialised_models_test_stream_class_il
from Thirteenth_implementation_attempt.Train_ResNet20_People_LwF_v13 import people_specialised_models_test_stream_class_il
from Twelfth_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_v12 import objects_and_vehicles_specialised_models_test_stream_class_il
#TODO: TEMPORARY!!!! Change to that of v13 once I have final Objects_and_Vehicles model!!!!!

def make_test_dataloader_for_evaluate_all_experiences_and_bwt_aggregation(
                                                              # experiences: Union[TExperienceType, Iterable[TExperienceType]],
                                                              test_adapted_dataset: Optional[Dataset],
                                                              # AvalancheDataset[T_co],
                                                              test_mb_size: int,
                                                              num_workers=0,
                                                              shuffle=False,
                                                              pin_memory=None,
                                                              persistent_workers=False,
                                                              drop_last=False,
                                                              **kwargs
                                                              ):
    assert test_adapted_dataset is not None
    # CODE FOR "other_dataloader_args = self._obtain_common_dataloader_parameters(..." DELETED, as it does not seem like I needed for the reasons explained above
    # CODE FOR "use task-balanced dataloader for task-aware benchmarks" DELETED since it was causing data type issues and I am not gonna be using task-aware/task-IL benchmark (AT LEAST NOT FOR NOW!)
    eval_dataloader = DataLoader(test_adapted_dataset,
                                 batch_size=test_mb_size,
                                 num_workers=num_workers,
                                 shuffle=shuffle,
                                 pin_memory=pin_memory,
                                 persistent_workers=persistent_workers,
                                 drop_last=drop_last, )
    return eval_dataloader

outdoors_specialised_models_test_stream_class_il_variable = outdoors_specialised_models_test_stream_class_il
nature_specialised_models_test_stream_class_il_variable = nature_specialised_models_test_stream_class_il
people_specialised_models_test_stream_class_il_variable = people_specialised_models_test_stream_class_il
objects_and_vehicles_specialised_models_test_stream_class_il_variable = objects_and_vehicles_specialised_models_test_stream_class_il

test_stream_dict ={}
test_stream_dict["outdoors_specialised_models_test_stream_class_il_variable"]= outdoors_specialised_models_test_stream_class_il_variable
test_stream_dict["nature_specialised_models_test_stream_class_il_variable"]= nature_specialised_models_test_stream_class_il_variable
test_stream_dict["people_specialised_models_test_stream_class_il_variable"]= people_specialised_models_test_stream_class_il_variable
test_stream_dict["objects_and_vehicles_specialised_models_test_stream_class_il_variable"]=objects_and_vehicles_specialised_models_test_stream_class_il_variable

from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

coarse_label_list = []
targets_list = []
for test_stream_name, test_stream in test_stream_dict.items():
    for test_exp in test_stream:
        if len(test_exp.dataset) != 0:
            print(
                f"{test_stream_name}.Exp{test_exp.current_experience}.dataset.coarse_label.uniques (coarse_labels of TEST)",
                test_exp.dataset.coarse_label.uniques
            )
            for test_exp_element in test_exp.dataset.coarse_label.uniques:
                if len(coarse_label_list) == 0:
                    coarse_label_list.append(test_exp_element)
                #if len(coarse_label_list) >= 2:
                if len(coarse_label_list) >= 1:
                    a = len(coarse_label_list) - 1
                    b = len(coarse_label_list) - 2
                    if test_exp_element != coarse_label_list[a] and test_exp_element != coarse_label_list[b]:
                        coarse_label_list.append(test_exp_element)#coarse_label_list.append(test_exp.dataset.coarse_label.uniques)

            print(
                f"{test_stream_name}.Exp{test_exp.current_experience}.dataset.targets.uniques (targets of TEST)",
                test_exp.dataset.targets.uniques
            )
            for test_exp_element in test_exp.dataset.targets.uniques:
                targets_list.append(test_exp_element)#targets_list.append(test_exp.dataset.targets.uniques)

    print(f"\ncoarse_label_list for {test_stream_name}", coarse_label_list)
    print(f"targets_list for {test_stream_name}", targets_list)

#torch.utils.data.dataset.Dataset(coarse_label_list)

'''dataset_coarse_label = ListDataset(coarse_label_list)
dataset_targets = ListDataset(targets_list)'''

coarse_label_data_attribute=make_avalanche_dataset(coarse_label_list)
targets_data_attribute=make_avalanche_dataset(targets_list)

#TODO: Uncomment this block later

make_avalanche_dataset(outdoors_specialised_models_test_stream_class_il[0])
make_avalanche_dataset(outdoors_specialised_models_test_stream_class_il[1])
make_avalanche_dataset(outdoors_specialised_models_test_stream_class_il[2])
make_avalanche_dataset(outdoors_specialised_models_test_stream_class_il[3])
make_avalanche_dataset(outdoors_specialised_models_test_stream_class_il[4])

test_set_global_model = outdoors_specialised_models_test_stream_class_il[0].concat(outdoors_specialised_models_test_stream_class_il[1])
print("test_set_global_model VERSION 1", test_set_global_model)

avl_test_outdoors = make_avalanche_dataset(outdoors_specialised_models_test_stream_class_il)
avl_test_nature = make_avalanche_dataset(nature_specialised_models_test_stream_class_il)
avl_test_people = make_avalanche_dataset(people_specialised_models_test_stream_class_il)
avl_test_objects_and_vehicles = make_avalanche_dataset(objects_and_vehicles_specialised_models_test_stream_class_il)

test_set_global_model = avl_test_outdoors.concat(avl_test_nature)
test_set_global_model = test_set_global_model.concat(avl_test_people)
test_set_global_model = test_set_global_model.concat(avl_test_objects_and_vehicles)

print("test_set_global_model", test_set_global_model)
'''
test_set_global_model = ClassificationDataset(
    [test_set_global_model],
    data_attributes=[DataAttribute(coarse_label_data_attribute, "coarse_label"),
                     DataAttribute(targets_data_attribute, "targets")],)'''


'''avl_test = ClassificationDataset(
    [avl_test],  # [nature_specialised_models_test_subset.dataset],
    data_attributes=[DataAttribute(nature_specialised_models_test_subset.dataset.coarse_label, "coarse_label"),
                     DataAttribute(nature_specialised_models_test_subset.dataset.targets, "targets")],
    # "fine_label")],
)'''

#TODO: Uncomment later
'''target_model.eval()

test_adapted_dataset = test_set_global_model.eval() #FAILED --> test_adapted_dataset = test_set_global_model.dataset.eval()
test_dataloader = make_test_dataloader_for_evaluate_all_experiences_and_bwt_aggregation(test_adapted_dataset,test_mb_size=64)
#TRY NEXT IF ABOVE FAILS --> test_adapted_dataset = test_set_global_model.eval()

with torch.no_grad():
    accuracy = 0.0
    for batch_data, batch_labels in test_dataloader:
        outputs = target_model(batch_data)
        accuracy += (outputs.argmax(1) == batch_labels).float().mean().item()

print(f"Fine-tuned model accuracy: {accuracy:.2%}")'''

'''# Compare with individual models
print("\\nComparison with individual models:")
for i, path in enumerate(model_paths):
    model = ContinualLearningModel(num_classes=10)
    model.load_state_dict(torch.load(path))
    model.eval()

    with torch.no_grad():
        acc = 0.0
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            acc += (outputs.argmax(1) == batch_labels).float().mean().item()

    print(f"  Model {i + 1} accuracy: {acc:.2%}")'''


'''if __name__ == "__main__":
    aggregated_model = scenario_uniform_aggregation_mine(model_paths)#, training_samples_per_model)'''