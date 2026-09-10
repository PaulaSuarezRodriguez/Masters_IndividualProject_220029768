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

def scenario_uniform_aggregation_mine(model_paths):#, training_samples_per_model):
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

    '''# Load models into aggregator
    print("\nLoading model weights...")
    aggregator.load_model_weights_from_files(model_paths)#, training_samples_per_model)'''
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

    return aggregated_model

if __name__ == "__main__":
    aggregated_model = scenario_uniform_aggregation_mine(model_paths)#, training_samples_per_model)