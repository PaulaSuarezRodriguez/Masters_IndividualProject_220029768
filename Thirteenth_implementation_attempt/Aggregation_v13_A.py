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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Thirteenth_implementation_attempt.FedAvg_code_v1.fedavg_aggregator import FedAvgAggregator
from Thirteenth_implementation_attempt.FedAvg_code_v2.fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier
from Thirteenth_implementation_attempt.ResNet20_v13 import ResNet20
from Thirteenth_implementation_attempt.FedAvg_code_v3.fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly
from Thirteenth_implementation_attempt.ResNet20_DynamicModel_v13 import ResNet20_DynamicModel

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

model_paths=[#f"../Thirteenth_implementation_attempt/final_specialised_models/final_outdoors_model.pt",
             f"../Thirteenth_implementation_attempt/final_specialised_models/final_nature_model.pt",
             f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model.pt",
             f"../Twelfth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model.pt"]

training_samples_per_model=[3600, 1200, 10000]#training_samples_per_model=[2400, 3600, 1200, 10000]

# Create aggregator
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',#'uniform',#'data_driven',
    exclude_layers=['linear', 'num_batches_tracked']#exclude_layers=['classifier']  # ← Skip classifier
)

# Load models
aggregator.load_model_weights_from_files(
    model_paths, #['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=training_samples_per_model
)

#TODO: This was the ORIGINAL LOCATION, I moved it below
# See aggregation details
#aggregator.print_report()
# Aggregate
#aggregator.save_aggregated_weights('aggregated_backbone.pth')

# TODO: BRANCH 1
'''combined_classifier = torch.nn.Linear(64, 43)

with torch.no_grad():
    combined_classifier.weight[:3].load_state_dict(torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model_CLASSIFIER.pt"))
    combined_classifier.bias[:3] = people_classifier.bias

    combined_classifier.weight[3:23] = nature_classifier.weight
    combined_classifier.bias[3:23] = nature_classifier.bias

    combined_classifier.weight[23:43] = outdoors_classifier.weight
    combined_classifier.bias[23:43] = outdoors_classifier.bias'''

#TODO: BRANCH 2
'''#load_state_dict(torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model_CLASSIFIER_WEIGHT.pt"))
combined_weight = torch.cat([
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_outdoors_model_CLASSIFIER_WEIGHT.pt"),#people_head.weight,
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_nature_model_CLASSIFIER_WEIGHT.pt"),#nature_head.weight,
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model_CLASSIFIER_WEIGHT.pt"),#people_head.weight,
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model_CLASSIFIER_WEIGHT.pt"),#objects_and_vehicles_head.weight,
], dim=0)

combined_bias = torch.cat([
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_outdoors_model_CLASSIFIER_BIAS.pt"),#people_head.BIAS,
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_nature_model_CLASSIFIER_BIAS.pt"),#nature_head.BIAS,
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model_CLASSIFIER_BIAS.pt"),#people_head.BIAS,
    torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model_CLASSIFIER_BIAS.pt"),#objects_and_vehicles_head.BIAS,
], dim=0)'''

#TODO: BRANCH 2, OPTION 2:
people = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_people_model_CLASSIFIER.pt")
nature = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_nature_model_CLASSIFIER.pt")
outdoors = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_outdoors_model_CLASSIFIER.pt")
#TODO:Temporary ###########objects_and_vehicles = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model_CLASSIFIER.pt")
objects_and_vehicles = torch.load(f"../Thirteenth_implementation_attempt/final_specialised_models/final_objects_and_vehicles_model_CLASSIFIER.pt")

#print("people[weight]", people["weight"])
print("people[weight]", people["weight"].shape) #lwf_trainer.model.linear Linear(in_features=64, out_features=3, bias=True)
#print("nature[weight]", nature["weight"])
print("nature[weight]", nature["weight"].shape)
#print("outdoors[weight]", outdoors["weight"])
print("outdoors[weight]", outdoors["weight"].shape)
print("objects_and_vehicles[weight]", objects_and_vehicles["weight"].shape)
#OUTPUTS OF THE ABOVE PRINTS:
#people[weight] torch.Size([3, 64])
#nature[weight] torch.Size([9, 64])
#outdoors[weight] torch.Size([6, 64])
#objects_and_vehicles[weight] torch.Size([25, 64])
#print(weights.shape) torch.Size([37, 64])
#biases.shape torch.Size([37])

weights = torch.cat([
    #outdoors["weight"],
    nature["weight"],
    people["weight"],
    objects_and_vehicles["weight"]
], dim=0)

biases = torch.cat([
    #outdoors["bias"],
    nature["bias"],
    people["bias"],
    objects_and_vehicles["bias"]
], dim=0)

print("print(weights.shape)",weights.shape)
print("biases.shape",biases.shape)

# Load into model
print("\nLoading aggregated backbone into target model...")
target_model = ResNet20_DynamicModel(num_classes=37)#TODO:Temporary ###########(num_classes=43)  # For new classes

'''target_model.linear.weight.data.copy_(combined_weight)
target_model.linear.bias.data.copy_(combined_bias)'''
##########target_model.linear = torch.nn.Linear(64, 43)#TODO:Temporary ###########(64, 43)

'''with torch.no_grad():
    target_model.linear.weight.copy_(weights)
    target_model.linear.bias.copy_(biases)

target_model = aggregator.aggregate_into_model(target_model)
print("  ✓ Aggregated backbone loaded")
print("  ✓ Classifier from target model preserved")'''

aggregator.aggregate_into_model(target_model)#target_model = aggregator.aggregate_into_model(target_model, keep_classifier=False)
print("  ✓ Aggregated backbone loaded")
print("  ✓ Classifier from target model preserved")

#target_model.linear = torch.nn.Linear(64, 18)#TODO:Temporary ###########(64, 43)

with torch.no_grad():
    target_model.linear.weight.copy_(weights)
    target_model.linear.bias.copy_(biases)

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
from torch.utils.data import ConcatDataset, DataLoader

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
    #combined_dataset_outdoors,
    combined_dataset_nature, combined_dataset_people, combined_dataset_objects_and_vehicles
])

test_loader = DataLoader(
    combined_dataset,
    batch_size=32,
    shuffle=False
)
print("print test_loader", test_loader)
print("print test_loader.dataset", test_loader.dataset)


for batch in test_loader:
    print(type(batch))
    print(len(batch))
    for i, item in enumerate(batch):
        print(i, type(item), getattr(item, "shape", None))
        print(type(batch))
    break


#print("test_loader.batch_sampler",test_loader.batch_sampler)
    @property
    def mb_x(self):
        """Current mini-batch input."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3, "Task label not found."
        return mbatch[-1]

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


correct = 0
total = 0

#target_model.eval()
with torch.no_grad():
    for batch_data, batch_labels, batch_task_ids in test_loader:
        #print(type(batch_data))
        #print(type(batch_labels))
        #print(type(batch_task_ids))

        #print("print(batch_labels)",batch_labels)
        #print("print(batch_task_ids)",batch_task_ids)

        outputs = target_model(batch_data)
        targets_all_experiences = batch_labels
        predictions = outputs.argmax(dim=1)


        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

accuracy = correct / total
print(f"Fine-tuned model accuracy: {accuracy*100:.2%}")#print(f"Fine-tuned model accuracy: {accuracy:.2%}")

from torch.nn import Module, CrossEntropyLoss
def evaluate_aggregated_model(model: nn.Module):
    total_loss_all_experiences = 0.0
    total_correct_all_experiences = 0
    total_samples_all_experiences = 0

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in test_loader:
            unpack_minibatch(batch_data)

            logits_all_experiences = model(batch_data)

            targets_all_experiences = batch_labels

            loss_all_experiences = CrossEntropyLoss(logits_all_experiences, targets_all_experiences, reduction='mean')
            # print("print(self.model.training) -- evaluate all experiences after self.criterion",self.model.training)

            total_loss_all_experiences += loss_all_experiences.item() * targets_all_experiences.size(0)
            predictions_all_experiences = torch.argmax(logits_all_experiences, dim=1)
            total_correct_all_experiences += (predictions_all_experiences == targets_all_experiences).sum().item()
            total_samples_all_experiences += targets_all_experiences.size(0)

    avg_loss_all_experiences = total_loss_all_experiences / total_samples_all_experiences if total_samples_all_experiences > 0 else 0.0
    accuracy_all_experiences = (total_correct_all_experiences / total_samples_all_experiences * 100) if total_samples_all_experiences > 0 else 0.0

    print(
        f"Loss = {avg_loss_all_experiences:.6f} | Accuracy = {accuracy_all_experiences:.2f}%"
    )

#evaluate_aggregated_model(target_model)

'''if __name__ == "__main__":
    aggregated_model = scenario_uniform_aggregation_mine(model_paths)#, training_samples_per_model)'''