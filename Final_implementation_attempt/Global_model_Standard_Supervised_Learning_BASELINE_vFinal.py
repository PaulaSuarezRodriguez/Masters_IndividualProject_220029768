from Final_implementation_attempt.Config_vFinal import set_seed

set_seed()  # supposedly ensuring seed in split_validation_class_balanced since this method does not have a seed parameter when split_validation_class_balanced is used and instead pulls from the PyTorch Global Number Generator
from Final_implementation_attempt.Config_vFinal import set_device

DEVICE = set_device()
from Final_implementation_attempt.Config_vFinal import SEED
from avalanche.training.determinism.rng_manager import RNGManager

RNGManager.set_random_seeds(SEED)
from Final_implementation_attempt.ResNet20_vFinal import ResNet20

import os.path
from typing import (
    BinaryIO,
    IO,
    Union,
)
from typing_extensions import TypeAlias

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]

import torch.optim as optim

# Imports from Evaluate_BASELINE_vFinal
import torch.nn as nn
import pickle
from pathlib import Path
from Final_implementation_attempt.ResNet20_DynamicModel_vFinal import ResNet20_DynamicModel
############ Imports for Evaluating: ##############################################################################################################################################

from Final_implementation_attempt.Train_ResNet20_Outdoors_LwF_vFinal import \
    outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il, \
    outdoors_specialised_models_val_stream_class_il, outdoors_specialised_models_train_subset
from Final_implementation_attempt.Train_ResNet20_Nature_LwF_vFinal import \
    nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il, \
    nature_specialised_models_val_stream_class_il, nature_specialised_models_train_subset
from Final_implementation_attempt.Train_ResNet20_People_LwF_vFinal import \
    people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il, \
    people_specialised_models_val_stream_class_il, people_specialised_models_train_subset
from Final_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vFinal import \
    objects_and_vehicles_specialised_models_test_stream_class_il, \
    objects_and_vehicles_specialised_models_train_stream_class_il, \
    objects_and_vehicles_specialised_models_val_stream_class_il, objects_and_vehicles_specialised_models_train_subset
from torch.utils.data import ConcatDataset, DataLoader
import torch

DEBUG = True
TRAINING_WITH_LOOP = False
TRAINING_WITHOUT_LOOP = False
DYNAMIC_MODELS = True
print(
    "\n\n\n################################ HERE BEGINS Global_model_Standard_Supervised_Learning_BASELINE_vFinal.py ################################ \n\n\n")

if DYNAMIC_MODELS == False:
    resnet20_global_model = ResNet20(num_classes=100)
    outdoors_specialised_model = ResNet20(
        num_classes=100)
    nature_specialised_model = ResNet20(
        num_classes=100)
    people_specialised_model = ResNet20(
        num_classes=100)
    animals_specialised_model = ResNet20(
        num_classes=100)
    objects_and_vehicles_specialised_model = ResNet20(
        num_classes=100)

if DYNAMIC_MODELS == True:
    resnet20_global_model = ResNet20_DynamicModel(num_classes=0)
    outdoors_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    nature_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    people_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    animals_specialised_model = ResNet20_DynamicModel(
        num_classes=0)
    objects_and_vehicles_specialised_model = ResNet20_DynamicModel(
        num_classes=0)

# TRAIN DATASET all specialised train subsets together
combined_dataset_outdoors_train = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il[0].dataset,
    outdoors_specialised_models_train_stream_class_il[1].dataset,
    outdoors_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_nature_train = ConcatDataset([
    nature_specialised_models_train_stream_class_il[0].dataset,
    nature_specialised_models_train_stream_class_il[1].dataset,
    nature_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_people_train = ConcatDataset([
    people_specialised_models_train_stream_class_il[0].dataset,
    people_specialised_models_train_stream_class_il[1].dataset,
    people_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_train = ConcatDataset([
    objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_train = ConcatDataset([
    combined_dataset_outdoors_train,
    combined_dataset_nature_train,
    combined_dataset_people_train,
    combined_dataset_objects_and_vehicles_train
])
train_loader = DataLoader(
    combined_dataset_train,
    batch_size=128,
    shuffle=True
)
print("print train_loader", train_loader)
print("print test_loader.dataset", train_loader.dataset)

# VAL DATASET all specialised val subsets together
combined_dataset_outdoors_val = ConcatDataset([
    outdoors_specialised_models_val_stream_class_il[0].dataset,
    outdoors_specialised_models_val_stream_class_il[1].dataset,
    outdoors_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_nature_val = ConcatDataset([
    nature_specialised_models_val_stream_class_il[0].dataset,
    nature_specialised_models_val_stream_class_il[1].dataset,
    nature_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_people_val = ConcatDataset([
    people_specialised_models_val_stream_class_il[0].dataset,
    people_specialised_models_val_stream_class_il[1].dataset,
    people_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_val = ConcatDataset([
    objects_and_vehicles_specialised_models_val_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_val_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_val = ConcatDataset([
    combined_dataset_outdoors_val,
    combined_dataset_nature_val,
    combined_dataset_people_val,
    combined_dataset_objects_and_vehicles_val
])
val_loader = DataLoader(
    combined_dataset_val,
    batch_size=32,
    shuffle=False
)
print("print val_loader", val_loader)
print("print val_loader.dataset", val_loader.dataset)

combined_dataset_outdoors_test = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
    outdoors_specialised_models_test_stream_class_il[1].dataset,
    outdoors_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_nature_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
    nature_specialised_models_test_stream_class_il[1].dataset,
    nature_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_people_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[0].dataset,
    people_specialised_models_test_stream_class_il[1].dataset,
    people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_test = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_test = ConcatDataset([
    combined_dataset_outdoors_test,
    combined_dataset_nature_test,
    combined_dataset_people_test,
    combined_dataset_objects_and_vehicles_test
])

test_loader = DataLoader(
    combined_dataset_test,
    batch_size=32,
    shuffle=False
)
print("print test_loader", test_loader)
print("print test_loader.dataset", test_loader.dataset)


def map_targets(batch_labels_tensor):
    targets_tensor = torch.tensor([class_to_idx_new[int(fine_label.item())] for fine_label in batch_labels_tensor],
                                  device=batch_labels_tensor.device,
                                  dtype=torch.long
                                  )
    return targets_tensor


def map_labels_expanding_classifier_benchmark(ds_list, class_to_idx_new=None):
    original_to_aggregated_dict = {17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5,  # OUTDOORS
                                   51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14,  # NATURE
                                   2: 15, 11: 16, 35: 17,  # PEOPLE
                                   16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27, 39: 28,
                                   28: 29, 25: 30, 22: 31, 20: 32, 94: 33,  # OBJECTS_AND_VEHICLES
                                   90: 34, 89: 35, 87: 36, 86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42}

    if class_to_idx_new is None:
        class_to_idx_new = {}

    for ds in ds_list:
        for fine_label in sorted(ds.targets):
            fine_label_item = fine_label
            if fine_label_item not in class_to_idx_new.keys() and fine_label_item in original_to_aggregated_dict.keys():
                class_to_idx_new[fine_label_item] = len(class_to_idx_new)

    return class_to_idx_new


class_to_idx_new = map_labels_expanding_classifier_benchmark(
    [outdoors_specialised_models_train_subset.dataset, nature_specialised_models_train_subset.dataset,
     people_specialised_models_train_subset.dataset, objects_and_vehicles_specialised_models_train_subset.dataset])
print("class_to_idx_new:", class_to_idx_new)


def new_to_original_creation(new_to_original=None):
    if new_to_original is None:
        new_to_original = {}

    for original_label, new_idx in class_to_idx_new.items():
        new_to_original[new_idx] = original_label
    return new_to_original


new_to_original = new_to_original_creation()
print("new_to_original:", new_to_original)


def train(model, train_loader, val_loader, num_epochs=200, lr=0.1, weight_decay=1e-4, patience=3, lr_decay_factor=0.1,
          device='cpu'):
    """Main training loop"""

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)

    model.train()

    total_loss = 0
    c = 0
    loss_c = 0
    prev_epoch_loss = 100
    prev_epoch_acc = 0
    best_acc_so_far = 0
    accuracies = []
    original_lr = lr
    acc_limit = 85
    best_val_acc = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f'\n=== Epoch [{epoch + 1}/{num_epochs}] ===')
        batch_idx = 0
        total_ce = 0.0
        total_dist = 0.0
        total_loss = 0.0
        n = 0
        mini_batch_loss = 0.0
        mini_batch_acc = 0.0
        num_batches = 0
        correct = 0
        total = 0

        for images, labels, batch_task_ids in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            labels = map_targets(labels)

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100 * correct / total:.2f}%')

            # Track metrics
            mini_batch_loss += loss.item()
            with torch.no_grad():
                acc = (output.argmax(1) == labels).float().mean()
                mini_batch_acc += acc.item()
            num_batches += 1

        # Log epoch results
        avg_loss = mini_batch_loss / num_batches
        avg_acc = mini_batch_acc / num_batches
        avg_acc = avg_acc * 100

        print(f"\n{'...' * 60}")
        print(
            f"ALTERNATIVE CODE (in testing) --> EVALUATING Epoch {epoch} OF epoch {epoch}")
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        print(f"{'...' * 60}")

        epoch_loss = avg_loss
        epoch_acc = avg_acc
        accuracies.append(epoch_acc)
        accuracies.sort(reverse=True)
        print(f"accuracies: {accuracies}")

        if epoch_loss >= prev_epoch_loss:
            loss_c += 1
            if loss_c == patience - 1:
                print(
                    f"lr before being increased at epoch {epoch}, Epoch {epoch}: {lr}")
                lr = lr * lr_decay_factor
                print(
                    f"lr after being increased at epoch {epoch}, Epoch {epoch}: {lr}")
                c = 0

        if epoch_acc > prev_epoch_acc and prev_epoch_acc < acc_limit:
            # Both metrics improved - reset patience counter
            c = 0
            loss_c = 0
        elif epoch_acc <= prev_epoch_acc:
            # No improvement - increment patience counter
            c += 1
            print(f"WARNING: Loss and accuracy did not improve during the previous epoch.")

            if c >= patience:
                print(
                    f"WARNING: The number of epochs equivalent to patience ({patience}) has been reached without improvement. Training will be stopped.")
                lr = original_lr
                c = 0
                loss_c = 0
                break
        elif epoch_acc >= acc_limit:
            print(
                f"WARNING: The accuracy of epoch {epoch} has reached 85%. Training will be stopped.")
            lr = original_lr
            c = 0
            loss_c = 0
            break

        prev_epoch_loss = epoch_loss
        prev_epoch_acc = epoch_acc

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)

    # Print results
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return model


def validate(model, val_loader, criterion, device='cpu'):
    """Validate on validation set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, batch_task_ids in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            labels = map_targets(labels)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


if __name__ == '__main__':
    # Initialize model
    model = ResNet20(num_classes=43)

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,
        lr=000.1,
        weight_decay=1e-4,
    )
    final_specialised_models_baseline_dir = Path("./final_specialised_models_BASELINE")
    final_specialised_models_baseline_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"./final_specialised_models_BASELINE/BASELINE_standard_supervised_global_model.pt")
    new_to_original = new_to_original_creation()
    class_to_idx_new = class_to_idx_new
    # Save the mappings used in training
    metadata = {
        'class_to_idx_new': class_to_idx_new,
        'new_to_original': new_to_original,
        'num_subset_classes': 43
    }
    with open('./final_specialised_models_BASELINE/BASELINE_standard_supervised_global_metadata.pkl',
              'wb') as f:
        pickle.dump(metadata, f)
    print("✓ Saved metadata for BASELINE EVALUATION")