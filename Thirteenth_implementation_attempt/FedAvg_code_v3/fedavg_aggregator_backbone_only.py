"""
Federated Averaging (FedAvg) for Non-Overlapping Class Models

Scenario: Models trained on completely different, non-overlapping class sets
- Model 1: Classes 0-9
- Model 2: Classes 10-19
- Model 3: Classes 20-29
(etc.)

Solution: Aggregate ONLY the backbone/feature extractor layers.
Skip classifier heads since they're trained on different classes.

Approach: Element-wise aggregation at individual tensor element level.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Union, Optional, Set
import numpy as np


class FedAvgAggregatorBackboneOnly:
    """
    Aggregates only backbone layers (feature extractor) using element-wise aggregation.
    
    Excludes classifier layers since models are trained on non-overlapping classes.
    
    Element-wise approach: For each individual element in each tensor,
    compute the weighted average across all models.
    """
    
    def __init__(self, 
                 weighting_strategy: str = 'uniform',
                 exclude_layers: Optional[List[str]] = None):
        """
        Initialize aggregator for backbone-only aggregation.
        
        Args:
            weighting_strategy (str): 'uniform' or 'data_driven'
            exclude_layers (List[str]): Layer name patterns to exclude
                Default: excludes anything with 'classifier' in the name
                Example: ['classifier', 'head', 'fc_final']
        """
        self.weighting_strategy = weighting_strategy
        self.dataset_sizes = []
        
        # Define layers to exclude (non-aggregable layers)
        if exclude_layers is None:
            self.exclude_layers = ['linear'] #['classifier']  # Default: exclude classifier #TODO: MY CHANGE TO THE CODE
        else:
            self.exclude_layers = exclude_layers
        
        # Store weights as list of dicts
        self.weights_list = []
        self.aggregated_weights = None
    
    def add_model_weights(self, 
                         model_weights: Dict[str, torch.Tensor],
                         dataset_size: Optional[int] = None):
        """
        Add model weights to aggregation pool.
        
        Args:
            model_weights (Dict[str, torch.Tensor]): State dict of model
            dataset_size (int, optional): Training samples for this model
        """
        self.weights_list.append(model_weights)
        
        if self.weighting_strategy == 'data_driven':
            if dataset_size is None:
                raise ValueError("dataset_size required for 'data_driven' weighting")
            self.dataset_sizes.append(dataset_size)
        else:
            self.dataset_sizes.append(1)
    
    def load_model_weights_from_files(self,
                                      weight_paths: List[Union[str, Path]],
                                      dataset_sizes: Optional[List[int]] = None):
        """
        Load model weights from saved checkpoint files.
        
        Args:
            weight_paths (List[str or Path]): Paths to saved model weights
            dataset_sizes (List[int], optional): Training samples per model
        """
        if dataset_sizes is not None:
            if len(dataset_sizes) != len(weight_paths):
                raise ValueError("Length mismatch between paths and dataset_sizes")
        else:
            dataset_sizes = [1] * len(weight_paths)
            print("len(dataset_sizes)",len(dataset_sizes)) # TODO: MY ADDITION TO THE CODE
            print("len(weight_paths)",len(weight_paths)) # TODO: MY ADDITION TO THE CODE
        
        for path, size in zip(weight_paths, dataset_sizes):
            weights = torch.load(path, map_location='cpu')
            self.add_model_weights(weights, size)
    
    def _should_exclude_layer(self, layer_name: str) -> bool:
        """
        Check if layer should be excluded from aggregation.
        
        Args:
            layer_name (str): Name of layer in state dict
            
        Returns:
            bool: True if layer should be excluded
        """
        for exclude_pattern in self.exclude_layers:
            if exclude_pattern in layer_name:
                return True
        return False
    
    def _get_aggregation_weights(self) -> np.ndarray:
        """
        Compute normalized aggregation weights for each model.
        
        Returns:
            np.ndarray: Normalized weights (sum = 1.0)
        """
        if self.weighting_strategy == 'uniform':
            n_models = len(self.weights_list)
            return np.ones(n_models) / n_models
        
        elif self.weighting_strategy == 'data_driven':
            dataset_sizes = np.array(self.dataset_sizes, dtype=np.float32)
            total_samples = dataset_sizes.sum()
            return dataset_sizes / total_samples
        
        else:
            raise ValueError(f"Unknown strategy: {self.weighting_strategy}")
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Perform element-wise aggregation of backbone layers only.
        
        Element-wise approach:
        - For each parameter name (layer)
        - Stack tensors from all models
        - For each individual element: weighted average
        - Result: aggregated_param[i,j,...] = Σ λ_k * model_k_param[i,j,...]
        
        Returns:
            Dict[str, torch.Tensor]: Aggregated backbone weights
        """
        if not self.weights_list:
            raise ValueError("No model weights loaded. Call load_model_weights_from_files() first.")
        
        agg_weights = self._get_aggregation_weights()
        
        print("\n" + "="*80)
        print("ELEMENT-WISE AGGREGATION (Backbone Only)")
        print("="*80)
        print(f"\nAggregation strategy: {self.weighting_strategy}")
        print(f"Model weights: {[f'{w:.4f}' for w in agg_weights]}")
        
        aggregated = {}
        excluded_count = 0
        aggregated_count = 0
        
        # Get layer names from first model
        first_model = self.weights_list[0]
        
        print(f"\nProcessing {len(first_model)} layers...\n")

        print("first_model.keys()", first_model.keys()) # TODO: MY ADDITION TO THE CODE
        # Process each layer
        for layer_name in first_model.keys():
            # Check if should exclude
            if self._should_exclude_layer(layer_name):
                excluded_count += 1
                print(f"  ✗ SKIP   : {layer_name}")
                continue
            
            aggregated_count += 1
            
            # Get tensors from all models for this layer
            tensors = [model[layer_name].float() for model in self.weights_list]
            
            # Verify all same shape (should be for backbone)
            shapes = [t.shape for t in tensors]
            if len(set(shapes)) > 1:
                print(f"  ⚠ WARNING: {layer_name} has mismatched shapes: {shapes}")
                print(f"             Skipping this layer")
                continue
            
            # Element-wise aggregation
            # Step 1: Stack tensors
            stacked = torch.stack(tensors, dim=0)  # Shape: [num_models, *param_shape]
            
            # Step 2: Prepare weights for broadcasting
            agg_weights_tensor = torch.tensor(
                agg_weights,
                dtype=torch.float32,
                device=stacked.device
            ).view(-1, *([1] * (stacked.dim() - 1))) # Shape: [num_models, 1, 1, ...] # TODO: MY ADDITiON TO THE CODE
            '''agg_weights_tensor = torch.tensor(
                1.0,
                dtype=torch.float32,
                device=stacked.device
            ).view(-1, *([1] * (stacked.dim() - 1)))'''
            '''agg_weights_tensor = torch.tensor(
                1.0,
                dtype=torch.float32,
                device=stacked.device
            )#.view(-1, *([1] * (stacked.dim() - 1)))'''

            # Step 3: Element-wise weighted sum
            # For each element [i,j,...]: result[i,j,...] = Σ λ_k * stacked[k,i,j,...]
            aggregated[layer_name] = (stacked * agg_weights_tensor).sum(dim=0)

            print(f"  ✓ AGG    : {layer_name:40s} {str(shapes[0]):30s}")
        
        print(f"\nSummary:")
        print(f"  Aggregated layers  : {aggregated_count}")
        print(f"  Excluded layers    : {excluded_count}")
        print(f"  Total layers       : {len(first_model)}")
        
        self.aggregated_weights = aggregated
        return aggregated
    
    def aggregate_into_model(self, 
                            model: nn.Module,
                            keep_classifier: bool = True) -> nn.Module:
        """
        Load aggregated backbone weights into a model.
        
        Args:
            model (nn.Module): Target model to load weights
            keep_classifier (bool): If True, keep target model's classifier.
                                   If False, may cause shape mismatch on classifier
        
        Returns:
            nn.Module: Model with aggregated backbone loaded
        """
        aggregated = self.aggregate()
        
        # Get target model's state dict
        target_state = model.state_dict()
        
        # Create new state dict with aggregated backbone + original classifier
        new_state = {}
        
        for key, value in target_state.items():
            if self._should_exclude_layer(key):
                # Keep original (don't aggregate classifier)
                new_state[key] = value
            else:
                # Load aggregated backbone weights
                if key in aggregated:
                    new_state[key] = aggregated[key]
                else:
                    new_state[key] = value
        
        model.load_state_dict(new_state)
        target_state_after = model.state_dict()
        return model
    
    def save_aggregated_weights(self, save_path: Union[str, Path]):
        """
        Save aggregated backbone weights to file.
        
        Args:
            save_path (str or Path): Where to save aggregated weights
        """
        if self.aggregated_weights is None:
            self.aggregate()
        
        torch.save(self.aggregated_weights, save_path)
        print(f"\n✓ Aggregated weights saved to {save_path}")
    
    def get_aggregation_report(self) -> Dict:
        """Generate aggregation report."""
        agg_weights = self._get_aggregation_weights()
        
        # Count layers
        first_model = self.weights_list[0]
        aggregated_count = sum(1 for k in first_model.keys() 
                              if not self._should_exclude_layer(k))
        excluded_count = len(first_model) - aggregated_count
        
        report = {
            'num_models': len(self.weights_list),
            'weighting_strategy': self.weighting_strategy,
            'model_weights': agg_weights.tolist(),
            'total_samples': sum(self.dataset_sizes),
            'per_model_samples': self.dataset_sizes,
            'excluded_layer_patterns': self.exclude_layers,
            'aggregated_layers': aggregated_count,
            'excluded_layers': excluded_count,
            'total_layers': len(first_model),
        }
        
        return report
    
    def print_report(self):
        """Print detailed aggregation report."""
        report = self.get_aggregation_report()
        
        print("\n" + "="*80)
        print("AGGREGATION REPORT")
        print("="*80)
        
        print(f"\nSetup:")
        print(f"  Number of models   : {report['num_models']}")
        print(f"  Total training data: {report['total_samples']} samples")
        print(f"  Weighting strategy : {report['weighting_strategy']}")
        
        print(f"\nModel weights (data-driven):")
        for i, weight in enumerate(report['model_weights']):
            print(f"  Model {i+1}: {weight:.4f} ({report['per_model_samples'][i]} samples)")
        
        print(f"\nLayer aggregation:")
        print(f"  Aggregated layers  : {report['aggregated_layers']}")
        print(f"  Excluded layers    : {report['excluded_layers']}")
        print(f"  Total layers       : {report['total_layers']}")
        
        print(f"\nExcluded layer patterns:")
        for pattern in report['excluded_layer_patterns']:
            print(f"  - '*{pattern}*'")
    
    def reset(self):
        """Clear all loaded weights."""
        self.weights_list = []
        self.dataset_sizes = []
        self.aggregated_weights = None


# ============================================================================
# Direct Element-Wise Aggregation Function (Your Code Style)
# ============================================================================

def fedavg_element_wise_aggregation(
    weights_list: List[Dict[str, torch.Tensor]],
    dataset_sizes: List[int],
    exclude_layers: Optional[List[str]] = None,
    weighting_strategy: str = 'data_driven'
) -> Dict[str, torch.Tensor]:
    """
    Direct element-wise aggregation function.
    
    Matches your code snippet style:
    1. Load weights into list
    2. Group by layer name
    3. Aggregate element-wise (weighted average per element)
    4. Skip excluded layers
    
    Args:
        weights_list: List of state dicts from each model
        dataset_sizes: Training samples per model
        exclude_layers: Layer patterns to exclude (default: ['classifier'])
        weighting_strategy: 'uniform' or 'data_driven'
    
    Returns:
        Dict[str, torch.Tensor]: Aggregated backbone weights
    
    Example:
        aggregated = fedavg_element_wise_aggregation(
            weights_list=[weights1, weights2, weights3],
            dataset_sizes=[1000, 1200, 900],
            exclude_layers=['classifier'],
            weighting_strategy='data_driven'
        )
    """
    
    if exclude_layers is None:
        exclude_layers = ['classifier']
    
    print("\n" + "="*80)
    print("ELEMENT-WISE AGGREGATION (Direct Function)")
    print("="*80)
    
    # Compute aggregation weights
    if weighting_strategy == 'uniform':
        n = len(weights_list)
        agg_weights = np.ones(n) / n
    else:  # data_driven
        dataset_sizes_arr = np.array(dataset_sizes, dtype=np.float32)
        agg_weights = dataset_sizes_arr / dataset_sizes_arr.sum()
    
    print(f"\nAggregation weights: {[f'{w:.4f}' for w in agg_weights]}")
    print(f"Total training data: {sum(dataset_sizes)} samples\n")
    
    # YOUR CODE SNIPPET APPROACH:
    # 1. Group weights by layer name
    print("Grouping weights by layer...")
    weights_by_layer = {}
    
    for key in weights_list[0].keys():
        # Only include layers (skip buffers that don't have "weight" in name)
        if "weight" in key or "bias" in key:
            weights_by_layer[key] = []
            
            for state_dict in weights_list:
                if key in state_dict:
                    weights_by_layer[key].append(state_dict[key])
    
    print(f"Found {len(weights_by_layer)} aggregable parameters\n")
    
    # 2. Aggregate element-wise
    print("Performing element-wise aggregation...\n")
    aggregated = {}
    excluded_count = 0
    aggregated_count = 0
    
    for layer_name in weights_by_layer.keys():
        # Check if should exclude
        should_exclude = any(pattern in layer_name for pattern in exclude_layers)
        
        if should_exclude:
            excluded_count += 1
            print(f"  ✗ SKIP   : {layer_name}")
            continue
        
        aggregated_count += 1
        tensors = weights_by_layer[layer_name]
        
        # Verify all same shape
        shapes = [t.shape for t in tensors]
        if len(set(shapes)) > 1:
            print(f"  ⚠ WARNING: {layer_name} shape mismatch: {shapes} - SKIPPING")
            continue
        
        # Element-wise aggregation
        # Stack: [model1, model2, model3] -> [3, *param_shape]
        stacked = torch.stack(tensors, dim=0)
        
        # Reshape weights for broadcasting
        agg_w = torch.tensor(agg_weights, dtype=torch.float32).view(
            -1, *([1] * (stacked.dim() - 1))
        )
        
        # Element-wise weighted sum
        # result[i,j,...] = λ₁*model1[i,j,...] + λ₂*model2[i,j,...] + λ₃*model3[i,j,...]
        aggregated[layer_name] = (stacked * agg_w).sum(dim=0)
        
        print(f"  ✓ AGG    : {layer_name:40s} {str(shapes[0]):30s}")
    
    print(f"\nSummary:")
    print(f"  Aggregated: {aggregated_count}")
    print(f"  Excluded  : {excluded_count}")
    print(f"  Total     : {len(weights_by_layer)}")
    
    return aggregated


# ============================================================================
# Example Usage
# ============================================================================

def example_non_overlapping_classes():
    """
    Example: Models trained on non-overlapping classes.
    
    - Model 1: Classes 0-9   (dataset_size=1000)
    - Model 2: Classes 10-19 (dataset_size=1200)
    - Model 3: Classes 20-29 (dataset_size=900)
    
    Backbone is identical size. Classifier heads are same size but different classes.
    Aggregate only backbone.
    """
    
    print("\n" + "="*80)
    print("Example: Non-Overlapping Classes - Backbone Only Aggregation")
    print("="*80)
    
    code_example = """
import torch
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

# Your setup: models trained on non-overlapping classes
# Model 1: Classes 0-9    (1000 samples)
# Model 2: Classes 10-19  (1200 samples)
# Model 3: Classes 20-29  (900 samples)

model_paths = [
    'models/task1_classes_0to9.pth',
    'models/task2_classes_10to19.pth',
    'models/task3_classes_20to29.pth',
]
dataset_sizes = [1000, 1200, 900]

# ============================================================================
# METHOD 1: Using the Aggregator Class
# ============================================================================

aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier']  # Exclude classifier
)

# Load models
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Print report
aggregator.print_report()

# Aggregate
aggregated_weights = aggregator.aggregate()

# Save
aggregator.save_aggregated_weights('aggregated_backbone.pth')

# Load into model
model = YourModel(num_classes=10)  # Can be any num_classes
model = aggregator.aggregate_into_model(model)

# ============================================================================
# METHOD 2: Direct Element-Wise Function (Your Code Style)
# ============================================================================

from fedavg_aggregator_backbone_only import fedavg_element_wise_aggregation

# Load weights manually
weights_list = [
    torch.load(model_paths[0]),
    torch.load(model_paths[1]),
    torch.load(model_paths[2]),
]

# Direct aggregation
aggregated = fedavg_element_wise_aggregation(
    weights_list=weights_list,
    dataset_sizes=dataset_sizes,
    exclude_layers=['classifier'],
    weighting_strategy='data_driven'
)

# Save aggregated backbone
torch.save(aggregated, 'aggregated_backbone.pth')
"""
    
    print(code_example)


def example_custom_exclude_patterns():
    """Example with custom layer exclusion patterns."""
    
    print("\n" + "="*80)
    print("Example: Custom Exclusion Patterns")
    print("="*80)
    
    code = """
# Exclude multiple layer types
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier', 'head', 'fc_final']  # Multiple patterns
)

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
aggregator.aggregate()

# This will skip ANY layer with 'classifier', 'head', or 'fc_final' in the name
# Examples of skipped layers:
#   - classifier.weight
#   - classifier.bias
#   - head.0.weight
#   - fc_final.bias
#
# Examples of aggregated layers:
#   - backbone.0.weight
#   - feature_extractor.weight
#   - encoder.layer1.weight
"""
    
    print(code)


if __name__ == "__main__":
    print("\nFedAvg Backbone-Only Aggregation")
    print("For models trained on non-overlapping classes")
    
    example_non_overlapping_classes()
    example_custom_exclude_patterns()
    
    print("\n" + "="*80)
    print("Key Points")
    print("="*80)
    print("""
    ✓ Element-wise aggregation: Each tensor element averaged independently
    ✓ Backbone only: Skips classifier layers (trained on different classes)
    ✓ Data-driven weighting: Weights models by training data size
    ✓ Simple approach: Matches your code snippet style
    
    FORMULA (Element-wise):
    ───────────────────────
    result[i,j,...] = λ₁*model₁[i,j,...] + λ₂*model₂[i,j,...] + λ₃*model₃[i,j,...]
    
    where λ = normalized weights (sum to 1.0)
    
    WHY SKIP CLASSIFIER:
    ────────────────────
    - Model 1 classifier: trained on classes 0-9
    - Model 2 classifier: trained on classes 10-19 (completely different!)
    - Model 3 classifier: trained on classes 20-29
    
    Since classes don't overlap, classifier heads can't be meaningfully aggregated.
    Only the shared backbone can be aggregated.
    
    USE AGGREGATED BACKBONE:
    ────────────────────────
    - For transfer learning on new task
    - For fine-tuning on new classes
    - As feature extractor
    - New classifier can be trained on new classes
    """)
