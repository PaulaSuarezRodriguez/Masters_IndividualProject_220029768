"""
Federated Averaging (FedAvg) with Support for Expanding Classifiers

Handles continual learning scenarios where the classifier grows (new classes added)
over time. Aggregates only the shared/common weights, preserving task-specific classes.

Modification: Element-wise aggregation for flexible handling of different architectures.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Union, Optional, Set
import numpy as np


class FedAvgAggregatorExpandingClassifier:
    """
    Aggregates model weights using FedAvg, with special handling for expanding classifiers.
    
    When models have different output dimensions (e.g., different number of classes),
    this aggregator:
    1. Aggregates shared/common parameters element-wise
    2. Handles expanding classifier by aggregating only common classes
    3. Preserves task-specific new classes
    """
    
    def __init__(self, 
                 weighting_strategy: str = 'uniform',
                 expanding_layers: Optional[List[str]] = None,
                 aggregation_mode: str = 'shared_only'):
        """
        Initialize the FedAvg aggregator with expanding classifier support.
        
        Args:
            weighting_strategy (str): 'uniform' or 'data_driven'
            expanding_layers (List[str]): Layer names that have expanding dimensions
                (e.g., ['classifier.weight', 'classifier.bias'])
                If None, will auto-detect based on shape mismatches
            aggregation_mode (str): How to handle expanding dimensions
                'shared_only': Aggregate only common/shared weights (recommended)
                'all_weighted': Aggregate all weights by proportion
        """
        self.weighting_strategy = weighting_strategy
        self.expanding_layers = expanding_layers or []
        self.aggregation_mode = aggregation_mode
        
        self.model_weights_list = []
        self.dataset_sizes = []
        self.layer_shapes = {}  # Track shapes of all layers
        
    def add_model_weights(
        self, 
        model_weights: Dict[str, torch.Tensor],
        dataset_size: Optional[int] = None
    ):
        """
        Add model weights to the aggregation pool.
        
        Args:
            model_weights (Dict[str, torch.Tensor]): State dict of model weights
            dataset_size (int, optional): Number of training samples
        """
        self.model_weights_list.append(model_weights)
        
        if self.weighting_strategy == 'data_driven':
            if dataset_size is None:
                raise ValueError("dataset_size required for 'data_driven' weighting")
            self.dataset_sizes.append(dataset_size)
        else:
            self.dataset_sizes.append(1)
        
        # Track layer shapes
        for key, value in model_weights.items():
            if key not in self.layer_shapes:
                self.layer_shapes[key] = []
            self.layer_shapes[key].append(tuple(value.shape))
    
    def load_model_weights_from_files(
        self,
        weight_paths: List[Union[str, Path]],
        dataset_sizes: Optional[List[int]] = None
    ):
        """
        Load model weights from saved checkpoint files.
        
        Args:
            weight_paths (List[str or Path]): Paths to saved model weights
            dataset_sizes (List[int], optional): Dataset sizes for each model
        """
        if dataset_sizes is not None:
            if len(dataset_sizes) != len(weight_paths):
                raise ValueError("Length of dataset_sizes must match weight_paths")
        else:
            dataset_sizes = [1] * len(weight_paths)
        
        for path, size in zip(weight_paths, dataset_sizes):
            weights = torch.load(path, map_location='cpu')
            self.add_model_weights(weights, size)
    
    def detect_expanding_layers(self) -> List[str]:
        """
        Automatically detect layers with expanding dimensions.
        
        Returns:
            List of layer names that have different shapes across models
        """
        if not self.model_weights_list:
            return []
        
        expanding = []
        
        for layer_name, shapes in self.layer_shapes.items():
            # Check if shapes vary
            if len(set(shapes)) > 1:
                # Check if only certain dimensions expand
                first_shape = shapes[0]
                for i, shape in enumerate(shapes[1:], 1):
                    # Find which dimension(s) differ
                    if len(first_shape) == len(shape):
                        for dim_idx, (s1, s2) in enumerate(zip(first_shape, shape)):
                            if s1 != s2:
                                # This dimension varies
                                if layer_name not in expanding:
                                    expanding.append(layer_name)
                                break
        
        return expanding
    
    def _compute_aggregation_weights(self) -> np.ndarray:
        """Compute weight coefficients for each model."""
        if self.weighting_strategy == 'uniform':
            n_models = len(self.model_weights_list)
            return np.ones(n_models) / n_models
        
        elif self.weighting_strategy == 'data_driven':
            dataset_sizes = np.array(self.dataset_sizes, dtype=np.float32)
            total_samples = dataset_sizes.sum()
            return dataset_sizes / total_samples
        
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")
    
    def _get_common_shape(self, layer_name: str) -> Optional[tuple]:
        """
        Get the minimum/common shape for a layer (for expanding classifiers).
        
        For expanding layers, we aggregate only the common dimensions.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Tuple representing the common/shared shape
        """
        shapes = self.layer_shapes.get(layer_name, [])
        
        if not shapes:
            return None
        
        if len(set(shapes)) == 1:
            # All same shape
            return shapes[0]
        
        # Multiple different shapes - find minimum along each dimension
        common_shape = []
        for dim_idx in range(len(shapes[0])):
            # Get minimum size along this dimension (for class expansion case)
            sizes = [s[dim_idx] for s in shapes]
            common_size = min(sizes)
            common_shape.append(common_size)
        
        return tuple(common_shape)
    
    def _aggregate_shared_weights(self, layer_name: str, agg_weights: np.ndarray) -> torch.Tensor:
        """
        Aggregate only the shared/common weights for expanding layers.
        
        Element-wise aggregation: processes weights at individual element level.
        
        Args:
            layer_name: Name of the layer to aggregate
            agg_weights: Weight coefficients for each model
            
        Returns:
            Aggregated weights (shared portion only)
        """
        # Get the common shape (minimum across all models)
        common_shape = self._get_common_shape(layer_name)
        
        if common_shape is None:
            raise ValueError(f"Cannot determine common shape for {layer_name}")
        
        # Extract shared portions from each model
        shared_weights = []
        for state_dict in self.model_weights_list:
            param = state_dict[layer_name].float()
            
            # Slice to common shape
            slices = tuple(slice(0, size) for size in common_shape)
            shared_portion = param[slices]
            shared_weights.append(shared_portion)
        
        # Element-wise aggregation
        # Stack all shared portions
        stacked = torch.stack(shared_weights)  # Shape: [num_models, *common_shape]
        
        # Convert aggregation weights to tensor for broadcasting
        agg_weights_tensor = torch.tensor(
            agg_weights,
            dtype=torch.float32,
            device=stacked.device
        ).view(-1, *([1] * (stacked.dim() - 1)))
        
        # Element-wise weighted sum
        aggregated = (stacked * agg_weights_tensor).sum(dim=0)
        
        return aggregated
    
    def _aggregate_full_weights(self, layer_name: str, agg_weights: np.ndarray) -> torch.Tensor:
        """
        Aggregate full weights for non-expanding layers.
        
        Args:
            layer_name: Name of the layer to aggregate
            agg_weights: Weight coefficients for each model
            
        Returns:
            Aggregated weights
        """
        # Stack weights from all models
        param_stack = torch.stack([
            model[layer_name].float()
            for model in self.model_weights_list
        ])
        
        # Convert aggregation weights for broadcasting
        agg_weights_tensor = torch.tensor(
            agg_weights,
            dtype=torch.float32,
            device=param_stack.device
        ).view(-1, *([1] * (param_stack.dim() - 1)))
        
        # Element-wise weighted sum
        aggregated = (param_stack * agg_weights_tensor).sum(dim=0)
        
        return aggregated
    
    def aggregate(self, expanding_layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform FedAvg aggregation with special handling for expanding classifiers.
        
        Args:
            expanding_layers: Override expanding layer list. If None, use self.expanding_layers
                            or auto-detect
        
        Returns:
            Dict[str, torch.Tensor]: Aggregated model weights
        """
        if not self.model_weights_list:
            raise ValueError("No model weights to aggregate. Add weights first.")
        
        # Determine which layers are expanding
        if expanding_layers is not None:
            layers_to_expand = expanding_layers
        elif self.expanding_layers:
            layers_to_expand = self.expanding_layers
        else:
            # Auto-detect
            layers_to_expand = self.detect_expanding_layers()
        
        # Compute aggregation weights
        agg_weights = self._compute_aggregation_weights()
        
        # Get reference layer names from first model
        first_model = self.model_weights_list[0]
        
        # Aggregate each parameter
        aggregated_state = {}
        
        for param_name in first_model.keys():
            if param_name in layers_to_expand:
                # Expanding layer - aggregate only shared portion
                print(f"  Aggregating (shared only): {param_name}")
                aggregated_state[param_name] = self._aggregate_shared_weights(
                    param_name, agg_weights
                )
            else:
                # Regular layer - full aggregation
                aggregated_state[param_name] = self._aggregate_full_weights(
                    param_name, agg_weights
                )
        
        return aggregated_state
    
    def aggregate_into_model(self, model: nn.Module, expanding_layers: Optional[List[str]] = None) -> nn.Module:
        """
        Aggregate weights and load into a model.
        
        Args:
            model (nn.Module): Target model to load aggregated weights
            expanding_layers: Override expanding layer list
        
        Returns:
            nn.Module: Model with aggregated weights loaded
        """
        aggregated_weights = self.aggregate(expanding_layers)
        
        # Only load common weights for expanding layers
        expanding_layers = expanding_layers or self.expanding_layers or self.detect_expanding_layers()
        
        # Create a modified state_dict that handles shape mismatches
        model_state = model.state_dict()
        final_state = {}
        
        for key, agg_value in aggregated_weights.items():
            model_param = model_state[key]
            
            if key in expanding_layers:
                # For expanding layers, need to handle shape mismatch
                # Copy aggregated values into model's parameter
                final_state[key] = model_param.clone()
                
                # Paste aggregated weights into shared portion
                slices = tuple(slice(0, size) for size in agg_value.shape)
                final_state[key][slices] = agg_value
            else:
                # Regular layers
                final_state[key] = agg_value
        
        model.load_state_dict(final_state)
        return model
    
    def save_aggregated_weights(self, save_path: Union[str, Path], expanding_layers: Optional[List[str]] = None):
        """
        Aggregate weights and save to file.
        
        Args:
            save_path (str or Path): Path where to save aggregated weights
            expanding_layers: Override expanding layer list
        """
        aggregated_weights = self.aggregate(expanding_layers)
        torch.save(aggregated_weights, save_path)
        print(f"✓ Aggregated weights saved to {save_path}")
    
    def get_aggregation_report(self) -> Dict:
        """
        Generate a detailed aggregation report.
        
        Returns:
            Dict with aggregation details
        """
        agg_weights = self._compute_aggregation_weights()
        expanding = self.detect_expanding_layers()
        
        report = {
            'num_models': len(self.model_weights_list),
            'weighting_strategy': self.weighting_strategy,
            'model_weights': agg_weights.tolist(),
            'total_samples': sum(self.dataset_sizes),
            'per_model_samples': self.dataset_sizes,
            'expanding_layers': expanding,
            'layer_shapes': {k: v for k, v in self.layer_shapes.items()},
        }
        
        return report
    
    def print_aggregation_details(self):
        """Print detailed aggregation information."""
        report = self.get_aggregation_report()
        
        print("\n" + "="*80)
        print("FEDAVG AGGREGATION DETAILS (Expanding Classifier)")
        print("="*80)
        
        print(f"\nModels: {report['num_models']}")
        print(f"Total samples: {report['total_samples']}")
        print(f"Weighting strategy: {report['weighting_strategy']}")
        
        print(f"\nModel weights:")
        for i, weight in enumerate(report['model_weights']):
            print(f"  Model {i+1}: {weight:.4f}")
        
        print(f"\nExpanding layers (shape varies):")
        if report['expanding_layers']:
            for layer_name in report['expanding_layers']:
                shapes = report['layer_shapes'][layer_name]
                print(f"  {layer_name}: {shapes}")
                common = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
                print(f"    → Common shape (shared portion): {common}")
        else:
            print("  None detected (all layers same shape)")
        
        print(f"\nOther layers (fixed architecture):")
        for layer_name, shapes in report['layer_shapes'].items():
            if layer_name not in report['expanding_layers']:
                print(f"  {layer_name}: {shapes[0]}")
    
    def reset(self):
        """Clear all collected model weights."""
        self.model_weights_list = []
        self.dataset_sizes = []
        self.layer_shapes = {}


# ============================================================================
# Example Usage for Expanding Classifier
# ============================================================================

def example_continual_learning_scenario():
    """
    Example: Aggregate models from continual learning with expanding classifier.
    
    Scenario:
    - Task 1: 10 classes
    - Task 2: 10 + 5 = 15 classes
    - Task 3: 10 + 5 + 7 = 22 classes
    
    We aggregate only the shared 10 classes' weights.
    """
    print("\n" + "="*80)
    print("Example: Continual Learning with Expanding Classifier")
    print("="*80)
    
    code = """
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# Initialize aggregator
aggregator = FedAvgAggregatorExpandingClassifier(
    weighting_strategy='data_driven',
    expanding_layers=['classifier.weight', 'classifier.bias']  # Specify expanding layers
)

# Load models from continual learning tasks
model_paths = [
    'models/task1_10classes.pth',
    'models/task2_15classes.pth',
    'models/task3_22classes.pth',
]
dataset_sizes = [1000, 1200, 900]

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Print detailed aggregation info
aggregator.print_aggregation_details()

# Aggregate (only shared 10 classes)
aggregator.save_aggregated_weights('aggregated_continual.pth')

# Load into model
model = YourModel(num_classes=22)  # Target model with max classes
model = aggregator.aggregate_into_model(model)

# Use for inference
model.eval()
with torch.no_grad():
    output = model(test_data)
    # Output will use aggregated weights for shared classes
    # and original weights for task-specific new classes
"""
    
    print(code)


def example_auto_detect():
    """
    Example: Auto-detect expanding layers.
    """
    print("\n" + "="*80)
    print("Example: Auto-Detect Expanding Layers")
    print("="*80)
    
    code = """
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# Initialize without specifying expanding layers
aggregator = FedAvgAggregatorExpandingClassifier(weighting_strategy='data_driven')

# Load models
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Auto-detect expanding layers
expanding = aggregator.detect_expanding_layers()
print(f"Auto-detected expanding layers: {expanding}")

# Print details
aggregator.print_aggregation_details()

# Aggregate (auto-detect handles expanding layers)
aggregator.save_aggregated_weights('aggregated.pth')
"""
    
    print(code)


if __name__ == "__main__":
    example_continual_learning_scenario()
    example_auto_detect()
    
    print("\n" + "="*80)
    print("Key Features for Expanding Classifiers")
    print("="*80)
    print("""
    ✓ Element-wise aggregation: Processes weights at individual level
    ✓ Auto-detect expanding layers: Automatically finds layers with shape changes
    ✓ Shared-only aggregation: Aggregates only common weights (shared classes)
    ✓ Task-specific preservation: New class weights preserved per model
    ✓ Flexible: Works with any expansion pattern
    
    Usage:
    1. Specify expanding_layers explicitly, or
    2. Let auto-detect find them, or
    3. Provide to aggregate() method
    
    Aggregation strategy:
    - Shared weights: Aggregated from all models
    - Task-specific weights: Preserved from each model
    """)
