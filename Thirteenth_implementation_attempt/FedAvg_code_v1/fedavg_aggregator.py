"""
Federated Averaging (FedAvg) Algorithm for Local Weight Aggregation

This module implements the FedAvg aggregation strategy to combine weights from
multiple locally trained models into a single aggregated model without requiring
server-client communication.

FedAvg Formula:
    w_t = Σ (n_i / n) * w_i
    
where:
    - w_i: weights from model i
    - n_i: number of training samples used for model i
    - n: total number of training samples across all models
    - w_t: aggregated weights at time t
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Union, Optional
import numpy as np


class FedAvgAggregator:
    """
    Aggregates model weights using the Federated Averaging algorithm.
    
    This implementation performs local weight aggregation without any
    server infrastructure, suitable for combining multiple locally trained
    models (e.g., trained with Learning without Forgetting).
    """
    
    def __init__(self, weighting_strategy: str = 'uniform'):
        """
        Initialize the FedAvg aggregator.
        
        Args:
            weighting_strategy (str): How to weight models during aggregation.
                'uniform': Equal weight for all models
                'data_driven': Weight by dataset size (requires dataset_sizes)
        """
        self.weighting_strategy = weighting_strategy
        self.model_weights = []
        self.dataset_sizes = []
        
    def add_model_weights(
        self, 
        model_weights: Dict[str, torch.Tensor],
        dataset_size: Optional[int] = None
    ):
        """
        Add model weights to the aggregation pool.
        
        Args:
            model_weights (Dict[str, torch.Tensor]): State dict of model weights
            dataset_size (int, optional): Number of samples used to train this model.
                Required if using 'data_driven' weighting strategy.
        """
        self.model_weights.append(model_weights)
        
        if self.weighting_strategy == 'data_driven':
            if dataset_size is None:
                raise ValueError(
                    "dataset_size required for 'data_driven' weighting strategy"
                )
            self.dataset_sizes.append(dataset_size)
        else:
            self.dataset_sizes.append(1)  # Equal weighting
    
    '''def load_model_weights_from_files(
        self,
        weight_paths: List[Union[str, Path]],
        dataset_sizes: Optional[List[int]] = None
    ):
        """
        Load model weights from saved checkpoint files.
        
        Args:
            weight_paths (List[str or Path]): Paths to saved model weights
            dataset_sizes (List[int], optional): Dataset sizes for each model.
                Required for 'data_driven' weighting.
        
        Example:
            >>> aggregator = FedAvgAggregator(weighting_strategy='data_driven')
            >>> paths = ['model_1.pth', 'model_2.pth', 'model_3.pth']
            >>> sizes = [1000, 1200, 900]  # training samples per model
            >>> aggregator.load_model_weights_from_files(paths, sizes)
        """
        if dataset_sizes is not None:
            if len(dataset_sizes) != len(weight_paths):
                raise ValueError(
                    "Length of dataset_sizes must match length of weight_paths"
                )
        else:
            dataset_sizes = [1] * len(weight_paths)
        
        for path, size in zip(weight_paths, dataset_sizes):
            weights = torch.load(path, map_location='cpu')
            self.add_model_weights(weights, size)'''
    
    def _compute_aggregation_weights(self) -> np.ndarray:
        """
        Compute the weight coefficients for each model.
        
        Returns:
            np.ndarray: Normalized weights for aggregation
        """
        if self.weighting_strategy == 'uniform':
            n_models = len(self.model_weights)
            return np.ones(n_models) / n_models
        
        elif self.weighting_strategy == 'data_driven':
            dataset_sizes = np.array(self.dataset_sizes, dtype=np.float32)
            total_samples = dataset_sizes.sum()
            return dataset_sizes / total_samples
        
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.weighting_strategy}"
            )
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Perform FedAvg aggregation on the collected model weights.
        
        Returns:
            Dict[str, torch.Tensor]: Aggregated model weights following FedAvg
        
        Raises:
            ValueError: If no model weights have been added
        """
        if not self.model_weights:
            raise ValueError("No model weights to aggregate. Add weights first.")
        
        # Compute aggregation weights
        agg_weights = self._compute_aggregation_weights()
        
        # Initialize aggregated weights with first model's structure
        aggregated_state = {}
        first_model = self.model_weights[0]
        
        # Aggregate each parameter
        for param_name in first_model.keys():
            # Stack all parameter values across models
            '''param_stack = torch.stack([
                model[param_name].float()
                for model in self.model_weights
            ])'''
            #average_linear_weight = torch.stack(weights["linear.weight"])
            param_stack = torch.stack([
                model[param_name].float()
                for model in self.model_weights
            ])
            
            # Apply weighted average
            agg_weights_tensor = torch.tensor(
                agg_weights, 
                dtype=torch.float32,
                device=param_stack.device
            ).view(-1, *([1] * (param_stack.dim() - 1)))
            
            aggregated_state[param_name] = (
                param_stack * agg_weights_tensor
            ).sum(dim=0)
        
        return aggregated_state
    
    def aggregate_into_model(self, model: nn.Module) -> nn.Module:
        """
        Aggregate weights and load them directly into a model.
        
        Args:
            model (nn.Module): Target model to load aggregated weights
        
        Returns:
            nn.Module: Model with aggregated weights loaded
        
        Example:
            >>> aggregator = FedAvgAggregator()
            >>> aggregator.load_model_weights_from_files([...])
            >>> aggregated_model = aggregator.aggregate_into_model(your_model)
        """
        aggregated_weights = self.aggregate()
        model.load_state_dict(aggregated_weights)
        return model
    
    def save_aggregated_weights(
        self,
        save_path: Union[str, Path]
    ):
        """
        Aggregate weights and save to file.
        
        Args:
            save_path (str or Path): Path where to save aggregated weights
        """
        aggregated_weights = self.aggregate()
        torch.save(aggregated_weights, save_path)
        print(f"Aggregated weights saved to {save_path}")
    
    def get_aggregation_report(self) -> Dict:
        """
        Generate a report of the aggregation process.
        
        Returns:
            Dict: Aggregation details and statistics
        """
        agg_weights = self._compute_aggregation_weights()
        
        report = {
            'num_models': len(self.model_weights),
            'weighting_strategy': self.weighting_strategy,
            'model_weights': agg_weights.tolist(),
            'total_samples': sum(self.dataset_sizes),
            'per_model_samples': self.dataset_sizes,
        }
        
        return report
    
    def reset(self):
        """Clear all collected model weights."""
        self.model_weights = []
        self.dataset_sizes = []


# ============================================================================
# Example Usage Functions
# ============================================================================

def example_uniform_aggregation():
    """
    Example: Aggregate weights with uniform weighting (each model contributes equally).
    """
    print("\n" + "="*60)
    print("Example 1: Uniform Weighting (Equal contribution from all models)")
    print("="*60)
    
    # Initialize aggregator with uniform weighting
    aggregator = FedAvgAggregator(weighting_strategy='uniform')
    
    # Simulate loading weights from 3 models
    model_paths = ['model_1_lwf.pth', 'model_2_lwf.pth', 'model_3_lwf.pth']
    
    # In practice, you would load actual model weights:
    # aggregator.load_model_weights_from_files(model_paths)
    
    print("Aggregation strategy: Uniform")
    print("Each model contributes 1/n to the final weights")
    print("Formula: w_aggregated = (1/3) * w_1 + (1/3) * w_2 + (1/3) * w_3")


def example_data_driven_aggregation():
    """
    Example: Aggregate weights with data-driven weighting (models trained on different dataset sizes).
    """
    print("\n" + "="*60)
    print("Example 2: Data-Driven Weighting (Weighted by dataset size)")
    print("="*60)
    
    # Initialize aggregator with data-driven weighting
    aggregator = FedAvgAggregator(weighting_strategy='data_driven')
    
    # Model training information
    model_configs = [
        ('model_task1_lwf.pth', 1000),  # Model 1: trained on 1000 samples
        ('model_task2_lwf.pth', 1500),  # Model 2: trained on 1500 samples
        ('model_task3_lwf.pth', 800),   # Model 3: trained on 800 samples
    ]
    
    paths = [config[0] for config in model_configs]
    sizes = [config[1] for config in model_configs]
    
    total = sum(sizes)
    weights = [s/total for s in sizes]
    
    print("Aggregation strategy: Data-Driven")
    print(f"Model 1: {sizes[0]} samples ({weights[0]:.1%} weight)")
    print(f"Model 2: {sizes[1]} samples ({weights[1]:.1%} weight)")
    print(f"Model 3: {sizes[2]} samples ({weights[2]:.1%} weight)")
    print(f"Total: {total} samples")
    
    # In practice:
    # aggregator.load_model_weights_from_files(paths, sizes)
    # aggregated_weights = aggregator.aggregate()


def complete_workflow_example():
    """
    Complete workflow: Load models, aggregate weights, and use aggregated model.
    """
    print("\n" + "="*60)
    print("Complete Workflow Example")
    print("="*60)
    
    code_example = """
# Step 1: Initialize aggregator
aggregator = FedAvgAggregator(weighting_strategy='data_driven')

# Step 2: Load model weights from training tasks
model_paths = [
    'checkpoints/task1_lwf_model.pth',
    'checkpoints/task2_lwf_model.pth',
    'checkpoints/task3_lwf_model.pth',
]
dataset_sizes = [1000, 1200, 900]  # samples per task

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Step 3: Generate aggregation report
report = aggregator.get_aggregation_report()
print("Aggregation Report:")
print(f"  Models aggregated: {report['num_models']}")
print(f"  Total training samples: {report['total_samples']}")
print(f"  Model weights: {report['model_weights']}")

# Step 4: Create aggregated model and load weights
aggregated_model = YourModelClass()
aggregated_model = aggregator.aggregate_into_model(aggregated_model)

# Step 5: Save aggregated weights
aggregator.save_aggregated_weights('aggregated_model_fedavg.pth')

# Step 6: Use aggregated model for inference
aggregated_model.eval()
with torch.no_grad():
    predictions = aggregated_model(test_data)
"""
    
    print(code_example)


if __name__ == "__main__":
    example_uniform_aggregation()
    example_data_driven_aggregation()
    complete_workflow_example()
