"""
LwFTrainer Integration Example
==============================

Shows how to integrate the GradientFreezeScheduler into your existing LwFTrainer
to test different gradient freezing strategies.

This example demonstrates:
1. How to modify the training loop
2. How to create experiment configurations
3. How to log and compare results
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from avalanche.training.templates import SupervisedTemplate

from Twelfth_implementation_attempt.GradientFreezeScheduler_v1.GradientFreezeScheduler import (
    GradientFreezeScheduler,
    apply_gradient_freezing
)
from Twelfth_implementation_attempt.Nature_specialised_model_v12.Train_ResNet20_Nature_LwF_v12 import lwf_trainer


class LwFTrainer_Experimental:
    """
    Extended version of LwFTrainer that supports gradient freeze strategies.
    
    This is a minimal example showing integration points.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        freeze_strategy: str = 'binary',
        freeze_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize with gradient freeze strategy support.
        
        Args:
            model: The neural network
            device: Device to use
            freeze_strategy: Strategy from GradientFreezeScheduler.FreezeStrategy
            freeze_config: Strategy-specific configuration
        """
        self.model = model
        self.device = torch.device(device)
        
        # Gradient freezing configuration
        self.freeze_strategy = freeze_strategy
        self.freeze_config = freeze_config or {}
        self.freeze_scheduler: Optional[GradientFreezeScheduler] = None
        
        # Tracking for analysis
        self.gradient_freeze_logs = {
            'epoch': [],
            'freeze_config': [],
            'loss': [],
            'accuracy': []
        }
        
        # These would be initialized from parent LwFTrainer
        self.train_epochs = kwargs.get('train_epochs', 10)
        self.ft_shared_after = kwargs.get('ft_shared_after', 8)
        self.old_classes = set()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_experience(self, experience, eval_streams=None, **kwargs):
        """
        Modified training loop with gradient freeze scheduler support.
        
        Shows key integration points marked with [INTEGRATION POINT].
        """
        # ===== [INTEGRATION POINT 1] Initialize scheduler for this experience =====
        self.freeze_scheduler = GradientFreezeScheduler(
            freeze_strategy=self.freeze_strategy,
            total_epochs=self.train_epochs,
            ft_shared_after=self.ft_shared_after,
            config=self.freeze_config,
            verbose=True  # Set to False in production
        )
        
        # Show schedule (for debugging)
        if self.freeze_strategy != 'binary':
            print(self.freeze_scheduler.get_schedule_visualization())
        
        # Training loop (simplified version of your actual code)
        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            for batch_idx, (x, y) in enumerate(self.get_batches(experience)):
                # Forward pass
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # ===== [INTEGRATION POINT 2] Apply gradient freezing =====
                if self.freeze_scheduler is not None:
                    freeze_config = self.freeze_scheduler.get_freeze_config(
                        current_epoch=epoch,
                        loss=loss.item()
                    )
                    
                    # Apply the freezing strategy
                    apply_gradient_freezing(
                        model=self.model,
                        freeze_config=freeze_config,
                        old_classes=self.old_classes,
                        device=self.device,
                        verbose=False  # Set to True to see per-batch freezing
                    )
                    
                    # Log the freeze configuration for analysis
                    self.gradient_freeze_logs['epoch'].append(epoch)
                    self.gradient_freeze_logs['freeze_config'].append(freeze_config)
                    self.gradient_freeze_logs['loss'].append(loss.item())
                else:
                    # Fallback to original behavior (binary freeze)
                    self._original_gradient_freezing()
                
                # Optimizer step
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                with torch.no_grad():
                    acc = (logits.argmax(1) == y).float().mean()
                    epoch_acc += acc.item()
                num_batches += 1
            
            # Log epoch results
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            print(f"Epoch {epoch:2d}/{self.train_epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    
    def _original_gradient_freezing(self):
        """
        Original gradient freezing logic from your LwFTrainer.
        This is the baseline for comparison.
        """
        # This is your original code from lines 828-872
        # (included here for reference)
        pass
    
    def get_batches(self, experience):
        """Dummy batch generator for example."""
        # This would iterate through actual data
        yield torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
    
    def get_gradient_statistics(self) -> Dict:
        """
        Analyze gradient statistics to understand freezing effectiveness.
        
        Returns:
            Dictionary with statistics about gradient changes
        """
        stats = {
            'freeze_intensity_by_epoch': [],
            'gradient_norms_by_epoch': [],
            'loss_trajectory': []
        }
        
        # Extract freeze intensities
        for config in self.gradient_freeze_logs['freeze_config']:
            stats['freeze_intensity_by_epoch'].append({
                'shared': config.shared_layer_intensity,
                'classifier': config.old_classifier_intensity
            })
        
        stats['loss_trajectory'] = self.gradient_freeze_logs['loss']
        
        return stats


# ==================== EXPERIMENT MANAGER ====================

class ExperimentManager:
    """
    Manages running multiple experiments with different freeze strategies.
    
    Helps compare performance across configurations.
    """
    
    def __init__(self, trainer, device: str = "cpu"):#def __init__(self, model_factory, device: str = "cpu"):
        """
        Initialize experiment manager.
        
        Args:
            model_factory: Callable that returns a fresh model instance
            device: Device to use for training
        """
        self.trainer:SupervisedTemplate
        #self.model_factory = model_factory
        self.device = device
        self.results = {}
    
    def run_experiment(
        self,
        exp_name: str,
        freeze_strategy: str,
        freeze_config: Optional[Dict] = None,
        train_epochs: int = 10,
        ft_shared_after: int = 8,
    ) -> Dict:
        """
        Run a single experiment with given configuration.
        
        Args:
            exp_name: Name for this experiment
            freeze_strategy: Strategy name
            freeze_config: Strategy-specific configuration
            train_epochs: Number of training epochs
            ft_shared_after: Baseline threshold for freezing
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*80}")
        print(f"Running Experiment: {exp_name}")
        print(f"freeze_strategy: {freeze_strategy}")
        print(f"Config: {freeze_config}")
        print(f"{'='*80}\n")
        
        # Create fresh model
        #model = self.trainer.model #self.model_factory()
        
        # Create trainer with specified strategy
        '''trainer = LwFTrainer_Experimental(
            model=model,
            device=self.device,
            freeze_strategy=freeze_strategy,
            freeze_config=freeze_config,
            train_epochs=train_epochs,
            ft_shared_after=ft_shared_after
        )'''
        trainer = self.trainer
        
        # Run training (placeholder - actual implementation would use real data)
        # trainer.train_experience(experience)
        
        # Extract results
        results = {
            'experiment_name': exp_name,
            'freeze_strategy': freeze_strategy,
            'config': freeze_config,
            'freeze_logs': trainer.gradient_freeze_logs,
            'gradient_stats': trainer.get_gradient_statistics()
        }
        
        self.results[exp_name] = results
        return results
    
    def compare_experiments(self) -> str:
        """
        Generate comparison summary of all experiments.
        
        Returns:
            String with formatted comparison results
        """
        lines = ["\n" + "="*80]
        lines.append("EXPERIMENT COMPARISON SUMMARY")
        lines.append("="*80 + "\n")
        
        lines.append("Experiment Name          | Strategy                | "
                     "Avg Freeze Intensity | Recommendation")
        lines.append("-" * 105)
        
        for exp_name, result in self.results.items():
            stats = result['gradient_stats']
            freeze_intensities = [
                config['shared'] for config in stats['freeze_intensity_by_epoch']
            ]
            avg_intensity = sum(freeze_intensities) / len(freeze_intensities)
            
            lines.append(
                f"{exp_name:24s} | {result['freeze_strategy']:23s} | "
                f"{avg_intensity:19.2f} | -"
            )
        
        lines.append("\n" + "="*80 + "\n")
        return "\n".join(lines)


# ==================== EXAMPLE CONFIGURATIONS ====================

def get_experiment_configs() -> List[tuple]:
    """
    Return list of experiment configurations to test.
    
    Each tuple is (exp_name, freeze_strategy, config_dict)
    
    This is where you define different stability-plasticity tradeoffs.
    """
    return [
        # Baseline: Original binary freeze freeze_strategy
        ('baseline_binary', 'binary', {
            'freeze_before': 8
        }),
        
        # Experiment 1: Gradual unfreezing
        ('exp_gradual_early_unfreeze', 'gradual_unfreezing', {
            'start_epoch': 0,
            'end_epoch': 8,
            'unfreeze_shared_first': True
        }),
        
        ('exp_gradual_late_unfreeze', 'gradual_unfreezing', {
            'start_epoch': 2,
            'end_epoch': 10,
            'unfreeze_shared_first': False
        }),
        
        # Experiment 2: Two-phase freeze_strategy
        ('exp_two_phase_tight3', 'two_phase', {
            'tight_freeze_epochs': 3,
            'gradual_unfreeze_epochs': 5
        }),
        
        ('exp_two_phase_tight5', 'two_phase', {
            'tight_freeze_epochs': 5,
            'gradual_unfreeze_epochs': 3
        }),
        
        # Experiment 3: Exponential decay
        ('exp_exponential_fast', 'exponential_decay', {
            'decay_rate': 0.3,
            'half_life_epochs': 2.0
        }),
        
        ('exp_exponential_slow', 'exponential_decay', {
            'decay_rate': 0.7,
            'half_life_epochs': 4.0
        }),
        
        # Experiment 4: Layer-wise curriculum
        ('exp_layer_wise_3stages', 'layer_wise_curriculum', {
            'num_stages': 3,
            'freeze_order': 'deep_to_shallow'
        }),
        
        # Experiment 5: Classifier-only freezing
        ('exp_classifier_only', 'classifier_only', {
            'freeze_classifier': True,
            'freeze_shared': False
        }),
        
        # Experiment 6: Cyclic freezing
        ('exp_cyclic_period2', 'cyclic', {
            'cycle_length': 2
        }),
        
        ('exp_cyclic_period3', 'cyclic', {
            'cycle_length': 3
        }),
    ]


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LwFTrainer Gradient Freeze Strategy Testing")
    print("="*80 + "\n")
    
    # Create a simple model factory
    '''def model_factory():
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )'''
    
    # Initialize experiment manager
    manager = ExperimentManager(lwf_trainer, device="cpu")#manager = ExperimentManager(model_factory, device="cpu")
    
    # Get experiment configurations
    configs = get_experiment_configs()
    
    print(f"Running {len(configs)} experiments...\n")
    
    # Run all experiments
    for exp_name, freeze_strategy, config in configs:
        try:
            manager.run_experiment(
                exp_name=exp_name,
                freeze_strategy=freeze_strategy,
                freeze_config=config,
                train_epochs=10,
                ft_shared_after=8
            )
        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")
    
    # Print comparison
    print(manager.compare_experiments())
    
    print("\nNext Steps:")
    print("-" * 80)
    print("1. Implement actual data loading in LwFTrainer_Experimental")
    print("2. Run experiments on your Nature dataset")
    print("3. Track BWT (Backward Transfer), FWT (Forward Transfer), and AvgAcc")
    print("4. Identify best freeze_strategy for your problem")
    print("5. Fine-tune hyperparameters of best freeze_strategy")
    print("6. Validate on test set")
