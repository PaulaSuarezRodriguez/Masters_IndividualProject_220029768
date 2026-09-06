"""
Gradient Freeze Scheduler for LwFTrainer
=========================================

A flexible framework to test different gradient freezing strategies to optimize
the stability-plasticity tradeoff in Class-Incremental Learning.

Usage:
------
scheduler = GradientFreezeScheduler(
    freeze_strategy='gradual_unfreezing',
    total_epochs=10,
    ft_shared_after=8,
    config={'initial_intensity': 1.0, 'final_intensity': 0.0}
)

# In training loop after loss.backward():
freeze_config = scheduler.get_freeze_config(current_epoch)
if freeze_config['should_freeze']:
    apply_gradient_freezing(model, freeze_config)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import warnings


@dataclass
class FreezeConfig:
    """Configuration for gradient freezing at a specific epoch."""
    should_freeze: bool
    shared_layer_intensity: float  # 1.0 = full freeze, 0.0 = no freeze
    old_classifier_intensity: float  # 1.0 = full freeze, 0.0 = no freeze
    layers_to_freeze: Optional[List[str]] = None  # Specific layer names
    freeze_bias_only: bool = False  # Only freeze bias terms
    freeze_weights_only: bool = False  # Only freeze weights
    
    def __repr__(self):
        return (f"FreezeConfig(epoch_freeze={self.should_freeze}, "
                f"shared={self.shared_layer_intensity:.2f}, "
                f"old_classifier={self.old_classifier_intensity:.2f})")


class FreezeStrategy(Enum):
    """Available gradient freezing strategies."""
    BINARY = "binary"  # Original: full freeze or no freeze
    GRADUAL_UNFREEZING = "gradual_unfreezing"  # Linear decay of freeze intensity
    LAYER_WISE_CURRICULUM = "layer_wise_curriculum"  # Unfreeze by layer depth
    TWO_PHASE = "two_phase"  # Tight freeze then gradual release
    EXPONENTIAL_DECAY = "exponential_decay"  # Exponential unfreezing schedule
    CLASSIFIER_ONLY = "classifier_only"  # Only freeze classifier, unfreeze shared
    ADAPTIVE_BOOSTING = "adaptive_boosting"  # Increase intensity if new task is hard
    CYCLIC = "cyclic"  # Periodic freeze/unfreeze cycles


class GradientFreezeScheduler:
    """
    Main scheduler for managing gradient freezing strategies.
    
    Provides a unified interface to test different stability-plasticity tradeoffs.
    """
    
    def __init__(
        self,
        freeze_strategy: str = 'binary',
        total_epochs: int = 10,
        ft_shared_after: int = 8,
        config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize the gradient freeze scheduler.
        
        Args:
            freeze_strategy: Name of the freezing freeze_strategy (see FreezeStrategy enum)
            total_epochs: Total number of training epochs per experience
            ft_shared_after: Default threshold for standard freezing behavior
            config: Strategy-specific configuration dictionary
            verbose: Print debug information
        """
        self.freeze_strategy = FreezeStrategy(freeze_strategy)
        self.total_epochs = total_epochs
        self.ft_shared_after = ft_shared_after
        self.config = config or {}
        self.verbose = verbose
        
        # Validate and set default configs
        self._validate_and_set_defaults()
        
        # For adaptive strategies
        self.epoch_losses = []
        self.loss_threshold = None
        
    def _validate_and_set_defaults(self):
        """Validate config and set freeze_strategy-specific defaults."""
        defaults = {
            FreezeStrategy.BINARY: {
                'freeze_before': self.ft_shared_after
            },
            FreezeStrategy.GRADUAL_UNFREEZING: {
                'start_epoch': 0,
                'end_epoch': self.ft_shared_after,
                'unfreeze_shared_first': True
            },
            FreezeStrategy.LAYER_WISE_CURRICULUM: {
                'num_stages': 3,
                'freeze_order': 'deep_to_shallow'  # Unfreeze from deepest layers
            },
            FreezeStrategy.TWO_PHASE: {
                'tight_freeze_epochs': 3,
                'gradual_unfreeze_epochs': self.ft_shared_after - 3,
                'shared_layer_intensity': 1.0
            },
            FreezeStrategy.EXPONENTIAL_DECAY: {
                'decay_rate': 0.5,
                'half_life_epochs': 3
            },
            FreezeStrategy.CLASSIFIER_ONLY: {
                'freeze_classifier': True,
                'freeze_shared': False
            },
            FreezeStrategy.ADAPTIVE_BOOSTING: {
                'base_intensity': 0.5,
                'loss_threshold': None  # Set dynamically
            },
            FreezeStrategy.CYCLIC: {
                'cycle_length': 2,  # Freeze every N epochs
                'freeze_probability': 0.5
            }
        }
        
        # Set defaults for this freeze_strategy
        freeze_strategy_defaults = defaults.get(self.freeze_strategy, {})
        for key, value in freeze_strategy_defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def get_freeze_config(self, current_epoch: int, loss: Optional[float] = None) -> FreezeConfig:
        """
        Get the freeze configuration for the current epoch.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            loss: Current batch loss (optional, used by some strategies)
            
        Returns:
            FreezeConfig object specifying what to freeze
        """
        if self.freeze_strategy == FreezeStrategy.BINARY:
            return self._binary_freeze(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.GRADUAL_UNFREEZING:
            return self._gradual_unfreezing(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.LAYER_WISE_CURRICULUM:
            return self._layer_wise_curriculum(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.TWO_PHASE:
            return self._two_phase(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.EXPONENTIAL_DECAY:
            return self._exponential_decay(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.CLASSIFIER_ONLY:
            return self._classifier_only(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.ADAPTIVE_BOOSTING:
            return self._adaptive_boosting(current_epoch, loss)
        elif self.freeze_strategy == FreezeStrategy.CYCLIC:
            return self._cyclic(current_epoch)
        else:
            raise ValueError(f"Unknown freeze_strategy: {self.freeze_strategy}")
    
    # ======================== Strategy Implementations ========================
    
    def _binary_freeze(self, epoch: int) -> FreezeConfig:
        """Original freeze_strategy: freeze until ft_shared_after, then fully trainable."""
        freeze_until = self.config.get('freeze_before', self.ft_shared_after)
        should_freeze = epoch <= freeze_until
        
        return FreezeConfig(
            should_freeze=should_freeze,
            shared_layer_intensity=1.0 if should_freeze else 0.0,
            old_classifier_intensity=1.0 if should_freeze else 0.0
        )
    
    def _gradual_unfreezing(self, epoch: int) -> FreezeConfig:
        """
        Linearly decrease freeze intensity over time.
        
        Epochs 0-N: gradually reduce from full freeze (1.0) to no freeze (0.0)
        """
        start = self.config.get('start_epoch', 0)
        end = self.config.get('end_epoch', self.ft_shared_after)
        unfreeze_shared_first = self.config.get('unfreeze_shared_first', True)
        
        if epoch < start:
            return FreezeConfig(
                should_freeze=True,
                shared_layer_intensity=1.0,
                old_classifier_intensity=1.0
            )
        elif epoch >= end:
            return FreezeConfig(
                should_freeze=False,
                shared_layer_intensity=0.0,
                old_classifier_intensity=0.0
            )
        else:
            # Linear interpolation
            progress = (epoch - start) / (end - start)  # 0 to 1
            
            if unfreeze_shared_first:
                # Unfreeze shared layers first, classifier rows last
                shared_intensity = max(0.0, 1.0 - progress * 1.5)
                classifier_intensity = max(0.0, 1.0 - progress * 0.5)
            else:
                # Unfreeze equally
                intensity = 1.0 - progress
                shared_intensity = intensity
                classifier_intensity = intensity
            
            return FreezeConfig(
                should_freeze=True,
                shared_layer_intensity=shared_intensity,
                old_classifier_intensity=classifier_intensity
            )
    
    def _layer_wise_curriculum(self, epoch: int) -> FreezeConfig:
        """
        Unfreeze layers in stages from deep to shallow.
        
        Deep layers (feature extraction): keep frozen longer
        Shallow layers (task adaptation): unfreeze earlier
        """
        num_stages = self.config.get('num_stages', 3)
        freeze_order = self.config.get('freeze_order', 'deep_to_shallow')
        
        stage = min(epoch // (self.total_epochs // num_stages), num_stages - 1)
        intensity = max(0.0, 1.0 - (stage / (num_stages - 1)))
        
        # For now, uniform intensity (could be expanded to track individual layers)
        layers_to_freeze = [f"stage_{i}" for i in range(num_stages - stage)]
        
        return FreezeConfig(
            should_freeze=intensity > 0,
            shared_layer_intensity=intensity,
            old_classifier_intensity=1.0,  # Keep classifier rows frozen longer
            layers_to_freeze=layers_to_freeze
        )
    
    def _two_phase(self, epoch: int) -> FreezeConfig:
        """
        Two-phase freezing: tight freeze then gradual release.
        
        Phase 1 (first N epochs): maximum freeze (stability)
        Phase 2 (remaining epochs): gradual unfreeze (plasticity)
        """
        tight_freeze_epochs = self.config.get('tight_freeze_epochs', 3)
        
        if epoch < tight_freeze_epochs:
            # Tight freeze phase
            return FreezeConfig(
                should_freeze=True,
                shared_layer_intensity=1.0,
                old_classifier_intensity=1.0
            )
        else:
            # Gradual unfreeze phase
            remaining_epochs = self.total_epochs - tight_freeze_epochs
            progress = (epoch - tight_freeze_epochs) / remaining_epochs
            intensity = max(0.0, 1.0 - progress)
            
            return FreezeConfig(
                should_freeze=intensity > 0,
                shared_layer_intensity=intensity,
                old_classifier_intensity=intensity * 0.5  # Classifier unfreezes faster
            )
    
    def _exponential_decay(self, epoch: int) -> FreezeConfig:
        """
        Exponential decay of freeze intensity: intensity = decay_rate^(epoch / half_life)
        """
        decay_rate = self.config.get('decay_rate', 0.5)
        half_life = self.config.get('half_life_epochs', 3)
        
        # Exponential: intensity = decay_rate^(epoch / half_life)
        intensity = decay_rate ** (epoch / half_life)
        
        return FreezeConfig(
            should_freeze=intensity > 0.01,  # Stop freezing when very small
            shared_layer_intensity=intensity,
            old_classifier_intensity=intensity ** 1.5  # Classifier decays faster
        )
    
    def _classifier_only(self, epoch: int) -> FreezeConfig:
        """Only freeze classifier rows, allow shared layers to train immediately."""
        freeze_classifier = self.config.get('freeze_classifier', True)
        freeze_shared = self.config.get('freeze_shared', False)
        
        freeze_until = self.ft_shared_after
        should_freeze = epoch <= freeze_until
        
        return FreezeConfig(
            should_freeze=should_freeze,
            shared_layer_intensity=1.0 if (should_freeze and freeze_shared) else 0.0,
            old_classifier_intensity=1.0 if (should_freeze and freeze_classifier) else 0.0
        )
    
    def _adaptive_boosting(self, epoch: int, loss: Optional[float] = None) -> FreezeConfig:
        """
        Adaptively adjust freeze intensity based on loss trajectory.
        
        If loss increases (overfitting), increase freezing to stabilize.
        If loss decreases steadily, gradually reduce freezing.
        """
        base_intensity = self.config.get('base_intensity', 0.5)
        
        if loss is not None:
            self.epoch_losses.append(loss)
        
        # Simple heuristic: if loss not improving, increase freeze intensity
        if len(self.epoch_losses) > 2:
            recent_loss = self.epoch_losses[-1]
            prev_loss = self.epoch_losses[-2]
            
            if recent_loss > prev_loss:
                # Loss increased, boost freezing
                intensity = base_intensity + 0.2
            else:
                # Loss decreased, reduce freezing
                intensity = max(0.0, base_intensity - (epoch / self.total_epochs) * 0.3)
        else:
            intensity = base_intensity
        
        return FreezeConfig(
            should_freeze=intensity > 0.1,
            shared_layer_intensity=min(1.0, intensity),
            old_classifier_intensity=min(1.0, intensity * 1.2)  # Keep classifier more stable
        )
    
    def _cyclic(self, epoch: int) -> FreezeConfig:
        """
        Cyclic freezing: alternate between frozen and unfrozen epochs.
        
        Periodically unfreeze to allow knowledge transfer from shared features.
        """
        cycle_length = self.config.get('cycle_length', 2)
        position_in_cycle = epoch % cycle_length
        
        # Freeze on first position of cycle, unfreeze on others
        should_freeze = position_in_cycle == 0
        
        return FreezeConfig(
            should_freeze=should_freeze,
            shared_layer_intensity=1.0 if should_freeze else 0.0,
            old_classifier_intensity=1.0 if should_freeze else 0.0
        )
    
    # ======================== Helper Methods ========================
    
    def get_schedule_visualization(self) -> str:
        """Generate a text visualization of the freeze schedule."""
        lines = [f"\n{self.freeze_strategy.value.upper()} Schedule (total_epochs={self.total_epochs}):\n"]
        lines.append("Epoch | Shared Intensity | Classifier Intensity")
        lines.append("------|-----------------|---------------------")
        
        for epoch in range(self.total_epochs):
            config = self.get_freeze_config(epoch)
            lines.append(
                f"{epoch:5d} | {config.shared_layer_intensity:>16.2f} | "
                f"{config.old_classifier_intensity:>20.2f}"
            )
        
        return "\n".join(lines)


# ======================== Integration with LwFTrainer ========================

def apply_gradient_freezing(
    model: nn.Module,
    freeze_config: FreezeConfig,
    old_classes: Set[int],
    device: torch.device = torch.device('cpu'),
    verbose: bool = False
):
    """
    Apply gradient freezing to model parameters based on freeze_config.
    
    This is the function to call AFTER loss.backward() in the training loop.
    
    Args:
        model: The neural network model
        freeze_config: FreezeConfig object specifying what to freeze
        old_classes: Set of class indices from previous experiences
        device: Torch device
        verbose: Print debug information
    """
    if not freeze_config.should_freeze:
        print("freeze_config.should_freeze NOT defined. No freezing will be applied.")
        return  # No freezing needed
    
    # Freeze shared layers
    if freeze_config.shared_layer_intensity > 0:
        freeze_shared_layers(
            model,
            intensity=freeze_config.shared_layer_intensity,
            specific_layers=freeze_config.layers_to_freeze,
            freeze_bias_only=freeze_config.freeze_bias_only,
            freeze_weights_only=freeze_config.freeze_weights_only,
            verbose=verbose
        )
    
    # Freeze old classifier rows
    if freeze_config.old_classifier_intensity > 0 and len(old_classes) > 0:
        freeze_classifier_rows(
            model,
            old_classes=old_classes,
            intensity=freeze_config.old_classifier_intensity,
            freeze_bias_only=freeze_config.freeze_bias_only,
            freeze_weights_only=freeze_config.freeze_weights_only,
            verbose=verbose
        )


def freeze_shared_layers(
    model: nn.Module,
    intensity: float = 1.0,
    specific_layers: Optional[List[str]] = None,
    freeze_bias_only: bool = False,
    freeze_weights_only: bool = False,
    verbose: bool = False
):
    """
    Freeze gradients of shared layers (all Conv2d and Linear except classifier).
    
    Args:
        model: The model
        intensity: Freeze intensity (1.0 = full freeze, 0.5 = half magnitude, etc.)
        specific_layers: Only freeze layers with names containing these strings
        freeze_bias_only: Only freeze bias terms
        freeze_weights_only: Only freeze weight terms
        verbose: Print debug information
    """
    if intensity <= 0:
        return
    
    num_frozen = 0
    for name, module in model.named_modules():
        # Skip classifier layer
        if hasattr(model, 'linear') and module is model.linear:
            continue
        
        # Filter by layer type
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        
        # Filter by specific layer names if provided
        if specific_layers and not any(layer in name for layer in specific_layers):
            continue
        
        # Apply freezing to weights
        if not freeze_bias_only and module.weight.grad is not None:
            module.weight.grad.mul_(1.0 - intensity)  # Scale gradient
            num_frozen += 1
        
        # Apply freezing to bias
        if hasattr(module, 'bias') and module.bias is not None and not freeze_weights_only:
            if module.bias.grad is not None:
                module.bias.grad.mul_(1.0 - intensity)
                num_frozen += 1
    
    if verbose:
        print(f"  [Freeze Shared] Reduced gradients for {num_frozen} parameters "
              f"(intensity={intensity:.2f})")


def freeze_classifier_rows(
    model: nn.Module,
    old_classes: Set[int],
    intensity: float = 1.0,
    freeze_bias_only: bool = False,
    freeze_weights_only: bool = False,
    verbose: bool = False
):
    """
    Freeze gradients of old classifier rows.
    
    Args:
        model: The model
        old_classes: Set of old class indices
        intensity: Freeze intensity (1.0 = full freeze, 0.5 = half magnitude, etc.)
        freeze_bias_only: Only freeze bias terms
        freeze_weights_only: Only freeze weight terms
        verbose: Print debug information
    """
    if intensity <= 0 or len(old_classes) == 0:
        return
    
    classifier = model.linear
    old_num_classes = len(old_classes)
    
    # Freeze weight gradients for old classes
    if not freeze_bias_only and classifier.weight.grad is not None:
        classifier.weight.grad[:old_num_classes].mul_(1.0 - intensity)
    
    # Freeze bias gradients for old classes
    if not freeze_weights_only and classifier.bias is not None:
        if classifier.bias.grad is not None:
            classifier.bias.grad[:old_num_classes].mul_(1.0 - intensity)
    
    if verbose:
        print(f"  [Freeze Classifier] Reduced gradients for {old_num_classes} old classes "
              f"(intensity={intensity:.2f})")


# ======================== Example Usage ========================

if __name__ == "__main__":
    # Example 1: Compare different strategies
    print("=" * 80)
    print("GRADIENT FREEZE SCHEDULER - EXAMPLE USAGE")
    print("=" * 80)
    
    strategies_to_test = [
        ('binary', {}),
        ('gradual_unfreezing', {'start_epoch': 0, 'end_epoch': 8}),
        ('two_phase', {'tight_freeze_epochs': 3, 'gradual_unfreeze_epochs': 5}),
        ('exponential_decay', {'decay_rate': 0.6, 'half_life_epochs': 2.5}),
        ('layer_wise_curriculum', {'num_stages': 3}),
    ]
    
    for freeze_strategy_name, config in strategies_to_test:
        scheduler = GradientFreezeScheduler(
            freeze_strategy=freeze_strategy_name,
            total_epochs=10,
            ft_shared_after=8,
            config=config,
            verbose=False
        )
        print(scheduler.get_schedule_visualization())
        print("\n")
    
    # Example 2: Integration in training loop
    print("=" * 80)
    print("INTEGRATION EXAMPLE")
    print("=" * 80)
    print("""
# In your LwFTrainer.train_experience() method, after loss.backward():

scheduler = GradientFreezeScheduler(
    freeze_strategy='gradual_unfreezing',
    total_epochs=self.train_epochs,
    ft_shared_after=self.ft_shared_after,
    config={'start_epoch': 0, 'end_epoch': self.ft_shared_after}
)

for epoch in range(self.train_epochs):
    for batch in dataloader:
        # ... forward pass, compute loss
        loss.backward()
        
        # Apply gradient freezing
        freeze_config = scheduler.get_freeze_config(epoch)
        apply_gradient_freezing(
            model=self.model,
            freeze_config=freeze_config,
            old_classes=self.old_classes,
            device=self.device,
            verbose=True
        )
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    """)
