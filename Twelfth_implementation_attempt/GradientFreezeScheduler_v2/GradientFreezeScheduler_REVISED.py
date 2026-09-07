"""
REVISED GradientFreezeScheduler - With Delayed-Start Strategies
================================================================

CRITICAL UPDATE: Based on experimental evidence, the optimal strategy is NOT
"gradually unfreeze from Exp 0" but rather "START FREEZING LATER" (from Exp 2+).

This revised version includes:
1. Delayed-Start Freezing (new, RECOMMENDED)
2. Progressive Freezing (new, RECOMMENDED)
3. Original strategies (for reference, less recommended based on evidence)
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
    # NEW STRATEGIES - RECOMMENDED BASED ON EVIDENCE
    DELAYED_START = "delayed_start"  # Start freezing from Exp N
    PROGRESSIVE_FREEZING = "progressive_freezing"  # Ramp up freezing over experiences
    DELAYED_GRADUAL = "delayed_gradual"  # Delayed start + gradual unfreeze
    
    # Original strategies (for reference)
    BINARY = "binary"
    GRADUAL_UNFREEZING = "gradual_unfreezing"
    LAYER_WISE_CURRICULUM = "layer_wise_curriculum"
    TWO_PHASE = "two_phase"
    EXPONENTIAL_DECAY = "exponential_decay"


class GradientFreezeScheduler:
    """
    Gradient freeze scheduler with support for delayed-start strategies.
    
    KEY INSIGHT: Freezing from Experience 0 is harmful because:
    1. Features are overfit to first task
    2. Cannot adapt to new tasks' domains
    3. Creates contradictory optimization objectives
    4. Result: Catastrophic forgetting (33% drop observed)
    
    Solution: Delay freezing until model has seen multiple experiences
    and learned generalizable features.
    """
    
    def __init__(
        self,
        freeze_strategy: str = 'delayed_start',
        total_epochs: int = 10,
        ft_shared_after: int = 8,
        config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize the gradient freeze scheduler.
        
        Args:
            strategy: Name of the freezing strategy
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
        
        self._validate_and_set_defaults()
    
    def _validate_and_set_defaults(self):
        """Validate config and set strategy-specific defaults."""
        defaults = {
            # NEW STRATEGIES
            FreezeStrategy.DELAYED_START: {
                'freeze_start_experience': 2,  # Start freezing from Exp 2
            },
            FreezeStrategy.PROGRESSIVE_FREEZING: {
                'ramp_up_experiences': 4,  # Reach full freeze by Exp 4
            },
            FreezeStrategy.DELAYED_GRADUAL: {
                'freeze_start_experience': 2,
                'unfreeze_shared_first': True,
            },
            # ORIGINAL STRATEGIES
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
                'freeze_order': 'deep_to_shallow'
            },
            FreezeStrategy.TWO_PHASE: {
                'tight_freeze_epochs': 3,
                'gradual_unfreeze_epochs': self.ft_shared_after - 3,
            },
            FreezeStrategy.EXPONENTIAL_DECAY: {
                'decay_rate': 0.5,
                'half_life_epochs': 3
            },
        }
        
        strategy_defaults = defaults.get(self.freeze_strategy, {})
        for key, value in strategy_defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def get_freeze_config(
        self,
        current_epoch: int,
        current_experience: int = 0,
        loss: Optional[float] = None
    ) -> FreezeConfig:
        """
        Get the freeze configuration for the current epoch and experience.
        
        Args:
            current_epoch: Current training epoch (0-indexed)
            current_experience: Current experience number (0-indexed) - IMPORTANT for delayed strategies
            loss: Current batch loss (optional)
            
        Returns:
            FreezeConfig object specifying what to freeze
        """
        if self.freeze_strategy == FreezeStrategy.DELAYED_START:
            return self._delayed_start(current_epoch, current_experience)
        elif self.freeze_strategy == FreezeStrategy.PROGRESSIVE_FREEZING:
            return self._progressive_freezing(current_epoch, current_experience)
        elif self.freeze_strategy == FreezeStrategy.DELAYED_GRADUAL:
            return self._delayed_gradual(current_epoch, current_experience)
        elif self.freeze_strategy == FreezeStrategy.BINARY:
            return self._binary_freeze(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.GRADUAL_UNFREEZING:
            return self._gradual_unfreezing(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.LAYER_WISE_CURRICULUM:
            return self._layer_wise_curriculum(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.TWO_PHASE:
            return self._two_phase(current_epoch)
        elif self.freeze_strategy == FreezeStrategy.EXPONENTIAL_DECAY:
            return self._exponential_decay(current_epoch)
        else:
            raise ValueError(f"Unknown freeze_strategy: {self.freeze_strategy}")
    
    # ======================== NEW STRATEGIES (RECOMMENDED) ========================
    
    def _delayed_start(self, epoch: int, current_experience: int) -> FreezeConfig:
        """
        RECOMMENDED STRATEGY #1: Delayed-Start Freezing
        
        Do NOT freeze until the model has seen N experiences.
        This allows the model to learn generalizable features before locking them.
        
        Rationale: Freezing from Exp 0 forces adaptation of task-specific features,
        causing catastrophic forgetting. Delay allows multi-task feature learning.
        
        Evidence: Your experiments show +6-10% improvement by freezing from Exp 2
        instead of Exp 0.
        """
        freeze_start_experience = self.config.get('freeze_start_experience', 2)
        
        # Don't freeze if we haven't reached the required experience yet
        if current_experience < freeze_start_experience:
            return FreezeConfig(
                should_freeze=False,
                shared_layer_intensity=0.0,
                old_classifier_intensity=0.0
            )
        
        # Once we reach the freeze start experience, use binary freezing
        should_freeze = epoch <= self.ft_shared_after
        
        return FreezeConfig(
            should_freeze=should_freeze,
            shared_layer_intensity=1.0 if should_freeze else 0.0,
            old_classifier_intensity=1.0 if should_freeze else 0.0
        )
    
    def _progressive_freezing(self, epoch: int, current_experience: int) -> FreezeConfig:
        """
        RECOMMENDED STRATEGY #2: Progressive Freezing
        
        Gradually INCREASE freezing intensity as the model sees more experiences.
        
        Rationale: Instead of "no freezing → full freezing", use "light → medium → strong".
        This prevents sudden commitment to early task-specific features while
        allowing gradual protection as you accumulate diverse tasks.
        
        Schedule:
        Exp 0: 0% freeze (learn freely)
        Exp 1: 25% freeze (light stabilization)
        Exp 2: 50% freeze (medium stabilization)
        Exp 3: 75% freeze (strong stabilization)
        Exp 4+: 100% freeze (full protection)
        """
        ramp_up_experiences = self.config.get('ramp_up_experiences', 4)
        
        # Calculate intensity based on how many experiences we've seen
        # intensity = (current_experience / ramp_up_experiences)
        # But we only increase during freezing epochs
        if epoch <= self.ft_shared_after:
            # Linearly ramp up from 0 to 1 over specified experiences
            intensity = min(1.0, (current_experience + 1) / ramp_up_experiences)
        else:
            # After ft_shared_after epochs, gradually reduce intensity
            intensity = max(0.0, 1.0 - ((epoch - self.ft_shared_after) / (self.total_epochs - self.ft_shared_after)))
        
        return FreezeConfig(
            should_freeze=intensity > 0.01,
            shared_layer_intensity=intensity,
            old_classifier_intensity=intensity
        )
    
    def _delayed_gradual(self, epoch: int, current_experience: int) -> FreezeConfig:
        """
        RECOMMENDED STRATEGY #3: Delayed-Start + Gradual Unfreeze
        
        Combines delayed-start (avoid early freezing) with gradual unfreezing
        (smooth transition from frozen to trainable).
        
        This is likely the BEST strategy based on your evidence:
        - Delayed start ensures good multi-task features
        - Gradual unfreeze prevents abrupt forgetting after full freeze
        - Sweet spot of stability and plasticity
        """
        freeze_start_experience = self.config.get('freeze_start_experience', 2)
        unfreeze_shared_first = self.config.get('unfreeze_shared_first', True)
        
        # If we haven't reached the freeze start experience, no freezing
        if current_experience < freeze_start_experience:
            return FreezeConfig(
                should_freeze=False,
                shared_layer_intensity=0.0,
                old_classifier_intensity=0.0
            )
        
        # Once we reach freeze start experience, apply gradual unfreezing
        if epoch < self.ft_shared_after:
            # Full freeze initially
            return FreezeConfig(
                should_freeze=True,
                shared_layer_intensity=1.0,
                old_classifier_intensity=1.0
            )
        else:
            # After ft_shared_after, gradually unfreeze
            remaining_epochs = self.total_epochs - self.ft_shared_after
            progress = (epoch - self.ft_shared_after) / remaining_epochs if remaining_epochs > 0 else 1.0
            
            if unfreeze_shared_first:
                shared_intensity = max(0.0, 1.0 - progress * 1.5)
                classifier_intensity = max(0.0, 1.0 - progress * 0.5)
            else:
                intensity = 1.0 - progress
                shared_intensity = intensity
                classifier_intensity = intensity
            
            return FreezeConfig(
                should_freeze=shared_intensity > 0.01 or classifier_intensity > 0.01,
                shared_layer_intensity=shared_intensity,
                old_classifier_intensity=classifier_intensity
            )
    
    # ======================== ORIGINAL STRATEGIES ========================
    
    def _binary_freeze(self, epoch: int) -> FreezeConfig:
        """Original strategy: freeze until ft_shared_after, then fully trainable."""
        freeze_until = self.config.get('freeze_before', self.ft_shared_after)
        should_freeze = epoch <= freeze_until
        
        return FreezeConfig(
            should_freeze=should_freeze,
            shared_layer_intensity=1.0 if should_freeze else 0.0,
            old_classifier_intensity=1.0 if should_freeze else 0.0
        )
    
    def _gradual_unfreezing(self, epoch: int) -> FreezeConfig:
        """Linearly decrease freeze intensity over time."""
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
            progress = (epoch - start) / (end - start)
            
            if unfreeze_shared_first:
                shared_intensity = max(0.0, 1.0 - progress * 1.5)
                classifier_intensity = max(0.0, 1.0 - progress * 0.5)
            else:
                intensity = 1.0 - progress
                shared_intensity = intensity
                classifier_intensity = intensity
            
            return FreezeConfig(
                should_freeze=True,
                shared_layer_intensity=shared_intensity,
                old_classifier_intensity=classifier_intensity
            )
    
    def _layer_wise_curriculum(self, epoch: int) -> FreezeConfig:
        """Unfreeze layers in stages from deep to shallow."""
        num_stages = self.config.get('num_stages', 3)
        stage = min(epoch // (self.total_epochs // num_stages), num_stages - 1)
        intensity = max(0.0, 1.0 - (stage / (num_stages - 1)))
        
        layers_to_freeze = [f"stage_{i}" for i in range(num_stages - stage)]
        
        return FreezeConfig(
            should_freeze=intensity > 0,
            shared_layer_intensity=intensity,
            old_classifier_intensity=1.0,
            layers_to_freeze=layers_to_freeze
        )
    
    def _two_phase(self, epoch: int) -> FreezeConfig:
        """Two-phase freezing: tight freeze then gradual release."""
        tight_freeze_epochs = self.config.get('tight_freeze_epochs', 3)
        
        if epoch < tight_freeze_epochs:
            return FreezeConfig(
                should_freeze=True,
                shared_layer_intensity=1.0,
                old_classifier_intensity=1.0
            )
        else:
            remaining_epochs = self.total_epochs - tight_freeze_epochs
            progress = (epoch - tight_freeze_epochs) / remaining_epochs
            intensity = max(0.0, 1.0 - progress)
            
            return FreezeConfig(
                should_freeze=intensity > 0,
                shared_layer_intensity=intensity,
                old_classifier_intensity=intensity * 0.5
            )
    
    def _exponential_decay(self, epoch: int) -> FreezeConfig:
        """Exponential decay of freeze intensity."""
        decay_rate = self.config.get('decay_rate', 0.5)
        half_life = self.config.get('half_life_epochs', 3)
        
        intensity = decay_rate ** (epoch / half_life)
        
        return FreezeConfig(
            should_freeze=intensity > 0.01,
            shared_layer_intensity=intensity,
            old_classifier_intensity=intensity ** 1.5
        )
    
    # ======================== Helper Methods ========================
    
    def get_schedule_visualization(
        self,
        num_experiences: int = 4,
        epochs_per_exp: int = 10
    ) -> str:
        """
        Generate visualization of freeze schedule across multiple experiences.
        
        Args:
            num_experiences: Number of experiences to visualize
            epochs_per_exp: Epochs per experience
        """
        lines = [f"\n{self.freeze_strategy.value.upper()} Schedule\n"]
        lines.append("(Exp = Experience, Epoch = Epoch within that experience)\n")
        
        for exp_id in range(num_experiences):
            lines.append(f"Experience {exp_id}:")
            lines.append("Epoch | Shared Intensity | Classifier Intensity")
            lines.append("------|-----------------|---------------------")
            
            for epoch in range(epochs_per_exp):
                config = self.get_freeze_config(epoch, exp_id)
                lines.append(
                    f"{epoch:5d} | {config.shared_layer_intensity:>16.2f} | "
                    f"{config.old_classifier_intensity:>20.2f}"
                )
            lines.append("")
        
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
    """
    if not freeze_config.should_freeze:
        return
    
    if freeze_config.shared_layer_intensity > 0:
        freeze_shared_layers(
            model,
            intensity=freeze_config.shared_layer_intensity,
            specific_layers=freeze_config.layers_to_freeze,
            freeze_bias_only=freeze_config.freeze_bias_only,
            freeze_weights_only=freeze_config.freeze_weights_only,
            verbose=verbose
        )
    
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
    """Freeze gradients of shared layers."""
    if intensity <= 0:
        return
    
    num_frozen = 0
    for name, module in model.named_modules():
        if hasattr(model, 'linear') and module is model.linear:
            continue
        
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        
        if specific_layers and not any(layer in name for layer in specific_layers):
            continue
        
        if not freeze_bias_only and module.weight.grad is not None:
            module.weight.grad.mul_(1.0 - intensity)
            num_frozen += 1
        
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
    """Freeze gradients of old classifier rows."""
    if intensity <= 0 or len(old_classes) == 0:
        return
    
    classifier = model.linear
    old_num_classes = len(old_classes)
    
    if not freeze_bias_only and classifier.weight.grad is not None:
        classifier.weight.grad[:old_num_classes].mul_(1.0 - intensity)
    
    if not freeze_weights_only and classifier.bias is not None:
        if classifier.bias.grad is not None:
            classifier.bias.grad[:old_num_classes].mul_(1.0 - intensity)
    
    if verbose:
        print(f"  [Freeze Classifier] Reduced gradients for {old_num_classes} old classes "
              f"(intensity={intensity:.2f})")


# ======================== Example Usage ========================

if __name__ == "__main__":
    print("=" * 80)
    print("GRADIENT FREEZE SCHEDULER - REVISED WITH DELAYED-START STRATEGIES")
    print("=" * 80)
    
    # Recommended strategies based on your experimental evidence
    print("\nRECOMMENDED STRATEGIES (Based on Evidence):\n")
    
    strategies_to_test = [
        ('delayed_start', {'freeze_start_experience': 2}),
        ('progressive_freezing', {'ramp_up_experiences': 4}),
        ('delayed_gradual', {'freeze_start_experience': 2, 'unfreeze_shared_first': True}),
    ]
    
    for strategy_name, config in strategies_to_test:
        scheduler = GradientFreezeScheduler(
            freeze_strategy=strategy_name,
            total_epochs=10,
            ft_shared_after=8,
            config=config,
            verbose=False
        )
        print(scheduler.get_schedule_visualization(num_experiences=3, epochs_per_exp=10))
        print("\n" + "="*80 + "\n")
