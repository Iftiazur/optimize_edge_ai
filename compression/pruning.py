"""
Model pruning implementations for edge AI optimization
Includes magnitude-based pruning, structured pruning, and gradual pruning
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import logging

from ..utils.config import PRUNING_RATIOS


class MagnitudePruning:
    """Magnitude-based pruning implementation"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize magnitude-based pruning
        
        Args:
            model: PyTorch model to prune
        """
        self.model = model
        self.original_model = copy.deepcopy(model)
        self.pruned_modules = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_pruning_threshold(self, pruning_ratio: float) -> float:
        """
        Calculate threshold for magnitude-based pruning
        
        Args:
            pruning_ratio: Fraction of weights to prune (0.0 to 1.0)
            
        Returns:
            Threshold value for pruning
        """
        all_weights = []
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                all_weights.extend(module.weight.data.abs().flatten().tolist())
        
        all_weights = torch.tensor(all_weights)
        threshold = torch.quantile(all_weights, pruning_ratio)
        
        return threshold.item()
    
    def apply_magnitude_pruning(self, pruning_ratio: float, 
                               structured: bool = False) -> Dict[str, float]:
        """
        Apply magnitude-based pruning to the model
        
        Args:
            pruning_ratio: Fraction of weights to prune
            structured: Whether to apply structured pruning
            
        Returns:
            Dictionary with pruning statistics
        """
        if structured:
            return self._apply_structured_pruning(pruning_ratio)
        else:
            return self._apply_unstructured_pruning(pruning_ratio)
    
    def _apply_unstructured_pruning(self, pruning_ratio: float) -> Dict[str, float]:
        """Apply unstructured magnitude-based pruning"""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune conv2d weights
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                self.pruned_modules.append((module, 'weight'))
                
                total_params += module.weight.numel()
                pruned_params += (module.weight == 0).sum().item()
                
                # Optionally prune bias
                if module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=pruning_ratio)
                    self.pruned_modules.append((module, 'bias'))
                    total_params += module.bias.numel()
                    pruned_params += (module.bias == 0).sum().item()
                    
            elif isinstance(module, nn.Linear):
                # Prune linear weights
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                self.pruned_modules.append((module, 'weight'))
                
                total_params += module.weight.numel()
                pruned_params += (module.weight == 0).sum().item()
                
                # Optionally prune bias
                if module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=pruning_ratio)
                    self.pruned_modules.append((module, 'bias'))
                    total_params += module.bias.numel()
                    pruned_params += (module.bias == 0).sum().item()
        
        actual_pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        
        # Calculate compression ratio using formula: S_compressed = N × (1 - p) × q
        # where N is original size, p is pruning ratio, q is quantization factor (1.0 for 32-bit)
        compression_ratio = (1 - actual_pruning_ratio) * 1.0
        
        self.logger.info(f"Applied unstructured pruning: {actual_pruning_ratio:.2%} of weights pruned")
        
        return {
            'target_pruning_ratio': pruning_ratio,
            'actual_pruning_ratio': actual_pruning_ratio,
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'compression_ratio': compression_ratio,
            'remaining_parameters': total_params - pruned_params
        }
    
    def _apply_structured_pruning(self, pruning_ratio: float) -> Dict[str, float]:
        """Apply structured pruning (remove entire filters/neurons)"""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Structured filter pruning (prune output channels)
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                self.pruned_modules.append((module, 'weight'))
                total_params += module.weight.numel()
                pruned_params += (module.weight == 0).sum().item()
            elif isinstance(module, nn.Linear):
                # Structured neuron pruning (prune output neurons)
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=1)
                self.pruned_modules.append((module, 'weight'))
                total_params += module.weight.numel()
                pruned_params += (module.weight == 0).sum().item()
        
        actual_pruning_ratio = pruned_params / total_params if total_params > 0 else 0
        compression_ratio = (1 - actual_pruning_ratio) * 1.0
        
        self.logger.info(f"Applied structured pruning: {actual_pruning_ratio:.2%} of weights pruned")
        
        return {
            'target_pruning_ratio': pruning_ratio,
            'actual_pruning_ratio': actual_pruning_ratio,
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'compression_ratio': compression_ratio,
            'remaining_parameters': total_params - pruned_params
        }
    
    def remove_pruning_masks(self):
        """Permanently remove pruning masks and make pruning permanent"""
        for module, param_name in self.pruned_modules:
            prune.remove(module, param_name)
        self.pruned_modules.clear()
    
    def get_sparsity_info(self) -> Dict[str, Dict]:
        """Get detailed sparsity information for each layer"""
        sparsity_info = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total_weights = module.weight.numel()
                zero_weights = (module.weight == 0).sum().item()
                sparsity = zero_weights / total_weights
                
                sparsity_info[name] = {
                    'total_weights': total_weights,
                    'zero_weights': zero_weights,
                    'sparsity': sparsity,
                    'layer_type': module.__class__.__name__
                }
        
        return sparsity_info


class GradualPruning:
    """Gradual pruning implementation for training-time optimization"""
    
    def __init__(self, model: nn.Module, initial_sparsity: float = 0.0, 
                 final_sparsity: float = 0.5, pruning_frequency: int = 100):
        """
        Initialize gradual pruning
        
        Args:
            model: PyTorch model to prune
            initial_sparsity: Starting sparsity level
            final_sparsity: Target sparsity level
            pruning_frequency: How often to update pruning (in steps)
        """
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
    
    def update_pruning(self, current_epoch: int, total_epochs: int):
        """
        Update pruning based on current training progress
        
        Args:
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs
        """
        if self.step_count % self.pruning_frequency == 0:
            # Calculate current sparsity using cubic schedule
            progress = current_epoch / total_epochs
            current_sparsity = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * \
                              (1 - progress) ** 3
            
            self._apply_global_magnitude_pruning(current_sparsity)
            
            self.logger.debug(f"Updated pruning to {current_sparsity:.3f} sparsity at epoch {current_epoch}")
        
        self.step_count += 1
    
    def _apply_global_magnitude_pruning(self, sparsity_level: float):
        """Apply global magnitude pruning at specified sparsity level"""
        parameters_to_prune = []
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity_level,
        )


class ChannelPruning:
    """Channel-wise pruning for CNN optimization"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize channel pruning
        
        Args:
            model: PyTorch CNN model to prune
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def calculate_channel_importance(self, module: nn.Conv2d) -> torch.Tensor:
        """
        Calculate channel importance scores
        
        Args:
            module: Conv2d module to analyze
            
        Returns:
            Tensor of importance scores for each output channel
        """
        # Use L2 norm of filters as importance metric
        with torch.no_grad():
            importance_scores = torch.norm(module.weight.data, p=2, dim=(1, 2, 3))
        
        return importance_scores
    
    def prune_channels(self, pruning_ratio: float) -> Dict[str, int]:
        """
        Prune channels based on importance scores
        
        Args:
            pruning_ratio: Fraction of channels to prune
            
        Returns:
            Dictionary with pruning statistics
        """
        pruning_stats = {}
        
        conv_modules = [(name, module) for name, module in self.model.named_modules() 
                       if isinstance(module, nn.Conv2d)]
        
        for name, module in conv_modules:
            original_channels = module.out_channels
            channels_to_prune = int(original_channels * pruning_ratio)
            
            if channels_to_prune > 0:
                importance_scores = self.calculate_channel_importance(module)
                
                # Get indices of least important channels
                _, indices_to_prune = torch.topk(importance_scores, channels_to_prune, largest=False)
                
                # Apply structured pruning
                prune.ln_structured(module, name='weight', amount=channels_to_prune, 
                               dim=0, n=2)
                
                remaining_channels = original_channels - channels_to_prune
                pruning_stats[name] = {
                    'original_channels': original_channels,
                    'pruned_channels': channels_to_prune,
                    'remaining_channels': remaining_channels,
                    'pruning_ratio': channels_to_prune / original_channels
                }
        
        return pruning_stats


def compare_pruning_methods(model: nn.Module, test_loader, pruning_ratios: List[float] = None) -> Dict:
    """
    Compare different pruning methods
    
    Args:
        model: Original model to compare
        test_loader: Test data loader for evaluation
        pruning_ratios: List of pruning ratios to test
        
    Returns:
        Dictionary with comparison results
    """
    if pruning_ratios is None:
        pruning_ratios = PRUNING_RATIOS
    
    results = {}
    
    for ratio in pruning_ratios:
        if ratio == 0.0:
            # Baseline (no pruning)
            results[f'baseline'] = {
                'pruning_ratio': 0.0,
                'model_size_mb': get_model_size_mb(model),
                'accuracy': evaluate_model(model, test_loader),
                'method': 'none'
            }
            continue
        
        # Test unstructured pruning
        model_unstructured = copy.deepcopy(model)
        pruner_unstructured = MagnitudePruning(model_unstructured)
        stats_unstructured = pruner_unstructured.apply_magnitude_pruning(ratio, structured=False)
        
        results[f'unstructured_{ratio}'] = {
            'pruning_ratio': ratio,
            'actual_pruning_ratio': stats_unstructured['actual_pruning_ratio'],
            'model_size_mb': get_model_size_mb(model_unstructured),
            'accuracy': evaluate_model(model_unstructured, test_loader),
            'method': 'unstructured_magnitude',
            'compression_ratio': stats_unstructured['compression_ratio']
        }
        
        # Test structured pruning
        model_structured = copy.deepcopy(model)
        pruner_structured = MagnitudePruning(model_structured)
        stats_structured = pruner_structured.apply_magnitude_pruning(ratio, structured=True)
        
        results[f'structured_{ratio}'] = {
            'pruning_ratio': ratio,
            'actual_pruning_ratio': stats_structured['actual_pruning_ratio'],
            'model_size_mb': get_model_size_mb(model_structured),
            'accuracy': evaluate_model(model_structured, test_loader),
            'method': 'structured_magnitude',
            'compression_ratio': stats_structured['compression_ratio']
        }
    
    return results


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def evaluate_model(model: nn.Module, test_loader) -> float:
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    # Example usage
    from ..models.base_model import create_model
    
    # Create test model
    model = create_model('base')
    print(f"Original model size: {get_model_size_mb(model):.2f} MB")
    
    # Apply magnitude pruning
    pruner = MagnitudePruning(model)
    stats = pruner.apply_magnitude_pruning(0.5, structured=False)
    
    print(f"Pruning statistics: {stats}")
    print(f"Pruned model size: {get_model_size_mb(model):.2f} MB")
    
    # Get sparsity information
    sparsity_info = pruner.get_sparsity_info()
    for layer_name, info in sparsity_info.items():
        print(f"{layer_name}: {info['sparsity']:.2%} sparse")