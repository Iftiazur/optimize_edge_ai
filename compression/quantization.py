"""
Model quantization implementations for edge AI optimization
Includes post-training quantization, quantization-aware training, and different bit-width support
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
import copy
import logging

from ..utils.config import QUANTIZATION_BITS, DEVICE


class PostTrainingQuantization:
    """Post-training quantization implementation"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize post-training quantization
        
        Args:
            model: PyTorch model to quantize
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def prepare_model_for_quantization(self) -> nn.Module:
        """Prepare model for quantization by adding quantization stubs"""
        class QuantizedWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.quant = QuantStub()
                self.model = model
                self.dequant = DeQuantStub()
            
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        wrapped_model = QuantizedWrapper(self.model)
        return wrapped_model
    
    def apply_dynamic_quantization(self, bit_width: int = 8) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply dynamic quantization to the model
        
        Args:
            bit_width: Target bit width for quantization
            
        Returns:
            Tuple of (quantized_model, quantization_stats)
        """
        # Set quantization configuration based on bit width
        if bit_width == 8:
            dtype = torch.qint8
        elif bit_width == 16:
            dtype = torch.qint32  # Using qint32 for 16-bit simulation
        else:
            raise ValueError(f"Unsupported bit width: {bit_width}")
        
        # Apply dynamic quantization to linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )
        
        # Calculate quantization statistics
        original_size = self._calculate_model_size(self.model)
        quantized_size = self._calculate_model_size(quantized_model)
        
        stats = {
            'bit_width': bit_width,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100,
            'quantization_type': 'dynamic'
        }
        
        self.logger.info(f"Applied dynamic quantization: {bit_width}-bit, "
                        f"{stats['size_reduction_percent']:.1f}% size reduction")
        
        return quantized_model, stats
    
    def apply_static_quantization(self, calibration_loader, bit_width: int = 8) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply static quantization with calibration
        
        Args:
            calibration_loader: Data loader for calibration
            bit_width: Target bit width for quantization
            
        Returns:
            Tuple of (quantized_model, quantization_stats)
        """
        # Prepare model
        wrapped_model = self.prepare_model_for_quantization()
        wrapped_model.eval()
        
        # Set quantization configuration
        if bit_width == 8:
            wrapped_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        elif bit_width == 16:
            # Custom 16-bit configuration
            wrapped_model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint32),
                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8)
            )
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(wrapped_model)
        
        # Calibration
        self._calibrate_model(prepared_model, calibration_loader)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        # Calculate statistics
        original_size = self._calculate_model_size(self.model)
        quantized_size = self._calculate_model_size(quantized_model)
        
        stats = {
            'bit_width': bit_width,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100,
            'quantization_type': 'static'
        }
        
        self.logger.info(f"Applied static quantization: {bit_width}-bit, "
                        f"{stats['size_reduction_percent']:.1f}% size reduction")
        
        return quantized_model, stats
    
    def _calibrate_model(self, model: nn.Module, calibration_loader, num_batches: int = 100):
        """Calibrate model using representative data"""
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= num_batches:
                    break
                _ = model(data)
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb


class QuantizationAwareTraining:
    """Quantization-aware training implementation"""
    
    def __init__(self, model: nn.Module, bit_width: int = 8):
        """
        Initialize quantization-aware training
        
        Args:
            model: PyTorch model for QAT
            bit_width: Target bit width for quantization
        """
        self.model = model
        self.bit_width = bit_width
        self.logger = logging.getLogger(__name__)
        
        # Prepare model for QAT
        self.qat_model = self._prepare_qat_model()
    
    def _prepare_qat_model(self) -> nn.Module:
        """Prepare model for quantization-aware training"""
        class QATWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.quant = QuantStub()
                self.model = model
                self.dequant = DeQuantStub()
            
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        wrapped_model = QATWrapper(self.model)
        
        # Set quantization configuration
        if self.bit_width == 8:
            wrapped_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        elif self.bit_width == 16:
            # Custom 16-bit QAT configuration
            wrapped_model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAverageMinMaxObserver,
                    quant_min=-32768, quant_max=32767, dtype=torch.qint32
                ),
                weight=torch.quantization.FakeQuantize.with_args(
                    observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128, quant_max=127, dtype=torch.qint8
                )
            )
        
        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(wrapped_model)
        return prepared_model
    
    def train_step(self, data: torch.Tensor, target: torch.Tensor, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """
        Perform one training step with quantization-aware training
        
        Args:
            data: Input data
            target: Target labels
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Dictionary with training metrics
        """
        self.qat_model.train()
        
        optimizer.zero_grad()
        output = self.qat_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        accuracy = (predicted == target).float().mean().item() * 100
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to fully quantized model"""
        self.qat_model.eval()
        quantized_model = torch.quantization.convert(self.qat_model)
        return quantized_model
    
    def get_fake_quantization_stats(self) -> Dict[str, Dict]:
        """Get statistics about fake quantization during training"""
        stats = {}
        
        for name, module in self.qat_model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                fake_quant = module.weight_fake_quant
                stats[f'{name}_weight'] = {
                    'scale': fake_quant.scale.item() if fake_quant.scale is not None else None,
                    'zero_point': fake_quant.zero_point.item() if fake_quant.zero_point is not None else None,
                    'quant_min': fake_quant.quant_min,
                    'quant_max': fake_quant.quant_max
                }
            
            if hasattr(module, 'activation_post_process'):
                act_fake_quant = module.activation_post_process
                stats[f'{name}_activation'] = {
                    'scale': act_fake_quant.scale.item() if hasattr(act_fake_quant, 'scale') and act_fake_quant.scale is not None else None,
                    'zero_point': act_fake_quant.zero_point.item() if hasattr(act_fake_quant, 'zero_point') and act_fake_quant.zero_point is not None else None
                }
        
        return stats


class CustomQuantization:
    """Custom quantization implementations for different bit widths"""
    
    @staticmethod
    def linear_quantization(tensor: torch.Tensor, bit_width: int, 
                          symmetric: bool = True) -> Tuple[torch.Tensor, float, int]:
        """
        Apply linear quantization to tensor
        
        Args:
            tensor: Input tensor to quantize
            bit_width: Number of bits for quantization
            symmetric: Whether to use symmetric quantization
            
        Returns:
            Tuple of (quantized_tensor, scale, zero_point)
        """
        if symmetric:
            # Symmetric quantization
            max_val = tensor.abs().max()
            scale = max_val / (2**(bit_width - 1) - 1)
            zero_point = 0
            
            quantized = torch.round(tensor / scale).clamp(
                -(2**(bit_width - 1)), 2**(bit_width - 1) - 1
            )
        else:
            # Asymmetric quantization
            min_val, max_val = tensor.min(), tensor.max()
            
            qmin = 0
            qmax = 2**bit_width - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, qmin, qmax).int()
            
            quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
        
        return quantized, scale.item(), zero_point.item() if isinstance(zero_point, torch.Tensor) else zero_point
    
    @staticmethod
    def dequantize(quantized_tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        """
        Dequantize tensor back to floating point
        
        Args:
            quantized_tensor: Quantized tensor
            scale: Quantization scale
            zero_point: Quantization zero point
            
        Returns:
            Dequantized tensor
        """
        return scale * (quantized_tensor - zero_point)


def compare_quantization_methods(model: nn.Module, test_loader, calibration_loader,
                               bit_widths: Optional[list] = None) -> Dict[str, Dict]:
    """
    Compare different quantization methods and bit widths
    
    Args:
        model: Original model to quantize
        test_loader: Test data loader for evaluation
        calibration_loader: Calibration data loader
        bit_widths: List of bit widths to test
        
    Returns:
        Dictionary with comparison results
    """
    if bit_widths is None:
        bit_widths = QUANTIZATION_BITS
    
    results = {}
    
    # Baseline (no quantization)
    results['baseline'] = {
        'bit_width': 32,
        'accuracy': evaluate_quantized_model(model, test_loader),
        'model_size_mb': calculate_model_size(model),
        'method': 'none'
    }
    
    for bit_width in bit_widths:
        if bit_width == 32:
            continue  # Skip baseline
        
        # Dynamic quantization
        try:
            model_copy = copy.deepcopy(model)
            quantizer = PostTrainingQuantization(model_copy)
            dynamic_model, dynamic_stats = quantizer.apply_dynamic_quantization(bit_width)
            
            results[f'dynamic_{bit_width}bit'] = {
                'bit_width': bit_width,
                'accuracy': evaluate_quantized_model(dynamic_model, test_loader),
                'model_size_mb': dynamic_stats['quantized_size_mb'],
                'compression_ratio': dynamic_stats['compression_ratio'],
                'method': 'dynamic'
            }
        except Exception as e:
            logging.warning(f"Dynamic quantization failed for {bit_width}-bit: {e}")
        
        # Static quantization
        try:
            model_copy = copy.deepcopy(model)
            quantizer = PostTrainingQuantization(model_copy)
            static_model, static_stats = quantizer.apply_static_quantization(calibration_loader, bit_width)
            
            results[f'static_{bit_width}bit'] = {
                'bit_width': bit_width,
                'accuracy': evaluate_quantized_model(static_model, test_loader),
                'model_size_mb': static_stats['quantized_size_mb'],
                'compression_ratio': static_stats['compression_ratio'],
                'method': 'static'
            }
        except Exception as e:
            logging.warning(f"Static quantization failed for {bit_width}-bit: {e}")
    
    return results


def evaluate_quantized_model(model: nn.Module, test_loader) -> float:
    """Evaluate quantized model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            try:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            except Exception as e:
                logging.warning(f"Evaluation error: {e}")
                continue
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


def calculate_model_size(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def analyze_quantization_sensitivity(model: nn.Module, test_loader, 
                                   layer_names: Optional[list] = None) -> Dict[str, float]:
    """
    Analyze sensitivity of different layers to quantization
    
    Args:
        model: Model to analyze
        test_loader: Test data loader
        layer_names: Specific layers to analyze (if None, analyze all)
        
    Returns:
        Dictionary mapping layer names to sensitivity scores
    """
    baseline_accuracy = evaluate_quantized_model(model, test_loader)
    sensitivity_scores = {}
    
    # Get all quantizable layers if not specified
    if layer_names is None:
        layer_names = [name for name, module in model.named_modules() 
                      if isinstance(module, (nn.Conv2d, nn.Linear))]
    
    for layer_name in layer_names:
        # Create copy and quantize only this layer
        model_copy = copy.deepcopy(model)
        
        for name, module in model_copy.named_modules():
            if name == layer_name and isinstance(module, (nn.Conv2d, nn.Linear)):
                # Apply aggressive quantization to this layer only
                with torch.no_grad():
                    quantized_weight, _, _ = CustomQuantization.linear_quantization(
                        module.weight, bit_width=4, symmetric=True
                    )
                    module.weight.data = CustomQuantization.dequantize(
                        quantized_weight, _, _
                    )
                break
        
        # Evaluate accuracy with this layer quantized
        quantized_accuracy = evaluate_quantized_model(model_copy, test_loader)
        sensitivity = baseline_accuracy - quantized_accuracy
        sensitivity_scores[layer_name] = sensitivity
    
    return sensitivity_scores


if __name__ == "__main__":
    # Example usage
    from ..models.base_model import create_model
    
    # Create test model
    model = create_model('base')
    print(f"Original model size: {calculate_model_size(model):.2f} MB")
    
    # Test dynamic quantization
    quantizer = PostTrainingQuantization(model)
    quantized_model, stats = quantizer.apply_dynamic_quantization(8)
    
    print(f"Quantization statistics: {stats}")
    print(f"Quantized model size: {calculate_model_size(quantized_model):.2f} MB")
