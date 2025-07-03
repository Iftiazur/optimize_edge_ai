"""
Compressed Models Implementation
Contains compressed versions of the base models after applying various optimization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, Any
import copy
from .base_model import BaseCNN, TeacherCNN, StudentCNN

logger = logging.getLogger(__name__)

class CompressedModel:
    """
    Container for a compressed model with metadata about compression techniques applied
    """
    
    def __init__(self, model: nn.Module, compression_info: Dict):
        self.model = model
        self.compression_info = compression_info
        self.original_size = compression_info.get('original_size', 0)
        self.compressed_size = self._calculate_model_size()
        
    def _calculate_model_size(self) -> int:
        """Calculate the size of the model in bytes"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return param_size + buffer_size
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio (original_size / compressed_size)"""
        if self.compressed_size == 0:
            return 0.0
        return self.original_size / self.compressed_size
    
    def get_size_reduction_percent(self) -> float:
        """Get the size reduction as a percentage"""
        if self.original_size == 0:
            return 0.0
        return ((self.original_size - self.compressed_size) / self.original_size) * 100
    
    def get_info(self) -> Dict:
        """Get comprehensive information about the compressed model"""
        return {
            'compression_techniques': self.compression_info.get('techniques', []),
            'original_size_bytes': self.original_size,
            'compressed_size_bytes': self.compressed_size,
            'compression_ratio': self.get_compression_ratio(),
            'size_reduction_percent': self.get_size_reduction_percent(),
            'parameters_count': sum(p.numel() for p in self.model.parameters()),
            'model_type': self.compression_info.get('model_type', 'unknown'),
            'compression_params': self.compression_info.get('params', {})
        }

class PrunedModel(CompressedModel):
    """
    Model compressed using pruning techniques
    """
    
    def __init__(self, original_model: nn.Module, pruning_ratio: float, 
                 pruning_type: str = "magnitude", structured: bool = False):
        
        # Apply pruning
        pruned_model, pruning_info = self._apply_pruning(
            original_model, pruning_ratio, pruning_type, structured
        )
        
        compression_info = {
            'techniques': ['pruning'],
            'model_type': 'pruned',
            'original_size': self._calculate_original_size(original_model),
            'params': {
                'pruning_ratio': pruning_ratio,
                'pruning_type': pruning_type,
                'structured': structured,
                'pruning_info': pruning_info
            }
        }
        
        super().__init__(pruned_model, compression_info)
        self.pruning_ratio = pruning_ratio
        self.pruning_type = pruning_type
        self.structured = structured
        self.mask_info = pruning_info.get('masks', {})
    
    def _calculate_original_size(self, model: nn.Module) -> int:
        """Calculate original model size"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size
    
    def _apply_pruning(self, model: nn.Module, ratio: float, 
                      pruning_type: str, structured: bool) -> Tuple[nn.Module, Dict]:
        """Apply pruning to the model"""
        import torch.nn.utils.prune as prune
        
        model_copy = copy.deepcopy(model)
        pruning_info = {
            'pruned_parameters': [],
            'masks': {},
            'sparsity_achieved': {}
        }
        
        if structured:
            # Structured pruning - remove entire filters/channels
            for name, module in model_copy.named_modules():
                if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                    # Prune filters based on L1 norm
                    prune.ln_structured(module, name='weight', amount=ratio, n=1, dim=0)
                    pruning_info['pruned_parameters'].append(f"{name}.weight")
                    
                    # Calculate actual sparsity
                    total_params = module.weight.numel()
                    zero_params = (module.weight == 0).sum().item()
                    sparsity = zero_params / total_params
                    pruning_info['sparsity_achieved'][f"{name}.weight"] = sparsity
        else:
            # Unstructured pruning
            parameters_to_prune = []
            
            for name, module in model_copy.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
                    pruning_info['pruned_parameters'].append(f"{name}.weight")
            
            if parameters_to_prune:
                if pruning_type == "magnitude":
                    prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=ratio
                    )
                elif pruning_type == "random":
                    prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=prune.RandomUnstructured,
                        amount=ratio
                    )
                
                # Calculate sparsity for each pruned parameter
                for module, param_name in parameters_to_prune:
                    param = getattr(module, param_name)
                    total_params = param.numel()
                    zero_params = (param == 0).sum().item()
                    sparsity = zero_params / total_params
                    pruning_info['sparsity_achieved'][f"{param_name}"] = sparsity
        
        # Remove pruning re-parametrization to make pruning permanent
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # No pruning mask to remove
        
        logger.info(f"Applied {pruning_type} pruning with ratio {ratio}, structured: {structured}")
        return model_copy, pruning_info
    
    def get_sparsity_info(self) -> Dict:
        """Get detailed sparsity information"""
        sparsity_info = {}
        
        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # Only for weight matrices
                total_params = param.numel()
                zero_params = (param.abs() < 1e-8).sum().item()  # Consider very small values as zero
                sparsity = zero_params / total_params if total_params > 0 else 0
                
                sparsity_info[name] = {
                    'total_parameters': total_params,
                    'zero_parameters': zero_params,
                    'sparsity_ratio': sparsity,
                    'effective_parameters': total_params - zero_params
                }
        
        return sparsity_info

class QuantizedModel(CompressedModel):
    """
    Model compressed using quantization techniques
    """
    
    def __init__(self, original_model: nn.Module, quantization_config: Dict):
        
        # Apply quantization
        quantized_model, quant_info = self._apply_quantization(original_model, quantization_config)
        
        compression_info = {
            'techniques': ['quantization'],
            'model_type': 'quantized',
            'original_size': self._calculate_original_size(original_model),
            'params': {
                'quantization_config': quantization_config,
                'quantization_info': quant_info
            }
        }
        
        super().__init__(quantized_model, compression_info)
        self.quantization_config = quantization_config
        self.bit_width = quantization_config.get('bit_width', 8)
        self.quantization_type = quantization_config.get('type', 'dynamic')
    
    def _calculate_original_size(self, model: nn.Module) -> int:
        """Calculate original model size"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size
    
    def _apply_quantization(self, model: nn.Module, config: Dict) -> Tuple[nn.Module, Dict]:
        """Apply quantization to the model"""
        quantization_type = config.get('type', 'dynamic')
        bit_width = config.get('bit_width', 8)
        
        quant_info = {
            'quantization_type': quantization_type,
            'bit_width': bit_width,
            'quantized_layers': []
        }
        
        if quantization_type == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8 if bit_width == 8 else torch.qint32
            )
            
            for name, module in quantized_model.named_modules():
                if hasattr(module, 'weight') and hasattr(module.weight(), 'dtype'):
                    if 'qint' in str(module.weight().dtype):
                        quant_info['quantized_layers'].append(name)
        
        elif quantization_type == 'static':
            # Static quantization (requires calibration data)
            model_copy = copy.deepcopy(model)
            model_copy.eval()
            
            # Prepare model for static quantization
            model_copy.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model_copy, inplace=True)
            
            # Note: In practice, you would run calibration data through the model here
            # For now, we'll just proceed with the preparation
            
            quantized_model = torch.quantization.convert(model_copy, inplace=False)
            
            for name, module in quantized_model.named_modules():
                if 'Quantized' in type(module).__name__:
                    quant_info['quantized_layers'].append(name)
        
        elif quantization_type == 'qat':
            # Quantization Aware Training (requires training loop)
            model_copy = copy.deepcopy(model)
            model_copy.train()
            
            # Prepare for QAT
            model_copy.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            quantized_model = torch.quantization.prepare_qat(model_copy)
            
            # Note: In practice, you would train the model here
            # For demonstration, we'll convert directly
            quantized_model.eval()
            quantized_model = torch.quantization.convert(quantized_model)
            
            for name, module in quantized_model.named_modules():
                if 'Quantized' in type(module).__name__:
                    quant_info['quantized_layers'].append(name)
        
        else:
            # Custom quantization
            quantized_model = self._custom_quantization(model, bit_width)
            quant_info['quantized_layers'] = ['all_layers']
        
        logger.info(f"Applied {quantization_type} quantization with {bit_width}-bit precision")
        return quantized_model, quant_info
    
    def _custom_quantization(self, model: nn.Module, bit_width: int) -> nn.Module:
        """Apply custom quantization"""
        model_copy = copy.deepcopy(model)
        
        for name, param in model_copy.named_parameters():
            if param.dim() > 1:  # Only quantize weight matrices
                # Simple linear quantization
                param_min = param.min()
                param_max = param.max()
                
                # Calculate scale and zero point
                qmin = -(2**(bit_width-1))
                qmax = 2**(bit_width-1) - 1
                
                scale = (param_max - param_min) / (qmax - qmin)
                zero_point = qmin - torch.round(param_min / scale)
                
                # Quantize and dequantize
                quantized = torch.clamp(torch.round(param / scale) + zero_point, qmin, qmax)
                dequantized = (quantized - zero_point) * scale
                
                param.data = dequantized
        
        return model_copy
    
    def get_quantization_info(self) -> Dict:
        """Get detailed quantization information"""
        info = {
            'quantization_type': self.quantization_type,
            'bit_width': self.bit_width,
            'quantized_layers': self.compression_info['params']['quantization_info'].get('quantized_layers', []),
            'size_reduction_factor': 32 / self.bit_width if self.bit_width > 0 else 1,
            'theoretical_speedup': 32 / self.bit_width if self.bit_width > 0 else 1
        }
        
        return info

class DistilledModel(CompressedModel):
    """
    Model compressed using knowledge distillation
    """
    
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, 
                 distillation_info: Dict):
        
        compression_info = {
            'techniques': ['knowledge_distillation'],
            'model_type': 'distilled',
            'original_size': self._calculate_original_size(teacher_model),
            'params': {
                'teacher_model_info': self._get_model_info(teacher_model),
                'student_model_info': self._get_model_info(student_model),
                'distillation_config': distillation_info
            }
        }
        
        super().__init__(student_model, compression_info)
        self.teacher_model = teacher_model
        self.distillation_config = distillation_info
        
    def _calculate_original_size(self, model: nn.Module) -> int:
        """Calculate teacher model size"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size
    
    def _get_model_info(self, model: nn.Module) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'architecture': type(model).__name__
        }
    
    def get_compression_comparison(self) -> Dict:
        """Compare teacher and student models"""
        teacher_info = self.compression_info['params']['teacher_model_info']
        student_info = self.compression_info['params']['student_model_info']
        
        param_reduction = ((teacher_info['total_parameters'] - student_info['total_parameters']) 
                          / teacher_info['total_parameters']) * 100
        
        size_reduction = ((teacher_info['model_size_bytes'] - student_info['model_size_bytes']) 
                         / teacher_info['model_size_bytes']) * 100
        
        return {
            'teacher_parameters': teacher_info['total_parameters'],
            'student_parameters': student_info['total_parameters'],
            'parameter_reduction_percent': param_reduction,
            'teacher_size_mb': teacher_info['model_size_bytes'] / (1024 * 1024),
            'student_size_mb': student_info['model_size_bytes'] / (1024 * 1024),
            'size_reduction_percent': size_reduction,
            'compression_ratio': teacher_info['total_parameters'] / student_info['total_parameters']
        }

class MultiTechniqueCompressedModel(CompressedModel):
    """
    Model compressed using multiple techniques (pruning + quantization + distillation)
    """
    
    def __init__(self, original_model: nn.Module, compression_techniques: Dict):
        """
        Args:
            original_model: The original uncompressed model
            compression_techniques: Dict with keys like 'pruning', 'quantization', 'distillation'
        """
        self.applied_techniques = []
        current_model = copy.deepcopy(original_model)
        technique_info = {}
        
        # Apply techniques in order: distillation -> pruning -> quantization
        if 'distillation' in compression_techniques:
            current_model, distill_info = self._apply_distillation(
                current_model, compression_techniques['distillation']
            )
            self.applied_techniques.append('distillation')
            technique_info['distillation'] = distill_info
        
        if 'pruning' in compression_techniques:
            current_model, prune_info = self._apply_pruning_step(
                current_model, compression_techniques['pruning']
            )
            self.applied_techniques.append('pruning')
            technique_info['pruning'] = prune_info
        
        if 'quantization' in compression_techniques:
            current_model, quant_info = self._apply_quantization_step(
                current_model, compression_techniques['quantization']
            )
            self.applied_techniques.append('quantization')
            technique_info['quantization'] = quant_info
        
        compression_info = {
            'techniques': self.applied_techniques,
            'model_type': 'multi_technique',
            'original_size': self._calculate_original_size(original_model),
            'params': {
                'compression_techniques': compression_techniques,
                'technique_info': technique_info,
                'application_order': self.applied_techniques
            }
        }
        
        super().__init__(current_model, compression_info)
    
    def _calculate_original_size(self, model: nn.Module) -> int:
        """Calculate original model size"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size
    
    def _apply_distillation_step(self, model: nn.Module, distill_config: Dict) -> Tuple[nn.Module, Dict]:
        """Apply distillation (returns student model)"""
        # For distillation, we need to return a smaller student model
        # This is a simplified version - in practice, distillation involves training
        
        if distill_config.get('student_model'):
            student_model = distill_config['student_model']
        else:
            # Create a smaller version of the model
            student_model = self._create_student_model(model)
        
        distill_info = {
            'teacher_params': sum(p.numel() for p in model.parameters()),
            'student_params': sum(p.numel() for p in student_model.parameters()),
            'compression_ratio': sum(p.numel() for p in model.parameters()) / sum(p.numel() for p in student_model.parameters())
        }
        
        return student_model, distill_info
    
    def _apply_pruning_step(self, model: nn.Module, prune_config: Dict) -> Tuple[nn.Module, Dict]:
        """Apply pruning to the model"""
        import torch.nn.utils.prune as prune
        
        model_copy = copy.deepcopy(model)
        ratio = prune_config.get('ratio', 0.5)
        
        # Global magnitude-based pruning
        parameters_to_prune = []
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=ratio
            )
            
            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                try:
                    prune.remove(module, param_name)
                except ValueError:
                    pass
        
        prune_info = {
            'pruning_ratio': ratio,
            'parameters_pruned': len(parameters_to_prune)
        }
        
        return model_copy, prune_info
    
    def _apply_quantization_step(self, model: nn.Module, quant_config: Dict) -> Tuple[nn.Module, Dict]:
        """Apply quantization to the model"""
        bit_width = quant_config.get('bit_width', 8)
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8 if bit_width == 8 else torch.qint32
        )
        
        quant_info = {
            'bit_width': bit_width,
            'quantization_type': 'dynamic'
        }
        
        return quantized_model, quant_info
    
    def _create_student_model(self, teacher_model: nn.Module) -> nn.Module:
        """Create a smaller student model based on the teacher"""
        # This is a simplified approach - create a model with fewer channels/layers
        if isinstance(teacher_model, BaseCNN):
            # Create a student with fewer channels
            return StudentCNN(num_classes=teacher_model.num_classes)
        else:
            # For other models, just return a copy (in practice, you'd design a smaller architecture)
            return copy.deepcopy(teacher_model)
    
    def get_technique_breakdown(self) -> Dict:
        """Get breakdown of compression achieved by each technique"""
        technique_info = self.compression_info['params']['technique_info']
        breakdown = {}
        
        for technique in self.applied_techniques:
            if technique in technique_info:
                info = technique_info[technique]
                if technique == 'distillation':
                    breakdown[technique] = {
                        'compression_ratio': info.get('compression_ratio', 1.0),
                        'parameter_reduction': ((info.get('teacher_params', 0) - info.get('student_params', 0)) 
                                              / info.get('teacher_params', 1)) * 100
                    }
                elif technique == 'pruning':
                    breakdown[technique] = {
                        'pruning_ratio': info.get('pruning_ratio', 0.0),
                        'parameters_affected': info.get('parameters_pruned', 0)
                    }
                elif technique == 'quantization':
                    breakdown[technique] = {
                        'bit_width': info.get('bit_width', 32),
                        'theoretical_compression': 32 / info.get('bit_width', 32)
                    }
        
        return breakdown

def create_compressed_model(original_model: nn.Module, compression_config: Dict) -> CompressedModel:
    """
    Factory function to create compressed models based on configuration
    
    Args:
        original_model: The original model to compress
        compression_config: Configuration specifying compression techniques
    
    Returns:
        CompressedModel instance
    """
    techniques = compression_config.get('techniques', [])
    
    if len(techniques) == 1:
        technique = techniques[0]
        
        if technique == 'pruning':
            return PrunedModel(
                original_model,
                compression_config.get('pruning_ratio', 0.5),
                compression_config.get('pruning_type', 'magnitude'),
                compression_config.get('structured', False)
            )
        
        elif technique == 'quantization':
            return QuantizedModel(
                original_model,
                compression_config.get('quantization_config', {'type': 'dynamic', 'bit_width': 8})
            )
        
        elif technique == 'distillation':
            student_model = compression_config.get('student_model')
            if student_model is None:
                raise ValueError("Student model must be provided for distillation")
            
            return DistilledModel(
                student_model,
                original_model,  # teacher
                compression_config.get('distillation_config', {})
            )
    
    elif len(techniques) > 1:
        # Multiple techniques
        return MultiTechniqueCompressedModel(original_model, compression_config)
    
    else:
        raise ValueError("No compression techniques specified")

def compare_compressed_models(models: Dict[str, CompressedModel]) -> Dict:
    """
    Compare multiple compressed models
    
    Args:
        models: Dictionary of model_name -> CompressedModel
    
    Returns:
        Comparison results
    """
    comparison = {
        'models': {},
        'best_compression_ratio': None,
        'best_size_reduction': None,
        'techniques_summary': {}
    }
    
    best_compression = 0
    best_size_reduction = 0
    
    for name, model in models.items():
        info = model.get_info()
        comparison['models'][name] = info
        
        # Track best performers
        if info['compression_ratio'] > best_compression:
            best_compression = info['compression_ratio']
            comparison['best_compression_ratio'] = name
        
        if info['size_reduction_percent'] > best_size_reduction:
            best_size_reduction = info['size_reduction_percent']
            comparison['best_size_reduction'] = name
        
        # Count technique usage
        for technique in info['compression_techniques']:
            if technique not in comparison['techniques_summary']:
                comparison['techniques_summary'][technique] = 0
            comparison['techniques_summary'][technique] += 1
    
    return comparison
