"""
Federated Learning Models
Contains model implementations specific to federated learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any
import copy
import numpy as np
from .base_model import BaseCNN, TeacherCNN, StudentCNN

logger = logging.getLogger(__name__)

class FederatedModel:
    """
    Wrapper for models used in federated learning with additional federated-specific functionality
    """
    
    def __init__(self, base_model: nn.Module, client_id: Optional[str] = None):
        self.model = base_model
        self.client_id = client_id
        self.round_number = 0
        self.local_epochs = 0
        self.training_history = []
        self.communication_history = []
        
        # Store initial weights for delta calculation
        self.initial_weights = self._get_model_weights()
        self.last_global_weights = copy.deepcopy(self.initial_weights)
        
    def _get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights as a dictionary"""
        return {name: param.clone() for name, param in self.model.named_parameters()}
    
    def _set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from a dictionary"""
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])
    
    def get_model_delta(self) -> Dict[str, torch.Tensor]:
        """
        Calculate the delta (difference) between current weights and last global weights
        Used for federated averaging
        """
        current_weights = self._get_model_weights()
        delta = {}
        
        for name in current_weights:
            if name in self.last_global_weights:
                delta[name] = current_weights[name] - self.last_global_weights[name]
            else:
                delta[name] = current_weights[name]
        
        return delta
    
    def apply_global_update(self, global_weights: Dict[str, torch.Tensor]):
        """Apply global model weights from federated averaging"""
        self._set_model_weights(global_weights)
        self.last_global_weights = copy.deepcopy(global_weights)
        self.round_number += 1
        
        # Record communication event
        self.communication_history.append({
            'round': self.round_number,
            'type': 'global_update',
            'weights_received': len(global_weights),
            'model_size_mb': sum(w.numel() * w.element_size() for w in global_weights.values()) / (1024 * 1024)
        })
    
    def local_train_step(self, loss: float, accuracy: float, epoch: int):
        """Record local training step information"""
        self.local_epochs += 1
        self.training_history.append({
            'global_round': self.round_number,
            'local_epoch': epoch,
            'total_local_epochs': self.local_epochs,
            'loss': loss,
            'accuracy': accuracy,
            'client_id': self.client_id
        })
    
    def get_communication_cost(self) -> Dict[str, float]:
        """Calculate communication costs"""
        if not self.communication_history:
            return {'total_mb': 0.0, 'rounds': 0, 'avg_mb_per_round': 0.0}
        
        total_mb = sum(event['model_size_mb'] for event in self.communication_history)
        rounds = len(self.communication_history)
        avg_mb = total_mb / rounds if rounds > 0 else 0.0
        
        return {
            'total_mb': total_mb,
            'rounds': rounds,
            'avg_mb_per_round': avg_mb
        }
    
    def get_training_summary(self) -> Dict:
        """Get summary of training progress"""
        if not self.training_history:
            return {}
        
        recent_history = self.training_history[-10:]  # Last 10 epochs
        
        return {
            'client_id': self.client_id,
            'global_rounds_participated': self.round_number,
            'total_local_epochs': self.local_epochs,
            'current_loss': self.training_history[-1]['loss'],
            'current_accuracy': self.training_history[-1]['accuracy'],
            'avg_recent_loss': np.mean([h['loss'] for h in recent_history]),
            'avg_recent_accuracy': np.mean([h['accuracy'] for h in recent_history]),
            'loss_trend': 'improving' if len(self.training_history) > 1 and 
                         self.training_history[-1]['loss'] < self.training_history[-2]['loss'] else 'stable',
            'communication_cost': self.get_communication_cost()
        }

class PersonalizedFederatedModel(FederatedModel):
    """
    Federated model with personalization capabilities
    Maintains both global and local model components
    """
    
    def __init__(self, base_model: nn.Module, client_id: str, personalization_ratio: float = 0.3):
        super().__init__(base_model, client_id)
        self.personalization_ratio = personalization_ratio
        
        # Create local personalization layers
        self.personal_layers = self._create_personal_layers()
        self.global_layers = self._identify_global_layers()
        
        # Track personalization performance
        self.personalization_history = []
    
    def _create_personal_layers(self) -> nn.ModuleDict:
        """Create personalized layers for this client"""
        personal_layers = nn.ModuleDict()
        
        # Add a personalized classifier head
        if hasattr(self.model, 'classifier'):
            original_classifier = self.model.classifier
            if isinstance(original_classifier, nn.Linear):
                personal_layers['classifier'] = nn.Linear(
                    original_classifier.in_features,
                    original_classifier.out_features
                )
            elif isinstance(original_classifier, nn.Sequential):
                personal_layers['classifier'] = copy.deepcopy(original_classifier)
        
        # Add personalized batch normalization layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                personal_layers[f'personal_{name}'] = nn.BatchNorm2d(module.num_features)
        
        return personal_layers
    
    def _identify_global_layers(self) -> List[str]:
        """Identify which layers remain global (shared across clients)"""
        global_layers = []
        
        for name, module in self.model.named_parameters():
            # Feature extraction layers remain global
            if 'conv' in name.lower() or 'features' in name.lower():
                if 'classifier' not in name.lower():
                    global_layers.append(name)
        
        return global_layers
    
    def personalized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using both global and personalized components"""
        # Use global feature extraction
        features = self.model.features(x) if hasattr(self.model, 'features') else x
        
        # Apply personalized batch norm if available
        for name, layer in self.personal_layers.items():
            if 'batchnorm' in name.lower() and isinstance(layer, nn.BatchNorm2d):
                features = layer(features)
        
        # Flatten features
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Use personalized classifier
        if 'classifier' in self.personal_layers:
            output = self.personal_layers['classifier'](features)
        else:
            output = self.model.classifier(features)
        
        return output
    
    def update_personalization(self, local_data_performance: Dict):
        """Update personalization based on local data performance"""
        self.personalization_history.append({
            'round': self.round_number,
            'local_performance': local_data_performance,
            'personalization_ratio': self.personalization_ratio
        })
        
        # Adaptive personalization ratio based on performance
        if len(self.personalization_history) > 2:
            recent_perf = local_data_performance.get('accuracy', 0.0)
            prev_perf = self.personalization_history[-2]['local_performance'].get('accuracy', 0.0)
            
            if recent_perf > prev_perf:
                # Increase personalization if local performance is improving
                self.personalization_ratio = min(0.8, self.personalization_ratio + 0.05)
            else:
                # Decrease personalization if local performance is declining
                self.personalization_ratio = max(0.1, self.personalization_ratio - 0.05)
    
    def get_personalized_weights(self) -> Dict[str, torch.Tensor]:
        """Get weights of personalized layers only"""
        personal_weights = {}
        
        for name, layer in self.personal_layers.items():
            for param_name, param in layer.named_parameters():
                personal_weights[f"{name}.{param_name}"] = param.clone()
        
        return personal_weights
    
    def apply_selective_global_update(self, global_weights: Dict[str, torch.Tensor]):
        """Apply global update only to global layers, keep personalized layers local"""
        for name, param in self.model.named_parameters():
            if name in self.global_layers and name in global_weights:
                param.data.copy_(global_weights[name])
        
        # Update tracking
        self.last_global_weights = {name: global_weights[name].clone() 
                                   for name in self.global_layers if name in global_weights}
        self.round_number += 1

class PrivacyPreservingFederatedModel(FederatedModel):
    """
    Federated model with privacy-preserving mechanisms
    Implements differential privacy and secure aggregation concepts
    """
    
    def __init__(self, base_model: nn.Module, client_id: str, 
                 privacy_budget: float = 1.0, noise_multiplier: float = 0.1):
        super().__init__(base_model, client_id)
        self.privacy_budget = privacy_budget
        self.noise_multiplier = noise_multiplier
        self.privacy_spent = 0.0
        self.dp_history = []
    
    def add_differential_privacy_noise(self, gradients: Dict[str, torch.Tensor], 
                                     sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to gradients
        
        Args:
            gradients: Model gradients
            sensitivity: L2 sensitivity of the gradients
        
        Returns:
            Noisy gradients
        """
        noisy_gradients = {}
        
        # Calculate noise scale based on differential privacy parameters
        noise_scale = sensitivity * self.noise_multiplier
        
        for name, grad in gradients.items():
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=grad.shape, device=grad.device)
            noisy_gradients[name] = grad + noise
        
        # Update privacy budget spent (simplified privacy accounting)
        privacy_cost = self.noise_multiplier ** 2 / 2  # Simplified calculation
        self.privacy_spent += privacy_cost
        
        self.dp_history.append({
            'round': self.round_number,
            'noise_scale': noise_scale,
            'privacy_cost': privacy_cost,
            'cumulative_privacy_spent': self.privacy_spent
        })
        
        logger.info(f"Added DP noise with scale {noise_scale:.4f}, "
                   f"privacy spent: {self.privacy_spent:.4f}/{self.privacy_budget}")
        
        return noisy_gradients
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor], 
                      clip_norm: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bound sensitivity for differential privacy
        
        Args:
            gradients: Model gradients
            clip_norm: Maximum L2 norm for gradient clipping
        
        Returns:
            Clipped gradients
        """
        clipped_gradients = {}
        
        # Calculate total gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += grad.norm() ** 2
        total_norm = total_norm ** 0.5
        
        # Apply clipping if necessary
        if total_norm > clip_norm:
            clip_factor = clip_norm / total_norm
            for name, grad in gradients.items():
                clipped_gradients[name] = grad * clip_factor
        else:
            clipped_gradients = gradients
        
        return clipped_gradients
    
    def secure_aggregation_mask(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply secure aggregation mask (simplified version)
        In practice, this would involve cryptographic protocols
        """
        masked_weights = {}
        
        for name, weight in weights.items():
            # Generate random mask (in practice, this would be shared with other clients)
            mask = torch.randn_like(weight) * 0.01  # Small random mask
            masked_weights[name] = weight + mask
        
        return masked_weights
    
    def check_privacy_budget(self) -> bool:
        """Check if privacy budget is exhausted"""
        return self.privacy_spent < self.privacy_budget
    
    def get_privacy_report(self) -> Dict:
        """Get privacy preservation report"""
        return {
            'client_id': self.client_id,
            'privacy_budget': self.privacy_budget,
            'privacy_spent': self.privacy_spent,
            'privacy_remaining': max(0, self.privacy_budget - self.privacy_spent),
            'noise_multiplier': self.noise_multiplier,
            'dp_rounds': len(self.dp_history),
            'avg_noise_scale': np.mean([h['noise_scale'] for h in self.dp_history]) if self.dp_history else 0,
            'can_participate': self.check_privacy_budget()
        }

class AdaptiveFederatedModel(FederatedModel):
    """
    Federated model that adapts its architecture and training based on client capabilities
    """
    
    def __init__(self, base_model: nn.Module, client_id: str, device_profile: Dict):
        super().__init__(base_model, client_id)
        self.device_profile = device_profile
        self.adapted_model = self._adapt_model_to_device()
        self.adaptation_history = []
    
    def _adapt_model_to_device(self) -> nn.Module:
        """Adapt model architecture based on device capabilities"""
        memory_mb = self.device_profile.get('memory_mb', 200)
        compute_power = self.device_profile.get('compute_power', 1.0)
        
        adapted_model = copy.deepcopy(self.model)
        
        # Reduce model size for low-memory devices
        if memory_mb < 100:  # Very low memory
            adapted_model = self._create_lightweight_model()
        elif memory_mb < 200:  # Low memory
            adapted_model = self._reduce_model_channels(adapted_model, factor=0.5)
        
        # Adjust batch norm momentum for low compute devices
        if compute_power < 0.5:
            for module in adapted_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = 0.01  # Slower adaptation for stability
        
        self.adaptation_history.append({
            'round': self.round_number,
            'memory_mb': memory_mb,
            'compute_power': compute_power,
            'adaptation_applied': 'lightweight' if memory_mb < 100 else 'reduced_channels' if memory_mb < 200 else 'none'
        })
        
        return adapted_model
    
    def _create_lightweight_model(self) -> nn.Module:
        """Create a very lightweight version of the model"""
        # Create a minimal CNN for very constrained devices
        if isinstance(self.model, BaseCNN):
            return StudentCNN(num_classes=self.model.num_classes)
        else:
            # Fallback: use original model but remove some layers
            return self._reduce_model_layers(self.model)
    
    def _reduce_model_channels(self, model: nn.Module, factor: float = 0.5) -> nn.Module:
        """Reduce the number of channels in convolutional layers"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated model pruning
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Reduce output channels
                new_out_channels = max(1, int(module.out_channels * factor))
                if new_out_channels != module.out_channels:
                    # Create new layer with reduced channels
                    new_conv = nn.Conv2d(
                        module.in_channels,
                        new_out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding
                    )
                    # Copy subset of weights
                    with torch.no_grad():
                        new_conv.weight.data = module.weight.data[:new_out_channels].clone()
                        if module.bias is not None:
                            new_conv.bias.data = module.bias.data[:new_out_channels].clone()
                    
                    # Replace the module (simplified - would need proper replacement in practice)
                    setattr(model, name.split('.')[-1], new_conv)
        
        return model
    
    def _reduce_model_layers(self, model: nn.Module) -> nn.Module:
        """Remove some layers to reduce model complexity"""
        # Simplified implementation - remove every other convolutional layer
        simplified_model = copy.deepcopy(model)
        
        # This would require more sophisticated layer removal logic
        # For now, just return the original model
        return simplified_model
    
    def adaptive_local_training(self, data_loader, optimizer, criterion, epochs: int):
        """
        Perform local training with adaptive parameters based on device capabilities
        """
        compute_power = self.device_profile.get('compute_power', 1.0)
        memory_mb = self.device_profile.get('memory_mb', 200)
        
        # Adapt training parameters
        if compute_power < 0.5:
            # Use smaller batch size for low compute devices
            effective_batch_size = max(1, data_loader.batch_size // 2)
        else:
            effective_batch_size = data_loader.batch_size
        
        # Adapt learning rate based on device stability
        if memory_mb < 100:
            lr_factor = 0.5  # Lower learning rate for unstable devices
        else:
            lr_factor = 1.0
        
        # Apply adaptations
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_factor
        
        # Record adaptation
        self.adaptation_history.append({
            'round': self.round_number,
            'training_adaptations': {
                'effective_batch_size': effective_batch_size,
                'lr_factor': lr_factor,
                'epochs': epochs
            }
        })
        
        return effective_batch_size, lr_factor
    
    def get_adaptation_summary(self) -> Dict:
        """Get summary of model adaptations"""
        if not self.adaptation_history:
            return {}
        
        latest_adaptation = self.adaptation_history[-1]
        
        return {
            'client_id': self.client_id,
            'device_profile': self.device_profile,
            'current_adaptation': latest_adaptation.get('adaptation_applied', 'none'),
            'model_size_reduction': self._calculate_size_reduction(),
            'training_adaptations': latest_adaptation.get('training_adaptations', {}),
            'adaptation_count': len(self.adaptation_history)
        }
    
    def _calculate_size_reduction(self) -> float:
        """Calculate model size reduction percentage"""
        original_params = sum(p.numel() for p in self.model.parameters())
        adapted_params = sum(p.numel() for p in self.adapted_model.parameters())
        
        if original_params == 0:
            return 0.0
        
        reduction = ((original_params - adapted_params) / original_params) * 100
        return reduction

def create_federated_model(base_model: nn.Module, client_id: str, 
                          model_type: str = "standard", **kwargs) -> FederatedModel:
    """
    Factory function to create different types of federated models
    
    Args:
        base_model: The base neural network model
        client_id: Unique identifier for the client
        model_type: Type of federated model ("standard", "personalized", "privacy_preserving", "adaptive")
        **kwargs: Additional arguments specific to the model type
    
    Returns:
        FederatedModel instance
    """
    if model_type == "standard":
        return FederatedModel(base_model, client_id)
    
    elif model_type == "personalized":
        personalization_ratio = kwargs.get('personalization_ratio', 0.3)
        return PersonalizedFederatedModel(base_model, client_id, personalization_ratio)
    
    elif model_type == "privacy_preserving":
        privacy_budget = kwargs.get('privacy_budget', 1.0)
        noise_multiplier = kwargs.get('noise_multiplier', 0.1)
        return PrivacyPreservingFederatedModel(base_model, client_id, privacy_budget, noise_multiplier)
    
    elif model_type == "adaptive":
        device_profile = kwargs.get('device_profile', {})
        return AdaptiveFederatedModel(base_model, client_id, device_profile)
    
    else:
        raise ValueError(f"Unknown federated model type: {model_type}")

def aggregate_federated_models(models: List[FederatedModel], weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    """
    Aggregate multiple federated models using weighted averaging (FedAvg)
    
    Args:
        models: List of FederatedModel instances
        weights: Optional weights for each model (default: uniform weighting)
    
    Returns:
        Aggregated model weights
    """
    if not models:
        return {}
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    if len(weights) != len(models):
        raise ValueError("Number of weights must match number of models")
    
    # Initialize aggregated weights
    aggregated_weights = {}
    
    # Get the first model's weight structure
    first_model_weights = models[0]._get_model_weights()
    
    for param_name in first_model_weights:
        aggregated_weights[param_name] = torch.zeros_like(first_model_weights[param_name])
    
    # Weighted averaging
    for model, weight in zip(models, weights):
        model_weights = model._get_model_weights()
        
        for param_name in aggregated_weights:
            if param_name in model_weights:
                aggregated_weights[param_name] += weight * model_weights[param_name]
    
    return aggregated_weights
