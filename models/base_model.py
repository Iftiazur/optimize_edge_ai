"""
Base CNN model implementations for CIFAR-10 classification
Includes both teacher (large) and student (small) models for knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

from ..utils.config import NUM_CLASSES, INPUT_CHANNELS


class BaseCNN(nn.Module):
    """Base CNN model for CIFAR-10 classification"""
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(BaseCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Global average pooling
        x = self.adaptive_pool(x)  # 4x4 -> 1x1
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Extract intermediate feature maps for analysis"""
        features = []
        
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        features.append(x.clone())
        x = F.relu(self.bn2(self.conv2(x)))
        features.append(x.clone())
        x = self.pool(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        features.append(x.clone())
        x = F.relu(self.bn4(self.conv4(x)))
        features.append(x.clone())
        x = self.pool(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        features.append(x.clone())
        x = F.relu(self.bn6(self.conv6(x)))
        features.append(x.clone())
        
        return tuple(features)


class TeacherModel(nn.Module):
    """Large teacher model for knowledge distillation"""
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(TeacherModel, self).__init__()
        
        # Larger architecture with more parameters
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(192)
        
        self.conv7 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(384)
        self.conv8 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(384)
        self.conv9 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(384)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Larger fully connected layers
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher network"""
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Second block
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        # Third block
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.pool(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class StudentModel(nn.Module):
    """Smaller student model for knowledge distillation"""
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(StudentModel, self).__init__()
        
        # Lightweight architecture
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student network"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MobileNetV2(nn.Module):
    """MobileNetV2-inspired lightweight model for mobile deployment"""
    
    def __init__(self, num_classes: int = NUM_CLASSES, width_multiplier: float = 1.0):
        super(MobileNetV2, self).__init__()
        
        # Depthwise separable convolutions
        def conv_bn_relu(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                # Pointwise
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        
        # Calculate channels based on width multiplier
        channels = [int(c * width_multiplier) for c in [32, 64, 128, 256, 512, 1024]]
        
        self.model = nn.Sequential(
            conv_bn_relu(3, channels[0], 2),
            conv_dw(channels[0], channels[1], 1),
            conv_dw(channels[1], channels[2], 2),
            conv_dw(channels[2], channels[2], 1),
            conv_dw(channels[2], channels[3], 2),
            conv_dw(channels[3], channels[3], 1),
            conv_dw(channels[3], channels[4], 2),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[5], 2),
            conv_dw(channels[5], channels[5], 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(channels[5], num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.model(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x


def get_model_info(model: nn.Module) -> dict:
    """Get comprehensive model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': size_mb,
        'architecture': model.__class__.__name__
    }


def create_model(model_type: str = 'base', **kwargs) -> nn.Module:
    """
    Factory function to create different model types
    
    Args:
        model_type: Type of model ('base', 'teacher', 'student', 'mobilenet')
        **kwargs: Additional arguments for model creation
    
    Returns:
        PyTorch model instance
    """
    if model_type == 'base':
        return BaseCNN(**kwargs)
    elif model_type == 'teacher':
        return TeacherModel(**kwargs)
    elif model_type == 'student':
        return StudentModel(**kwargs)
    elif model_type == 'mobilenet':
        return MobileNetV2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and info
    models = {
        'Base CNN': create_model('base'),
        'Teacher Model': create_model('teacher'),
        'Student Model': create_model('student'),
        'MobileNetV2': create_model('mobilenet')
    }
    
    for name, model in models.items():
        info = get_model_info(model)
        print(f"\n{name}:")
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Size: {info['model_size_mb']:.2f} MB")
        
        # Test forward pass
        test_input = torch.randn(1, 3, 32, 32)
        output = model(test_input)
        print(f"  Output shape: {output.shape}")
