"""
Knowledge distillation implementation for model compression
Includes teacher-student training with temperature scaling and soft targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import logging
import copy

from ..utils.config import DISTILLATION_TEMPERATURE, DISTILLATION_ALPHA


class KnowledgeDistillation:
    """Knowledge distillation implementation"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = DISTILLATION_TEMPERATURE, 
                 alpha: float = DISTILLATION_ALPHA):
        """
        Initialize knowledge distillation
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            temperature: Temperature for softmax scaling
            alpha: Weight for combining hard and soft losses
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_outputs: torch.Tensor, 
                         teacher_outputs: torch.Tensor,
                         targets: torch.Tensor,
                         hard_loss_fn: nn.Module = nn.CrossEntropyLoss()) -> Dict[str, torch.Tensor]:
        """
        Calculate knowledge distillation loss
        
        Formula: L = α * L_hard + (1-α) * L_soft
        where L_soft = KL_divergence(soft_student, soft_teacher)
        
        Args:
            student_outputs: Student model outputs (logits)
            teacher_outputs: Teacher model outputs (logits)
            targets: Ground truth labels
            hard_loss_fn: Loss function for hard targets
            
        Returns:
            Dictionary containing different loss components
        """
        # Hard loss (student predictions vs ground truth)
        hard_loss = hard_loss_fn(student_outputs, targets)
        
        # Soft loss (student vs teacher with temperature scaling)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        
        # Scale soft loss by temperature squared (as per Hinton et al.)
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'hard_weight': self.alpha,
            'soft_weight': 1.0 - self.alpha
        }
    
    def train_step(self, data: torch.Tensor, targets: torch.Tensor,
                   student_optimizer: torch.optim.Optimizer,
                   hard_loss_fn: nn.Module = nn.CrossEntropyLoss()) -> Dict[str, float]:
        """
        Perform one training step with knowledge distillation
        
        Args:
            data: Input batch
            targets: Ground truth labels
            student_optimizer: Optimizer for student model
            hard_loss_fn: Loss function for hard targets
            
        Returns:
            Dictionary with training metrics
        """
        self.student_model.train()
        self.teacher_model.eval()
        
        # Forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model(data)
        
        student_outputs = self.student_model(data)
        
        # Calculate distillation loss
        loss_dict = self.distillation_loss(student_outputs, teacher_outputs, targets, hard_loss_fn)
        
        # Backward pass
        student_optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        student_optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(student_outputs.data, 1)
        accuracy = (predicted == targets).float().mean().item() * 100
        
        return {
            'total_loss': loss_dict['total_loss'].item(),
            'hard_loss': loss_dict['hard_loss'].item(),
            'soft_loss': loss_dict['soft_loss'].item(),
            'accuracy': accuracy,
            'temperature': self.temperature,
            'alpha': self.alpha
        }
    
    def evaluate_models(self, test_loader) -> Dict[str, float]:
        """
        Evaluate both teacher and student models
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics for both models
        """
        teacher_accuracy = self._evaluate_single_model(self.teacher_model, test_loader)
        student_accuracy = self._evaluate_single_model(self.student_model, test_loader)
        
        return {
            'teacher_accuracy': teacher_accuracy,
            'student_accuracy': student_accuracy,
            'accuracy_gap': teacher_accuracy - student_accuracy,
            'knowledge_transfer_efficiency': student_accuracy / teacher_accuracy if teacher_accuracy > 0 else 0
        }
    
    def _evaluate_single_model(self, model: nn.Module, test_loader) -> float:
        """Evaluate single model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy
    
    def analyze_knowledge_transfer(self, test_loader, num_classes: int = 10) -> Dict[str, np.ndarray]:
        """
        Analyze how knowledge is transferred from teacher to student
        
        Args:
            test_loader: Test data loader
            num_classes: Number of classes
            
        Returns:
            Dictionary with transfer analysis metrics
        """
        self.teacher_model.eval()
        self.student_model.eval()
        
        teacher_confidences = []
        student_confidences = []
        agreement_matrix = np.zeros((num_classes, num_classes))
        
        with torch.no_grad():
            for data, targets in test_loader:
                teacher_outputs = self.teacher_model(data)
                student_outputs = self.student_model(data)
                
                # Get prediction confidences
                teacher_probs = F.softmax(teacher_outputs, dim=1)
                student_probs = F.softmax(student_outputs, dim=1)
                
                teacher_confidences.extend(teacher_probs.max(dim=1)[0].cpu().numpy())
                student_confidences.extend(student_probs.max(dim=1)[0].cpu().numpy())
                
                # Build agreement matrix
                teacher_preds = teacher_outputs.argmax(dim=1)
                student_preds = student_outputs.argmax(dim=1)
                
                for t_pred, s_pred in zip(teacher_preds.cpu().numpy(), student_preds.cpu().numpy()):
                    agreement_matrix[t_pred, s_pred] += 1
        
        # Normalize agreement matrix
        agreement_matrix = agreement_matrix / agreement_matrix.sum()
        
        return {
            'teacher_confidences': np.array(teacher_confidences),
            'student_confidences': np.array(student_confidences),
            'agreement_matrix': agreement_matrix,
            'average_teacher_confidence': np.mean(teacher_confidences),
            'average_student_confidence': np.mean(student_confidences)
        }


class AdvancedDistillation:
    """Advanced distillation techniques"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        """
        Initialize advanced distillation
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.logger = logging.getLogger(__name__)
    
    def feature_distillation_loss(self, teacher_features: torch.Tensor,
                                 student_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate feature distillation loss (matching intermediate representations)
        
        Args:
            teacher_features: Teacher intermediate features
            student_features: Student intermediate features
            
        Returns:
            Feature distillation loss
        """
        # Ensure features have same dimensions (use adaptive pooling if needed)
        if teacher_features.shape != student_features.shape:
            # Adaptive pooling to match dimensions
            target_size = student_features.shape[2:]
            teacher_features = F.adaptive_avg_pool2d(teacher_features, target_size)
        
        # L2 loss between features
        loss = F.mse_loss(student_features, teacher_features)
        return loss
    
    def attention_transfer_loss(self, teacher_features: torch.Tensor,
                               student_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention transfer loss
        
        Args:
            teacher_features: Teacher feature maps
            student_features: Student feature maps
            
        Returns:
            Attention transfer loss
        """
        # Calculate attention maps (spatial attention)
        def get_attention_map(features):
            # Sum across channels and normalize
            attention = torch.sum(features ** 2, dim=1, keepdim=True)
            attention = F.normalize(attention.view(attention.size(0), -1), p=2, dim=1)
            return attention.view(attention.size(0), 1, features.size(2), features.size(3))
        
        teacher_attention = get_attention_map(teacher_features)
        student_attention = get_attention_map(student_features)
        
        # Ensure same spatial dimensions
        if teacher_attention.shape != student_attention.shape:
            target_size = student_attention.shape[2:]
            teacher_attention = F.adaptive_avg_pool2d(teacher_attention, target_size)
        
        loss = F.mse_loss(student_attention, teacher_attention)
        return loss
    
    def progressive_distillation(self, data: torch.Tensor, targets: torch.Tensor,
                               current_epoch: int, total_epochs: int,
                               optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Progressive knowledge distillation with changing temperature
        
        Args:
            data: Input data
            targets: Ground truth targets
            current_epoch: Current training epoch
            total_epochs: Total training epochs
            optimizer: Student optimizer
            
        Returns:
            Training metrics
        """
        # Progressive temperature (starts high, decreases over time)
        progress = current_epoch / total_epochs
        temperature = DISTILLATION_TEMPERATURE * (1 - progress) + 1.0 * progress
        
        # Progressive alpha (starts with more emphasis on soft targets)
        alpha = DISTILLATION_ALPHA * progress + 0.1 * (1 - progress)
        
        self.student_model.train()
        self.teacher_model.eval()
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(data)
        
        student_outputs = self.student_model(data)
        
        # Standard distillation loss with progressive parameters
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        soft_student = F.log_softmax(student_outputs / temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
        
        total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(student_outputs, 1)
        accuracy = (predicted == targets).float().mean().item() * 100
        
        return {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'accuracy': accuracy,
            'temperature': temperature,
            'alpha': alpha
        }


class OnlineDistillation:
    """Online distillation where multiple models teach each other"""
    
    def __init__(self, models: list, temperature: float = 4.0):
        """
        Initialize online distillation
        
        Args:
            models: List of models for mutual learning
            temperature: Temperature for softmax scaling
        """
        self.models = models
        self.temperature = temperature
        self.num_models = len(models)
        self.logger = logging.getLogger(__name__)
    
    def mutual_learning_loss(self, outputs_list: list, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate mutual learning loss for online distillation
        
        Args:
            outputs_list: List of outputs from all models
            targets: Ground truth targets
            
        Returns:
            Dictionary with loss components
        """
        total_losses = []
        
        for i, student_outputs in enumerate(outputs_list):
            # Hard loss
            hard_loss = F.cross_entropy(student_outputs, targets)
            
            # Soft losses from other models (acting as teachers)
            soft_losses = []
            for j, teacher_outputs in enumerate(outputs_list):
                if i != j:  # Don't use self as teacher
                    soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
                    soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
                    soft_loss = F.kl_div(soft_student, soft_teacher.detach(), reduction='batchmean')
                    soft_losses.append(soft_loss)
            
            # Average soft losses
            avg_soft_loss = torch.stack(soft_losses).mean() if soft_losses else torch.tensor(0.0)
            
            # Combined loss
            total_loss = hard_loss + avg_soft_loss * (self.temperature ** 2)
            total_losses.append(total_loss)
        
        return {
            'losses': total_losses,
            'total_loss': torch.stack(total_losses).mean()
        }


def compare_distillation_strategies(teacher_model: nn.Module, student_model: nn.Module,
                                  train_loader, test_loader, 
                                  strategies: Optional[list] = None) -> Dict[str, Dict]:
    """
    Compare different knowledge distillation strategies
    
    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model architecture
        train_loader: Training data loader
        test_loader: Test data loader
        strategies: List of strategies to compare
        
    Returns:
        Comparison results
    """
    if strategies is None:
        strategies = ['standard', 'progressive', 'high_temp', 'low_temp']
    
    results = {}
    
    # Baseline: Train student without distillation
    baseline_student = copy.deepcopy(student_model)
    baseline_optimizer = torch.optim.Adam(baseline_student.parameters(), lr=0.001)
    
    # Train baseline for a few epochs
    baseline_student.train()
    for epoch in range(5):
        for data, targets in train_loader:
            baseline_optimizer.zero_grad()
            outputs = baseline_student(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            baseline_optimizer.step()
    
    baseline_accuracy = evaluate_model_accuracy(baseline_student, test_loader)
    results['baseline'] = {'accuracy': baseline_accuracy, 'strategy': 'no_distillation'}
    
    # Test different distillation strategies
    for strategy in strategies:
        if strategy == 'standard':
            # Standard distillation
            student_copy = copy.deepcopy(student_model)
            distiller = KnowledgeDistillation(teacher_model, student_copy)
            optimizer = torch.optim.Adam(student_copy.parameters(), lr=0.001)
            
            # Train for a few epochs
            for epoch in range(5):
                for data, targets in train_loader:
                    metrics = distiller.train_step(data, targets, optimizer)
            
            accuracy = evaluate_model_accuracy(student_copy, test_loader)
            results[strategy] = {'accuracy': accuracy, 'strategy': 'standard_distillation'}
        
        elif strategy == 'high_temp':
            # High temperature distillation
            student_copy = copy.deepcopy(student_model)
            distiller = KnowledgeDistillation(teacher_model, student_copy, temperature=8.0)
            optimizer = torch.optim.Adam(student_copy.parameters(), lr=0.001)
            
            for epoch in range(5):
                for data, targets in train_loader:
                    metrics = distiller.train_step(data, targets, optimizer)
            
            accuracy = evaluate_model_accuracy(student_copy, test_loader)
            results[strategy] = {'accuracy': accuracy, 'strategy': 'high_temperature'}
        
        elif strategy == 'low_temp':
            # Low temperature distillation
            student_copy = copy.deepcopy(student_model)
            distiller = KnowledgeDistillation(teacher_model, student_copy, temperature=2.0)
            optimizer = torch.optim.Adam(student_copy.parameters(), lr=0.001)
            
            for epoch in range(5):
                for data, targets in train_loader:
                    metrics = distiller.train_step(data, targets, optimizer)
            
            accuracy = evaluate_model_accuracy(student_copy, test_loader)
            results[strategy] = {'accuracy': accuracy, 'strategy': 'low_temperature'}
    
    return results


def evaluate_model_accuracy(model: nn.Module, test_loader) -> float:
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


if __name__ == "__main__":
    # Example usage
    from ..models.base_model import create_model
    
    # Create teacher and student models
    teacher = create_model('teacher')
    student = create_model('student')
    
    print(f"Teacher model parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student model parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Initialize distillation
    distiller = KnowledgeDistillation(teacher, student)
    
    # Test forward pass
    test_input = torch.randn(4, 3, 32, 32)
    test_targets = torch.randint(0, 10, (4,))
    
    with torch.no_grad():
        teacher_outputs = teacher(test_input)
        student_outputs = student(test_input)
    
    loss_dict = distiller.distillation_loss(student_outputs, teacher_outputs, test_targets)
    print(f"Distillation loss components: {loss_dict}")
