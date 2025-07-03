"""
Federated learning client implementation
Handles local training, model updates, and communication with server
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy
import logging
import time
from collections import OrderedDict

from ..utils.config import LOCAL_EPOCHS, LEARNING_RATE, DEVICE
from ..utils.device_simulator import EdgeDeviceSimulator


class FederatedClient:
    """Individual federated learning client"""
    
    def __init__(self, client_id: int, model: nn.Module, train_loader,
                 device_type: str = 'desktop', learning_rate: float = LEARNING_RATE):
        """
        Initialize federated client
        
        Args:
            client_id: Unique identifier for the client
            model: Local model (copy of global model)
            train_loader: Local training data loader
            device_type: Type of edge device this client simulates
            learning_rate: Learning rate for local training
        """
        self.client_id = client_id
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.device_type = device_type
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), 
                                 lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # Device simulator for hardware constraints
        self.device_simulator = EdgeDeviceSimulator(device_type)
        
        # Training metrics
        self.local_epochs_completed = 0
        self.total_samples_processed = 0
        self.training_history = []
        
        self.logger = logging.getLogger(f"Client_{client_id}")
        
    def local_training(self, epochs: int = LOCAL_EPOCHS, 
                      round_number: int = 0) -> Dict[str, Any]:
        """
        Perform local training on client data
        
        Args:
            epochs: Number of local epochs to train
            round_number: Current federated round number
            
        Returns:
            Dictionary with training metrics and statistics
        """
        self.model.train()
        
        training_metrics = {
            'losses': [],
            'accuracies': [],
            'training_time': 0,
            'samples_processed': 0,
            'epochs_completed': epochs,
            'round_number': round_number
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # Apply device constraints (simulate slower computation)
                compute_delay = 1.0 / self.device_simulator.config['compute_power']
                if compute_delay > 1.0:
                    time.sleep((compute_delay - 1.0) * 0.001)  # Small delay for simulation
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
                
                training_metrics['samples_processed'] += target.size(0)
            
            # Calculate epoch metrics
            epoch_accuracy = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            
            training_metrics['losses'].append(avg_epoch_loss)
            training_metrics['accuracies'].append(epoch_accuracy)
            
            self.logger.debug(f"Client {self.client_id}, Epoch {epoch + 1}/{epochs}, "
                            f"Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        training_metrics['training_time'] = time.time() - start_time
        
        # Update client state
        self.local_epochs_completed += epochs
        self.total_samples_processed += training_metrics['samples_processed']
        self.training_history.append(training_metrics)
        
        return training_metrics
    
    def get_model_parameters(self) -> OrderedDict:
        """
        Get current model parameters
        
        Returns:
            OrderedDict of model parameters
        """
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_parameters(self, parameters: OrderedDict):
        """
        Set model parameters from global model
        
        Args:
            parameters: Model parameters to set
        """
        self.model.load_state_dict(parameters)
    
    def calculate_model_delta(self, global_parameters: OrderedDict) -> OrderedDict:
        """
        Calculate the difference between local and global model parameters
        
        Args:
            global_parameters: Global model parameters
            
        Returns:
            Parameter differences (delta)
        """
        local_params = self.get_model_parameters()
        delta = OrderedDict()
        
        for key in local_params.keys():
            delta[key] = local_params[key] - global_parameters[key]
        
        return delta
    
    def apply_differential_privacy(self, parameters: OrderedDict, 
                                 noise_scale: float = 0.1) -> OrderedDict:
        """
        Apply differential privacy by adding noise to parameters
        
        Args:
            parameters: Model parameters
            noise_scale: Scale of Gaussian noise to add
            
        Returns:
            Parameters with added noise
        """
        noisy_parameters = OrderedDict()
        
        for key, param in parameters.items():
            noise = torch.normal(0, noise_scale, size=param.shape).to(param.device)
            noisy_parameters[key] = param + noise
        
        return noisy_parameters
    
    def evaluate_local_model(self, test_loader=None) -> Dict[str, float]:
        """
        Evaluate local model performance
        
        Args:
            test_loader: Test data loader (if None, uses training data)
            
        Returns:
            Evaluation metrics
        """
        if test_loader is None:
            test_loader = self.train_loader
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = self.model(data)
                test_loss += self.criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = test_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total,
            'correct_predictions': correct
        }
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about local data distribution
        
        Returns:
            Data distribution statistics
        """
        label_counts = {}
        total_samples = 0
        
        for _, targets in self.train_loader:
            for target in targets:
                label = target.item()
                label_counts[label] = label_counts.get(label, 0) + 1
                total_samples += 1
        
        # Calculate class distribution
        class_distribution = {k: v / total_samples for k, v in label_counts.items()}
        
        # Calculate data heterogeneity (entropy)
        entropy = -sum(p * np.log2(p) for p in class_distribution.values() if p > 0)
        
        return {
            'total_samples': total_samples,
            'num_classes': len(label_counts),
            'class_counts': label_counts,
            'class_distribution': class_distribution,
            'entropy': entropy,
            'client_id': self.client_id,
            'device_type': self.device_type
        }
    
    def simulate_communication_constraints(self, parameters: OrderedDict, 
                                         bandwidth_mbps: float = 10.0,
                                         latency_ms: float = 100.0) -> Dict[str, float]:
        """
        Simulate communication constraints for parameter transmission
        
        Args:
            parameters: Model parameters to transmit
            bandwidth_mbps: Available bandwidth in Mbps
            latency_ms: Network latency in milliseconds
            
        Returns:
            Communication metrics
        """
        # Calculate parameter size
        param_size_bytes = 0
        for param in parameters.values():
            param_size_bytes += param.nelement() * param.element_size()
        
        param_size_mb = param_size_bytes / (1024 ** 2)
        
        # Calculate transmission time
        transmission_time = (param_size_mb * 8) / bandwidth_mbps  # Convert to seconds
        total_time = transmission_time + (latency_ms / 1000)  # Add latency
        
        # Apply device-specific communication constraints
        device_multiplier = 2.0 - self.device_simulator.config['compute_power']
        total_time *= device_multiplier
        
        return {
            'parameter_size_mb': param_size_mb,
            'transmission_time_s': transmission_time,
            'total_communication_time_s': total_time,
            'effective_bandwidth_mbps': bandwidth_mbps / device_multiplier
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary
        
        Returns:
            Training summary with all metrics
        """
        if not self.training_history:
            return {'status': 'no_training_completed'}
        
        # Aggregate metrics across all rounds
        total_training_time = sum(round_data['training_time'] for round_data in self.training_history)
        total_samples = sum(round_data['samples_processed'] for round_data in self.training_history)
        
        # Latest metrics
        latest_round = self.training_history[-1]
        
        return {
            'client_id': self.client_id,
            'device_type': self.device_type,
            'total_rounds_participated': len(self.training_history),
            'total_local_epochs': self.local_epochs_completed,
            'total_samples_processed': total_samples,
            'total_training_time_s': total_training_time,
            'latest_accuracy': latest_round['accuracies'][-1] if latest_round['accuracies'] else 0,
            'latest_loss': latest_round['losses'][-1] if latest_round['losses'] else 0,
            'average_training_time_per_round': total_training_time / len(self.training_history),
            'samples_per_second': total_samples / total_training_time if total_training_time > 0 else 0,
            'data_statistics': self.get_data_statistics()
        }


class ClientManager:
    """Manages multiple federated learning clients"""
    
    def __init__(self, num_clients: int, model_template: nn.Module, 
                 client_data_loaders: List, device_types: Optional[List[str]] = None):
        """
        Initialize client manager
        
        Args:
            num_clients: Number of clients to create
            model_template: Template model to copy for each client
            client_data_loaders: List of data loaders for each client
            device_types: List of device types for each client
        """
        self.num_clients = num_clients
        self.model_template = model_template
        
        if device_types is None:
            # Distribute device types evenly
            available_types = ['desktop', 'jetson_nano', 'raspberry_pi', 'mobile_cpu']
            device_types = [available_types[i % len(available_types)] for i in range(num_clients)]
        
        # Create clients
        self.clients = []
        for i in range(num_clients):
            client_model = copy.deepcopy(model_template)
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_loader=client_data_loaders[i],
                device_type=device_types[i]
            )
            self.clients.append(client)
        
        self.logger = logging.getLogger("ClientManager")
        
    def select_clients(self, fraction: float = 0.5, 
                      selection_strategy: str = 'random') -> List[FederatedClient]:
        """
        Select subset of clients for training round
        
        Args:
            fraction: Fraction of clients to select
            selection_strategy: Strategy for client selection
            
        Returns:
            List of selected clients
        """
        num_selected = max(1, int(self.num_clients * fraction))
        
        if selection_strategy == 'random':
            selected_indices = np.random.choice(self.num_clients, num_selected, replace=False)
            selected_clients = [self.clients[i] for i in selected_indices]
        
        elif selection_strategy == 'high_resource':
            # Prefer clients with higher compute resources
            resource_scores = []
            for client in self.clients:
                score = client.device_simulator.config['compute_power']
                resource_scores.append(score)
            
            selected_indices = np.argsort(resource_scores)[-num_selected:]
            selected_clients = [self.clients[i] for i in selected_indices]
        
        elif selection_strategy == 'balanced':
            # Balance different device types
            device_type_counts = {}
            for client in self.clients:
                device_type = client.device_type
                device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1
            
            # Select proportionally from each device type
            selected_clients = []
            for device_type in device_type_counts.keys():
                type_clients = [c for c in self.clients if c.device_type == device_type]
                type_selection = max(1, num_selected // len(device_type_counts))
                selected = np.random.choice(type_clients, 
                                          min(type_selection, len(type_clients)), 
                                          replace=False)
                selected_clients.extend(selected)
            
            # Fill remaining slots randomly
            while len(selected_clients) < num_selected:
                remaining_clients = [c for c in self.clients if c not in selected_clients]
                if remaining_clients:
                    selected_clients.append(np.random.choice(remaining_clients))
                else:
                    break
        
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")
        
        self.logger.info(f"Selected {len(selected_clients)} clients using {selection_strategy} strategy")
        return selected_clients
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all clients
        
        Returns:
            Aggregated client statistics
        """
        device_type_distribution = {}
        total_data_samples = 0
        client_data_sizes = []
        
        for client in self.clients:
            device_type = client.device_type
            device_type_distribution[device_type] = device_type_distribution.get(device_type, 0) + 1
            
            data_stats = client.get_data_statistics()
            total_data_samples += data_stats['total_samples']
            client_data_sizes.append(data_stats['total_samples'])
        
        return {
            'total_clients': self.num_clients,
            'device_type_distribution': device_type_distribution,
            'total_data_samples': total_data_samples,
            'average_data_per_client': total_data_samples / self.num_clients,
            'min_data_per_client': min(client_data_sizes),
            'max_data_per_client': max(client_data_sizes),
            'data_distribution_std': np.std(client_data_sizes)
        }
    
    def simulate_client_availability(self, availability_prob: float = 0.8) -> List[bool]:
        """
        Simulate client availability for federated round
        
        Args:
            availability_prob: Probability that each client is available
            
        Returns:
            List of boolean values indicating client availability
        """
        availability = []
        for client in self.clients:
            # Device-dependent availability (more powerful devices are more reliable)
            device_reliability = client.device_simulator.config['compute_power']
            adjusted_prob = availability_prob * (0.5 + 0.5 * device_reliability)
            available = np.random.random() < adjusted_prob
            availability.append(available)
        
        return availability


if __name__ == "__main__":
    # Example usage
    from ..models.base_model import create_model
    from ..utils.data_loader import DataLoader
    
    # Create model and data
    model = create_model('base')
    data_loader = DataLoader()
    
    # Create mock data loaders for testing
    train_loader, _ = data_loader.load_centralized_data()
    
    # Create a single client for testing
    client = FederatedClient(
        client_id=0,
        model=model,
        train_loader=train_loader,
        device_type='raspberry_pi'
    )
    
    print(f"Client {client.client_id} created with {client.device_type} device")
    print(f"Data statistics: {client.get_data_statistics()}")
    
    # Test local training
    metrics = client.local_training(epochs=1)
    print(f"Training metrics: {metrics}")
    
    # Test evaluation
    eval_metrics = client.evaluate_local_model()
    print(f"Evaluation metrics: {eval_metrics}")
