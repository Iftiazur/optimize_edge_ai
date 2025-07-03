"""
Federated learning server implementation
Handles model aggregation, client coordination, and global model management
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import copy
import logging
import time
import json

from .client import FederatedClient, ClientManager
from ..utils.config import FEDERATED_ROUNDS, CLIENT_FRACTION, DEVICE


class FederatedServer:
    """Central federated learning server"""
    
    def __init__(self, global_model: nn.Module, test_loader,
                 aggregation_method: str = 'fedavg'):
        """
        Initialize federated server
        
        Args:
            global_model: Global model template
            test_loader: Test data for global evaluation
            aggregation_method: Method for aggregating client updates
        """
        self.global_model = global_model.to(DEVICE)
        self.test_loader = test_loader
        self.aggregation_method = aggregation_method
        
        # Server state
        self.current_round = 0
        self.training_history = []
        self.client_metrics_history = []
        
        # Aggregation statistics
        self.aggregation_stats = {
            'total_parameters': sum(p.numel() for p in self.global_model.parameters()),
            'parameter_updates_received': 0,
            'successful_aggregations': 0
        }
        
        self.logger = logging.getLogger("FederatedServer")
        
    def federated_averaging(self, client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        """
        Perform FedAvg aggregation of client model updates
        
        Args:
            client_updates: List of (model_parameters, num_samples) tuples
            
        Returns:
            Aggregated global model parameters
        """
        if not client_updates:
            self.logger.warning("No client updates received for aggregation")
            return self.global_model.state_dict()
        
        # Calculate total samples for weighted averaging
        total_samples = sum(num_samples for _, num_samples in client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = OrderedDict()
        
        # Get parameter names from first client
        param_names = client_updates[0][0].keys()
        
        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(client_updates[0][0][param_name])
            
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                weighted_sum += weight * client_params[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        self.aggregation_stats['parameter_updates_received'] += len(client_updates)
        self.aggregation_stats['successful_aggregations'] += 1
        
        return aggregated_params
    
    def federated_median(self, client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        """
        Perform coordinate-wise median aggregation (robust to outliers)
        
        Args:
            client_updates: List of (model_parameters, num_samples) tuples
            
        Returns:
            Aggregated global model parameters using median
        """
        if not client_updates:
            return self.global_model.state_dict()
        
        aggregated_params = OrderedDict()
        param_names = client_updates[0][0].keys()
        
        for param_name in param_names:
            # Stack parameters from all clients
            param_stack = torch.stack([
                client_params[param_name] for client_params, _ in client_updates
            ])
            
            # Compute coordinate-wise median
            median_params = torch.median(param_stack, dim=0)[0]
            aggregated_params[param_name] = median_params
        
        return aggregated_params
    
    def federated_trimmed_mean(self, client_updates: List[Tuple[OrderedDict, int]], 
                              trim_ratio: float = 0.1) -> OrderedDict:
        """
        Perform trimmed mean aggregation (remove extreme values)
        
        Args:
            client_updates: List of (model_parameters, num_samples) tuples
            trim_ratio: Fraction of extreme values to trim
            
        Returns:
            Aggregated parameters using trimmed mean
        """
        if not client_updates:
            return self.global_model.state_dict()
        
        aggregated_params = OrderedDict()
        param_names = client_updates[0][0].keys()
        
        for param_name in param_names:
            param_stack = torch.stack([
                client_params[param_name] for client_params, _ in client_updates
            ])
            
            # Calculate trimmed mean
            num_trim = int(len(client_updates) * trim_ratio)
            if num_trim > 0:
                sorted_params, _ = torch.sort(param_stack, dim=0)
                trimmed_params = sorted_params[num_trim:-num_trim] if num_trim < len(client_updates)//2 else sorted_params
                aggregated_params[param_name] = torch.mean(trimmed_params, dim=0)
            else:
                aggregated_params[param_name] = torch.mean(param_stack, dim=0)
        
        return aggregated_params
    
    def aggregate_client_updates(self, client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        """
        Aggregate client model updates using specified method
        
        Args:
            client_updates: List of (model_parameters, num_samples) tuples
            
        Returns:
            Aggregated global model parameters
        """
        if self.aggregation_method == 'fedavg':
            return self.federated_averaging(client_updates)
        elif self.aggregation_method == 'fedmedian':
            return self.federated_median(client_updates)
        elif self.aggregation_method == 'fedtrimmed':
            return self.federated_trimmed_mean(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate global model on test set
        
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = self.global_model(data)
                test_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = test_loss / len(self.test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total,
            'correct_predictions': correct
        }
    
    def federated_training_round(self, client_manager: ClientManager,
                               client_fraction: float = CLIENT_FRACTION,
                               local_epochs: int = 5) -> Dict[str, Any]:
        """
        Execute one round of federated training
        
        Args:
            client_manager: Manager for federated clients
            client_fraction: Fraction of clients to participate
            local_epochs: Number of local epochs for each client
            
        Returns:
            Round statistics and metrics
        """
        round_start_time = time.time()
        
        # Select clients for this round
        selected_clients = client_manager.select_clients(client_fraction, 'random')
        self.logger.info(f"Round {self.current_round + 1}: Selected {len(selected_clients)} clients")
        
        # Distribute global model to selected clients
        global_params = self.global_model.state_dict()
        for client in selected_clients:
            client.set_model_parameters(global_params)
        
        # Collect client updates
        client_updates = []
        client_metrics = []
        
        for client in selected_clients:
            try:
                # Local training
                training_metrics = client.local_training(local_epochs, self.current_round)
                
                # Get updated parameters and sample count
                updated_params = client.get_model_parameters()
                num_samples = training_metrics['samples_processed']
                
                client_updates.append((updated_params, num_samples))
                
                # Collect client metrics
                client_summary = client.get_training_summary()
                client_summary.update(training_metrics)
                client_metrics.append(client_summary)
                
            except Exception as e:
                self.logger.error(f"Client {client.client_id} failed in round {self.current_round + 1}: {e}")
                continue
        
        if not client_updates:
            self.logger.error(f"No successful client updates in round {self.current_round + 1}")
            return {'status': 'failed', 'error': 'no_client_updates'}
        
        # Aggregate client updates
        aggregated_params = self.aggregate_client_updates(client_updates)
        self.global_model.load_state_dict(aggregated_params)
        
        # Evaluate global model
        global_metrics = self.evaluate_global_model()
        
        # Calculate round statistics
        round_time = time.time() - round_start_time
        
        round_stats = {
            'round_number': self.current_round + 1,
            'participating_clients': len(selected_clients),
            'successful_updates': len(client_updates),
            'global_accuracy': global_metrics['accuracy'],
            'global_loss': global_metrics['loss'],
            'round_time_s': round_time,
            'aggregation_method': self.aggregation_method,
            'client_metrics': client_metrics
        }
        
        # Store history
        self.training_history.append(round_stats)
        self.client_metrics_history.extend(client_metrics)
        
        self.current_round += 1
        
        self.logger.info(f"Round {self.current_round} completed. "
                        f"Global accuracy: {global_metrics['accuracy']:.2f}%, "
                        f"Time: {round_time:.2f}s")
        
        return round_stats
    
    def run_federated_training(self, client_manager: ClientManager,
                             num_rounds: int = FEDERATED_ROUNDS,
                             client_fraction: float = CLIENT_FRACTION,
                             local_epochs: int = 5) -> Dict[str, Any]:
        """
        Run complete federated training process
        
        Args:
            client_manager: Manager for federated clients
            num_rounds: Number of federated rounds
            client_fraction: Fraction of clients per round
            local_epochs: Local epochs per client
            
        Returns:
            Complete training results
        """
        self.logger.info(f"Starting federated training: {num_rounds} rounds, "
                        f"{client_fraction:.1%} client participation")
        
        training_start_time = time.time()
        
        # Initial evaluation
        initial_metrics = self.evaluate_global_model()
        self.logger.info(f"Initial global accuracy: {initial_metrics['accuracy']:.2f}%")
        
        # Training rounds
        for round_num in range(num_rounds):
            round_stats = self.federated_training_round(
                client_manager, client_fraction, local_epochs
            )
            
            if round_stats.get('status') == 'failed':
                self.logger.error(f"Training failed at round {round_num + 1}")
                break
            
            # Early stopping check (optional)
            if self._should_early_stop():
                self.logger.info(f"Early stopping at round {round_num + 1}")
                break
        
        total_training_time = time.time() - training_start_time
        
        # Final evaluation and summary
        final_metrics = self.evaluate_global_model()
        
        training_summary = {
            'initial_accuracy': initial_metrics['accuracy'],
            'final_accuracy': final_metrics['accuracy'],
            'accuracy_improvement': final_metrics['accuracy'] - initial_metrics['accuracy'],
            'total_rounds': len(self.training_history),
            'total_training_time_s': total_training_time,
            'average_round_time_s': total_training_time / len(self.training_history) if self.training_history else 0,
            'aggregation_method': self.aggregation_method,
            'training_history': self.training_history,
            'aggregation_stats': self.aggregation_stats
        }
        
        self.logger.info(f"Federated training completed. "
                        f"Final accuracy: {final_metrics['accuracy']:.2f}% "
                        f"(+{training_summary['accuracy_improvement']:.2f}%)")
        
        return training_summary
    
    def _should_early_stop(self, patience: int = 5, min_improvement: float = 0.1) -> bool:
        """
        Check if training should stop early based on convergence
        
        Args:
            patience: Number of rounds to wait for improvement
            min_improvement: Minimum accuracy improvement required
            
        Returns:
            True if should stop early
        """
        if len(self.training_history) < patience + 1:
            return False
        
        recent_accuracies = [round_data['global_accuracy'] 
                           for round_data in self.training_history[-patience-1:]]
        
        best_recent = max(recent_accuracies[:-1])
        current = recent_accuracies[-1]
        
        return (current - best_recent) < min_improvement
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence properties of federated training
        
        Returns:
            Convergence analysis results
        """
        if not self.training_history:
            return {'status': 'no_training_data'}
        
        accuracies = [round_data['global_accuracy'] for round_data in self.training_history]
        losses = [round_data['global_loss'] for round_data in self.training_history]
        
        # Calculate convergence metrics
        initial_accuracy = accuracies[0] if accuracies else 0
        final_accuracy = accuracies[-1] if accuracies else 0
        max_accuracy = max(accuracies) if accuracies else 0
        
        # Convergence rate (rounds to reach 90% of final accuracy)
        target_accuracy = initial_accuracy + 0.9 * (final_accuracy - initial_accuracy)
        convergence_round = len(accuracies)
        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                convergence_round = i + 1
                break
        
        # Stability analysis (variance in final rounds)
        final_rounds = min(5, len(accuracies))
        final_variance = np.var(accuracies[-final_rounds:]) if final_rounds > 1 else 0
        
        return {
            'initial_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy,
            'max_accuracy': max_accuracy,
            'accuracy_improvement': final_accuracy - initial_accuracy,
            'convergence_round': convergence_round,
            'convergence_rate': convergence_round / len(accuracies) if accuracies else 1,
            'final_stability_variance': final_variance,
            'training_efficiency': final_accuracy / len(accuracies) if accuracies else 0
        }
    
    def get_communication_costs(self) -> Dict[str, float]:
        """
        Calculate communication costs for federated training
        
        Returns:
            Communication cost analysis
        """
        model_size_mb = self._calculate_model_size_mb()
        
        # Communication per round: model download + model upload for each client
        # Assuming symmetric up/down communication
        avg_clients_per_round = np.mean([round_data['participating_clients'] 
                                       for round_data in self.training_history]) if self.training_history else 0
        
        communication_per_round = 2 * model_size_mb * avg_clients_per_round  # Up + Down
        total_communication = communication_per_round * len(self.training_history)
        
        return {
            'model_size_mb': model_size_mb,
            'communication_per_round_mb': communication_per_round,
            'total_communication_mb': total_communication,
            'communication_per_accuracy_point': total_communication / max(1, 
                self.training_history[-1]['global_accuracy'] - self.training_history[0]['global_accuracy']
            ) if len(self.training_history) >= 2 else 0,
            'average_clients_per_round': avg_clients_per_round
        }
    
    def _calculate_model_size_mb(self) -> float:
        """Calculate model size in MB"""
        param_size = sum(p.nelement() * p.element_size() for p in self.global_model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.global_model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def save_training_results(self, filepath: str):
        """
        Save training results to file
        
        Args:
            filepath: Path to save results
        """
        results = {
            'training_summary': {
                'total_rounds': len(self.training_history),
                'aggregation_method': self.aggregation_method,
                'final_accuracy': self.training_history[-1]['global_accuracy'] if self.training_history else 0,
                'convergence_analysis': self.analyze_convergence(),
                'communication_costs': self.get_communication_costs()
            },
            'training_history': self.training_history,
            'aggregation_stats': self.aggregation_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Training results saved to {filepath}")


def compare_aggregation_methods(global_model: nn.Module, client_manager: ClientManager,
                              test_loader, methods: Optional[List[str]] = None,
                              num_rounds: int = 10) -> Dict[str, Dict]:
    """
    Compare different federated aggregation methods
    
    Args:
        global_model: Global model template
        client_manager: Client manager
        test_loader: Test data loader
        methods: List of aggregation methods to compare
        num_rounds: Number of rounds for comparison
        
    Returns:
        Comparison results
    """
    if methods is None:
        methods = ['fedavg', 'fedmedian', 'fedtrimmed']
    
    results = {}
    
    for method in methods:
        # Create fresh server for each method
        server = FederatedServer(copy.deepcopy(global_model), test_loader, method)
        
        # Run training
        training_results = server.run_federated_training(
            client_manager, num_rounds, client_fraction=0.5, local_epochs=3
        )
        
        results[method] = {
            'final_accuracy': training_results['final_accuracy'],
            'accuracy_improvement': training_results['accuracy_improvement'],
            'convergence_analysis': server.analyze_convergence(),
            'communication_costs': server.get_communication_costs(),
            'training_time': training_results['total_training_time_s']
        }
    
    return results


if __name__ == "__main__":
    # Example usage
    from ..models.base_model import create_model
    from ..utils.data_loader import DataLoader
    
    # Create model and data
    model = create_model('base')
    data_loader = DataLoader()
    
    # Create mock federated setup
    train_loader, test_loader = data_loader.load_centralized_data()
    
    # Create server
    server = FederatedServer(model, test_loader, 'fedavg')
    
    # Test evaluation
    initial_metrics = server.evaluate_global_model()
    print(f"Initial model metrics: {initial_metrics}")
    
    print("Federated server initialized successfully")
