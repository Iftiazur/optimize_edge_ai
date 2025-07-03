"""
Model Partitioning for Distributed Inference
Implements various strategies to partition CNN models across edge devices.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Tuple, Optional
import copy
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PartitionConfig:
    """Configuration for model partitioning"""
    strategy: str = "layer_wise"  # "layer_wise", "computation_based", "memory_based"
    num_partitions: int = 2
    device_capabilities: List[Dict] = None
    communication_cost: float = 10.0  # ms per MB
    
class ModelPartitioner:
    """
    Handles partitioning of neural network models for distributed inference
    """
    
    def __init__(self, model: nn.Module, config: PartitionConfig):
        self.model = model
        self.config = config
        self.partitions = []
        self.partition_points = []
        
    def analyze_model_structure(self) -> Dict:
        """Analyze model to understand its structure and computational requirements"""
        analysis = {
            'layers': [],
            'total_params': 0,
            'total_flops': 0,
            'memory_usage': []
        }
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                params = sum(p.numel() for p in module.parameters())
                analysis['layers'].append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params,
                    'memory_mb': params * 4 / (1024 * 1024)  # Assuming fp32
                })
                analysis['total_params'] += params
        
        logger.info(f"Model analysis: {len(analysis['layers'])} layers, "
                   f"{analysis['total_params']:,} parameters")
        return analysis
    
    def layer_wise_partition(self) -> List[nn.Module]:
        """Partition model by distributing layers across devices"""
        layers = list(self.model.children())
        partition_size = len(layers) // self.config.num_partitions
        
        partitions = []
        for i in range(self.config.num_partitions):
            start_idx = i * partition_size
            if i == self.config.num_partitions - 1:
                end_idx = len(layers)  # Last partition gets remaining layers
            else:
                end_idx = (i + 1) * partition_size
            
            partition_layers = layers[start_idx:end_idx]
            partition = nn.Sequential(*partition_layers)
            partitions.append(partition)
            self.partition_points.append((start_idx, end_idx))
        
        logger.info(f"Created {len(partitions)} layer-wise partitions")
        return partitions
    
    def computation_based_partition(self) -> List[nn.Module]:
        """Partition based on computational load balancing"""
        analysis = self.analyze_model_structure()
        layers = list(self.model.children())
        
        # Estimate computation cost for each layer
        layer_costs = []
        for i, layer in enumerate(layers):
            if i < len(analysis['layers']):
                cost = analysis['layers'][i]['params']  # Simple cost model
            else:
                cost = 1000  # Default cost
            layer_costs.append(cost)
        
        total_cost = sum(layer_costs)
        target_cost = total_cost / self.config.num_partitions
        
        partitions = []
        current_partition = []
        current_cost = 0
        
        for i, (layer, cost) in enumerate(zip(layers, layer_costs)):
            current_partition.append(layer)
            current_cost += cost
            
            if (current_cost >= target_cost and len(partitions) < self.config.num_partitions - 1) or i == len(layers) - 1:
                partition = nn.Sequential(*current_partition)
                partitions.append(partition)
                current_partition = []
                current_cost = 0
        
        logger.info(f"Created {len(partitions)} computation-balanced partitions")
        return partitions
    
    def memory_based_partition(self, memory_limits: List[float]) -> List[nn.Module]:
        """Partition based on device memory constraints"""
        analysis = self.analyze_model_structure()
        layers = list(self.model.children())
        
        partitions = []
        current_partition = []
        current_memory = 0
        partition_idx = 0
        
        for i, layer in enumerate(layers):
            if i < len(analysis['layers']):
                layer_memory = analysis['layers'][i]['memory_mb']
            else:
                layer_memory = 1.0  # Default 1MB
            
            if (current_memory + layer_memory <= memory_limits[partition_idx] 
                or len(current_partition) == 0):
                current_partition.append(layer)
                current_memory += layer_memory
            else:
                # Finalize current partition
                if current_partition:
                    partition = nn.Sequential(*current_partition)
                    partitions.append(partition)
                
                # Start new partition
                partition_idx += 1
                if partition_idx >= len(memory_limits):
                    partition_idx = len(memory_limits) - 1
                
                current_partition = [layer]
                current_memory = layer_memory
        
        # Add final partition
        if current_partition:
            partition = nn.Sequential(*current_partition)
            partitions.append(partition)
        
        logger.info(f"Created {len(partitions)} memory-constrained partitions")
        return partitions
    
    def partition_model(self) -> List[nn.Module]:
        """Main method to partition the model based on strategy"""
        if self.config.strategy == "layer_wise":
            self.partitions = self.layer_wise_partition()
        elif self.config.strategy == "computation_based":
            self.partitions = self.computation_based_partition()
        elif self.config.strategy == "memory_based":
            memory_limits = [cap.get('memory_mb', 100) for cap in (self.config.device_capabilities or [])]
            if not memory_limits:
                memory_limits = [100] * self.config.num_partitions  # Default 100MB per device
            self.partitions = self.memory_based_partition(memory_limits)
        else:
            raise ValueError(f"Unknown partitioning strategy: {self.config.strategy}")
        
        return self.partitions
    
    def estimate_communication_cost(self, input_tensor: torch.Tensor) -> Dict:
        """Estimate communication costs between partitions"""
        costs = {
            'total_cost_ms': 0,
            'partition_costs': [],
            'data_sizes_mb': []
        }
        
        if not self.partitions:
            return costs
        
        current_tensor = input_tensor
        
        for i, partition in enumerate(self.partitions[:-1]):  # Exclude last partition
            # Forward pass to get intermediate output
            with torch.no_grad():
                current_tensor = partition(current_tensor)
            
            # Calculate data transfer size
            data_size_mb = current_tensor.numel() * current_tensor.element_size() / (1024 * 1024)
            transfer_cost_ms = data_size_mb * self.config.communication_cost
            
            costs['partition_costs'].append(transfer_cost_ms)
            costs['data_sizes_mb'].append(data_size_mb)
            costs['total_cost_ms'] += transfer_cost_ms
        
        logger.info(f"Total communication cost: {costs['total_cost_ms']:.2f} ms")
        return costs
    
    def get_partition_info(self) -> Dict:
        """Get detailed information about the partitions"""
        if not self.partitions:
            return {}
        
        info = {
            'num_partitions': len(self.partitions),
            'strategy': self.config.strategy,
            'partitions': []
        }
        
        for i, partition in enumerate(self.partitions):
            partition_info = {
                'partition_id': i,
                'num_layers': len(list(partition.children())),
                'num_parameters': sum(p.numel() for p in partition.parameters()),
                'memory_mb': sum(p.numel() * p.element_size() for p in partition.parameters()) / (1024 * 1024)
            }
            info['partitions'].append(partition_info)
        
        return info

class AdaptivePartitioner:
    """
    Advanced partitioner that adapts to device capabilities and network conditions
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device_profiles = {}
        self.network_conditions = {}
        
    def profile_device(self, device_id: str, capabilities: Dict):
        """Profile a device's capabilities"""
        self.device_profiles[device_id] = {
            'compute_power': capabilities.get('compute_power', 1.0),
            'memory_mb': capabilities.get('memory_mb', 100),
            'bandwidth_mbps': capabilities.get('bandwidth_mbps', 10),
            'latency_ms': capabilities.get('latency_ms', 50),
            'energy_budget': capabilities.get('energy_budget', 1000)  # mW
        }
        
    def update_network_conditions(self, conditions: Dict):
        """Update current network conditions"""
        self.network_conditions = {
            'bandwidth_mbps': conditions.get('bandwidth_mbps', 10),
            'latency_ms': conditions.get('latency_ms', 50),
            'packet_loss': conditions.get('packet_loss', 0.01),
            'congestion': conditions.get('congestion', 0.1)
        }
        
    def optimize_partitioning(self, optimization_target: str = "latency") -> PartitionConfig:
        """
        Optimize partitioning strategy based on target metric
        
        Args:
            optimization_target: "latency", "energy", "accuracy", "throughput"
        """
        if optimization_target == "latency":
            return self._optimize_for_latency()
        elif optimization_target == "energy":
            return self._optimize_for_energy()
        elif optimization_target == "throughput":
            return self._optimize_for_throughput()
        else:
            # Default to computation-based partitioning
            return PartitionConfig(
                strategy="computation_based",
                num_partitions=len(self.device_profiles),
                device_capabilities=list(self.device_profiles.values())
            )
    
    def _optimize_for_latency(self) -> PartitionConfig:
        """Optimize partitioning to minimize end-to-end latency"""
        # Choose strategy that minimizes communication overhead
        num_devices = len(self.device_profiles)
        avg_bandwidth = sum(d['bandwidth_mbps'] for d in self.device_profiles.values()) / num_devices
        
        if avg_bandwidth > 50:  # High bandwidth - can afford more partitions
            strategy = "computation_based"
            num_partitions = min(num_devices, 4)
        else:  # Low bandwidth - minimize communication
            strategy = "layer_wise"
            num_partitions = min(num_devices, 2)
        
        return PartitionConfig(
            strategy=strategy,
            num_partitions=num_partitions,
            device_capabilities=list(self.device_profiles.values()),
            communication_cost=50.0 / avg_bandwidth  # Inverse relationship
        )
    
    def _optimize_for_energy(self) -> PartitionConfig:
        """Optimize partitioning to minimize energy consumption"""
        # Balance computation across devices to minimize peak energy usage
        return PartitionConfig(
            strategy="computation_based",
            num_partitions=len(self.device_profiles),
            device_capabilities=list(self.device_profiles.values()),
            communication_cost=5.0  # Lower priority on communication cost
        )
    
    def _optimize_for_throughput(self) -> PartitionConfig:
        """Optimize partitioning to maximize inference throughput"""
        # Use all available devices to maximize parallel processing
        return PartitionConfig(
            strategy="memory_based",
            num_partitions=len(self.device_profiles),
            device_capabilities=list(self.device_profiles.values()),
            communication_cost=20.0
        )

def benchmark_partitioning_strategies(model: nn.Module, input_tensor: torch.Tensor, 
                                    device_configs: List[Dict]) -> Dict:
    """
    Benchmark different partitioning strategies
    
    Args:
        model: Model to partition
        input_tensor: Sample input for testing
        device_configs: List of device capability dictionaries
    
    Returns:
        Dictionary with benchmark results for each strategy
    """
    strategies = ["layer_wise", "computation_based", "memory_based"]
    results = {}
    
    for strategy in strategies:
        config = PartitionConfig(
            strategy=strategy,
            num_partitions=len(device_configs),
            device_capabilities=device_configs
        )
        
        partitioner = ModelPartitioner(model, config)
        partitions = partitioner.partition_model()
        
        # Measure partitioning time
        start_time = time.time()
        comm_costs = partitioner.estimate_communication_cost(input_tensor)
        partition_time = time.time() - start_time
        
        partition_info = partitioner.get_partition_info()
        
        results[strategy] = {
            'partitioning_time_ms': partition_time * 1000,
            'communication_cost_ms': comm_costs['total_cost_ms'],
            'num_partitions': len(partitions),
            'partition_info': partition_info,
            'load_balance_score': calculate_load_balance_score(partition_info)
        }
    
    logger.info(f"Benchmarked {len(strategies)} partitioning strategies")
    return results

def calculate_load_balance_score(partition_info: Dict) -> float:
    """Calculate load balance score (1.0 = perfectly balanced, 0.0 = completely unbalanced)"""
    if not partition_info.get('partitions'):
        return 0.0
    
    param_counts = [p['num_parameters'] for p in partition_info['partitions']]
    if not param_counts:
        return 0.0
    
    mean_params = sum(param_counts) / len(param_counts)
    if mean_params == 0:
        return 1.0
    
    # Calculate coefficient of variation (lower is better balanced)
    variance = sum((count - mean_params) ** 2 for count in param_counts) / len(param_counts)
    std_dev = variance ** 0.5
    cv = std_dev / mean_params
    
    # Convert to score (1 - normalized CV)
    score = max(0.0, 1.0 - cv)
    return score
