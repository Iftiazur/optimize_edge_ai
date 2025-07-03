"""
Edge device simulation utilities
Simulates different hardware constraints and performance characteristics
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import logging

from .config import EDGE_DEVICE_CONFIGS, ENERGY_PARAMS


@dataclass
class DeviceMetrics:
    """Data class to store device performance metrics"""
    inference_time: float
    memory_usage: float
    energy_consumption: float
    cpu_utilization: float
    accuracy: float
    throughput: float


class EdgeDeviceSimulator:
    """Simulates different edge device characteristics"""
    
    def __init__(self, device_type: str = 'desktop'):
        """
        Initialize device simulator
        
        Args:
            device_type: Type of device to simulate ('raspberry_pi', 'jetson_nano', 'mobile_cpu', 'desktop')
        """
        if device_type not in EDGE_DEVICE_CONFIGS:
            raise ValueError(f"Unknown device type: {device_type}")
        
        self.device_type = device_type
        self.config = EDGE_DEVICE_CONFIGS[device_type]
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring
        self.process = psutil.Process()
        
    def simulate_memory_constraint(self, model: torch.nn.Module) -> bool:
        """
        Check if model fits within device memory constraints
        
        Args:
            model: PyTorch model to check
            
        Returns:
            True if model fits in memory, False otherwise
        """
        model_size_mb = self.estimate_model_memory(model)
        available_memory = self.config['memory_limit']
        
        fits = model_size_mb <= available_memory
        
        if not fits:
            self.logger.warning(
                f"Model size ({model_size_mb:.1f} MB) exceeds device memory limit "
                f"({available_memory} MB) for {self.device_type}"
            )
        
        return fits
    
    def estimate_model_memory(self, model: torch.nn.Module) -> float:
        """
        Estimate model memory usage in MB
        
        Args:
            model: PyTorch model
            
        Returns:
            Estimated memory usage in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        # Convert to MB and add overhead
        memory_mb = (param_size + buffer_size) / (1024 ** 2)
        memory_mb *= 1.5  # Add 50% overhead for activations
        
        return memory_mb
    
    def measure_inference_time(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
                             num_runs: int = 100) -> Tuple[float, float]:
        """
        Measure inference time with device constraints
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor for inference
            num_runs: Number of inference runs for averaging
            
        Returns:
            Tuple of (mean_time, std_time) in milliseconds
        """
        model.eval()
        times = []
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measured runs
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                
                # Apply device performance multiplier
                actual_time = (end_time - start_time) / self.config['compute_power']
                times.append(actual_time * 1000)  # Convert to milliseconds
        
        return np.mean(times), np.std(times)
    
    def estimate_energy_consumption(self, inference_time_ms: float, 
                                  model_flops: float) -> float:
        """
        Estimate energy consumption for inference
        
        Args:
            inference_time_ms: Inference time in milliseconds
            model_flops: Number of FLOPs in the model
            
        Returns:
            Estimated energy consumption in mW
        """
        # Base energy consumption
        base_energy = ENERGY_PARAMS['cpu_active'] * (inference_time_ms / 1000)
        
        # Energy based on operations
        ops_energy = model_flops / ENERGY_PARAMS['ops_per_watt']
        
        # Apply device multiplier
        total_energy = (base_energy + ops_energy) * self.config['energy_multiplier']
        
        return total_energy
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """
        Monitor current system resource usage
        
        Returns:
            Dictionary with resource usage metrics
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        # Get process-specific metrics
        process_memory = self.process.memory_info().rss / (1024 ** 2)  # MB
        process_cpu = self.process.cpu_percent()
        
        return {
            'system_cpu_percent': cpu_percent,
            'system_memory_percent': memory_info.percent,
            'system_memory_available_mb': memory_info.available / (1024 ** 2),
            'process_memory_mb': process_memory,
            'process_cpu_percent': process_cpu
        }
    
    def calculate_throughput(self, batch_size: int, inference_time_ms: float) -> float:
        """
        Calculate inference throughput
        
        Args:
            batch_size: Size of input batch
            inference_time_ms: Inference time in milliseconds
            
        Returns:
            Throughput in samples per second
        """
        throughput = (batch_size * 1000) / inference_time_ms
        return throughput / self.config['compute_power']  # Apply device constraint
    
    def run_comprehensive_benchmark(self, model: torch.nn.Module, 
                                  test_loader: torch.utils.data.DataLoader) -> DeviceMetrics:
        """
        Run comprehensive benchmark on the device
        
        Args:
            model: PyTorch model to benchmark
            test_loader: Test data loader
            
        Returns:
            DeviceMetrics with comprehensive performance data
        """
        model.eval()
        correct = 0
        total = 0
        inference_times = []
        energy_consumptions = []
        
        # Get model FLOPs estimate
        model_flops = self.estimate_model_flops(model)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx >= 50:  # Limit for faster benchmarking
                    break
                
                # Measure inference time
                start_time = time.time()
                outputs = model(data)
                end_time = time.time()
                
                # Apply device performance constraint
                inference_time = (end_time - start_time) / self.config['compute_power']
                inference_times.append(inference_time * 1000)  # Convert to ms
                
                # Estimate energy consumption
                energy = self.estimate_energy_consumption(
                    inference_time * 1000, model_flops * data.size(0)
                )
                energy_consumptions.append(energy)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Calculate metrics
        accuracy = 100 * correct / total
        mean_inference_time = np.mean(inference_times)
        mean_energy = np.mean(energy_consumptions)
        memory_usage = self.estimate_model_memory(model)
        throughput = self.calculate_throughput(test_loader.batch_size, mean_inference_time)
        
        # Get system resources
        resources = self.monitor_system_resources()
        
        return DeviceMetrics(
            inference_time=mean_inference_time,
            memory_usage=memory_usage,
            energy_consumption=mean_energy,
            cpu_utilization=resources['process_cpu_percent'],
            accuracy=accuracy,
            throughput=throughput
        )
    
    def estimate_model_flops(self, model: torch.nn.Module, 
                           input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> float:
        """
        Estimate model FLOPs (simplified calculation)
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Estimated FLOPs
        """
        total_flops = 0
        
        def count_conv_flops(module):
            if isinstance(module, torch.nn.Conv2d):
                # Simplified FLOP calculation for conv layers
                kernel_flops = np.prod(module.kernel_size) * module.in_channels
                output_elements = np.prod(input_shape[2:]) * module.out_channels
                return kernel_flops * output_elements
            return 0
        
        def count_linear_flops(module):
            if isinstance(module, torch.nn.Linear):
                return module.in_features * module.out_features
            return 0
        
        for module in model.modules():
            total_flops += count_conv_flops(module)
            total_flops += count_linear_flops(module)
        
        return total_flops
    
    def compare_devices(self, model: torch.nn.Module, 
                       test_loader: torch.utils.data.DataLoader) -> Dict[str, DeviceMetrics]:
        """
        Compare model performance across different device types
        
        Args:
            model: PyTorch model to benchmark
            test_loader: Test data loader
            
        Returns:
            Dictionary mapping device types to their metrics
        """
        results = {}
        
        for device_type in EDGE_DEVICE_CONFIGS.keys():
            simulator = EdgeDeviceSimulator(device_type)
            metrics = simulator.run_comprehensive_benchmark(model, test_loader)
            results[device_type] = metrics
        
        return results


class NetworkSimulator:
    """Simulates network conditions for federated learning"""
    
    def __init__(self, bandwidth_mbps: float = 10.0, latency_ms: float = 100.0):
        """
        Initialize network simulator
        
        Args:
            bandwidth_mbps: Network bandwidth in Mbps
            latency_ms: Network latency in milliseconds
        """
        self.bandwidth_mbps = bandwidth_mbps
        self.latency_ms = latency_ms
    
    def calculate_transmission_time(self, model_size_mb: float) -> float:
        """
        Calculate time to transmit model
        
        Args:
            model_size_mb: Model size in MB
            
        Returns:
            Transmission time in seconds
        """
        transmission_time = (model_size_mb * 8) / self.bandwidth_mbps  # Convert to seconds
        total_time = transmission_time + (self.latency_ms / 1000)  # Add latency
        
        return total_time
    
    def simulate_packet_loss(self, success_rate: float = 0.95) -> bool:
        """
        Simulate packet loss
        
        Args:
            success_rate: Probability of successful transmission
            
        Returns:
            True if transmission successful, False otherwise
        """
        return np.random.random() < success_rate
