"""
Distributed Inference Engine
Implements distributed inference across multiple edge devices with model partitioning.
"""

import torch
import torch.nn as nn
import asyncio
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import numpy as np
from .partitioning import ModelPartitioner, PartitionConfig

logger = logging.getLogger(__name__)

@dataclass
class DeviceProfile:
    """Profile of an edge device"""
    device_id: str
    compute_power: float  # Relative compute capability (1.0 = baseline)
    memory_mb: float
    bandwidth_mbps: float
    latency_ms: float
    energy_budget_mw: float
    is_available: bool = True

@dataclass
class InferenceResult:
    """Result of distributed inference"""
    output: torch.Tensor
    total_time_ms: float
    device_times: Dict[str, float]
    communication_time_ms: float
    energy_consumed_mw: float
    success: bool
    error_message: str = ""

class EdgeDevice:
    """Simulates an edge device for distributed inference"""
    
    def __init__(self, profile: DeviceProfile):
        self.profile = profile
        self.current_load = 0.0
        self.partition = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.is_busy = False
        self.stats = {
            'inferences_completed': 0,
            'total_compute_time_ms': 0,
            'total_energy_consumed_mw': 0,
            'errors': 0
        }
    
    def load_partition(self, partition: nn.Module):
        """Load a model partition onto this device"""
        self.partition = partition
        if self.partition:
            self.partition.eval()  # Set to evaluation mode
        logger.info(f"Device {self.profile.device_id} loaded partition with "
                   f"{sum(p.numel() for p in partition.parameters()):,} parameters")
    
    def estimate_inference_time(self, input_tensor: torch.Tensor) -> float:
        """Estimate inference time based on device capabilities"""
        if self.partition is None:
            return float('inf')
        
        # Simple model: time = base_time / compute_power + load_penalty
        flops = self._estimate_flops(input_tensor)
        base_time_ms = flops / (self.profile.compute_power * 1e6)  # Assuming 1M FLOPS baseline
        load_penalty = self.current_load * 50  # 50ms penalty per load unit
        
        return base_time_ms + load_penalty
    
    def estimate_energy_consumption(self, compute_time_ms: float) -> float:
        """Estimate energy consumption for given compute time"""
        # Simple linear model: energy = base_power * time * efficiency
        base_power_mw = 500  # 500mW baseline
        efficiency = self.profile.compute_power  # Higher compute power = more efficient
        return (base_power_mw / efficiency) * (compute_time_ms / 1000)
    
    def _estimate_flops(self, input_tensor: torch.Tensor) -> float:
        """Estimate FLOPs for the partition with given input"""
        if self.partition is None:
            return 0
        
        flops = 0
        for module in self.partition.modules():
            if isinstance(module, nn.Conv2d):
                # For conv2d: output_elements * kernel_flops
                h_out = (input_tensor.shape[2] + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
                w_out = (input_tensor.shape[3] + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1
                output_elements = input_tensor.shape[0] * module.out_channels * h_out * w_out
                kernel_flops = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                flops += output_elements * kernel_flops
            elif isinstance(module, nn.Linear):
                flops += input_tensor.shape[0] * module.in_features * module.out_features
        
        return flops
    
    async def inference(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Perform inference on this device"""
        if self.partition is None:
            raise RuntimeError(f"Device {self.profile.device_id} has no partition loaded")
        
        if self.is_busy:
            raise RuntimeError(f"Device {self.profile.device_id} is busy")
        
        self.is_busy = True
        start_time = time.time()
        
        try:
            # Simulate computation delay
            estimated_time = self.estimate_inference_time(input_tensor)
            await asyncio.sleep(estimated_time / 1000)  # Convert ms to seconds
            
            # Perform actual inference
            with torch.no_grad():
                output = self.partition(input_tensor)
            
            compute_time_ms = (time.time() - start_time) * 1000
            energy_consumed = self.estimate_energy_consumption(compute_time_ms)
            
            # Update statistics
            self.stats['inferences_completed'] += 1
            self.stats['total_compute_time_ms'] += compute_time_ms
            self.stats['total_energy_consumed_mw'] += energy_consumed
            
            result_info = {
                'device_id': self.profile.device_id,
                'compute_time_ms': compute_time_ms,
                'energy_consumed_mw': energy_consumed,
                'output_shape': list(output.shape),
                'success': True
            }
            
            return output, result_info
            
        except Exception as e:
            self.stats['errors'] += 1
            error_info = {
                'device_id': self.profile.device_id,
                'compute_time_ms': 0,
                'energy_consumed_mw': 0,
                'output_shape': [],
                'success': False,
                'error': str(e)
            }
            raise RuntimeError(f"Inference failed on device {self.profile.device_id}: {e}")
            
        finally:
            self.is_busy = False
    
    def get_stats(self) -> Dict:
        """Get device statistics"""
        return {
            'device_id': self.profile.device_id,
            'profile': self.profile,
            'current_load': self.current_load,
            'is_busy': self.is_busy,
            'stats': self.stats.copy()
        }

class DistributedInferenceEngine:
    """Main engine for coordinating distributed inference across edge devices"""
    
    def __init__(self, devices: List[EdgeDevice]):
        self.devices = {device.profile.device_id: device for device in devices}
        self.partitioner = None
        self.partition_config = None
        self.inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_time_ms': 0,
            'total_energy_mw': 0
        }
    
    def setup_model_partitioning(self, model: nn.Module, partition_config: PartitionConfig):
        """Setup model partitioning across devices"""
        self.partition_config = partition_config
        self.partitioner = ModelPartitioner(model, partition_config)
        partitions = self.partitioner.partition_model()
        
        if len(partitions) > len(self.devices):
            raise ValueError(f"Not enough devices ({len(self.devices)}) for partitions ({len(partitions)})")
        
        # Assign partitions to devices
        device_list = list(self.devices.values())
        for i, partition in enumerate(partitions):
            if i < len(device_list):
                device_list[i].load_partition(partition)
        
        logger.info(f"Distributed {len(partitions)} partitions across {len(device_list)} devices")
    
    async def distributed_inference(self, input_tensor: torch.Tensor) -> InferenceResult:
        """Perform distributed inference across all devices"""
        start_time = time.time()
        
        try:
            current_tensor = input_tensor
            total_communication_time = 0
            total_energy = 0
            device_times = {}
            
            # Get devices with loaded partitions
            active_devices = [device for device in self.devices.values() if device.partition is not None]
            
            if not active_devices:
                raise RuntimeError("No devices have partitions loaded")
            
            # Sequential inference across partitions
            for device in active_devices:
                # Simulate data transfer time (except for first device)
                if device != active_devices[0]:
                    data_size_mb = current_tensor.numel() * current_tensor.element_size() / (1024 * 1024)
                    transfer_time_ms = data_size_mb * (1000 / device.profile.bandwidth_mbps)
                    total_communication_time += transfer_time_ms
                    await asyncio.sleep(transfer_time_ms / 1000)
                
                # Perform inference on current device
                current_tensor, device_info = await device.inference(current_tensor)
                
                device_times[device.profile.device_id] = device_info['compute_time_ms']
                total_energy += device_info['energy_consumed_mw']
            
            total_time = (time.time() - start_time) * 1000
            
            # Update global statistics
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['successful_inferences'] += 1
            self.inference_stats['total_time_ms'] += total_time
            self.inference_stats['total_energy_mw'] += total_energy
            
            return InferenceResult(
                output=current_tensor,
                total_time_ms=total_time,
                device_times=device_times,
                communication_time_ms=total_communication_time,
                energy_consumed_mw=total_energy,
                success=True
            )
            
        except Exception as e:
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['failed_inferences'] += 1
            
            return InferenceResult(
                output=torch.tensor([]),
                total_time_ms=(time.time() - start_time) * 1000,
                device_times={},
                communication_time_ms=0,
                energy_consumed_mw=0,
                success=False,
                error_message=str(e)
            )
    
    async def batch_inference(self, input_batch: torch.Tensor, batch_size: int = 1) -> List[InferenceResult]:
        """Perform batch inference with potential parallelization"""
        results = []
        
        # Split batch into individual samples or smaller batches
        num_samples = input_batch.shape[0]
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_input = input_batch[i:end_idx]
            
            result = await self.distributed_inference(batch_input)
            results.append(result)
        
        return results
    
    def benchmark_inference(self, test_inputs: List[torch.Tensor], runs: int = 10) -> Dict:
        """Benchmark distributed inference performance"""
        benchmark_results = {
            'average_latency_ms': 0,
            'throughput_fps': 0,
            'energy_efficiency_mw_per_inference': 0,
            'success_rate': 0,
            'device_utilization': {},
            'communication_overhead_percent': 0
        }
        
        async def run_benchmark():
            all_results = []
            
            for run in range(runs):
                for test_input in test_inputs:
                    result = await self.distributed_inference(test_input)
                    all_results.append(result)
            
            # Calculate metrics
            successful_results = [r for r in all_results if r.success]
            
            if successful_results:
                avg_latency = sum(r.total_time_ms for r in successful_results) / len(successful_results)
                avg_energy = sum(r.energy_consumed_mw for r in successful_results) / len(successful_results)
                avg_comm_time = sum(r.communication_time_ms for r in successful_results) / len(successful_results)
                
                benchmark_results['average_latency_ms'] = avg_latency
                benchmark_results['throughput_fps'] = 1000 / avg_latency if avg_latency > 0 else 0
                benchmark_results['energy_efficiency_mw_per_inference'] = avg_energy
                benchmark_results['success_rate'] = len(successful_results) / len(all_results)
                benchmark_results['communication_overhead_percent'] = (avg_comm_time / avg_latency) * 100
            
            return benchmark_results
        
        # Run benchmark
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_benchmark())
        finally:
            loop.close()
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        device_stats = {device_id: device.get_stats() for device_id, device in self.devices.items()}
        
        return {
            'inference_stats': self.inference_stats.copy(),
            'device_stats': device_stats,
            'num_devices': len(self.devices),
            'partition_info': self.partitioner.get_partition_info() if self.partitioner else {}
        }
    
    def optimize_load_balancing(self):
        """Optimize load balancing across devices"""
        # Simple load balancing: redistribute partitions based on device capabilities
        if not self.partitioner or not self.partition_config:
            return
        
        # Get current load on each device
        device_loads = {device_id: device.current_load for device_id, device in self.devices.items()}
        
        # Find most and least loaded devices
        max_load_device = max(device_loads, key=device_loads.get)
        min_load_device = min(device_loads, key=device_loads.get)
        
        load_difference = device_loads[max_load_device] - device_loads[min_load_device]
        
        # If load difference is significant, consider rebalancing
        if load_difference > 0.3:  # 30% load difference threshold
            logger.info(f"Load imbalance detected. Consider redistributing partitions.")
            # In a real implementation, this would trigger repartitioning

class InferenceScheduler:
    """Schedules inference requests across multiple devices"""
    
    def __init__(self, inference_engine: DistributedInferenceEngine):
        self.engine = inference_engine
        self.request_queue = asyncio.Queue()
        self.is_running = False
        
    async def add_request(self, input_tensor: torch.Tensor, priority: int = 0) -> InferenceResult:
        """Add an inference request to the queue"""
        request = {
            'input': input_tensor,
            'priority': priority,
            'timestamp': time.time(),
            'future': asyncio.Future()
        }
        
        await self.request_queue.put(request)
        return await request['future']
    
    async def start_scheduler(self):
        """Start the inference scheduler"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Get next request (with timeout to allow checking is_running)
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                
                # Process the request
                result = await self.engine.distributed_inference(request['input'])
                request['future'].set_result(result)
                
            except asyncio.TimeoutError:
                continue  # No requests in queue, continue
            except Exception as e:
                if 'request' in locals():
                    request['future'].set_exception(e)
    
    def stop_scheduler(self):
        """Stop the inference scheduler"""
        self.is_running = False

def create_simulated_devices(num_devices: int = 3, device_variety: str = "mixed") -> List[EdgeDevice]:
    """Create simulated edge devices with different capabilities"""
    devices = []
    
    if device_variety == "homogeneous":
        # All devices have similar capabilities
        for i in range(num_devices):
            profile = DeviceProfile(
                device_id=f"device_{i}",
                compute_power=1.0,
                memory_mb=200,
                bandwidth_mbps=20,
                latency_ms=30,
                energy_budget_mw=1000
            )
            devices.append(EdgeDevice(profile))
    
    elif device_variety == "heterogeneous":
        # Devices have varied capabilities
        capabilities = [
            (2.0, 500, 50, 20, 2000),  # High-end device
            (1.0, 200, 20, 40, 1000),  # Mid-range device
            (0.5, 100, 10, 60, 500),   # Low-end device
        ]
        
        for i in range(num_devices):
            cap_idx = i % len(capabilities)
            compute, memory, bandwidth, latency, energy = capabilities[cap_idx]
            
            profile = DeviceProfile(
                device_id=f"device_{i}",
                compute_power=compute,
                memory_mb=memory,
                bandwidth_mbps=bandwidth,
                latency_ms=latency,
                energy_budget_mw=energy
            )
            devices.append(EdgeDevice(profile))
    
    else:  # mixed
        # Random mix of capabilities
        for i in range(num_devices):
            profile = DeviceProfile(
                device_id=f"device_{i}",
                compute_power=np.random.uniform(0.5, 2.0),
                memory_mb=np.random.uniform(100, 500),
                bandwidth_mbps=np.random.uniform(10, 50),
                latency_ms=np.random.uniform(20, 80),
                energy_budget_mw=np.random.uniform(500, 2000)
            )
            devices.append(EdgeDevice(profile))
    
    logger.info(f"Created {num_devices} simulated {device_variety} devices")
    return devices
