"""
Performance metrics tracking and analysis for edge AI optimization
Comprehensive metrics collection, analysis, and reporting
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import DEVICE, ENERGY_PARAMS, PERFORMANCE_THRESHOLDS


@dataclass
class PerformanceMetrics:
    """Data class for comprehensive performance metrics"""
    # Model information
    model_name: str
    model_size_mb: float
    parameter_count: int
    
    # Performance metrics
    inference_time_ms: float
    inference_time_std: float
    accuracy: float
    throughput_fps: float
    
    # Resource usage
    memory_usage_mb: float
    cpu_utilization: float
    energy_consumption_mw: float
    
    # Compression metrics
    compression_ratio: float
    flops_reduction: float
    accuracy_drop: float
    
    # Device information
    device_type: str
    optimization_technique: str
    
    # Timestamp
    timestamp: str


class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics_history = []
        self.baseline_metrics = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize system monitoring
        self.process = psutil.Process()
        
    def measure_inference_performance(self, model: torch.nn.Module, 
                                    test_loader, device_type: str = 'desktop',
                                    num_iterations: int = 100) -> Dict[str, float]:
        """
        Measure comprehensive inference performance
        
        Args:
            model: PyTorch model to evaluate
            test_loader: Test data loader
            device_type: Type of device being simulated
            num_iterations: Number of inference iterations for timing
            
        Returns:
            Dictionary with performance metrics
        """
        model.eval()
        model = model.to(DEVICE)
        
        # Inference timing
        inference_times = []
        correct_predictions = 0
        total_samples = 0
        
        # Memory usage before inference
        torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
        memory_before = self._get_memory_usage()
        
        with torch.no_grad():
            # Warmup
            for i, (data, target) in enumerate(test_loader):
                if i >= 5:  # 5 warmup iterations
                    break
                data = data.to(DEVICE)
                _ = model(data)
            
            # Actual measurement
            for i, (data, target) in enumerate(test_loader):
                if i >= num_iterations:
                    break
                
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                # Measure inference time
                start_time = time.perf_counter()
                outputs = model(data)
                torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                # Accuracy calculation
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == target).sum().item()
                total_samples += target.size(0)
        
        # Memory usage after inference
        memory_after = self._get_memory_usage()
        memory_usage = memory_after - memory_before
        
        # Calculate statistics
        mean_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        accuracy = 100 * correct_predictions / total_samples if total_samples > 0 else 0
        
        # Calculate throughput (samples per second)
        batch_size = test_loader.batch_size
        throughput = (batch_size * 1000) / mean_inference_time if mean_inference_time > 0 else 0
        
        return {
            'inference_time_ms': mean_inference_time,
            'inference_time_std': std_inference_time,
            'accuracy': accuracy,
            'throughput_fps': throughput,
            'memory_usage_mb': memory_usage,
            'total_samples_tested': total_samples
        }
    
    def measure_model_characteristics(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Measure model characteristics and complexity
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary with model characteristics
        """
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size in MB
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Estimate FLOPs
        flops = self._estimate_model_flops(model)
        
        # Calculate sparsity (for pruned models)
        total_weights = 0
        zero_weights = 0
        for param in model.parameters():
            if param.dim() > 1:  # Only consider weight tensors
                total_weights += param.numel()
                zero_weights += (param.abs() < 1e-8).sum().item()
        
        sparsity = zero_weights / total_weights if total_weights > 0 else 0
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': flops,
            'sparsity': sparsity,
            'architecture': model.__class__.__name__
        }
    
    def measure_energy_consumption(self, model: torch.nn.Module, 
                                 inference_time_ms: float,
                                 device_type: str = 'desktop') -> float:
        """
        Estimate energy consumption for model inference
        
        Args:
            model: PyTorch model
            inference_time_ms: Measured inference time in milliseconds
            device_type: Type of device
            
        Returns:
            Estimated energy consumption in milliwatts
        """
        # Base energy consumption
        base_power = ENERGY_PARAMS['cpu_active']
        
        # Energy based on computation (FLOPs)
        flops = self._estimate_model_flops(model)
        compute_energy = flops / ENERGY_PARAMS['ops_per_watt'] * 1000  # Convert to mW
        
        # Time-based energy
        time_energy = base_power * (inference_time_ms / 1000)
        
        # Device-specific multiplier
        from ..utils.config import EDGE_DEVICE_CONFIGS
        device_multiplier = EDGE_DEVICE_CONFIGS.get(device_type, {}).get('energy_multiplier', 1.0)
        
        total_energy = (time_energy + compute_energy) * device_multiplier
        
        return total_energy
    
    def collect_comprehensive_metrics(self, model: torch.nn.Module, 
                                    test_loader, 
                                    model_name: str,
                                    optimization_technique: str,
                                    device_type: str = 'desktop',
                                    baseline_model: Optional[torch.nn.Module] = None) -> PerformanceMetrics:
        """
        Collect comprehensive performance metrics for a model
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name/identifier for the model
            optimization_technique: Optimization technique applied
            device_type: Device type for evaluation
            baseline_model: Baseline model for comparison
            
        Returns:
            PerformanceMetrics object with all collected metrics
        """
        self.logger.info(f"Collecting metrics for {model_name} with {optimization_technique}")
        
        # Measure performance
        perf_metrics = self.measure_inference_performance(model, test_loader, device_type)
        
        # Measure model characteristics
        model_chars = self.measure_model_characteristics(model)
        
        # Measure energy consumption
        energy = self.measure_energy_consumption(
            model, perf_metrics['inference_time_ms'], device_type
        )
        
        # Calculate compression metrics (if baseline available)
        compression_ratio = 1.0
        flops_reduction = 0.0
        accuracy_drop = 0.0
        
        if baseline_model is not None:
            baseline_chars = self.measure_model_characteristics(baseline_model)
            baseline_perf = self.measure_inference_performance(baseline_model, test_loader, device_type)
            
            compression_ratio = baseline_chars['model_size_mb'] / model_chars['model_size_mb']
            flops_reduction = 1.0 - (model_chars['estimated_flops'] / baseline_chars['estimated_flops'])
            accuracy_drop = baseline_perf['accuracy'] - perf_metrics['accuracy']
        
        # CPU utilization
        cpu_util = self.process.cpu_percent()
        
        # Create comprehensive metrics object
        metrics = PerformanceMetrics(
            model_name=model_name,
            model_size_mb=model_chars['model_size_mb'],
            parameter_count=model_chars['total_parameters'],
            inference_time_ms=perf_metrics['inference_time_ms'],
            inference_time_std=perf_metrics['inference_time_std'],
            accuracy=perf_metrics['accuracy'],
            throughput_fps=perf_metrics['throughput_fps'],
            memory_usage_mb=perf_metrics['memory_usage_mb'],
            cpu_utilization=cpu_util,
            energy_consumption_mw=energy,
            compression_ratio=compression_ratio,
            flops_reduction=flops_reduction,
            accuracy_drop=accuracy_drop,
            device_type=device_type,
            optimization_technique=optimization_technique,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Set baseline if this is the first measurement
        if self.baseline_metrics is None and optimization_technique == 'baseline':
            self.baseline_metrics = metrics
        
        return metrics
    
    def analyze_trade_offs(self) -> Dict[str, Any]:
        """
        Analyze trade-offs between different optimization techniques
        
        Returns:
            Analysis of trade-offs
        """
        if not self.metrics_history:
            return {'error': 'No metrics collected'}
        
        # Group metrics by optimization technique
        technique_groups = defaultdict(list)
        for metrics in self.metrics_history:
            technique_groups[metrics.optimization_technique].append(metrics)
        
        # Analyze each technique
        analysis = {}
        for technique, metrics_list in technique_groups.items():
            if not metrics_list:
                continue
            
            # Average metrics across runs
            avg_metrics = {
                'accuracy': np.mean([m.accuracy for m in metrics_list]),
                'inference_time_ms': np.mean([m.inference_time_ms for m in metrics_list]),
                'model_size_mb': np.mean([m.model_size_mb for m in metrics_list]),
                'energy_consumption_mw': np.mean([m.energy_consumption_mw for m in metrics_list]),
                'compression_ratio': np.mean([m.compression_ratio for m in metrics_list]),
                'accuracy_drop': np.mean([m.accuracy_drop for m in metrics_list])
            }
            
            # Calculate efficiency metrics
            efficiency_score = self._calculate_efficiency_score(avg_metrics)
            
            analysis[technique] = {
                'average_metrics': avg_metrics,
                'efficiency_score': efficiency_score,
                'num_evaluations': len(metrics_list),
                'meets_thresholds': self._check_performance_thresholds(avg_metrics)
            }
        
        return analysis
    
    def generate_comparison_table(self) -> str:
        """
        Generate a formatted comparison table of all collected metrics
        
        Returns:
            Formatted table string
        """
        if not self.metrics_history:
            return "No metrics collected"
        
        # Create table header
        header = "| Technique | Accuracy (%) | Inference (ms) | Size (MB) | Energy (mW) | Compression | Accuracy Drop |\n"
        header += "|-----------|--------------|----------------|-----------|-------------|-------------|---------------|\n"
        
        # Group by technique and calculate averages
        technique_groups = defaultdict(list)
        for metrics in self.metrics_history:
            technique_groups[metrics.optimization_technique].append(metrics)
        
        rows = []
        for technique, metrics_list in technique_groups.items():
            avg_accuracy = np.mean([m.accuracy for m in metrics_list])
            avg_inference = np.mean([m.inference_time_ms for m in metrics_list])
            avg_size = np.mean([m.model_size_mb for m in metrics_list])
            avg_energy = np.mean([m.energy_consumption_mw for m in metrics_list])
            avg_compression = np.mean([m.compression_ratio for m in metrics_list])
            avg_accuracy_drop = np.mean([m.accuracy_drop for m in metrics_list])
            
            row = f"| {technique:<9} | {avg_accuracy:>10.2f} | {avg_inference:>12.2f} | {avg_size:>7.2f} | {avg_energy:>9.2f} | {avg_compression:>9.2f}x | {avg_accuracy_drop:>11.2f} |\n"
            rows.append(row)
        
        return header + "".join(rows)
    
    def save_metrics_to_json(self, filepath: str):
        """
        Save collected metrics to JSON file
        
        Args:
            filepath: Path to save metrics
        """
        metrics_data = {
            'baseline_metrics': asdict(self.baseline_metrics) if self.baseline_metrics else None,
            'all_metrics': [asdict(m) for m in self.metrics_history],
            'summary': self.analyze_trade_offs(),
            'collection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to {filepath}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if DEVICE.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            return self.process.memory_info().rss / (1024 ** 2)
    
    def _estimate_model_flops(self, model: torch.nn.Module, 
                            input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> int:
        """
        Estimate model FLOPs (simplified calculation)
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Estimated FLOPs
        """
        total_flops = 0
        
        def flop_count(module, input, output):
            nonlocal total_flops
            if isinstance(module, torch.nn.Conv2d):
                # FLOPs = kernel_height * kernel_width * input_channels * output_height * output_width * output_channels
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = output.numel() // output.size(0)  # Remove batch dimension
                total_flops += kernel_flops * output_elements
            elif isinstance(module, torch.nn.Linear):
                # FLOPs = input_features * output_features
                total_flops += module.in_features * module.out_features
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(flop_count))
        
        # Forward pass to count FLOPs
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape).to(DEVICE)
            _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops
    
    def _calculate_efficiency_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall efficiency score
        
        Args:
            metrics: Dictionary of average metrics
            
        Returns:
            Efficiency score (higher is better)
        """
        # Normalize metrics (higher accuracy, lower inference time, size, energy is better)
        accuracy_score = metrics['accuracy'] / 100.0  # Normalize to 0-1
        speed_score = 1.0 / (1.0 + metrics['inference_time_ms'] / 100.0)  # Inverse relationship
        size_score = 1.0 / (1.0 + metrics['model_size_mb'] / 10.0)  # Inverse relationship
        energy_score = 1.0 / (1.0 + metrics['energy_consumption_mw'] / 1000.0)  # Inverse relationship
        
        # Weighted combination
        efficiency_score = (0.4 * accuracy_score + 0.2 * speed_score + 
                          0.2 * size_score + 0.2 * energy_score)
        
        return efficiency_score
    
    def _check_performance_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if metrics meet performance thresholds
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary indicating which thresholds are met
        """
        return {
            'accuracy_drop_ok': metrics.get('accuracy_drop', 0) <= PERFORMANCE_THRESHOLDS['accuracy_drop_max'],
            'inference_time_ok': metrics.get('inference_time_ms', float('inf')) <= PERFORMANCE_THRESHOLDS['inference_time_max'],
            'memory_usage_ok': metrics.get('memory_usage_mb', float('inf')) <= PERFORMANCE_THRESHOLDS['memory_usage_max'],
            'energy_consumption_ok': metrics.get('energy_consumption_mw', float('inf')) <= PERFORMANCE_THRESHOLDS['energy_consumption_max']
        }


class BenchmarkSuite:
    """Comprehensive benchmark suite for edge AI optimization"""
    
    def __init__(self, baseline_model: torch.nn.Module, test_loader):
        """
        Initialize benchmark suite
        
        Args:
            baseline_model: Baseline model for comparison
            test_loader: Test data loader
        """
        self.baseline_model = baseline_model
        self.test_loader = test_loader
        self.metrics_collector = MetricsCollector()
        self.results = {}
        
    def run_comprehensive_benchmark(self, models_dict: Dict[str, torch.nn.Module],
                                  device_types: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all models and devices
        
        Args:
            models_dict: Dictionary of {technique_name: model} pairs
            device_types: List of device types to test
            
        Returns:
            Comprehensive benchmark results
        """
        if device_types is None:
            device_types = ['desktop', 'jetson_nano', 'raspberry_pi']
        
        # Collect baseline metrics first
        baseline_metrics = self.metrics_collector.collect_comprehensive_metrics(
            self.baseline_model, self.test_loader, 'baseline', 'baseline', 'desktop'
        )
        
        # Collect metrics for all models and devices
        for device_type in device_types:
            for technique_name, model in models_dict.items():
                if technique_name == 'baseline':
                    continue  # Already collected
                
                metrics = self.metrics_collector.collect_comprehensive_metrics(
                    model, self.test_loader, technique_name, technique_name, 
                    device_type, self.baseline_model
                )
                
                self.results[f"{technique_name}_{device_type}"] = metrics
        
        # Generate analysis
        trade_off_analysis = self.metrics_collector.analyze_trade_offs()
        comparison_table = self.metrics_collector.generate_comparison_table()
        
        return {
            'baseline_metrics': baseline_metrics,
            'all_results': self.results,
            'trade_off_analysis': trade_off_analysis,
            'comparison_table': comparison_table,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> Dict[str, str]:
        """Generate recommendations based on benchmark results"""
        analysis = self.metrics_collector.analyze_trade_offs()
        
        recommendations = {}
        
        # Find best technique for different use cases
        best_accuracy = max(analysis.items(), key=lambda x: x[1]['average_metrics']['accuracy'])
        best_speed = min(analysis.items(), key=lambda x: x[1]['average_metrics']['inference_time_ms'])
        best_size = min(analysis.items(), key=lambda x: x[1]['average_metrics']['model_size_mb'])
        best_efficiency = max(analysis.items(), key=lambda x: x[1]['efficiency_score'])
        
        recommendations.update({
            'best_accuracy': f"{best_accuracy[0]} (Accuracy: {best_accuracy[1]['average_metrics']['accuracy']:.2f}%)",
            'best_speed': f"{best_speed[0]} (Inference: {best_speed[1]['average_metrics']['inference_time_ms']:.2f}ms)",
            'best_size': f"{best_size[0]} (Size: {best_size[1]['average_metrics']['model_size_mb']:.2f}MB)",
            'best_overall': f"{best_efficiency[0]} (Efficiency Score: {best_efficiency[1]['efficiency_score']:.3f})"
        })
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    from ..models.base_model import create_model
    from ..utils.data_loader import DataLoader
    
    # Create models and data
    baseline_model = create_model('base')
    student_model = create_model('student')
    data_loader = DataLoader()
    _, test_loader = data_loader.load_centralized_data()
    
    # Initialize metrics collector
    collector = MetricsCollector()
    
    # Collect metrics for baseline
    baseline_metrics = collector.collect_comprehensive_metrics(
        baseline_model, test_loader, 'baseline_cnn', 'baseline', 'desktop'
    )
    
    print(f"Baseline metrics collected:")
    print(f"Accuracy: {baseline_metrics.accuracy:.2f}%")
    print(f"Inference time: {baseline_metrics.inference_time_ms:.2f}ms")
    print(f"Model size: {baseline_metrics.model_size_mb:.2f}MB")
    
    # Collect metrics for student model
    student_metrics = collector.collect_comprehensive_metrics(
        student_model, test_loader, 'student_cnn', 'knowledge_distillation', 
        'desktop', baseline_model
    )
    
    print(f"\nStudent model metrics:")
    print(f"Accuracy: {student_metrics.accuracy:.2f}%")
    print(f"Compression ratio: {student_metrics.compression_ratio:.2f}x")
    print(f"Accuracy drop: {student_metrics.accuracy_drop:.2f}%")
    
    # Generate comparison table
    comparison_table = collector.generate_comparison_table()
    print(f"\nComparison Table:\n{comparison_table}")
