"""
Model and Technique Comparison Utilities
Provides comprehensive comparison tools for different optimization techniques.
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import time
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    accuracy: float
    inference_time_ms: float
    model_size_mb: float
    memory_usage_mb: float
    energy_consumption_mw: float
    flops: int
    parameters_count: int
    compression_ratio: float = 1.0
    technique: str = "baseline"
    additional_metrics: Dict = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}

class TechniqueComparator:
    """
    Compare different optimization techniques and their trade-offs
    """
    
    def __init__(self):
        self.results = {}
        self.baseline_metrics = None
        self.comparison_cache = {}
        
    def add_baseline(self, name: str, metrics: ModelMetrics):
        """Add baseline model metrics"""
        self.baseline_metrics = metrics
        self.results[name] = metrics
        logger.info(f"Added baseline model: {name}")
    
    def add_technique_result(self, name: str, metrics: ModelMetrics):
        """Add results for a specific technique"""
        self.results[name] = metrics
        logger.info(f"Added technique result: {name}")
    
    def compare_all_techniques(self) -> pd.DataFrame:
        """
        Compare all techniques in a comprehensive table
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, metrics in self.results.items():
            # Calculate relative metrics compared to baseline
            if self.baseline_metrics and name != "baseline":
                accuracy_change = ((metrics.accuracy - self.baseline_metrics.accuracy) 
                                 / self.baseline_metrics.accuracy) * 100
                speed_improvement = ((self.baseline_metrics.inference_time_ms - metrics.inference_time_ms) 
                                   / self.baseline_metrics.inference_time_ms) * 100
                size_reduction = ((self.baseline_metrics.model_size_mb - metrics.model_size_mb) 
                                / self.baseline_metrics.model_size_mb) * 100
                energy_reduction = ((self.baseline_metrics.energy_consumption_mw - metrics.energy_consumption_mw) 
                                  / self.baseline_metrics.energy_consumption_mw) * 100
            else:
                accuracy_change = 0.0
                speed_improvement = 0.0
                size_reduction = 0.0
                energy_reduction = 0.0
            
            row = {
                'Technique': name,
                'Accuracy (%)': metrics.accuracy,
                'Accuracy Change (%)': accuracy_change,
                'Inference Time (ms)': metrics.inference_time_ms,
                'Speed Improvement (%)': speed_improvement,
                'Model Size (MB)': metrics.model_size_mb,
                'Size Reduction (%)': size_reduction,
                'Memory Usage (MB)': metrics.memory_usage_mb,
                'Energy (mW)': metrics.energy_consumption_mw,
                'Energy Reduction (%)': energy_reduction,
                'Parameters (K)': metrics.parameters_count / 1000,
                'FLOPS (M)': metrics.flops / 1e6,
                'Compression Ratio': metrics.compression_ratio,
                'Efficiency Score': self._calculate_efficiency_score(metrics)
            }
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.round(3)
    
    def _calculate_efficiency_score(self, metrics: ModelMetrics) -> float:
        """
        Calculate overall efficiency score
        Combines accuracy, speed, and size metrics
        """
        if self.baseline_metrics is None:
            return 0.0
        
        # Normalize metrics relative to baseline
        accuracy_score = metrics.accuracy / 100  # Convert to 0-1 scale
        
        speed_score = self.baseline_metrics.inference_time_ms / max(metrics.inference_time_ms, 0.1)
        size_score = self.baseline_metrics.model_size_mb / max(metrics.model_size_mb, 0.1)
        
        # Weighted combination (accuracy: 50%, speed: 25%, size: 25%)
        efficiency_score = (0.5 * accuracy_score + 0.25 * speed_score + 0.25 * size_score)
        
        return efficiency_score
    
    def generate_pareto_frontier(self, x_metric: str = 'model_size_mb', 
                               y_metric: str = 'accuracy') -> Dict:
        """
        Generate Pareto frontier for two metrics
        
        Args:
            x_metric: X-axis metric name
            y_metric: Y-axis metric name
        
        Returns:
            Dictionary with Pareto frontier data
        """
        if not self.results:
            return {}
        
        points = []
        labels = []
        
        for name, metrics in self.results.items():
            x_val = getattr(metrics, x_metric, 0)
            y_val = getattr(metrics, y_metric, 0)
            points.append((x_val, y_val))
            labels.append(name)
        
        # Find Pareto frontier
        pareto_points = []
        pareto_labels = []
        
        for i, (x1, y1) in enumerate(points):
            is_pareto = True
            for j, (x2, y2) in enumerate(points):
                if i != j:
                    # For minimization of x and maximization of y
                    if (x2 <= x1 and y2 >= y1) and (x2 < x1 or y2 > y1):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append((x1, y1))
                pareto_labels.append(labels[i])
        
        return {
            'all_points': points,
            'all_labels': labels,
            'pareto_points': pareto_points,
            'pareto_labels': pareto_labels,
            'x_metric': x_metric,
            'y_metric': y_metric
        }
    
    def calculate_trade_off_metrics(self) -> Dict:
        """
        Calculate trade-off metrics between accuracy and efficiency
        
        Returns:
            Dictionary with trade-off analysis
        """
        if not self.results or not self.baseline_metrics:
            return {}
        
        trade_offs = {}
        
        for name, metrics in self.results.items():
            if name == "baseline":
                continue
            
            # Accuracy vs Size trade-off
            accuracy_loss = self.baseline_metrics.accuracy - metrics.accuracy
            size_reduction = ((self.baseline_metrics.model_size_mb - metrics.model_size_mb) 
                            / self.baseline_metrics.model_size_mb) * 100
            
            # Accuracy vs Speed trade-off
            speed_gain = ((self.baseline_metrics.inference_time_ms - metrics.inference_time_ms) 
                         / self.baseline_metrics.inference_time_ms) * 100
            
            # Accuracy vs Energy trade-off
            energy_savings = ((self.baseline_metrics.energy_consumption_mw - metrics.energy_consumption_mw) 
                            / self.baseline_metrics.energy_consumption_mw) * 100
            
            trade_offs[name] = {
                'accuracy_loss': accuracy_loss,
                'size_reduction': size_reduction,
                'speed_gain': speed_gain,
                'energy_savings': energy_savings,
                'accuracy_per_mb_saved': accuracy_loss / max(size_reduction, 0.1),
                'accuracy_per_ms_saved': accuracy_loss / max(speed_gain, 0.1),
                'accuracy_per_mw_saved': accuracy_loss / max(energy_savings, 0.1)
            }
        
        return trade_offs
    
    def rank_techniques(self, criteria: str = "efficiency") -> List[Tuple[str, float]]:
        """
        Rank techniques based on specified criteria
        
        Args:
            criteria: Ranking criteria ("efficiency", "accuracy", "speed", "size", "energy")
        
        Returns:
            List of (technique_name, score) tuples, sorted by score
        """
        if not self.results:
            return []
        
        rankings = []
        
        for name, metrics in self.results.items():
            if criteria == "efficiency":
                score = self._calculate_efficiency_score(metrics)
            elif criteria == "accuracy":
                score = metrics.accuracy
            elif criteria == "speed":
                score = 1000 / max(metrics.inference_time_ms, 0.1)  # Higher is better
            elif criteria == "size":
                score = 1 / max(metrics.model_size_mb, 0.1)  # Smaller is better
            elif criteria == "energy":
                score = 1 / max(metrics.energy_consumption_mw, 0.1)  # Lower is better
            else:
                score = 0.0
            
            rankings.append((name, score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_best_techniques(self, top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top-k techniques for different criteria
        
        Args:
            top_k: Number of top techniques to return
        
        Returns:
            Dictionary with top techniques for each criterion
        """
        criteria_list = ["efficiency", "accuracy", "speed", "size", "energy"]
        best_techniques = {}
        
        for criterion in criteria_list:
            rankings = self.rank_techniques(criterion)
            best_techniques[criterion] = rankings[:top_k]
        
        return best_techniques
    
    def save_comparison_report(self, output_dir: Path, report_name: str = "comparison_report"):
        """
        Save comprehensive comparison report
        
        Args:
            output_dir: Directory to save the report
            report_name: Name of the report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison table
        comparison_df = self.compare_all_techniques()
        comparison_df.to_csv(output_dir / f"{report_name}.csv", index=False)
        
        # Generate trade-off analysis
        trade_offs = self.calculate_trade_off_metrics()
        
        # Generate rankings
        best_techniques = self.get_best_techniques()
        
        # Create summary report
        report_content = f"""
# Technique Comparison Report

## Summary
- Total techniques compared: {len(self.results)}
- Baseline model: {self.baseline_metrics.technique if self.baseline_metrics else 'None'}

## Overall Rankings

### By Efficiency Score
{self._format_rankings(best_techniques.get('efficiency', []))}

### By Accuracy
{self._format_rankings(best_techniques.get('accuracy', []))}

### By Speed
{self._format_rankings(best_techniques.get('speed', []))}

### By Model Size
{self._format_rankings(best_techniques.get('size', []))}

### By Energy Efficiency
{self._format_rankings(best_techniques.get('energy', []))}

## Trade-off Analysis
{self._format_trade_offs(trade_offs)}

## Detailed Comparison
{comparison_df.to_string(index=False)}
"""
        
        with open(output_dir / f"{report_name}.md", 'w') as f:
            f.write(report_content)
        
        logger.info(f"Comparison report saved to {output_dir / report_name}")
    
    def _format_rankings(self, rankings: List[Tuple[str, float]]) -> str:
        """Format rankings for report"""
        if not rankings:
            return "No data available"
        
        formatted = []
        for i, (name, score) in enumerate(rankings, 1):
            formatted.append(f"{i}. {name}: {score:.3f}")
        
        return "\n".join(formatted)
    
    def _format_trade_offs(self, trade_offs: Dict) -> str:
        """Format trade-off analysis for report"""
        if not trade_offs:
            return "No trade-off data available"
        
        formatted = []
        for technique, metrics in trade_offs.items():
            formatted.append(f"\n### {technique}")
            formatted.append(f"- Accuracy loss: {metrics['accuracy_loss']:.2f}%")
            formatted.append(f"- Size reduction: {metrics['size_reduction']:.2f}%")
            formatted.append(f"- Speed gain: {metrics['speed_gain']:.2f}%")
            formatted.append(f"- Energy savings: {metrics['energy_savings']:.2f}%")
        
        return "\n".join(formatted)

class FederatedLearningComparator:
    """
    Compare federated learning configurations and results
    """
    
    def __init__(self):
        self.federated_results = {}
        self.centralized_baseline = None
    
    def add_centralized_baseline(self, metrics: Dict):
        """Add centralized learning baseline"""
        self.centralized_baseline = metrics
        logger.info("Added centralized learning baseline")
    
    def add_federated_result(self, config_name: str, results: Dict):
        """
        Add federated learning results
        
        Args:
            config_name: Name of the federated configuration
            results: Dictionary with federated learning metrics
        """
        self.federated_results[config_name] = results
        logger.info(f"Added federated learning result: {config_name}")
    
    def compare_federated_configurations(self) -> pd.DataFrame:
        """
        Compare different federated learning configurations
        
        Returns:
            DataFrame with comparison results
        """
        if not self.federated_results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for config_name, results in self.federated_results.items():
            # Extract metrics from results
            final_accuracy = results.get('final_accuracy', 0.0)
            convergence_rounds = results.get('convergence_rounds', 0)
            communication_cost_mb = results.get('total_communication_mb', 0.0)
            training_time_minutes = results.get('training_time_minutes', 0.0)
            num_clients = results.get('num_clients', 0)
            data_distribution = results.get('data_distribution', 'unknown')
            
            # Calculate efficiency metrics
            accuracy_per_round = final_accuracy / max(convergence_rounds, 1)
            communication_efficiency = final_accuracy / max(communication_cost_mb, 0.1)
            
            row = {
                'Configuration': config_name,
                'Final Accuracy (%)': final_accuracy,
                'Convergence Rounds': convergence_rounds,
                'Communication Cost (MB)': communication_cost_mb,
                'Training Time (min)': training_time_minutes,
                'Clients': num_clients,
                'Data Distribution': data_distribution,
                'Accuracy/Round': accuracy_per_round,
                'Comm. Efficiency': communication_efficiency
            }
            
            # Add comparison with centralized baseline if available
            if self.centralized_baseline:
                centralized_accuracy = self.centralized_baseline.get('accuracy', 0.0)
                accuracy_gap = centralized_accuracy - final_accuracy
                row['Accuracy Gap from Centralized'] = accuracy_gap
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.round(3)
    
    def analyze_convergence_patterns(self) -> Dict:
        """
        Analyze convergence patterns across configurations
        
        Returns:
            Dictionary with convergence analysis
        """
        convergence_analysis = {}
        
        for config_name, results in self.federated_results.items():
            accuracy_history = results.get('accuracy_history', [])
            
            if len(accuracy_history) < 2:
                continue
            
            # Calculate convergence metrics
            final_accuracy = accuracy_history[-1]
            initial_accuracy = accuracy_history[0]
            improvement = final_accuracy - initial_accuracy
            
            # Find convergence point (when accuracy stops improving significantly)
            convergence_round = len(accuracy_history)
            threshold = 0.001  # 0.1% improvement threshold
            
            for i in range(len(accuracy_history) - 1, 0, -1):
                if i < len(accuracy_history) - 1:
                    improvement_rate = accuracy_history[i] - accuracy_history[i-1]
                    if improvement_rate > threshold:
                        convergence_round = i
                        break
            
            # Calculate stability (variance in last 20% of rounds)
            stability_window = max(1, len(accuracy_history) // 5)
            recent_accuracies = accuracy_history[-stability_window:]
            stability = 1.0 / (1.0 + np.var(recent_accuracies))  # Higher is more stable
            
            convergence_analysis[config_name] = {
                'total_improvement': improvement,
                'convergence_round': convergence_round,
                'convergence_rate': improvement / max(convergence_round, 1),
                'stability_score': stability,
                'final_accuracy': final_accuracy
            }
        
        return convergence_analysis

class HardwareComparator:
    """
    Compare performance across different hardware configurations
    """
    
    def __init__(self):
        self.hardware_results = {}
    
    def add_hardware_result(self, hardware_name: str, device_specs: Dict, performance_metrics: Dict):
        """
        Add performance results for specific hardware
        
        Args:
            hardware_name: Name/identifier for the hardware
            device_specs: Hardware specifications
            performance_metrics: Performance metrics on this hardware
        """
        self.hardware_results[hardware_name] = {
            'specs': device_specs,
            'performance': performance_metrics
        }
        logger.info(f"Added hardware result: {hardware_name}")
    
    def compare_hardware_performance(self) -> pd.DataFrame:
        """
        Compare performance across different hardware
        
        Returns:
            DataFrame with hardware comparison
        """
        if not self.hardware_results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for hardware_name, data in self.hardware_results.items():
            specs = data['specs']
            performance = data['performance']
            
            row = {
                'Hardware': hardware_name,
                'CPU/GPU': specs.get('processor', 'Unknown'),
                'Memory (GB)': specs.get('memory_gb', 0),
                'Power Budget (W)': specs.get('power_budget_w', 0),
                'Inference Time (ms)': performance.get('inference_time_ms', 0),
                'Throughput (FPS)': performance.get('throughput_fps', 0),
                'Energy per Inference (mJ)': performance.get('energy_per_inference_mj', 0),
                'Accuracy (%)': performance.get('accuracy', 0),
                'Memory Usage (MB)': performance.get('memory_usage_mb', 0),
                'Performance/Watt': performance.get('throughput_fps', 0) / max(specs.get('power_budget_w', 1), 1)
            }
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.round(3)
    
    def recommend_hardware(self, constraints: Dict) -> List[Tuple[str, float]]:
        """
        Recommend hardware based on constraints
        
        Args:
            constraints: Dictionary with constraint requirements
        
        Returns:
            List of (hardware_name, suitability_score) tuples
        """
        recommendations = []
        
        for hardware_name, data in self.hardware_results.items():
            specs = data['specs']
            performance = data['performance']
            
            # Calculate suitability score based on constraints
            score = 0.0
            penalty = 0.0
            
            # Check constraints
            max_power = constraints.get('max_power_w', float('inf'))
            min_accuracy = constraints.get('min_accuracy', 0.0)
            max_latency = constraints.get('max_latency_ms', float('inf'))
            max_memory = constraints.get('max_memory_gb', float('inf'))
            
            # Apply penalties for constraint violations
            if specs.get('power_budget_w', 0) > max_power:
                penalty += 0.5
            
            if performance.get('accuracy', 0) < min_accuracy:
                penalty += 1.0
            
            if performance.get('inference_time_ms', 0) > max_latency:
                penalty += 0.3
            
            if specs.get('memory_gb', 0) > max_memory:
                penalty += 0.2
            
            # Calculate positive score based on performance
            normalized_accuracy = performance.get('accuracy', 0) / 100.0
            normalized_efficiency = min(1.0, 100.0 / max(performance.get('inference_time_ms', 100), 1))
            normalized_power_efficiency = min(1.0, 10.0 / max(specs.get('power_budget_w', 10), 1))
            
            score = (0.4 * normalized_accuracy + 
                    0.3 * normalized_efficiency + 
                    0.3 * normalized_power_efficiency)
            
            # Apply penalty
            final_score = max(0.0, score - penalty)
            
            recommendations.append((hardware_name, final_score))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

def create_comprehensive_comparison(model_results: Dict[str, ModelMetrics],
                                  federated_results: Dict = None,
                                  hardware_results: Dict = None) -> Dict:
    """
    Create a comprehensive comparison across all aspects
    
    Args:
        model_results: Dictionary of model technique results
        federated_results: Optional federated learning results
        hardware_results: Optional hardware comparison results
    
    Returns:
        Comprehensive comparison dictionary
    """
    # Model technique comparison
    technique_comparator = TechniqueComparator()
    
    baseline_added = False
    for name, metrics in model_results.items():
        if not baseline_added and "baseline" in name.lower():
            technique_comparator.add_baseline(name, metrics)
            baseline_added = True
        else:
            technique_comparator.add_technique_result(name, metrics)
    
    model_comparison = {
        'techniques_table': technique_comparator.compare_all_techniques(),
        'trade_offs': technique_comparator.calculate_trade_off_metrics(),
        'rankings': technique_comparator.get_best_techniques(),
        'pareto_frontier': technique_comparator.generate_pareto_frontier()
    }
    
    comprehensive_results = {
        'model_comparison': model_comparison,
        'summary': {
            'total_techniques_compared': len(model_results),
            'best_overall': technique_comparator.rank_techniques("efficiency")[0] if model_results else None,
            'best_accuracy': technique_comparator.rank_techniques("accuracy")[0] if model_results else None,
            'best_efficiency': technique_comparator.rank_techniques("speed")[0] if model_results else None
        }
    }
    
    # Add federated learning comparison if provided
    if federated_results:
        fl_comparator = FederatedLearningComparator()
        for config_name, results in federated_results.items():
            fl_comparator.add_federated_result(config_name, results)
        
        comprehensive_results['federated_comparison'] = {
            'configurations_table': fl_comparator.compare_federated_configurations(),
            'convergence_analysis': fl_comparator.analyze_convergence_patterns()
        }
    
    # Add hardware comparison if provided
    if hardware_results:
        hw_comparator = HardwareComparator()
        for hardware_name, data in hardware_results.items():
            hw_comparator.add_hardware_result(
                hardware_name, 
                data.get('specs', {}), 
                data.get('performance', {})
            )
        
        comprehensive_results['hardware_comparison'] = {
            'hardware_table': hw_comparator.compare_hardware_performance()
        }
    
    return comprehensive_results
