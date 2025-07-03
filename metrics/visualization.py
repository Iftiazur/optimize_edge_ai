"""
Visualization utilities for edge AI optimization results
Creates plots, charts, and visual comparisons of optimization techniques
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VisualizationEngine:
    """Comprehensive visualization engine for edge AI optimization results"""
    
    def __init__(self, save_path: str = "./plots"):
        """
        Initialize visualization engine
        
        Args:
            save_path: Directory to save plots
        """
        self.save_path = save_path
        self.logger = logging.getLogger(__name__)
        
        # Create save directory if it doesn't exist
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_compression_comparison(self, metrics_data: Dict[str, Any], 
                                  save_filename: str = "compression_comparison.png") -> str:
        """
        Create compression comparison plot showing trade-offs
        
        Args:
            metrics_data: Dictionary containing metrics for different techniques
            save_filename: Filename to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        techniques = list(metrics_data.keys())
        accuracies = [metrics_data[t]['average_metrics']['accuracy'] for t in techniques]
        model_sizes = [metrics_data[t]['average_metrics']['model_size_mb'] for t in techniques]
        inference_times = [metrics_data[t]['average_metrics']['inference_time_ms'] for t in techniques]
        energy_consumptions = [metrics_data[t]['average_metrics']['energy_consumption_mw'] for t in techniques]
        
        # Plot 1: Accuracy vs Model Size
        scatter1 = ax1.scatter(model_sizes, accuracies, s=100, alpha=0.7, c=range(len(techniques)), cmap='viridis')
        ax1.set_xlabel('Model Size (MB)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy vs Model Size Trade-off')
        
        # Add technique labels
        for i, technique in enumerate(techniques):
            ax1.annotate(technique, (model_sizes[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 2: Accuracy vs Inference Time
        scatter2 = ax2.scatter(inference_times, accuracies, s=100, alpha=0.7, c=range(len(techniques)), cmap='viridis')
        ax2.set_xlabel('Inference Time (ms)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy vs Inference Time Trade-off')
        
        for i, technique in enumerate(techniques):
            ax2.annotate(technique, (inference_times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 3: Model Size Reduction
        compression_ratios = [metrics_data[t]['average_metrics'].get('compression_ratio', 1.0) for t in techniques]
        bars3 = ax3.bar(techniques, compression_ratios, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(techniques))))
        ax3.set_ylabel('Compression Ratio')
        ax3.set_title('Model Size Compression Ratios')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, ratio in zip(bars3, compression_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{ratio:.2f}x', ha='center', va='bottom')
        
        # Plot 4: Energy Consumption
        bars4 = ax4.bar(techniques, energy_consumptions, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(techniques))))
        ax4.set_ylabel('Energy Consumption (mW)')
        ax4.set_title('Energy Consumption Comparison')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, energy in zip(bars4, energy_consumptions):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(energy_consumptions)*0.01,
                    f'{energy:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = f"{self.save_path}/{save_filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Compression comparison plot saved to {save_path}")
        return save_path
    
    def plot_pareto_frontier(self, metrics_data: Dict[str, Any], 
                           x_metric: str = 'inference_time_ms',
                           y_metric: str = 'accuracy',
                           save_filename: str = "pareto_frontier.png") -> str:
        """
        Create Pareto frontier plot showing optimal trade-offs
        
        Args:
            metrics_data: Dictionary containing metrics
            x_metric: Metric for x-axis (lower is better)
            y_metric: Metric for y-axis (higher is better)
            save_filename: Filename to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        techniques = list(metrics_data.keys())
        x_values = [metrics_data[t]['average_metrics'][x_metric] for t in techniques]
        y_values = [metrics_data[t]['average_metrics'][y_metric] for t in techniques]
        
        # Create scatter plot
        scatter = ax.scatter(x_values, y_values, s=150, alpha=0.7, c=range(len(techniques)), cmap='viridis')
        
        # Add technique labels
        for i, technique in enumerate(techniques):
            ax.annotate(technique, (x_values[i], y_values[i]), 
                       xytext=(10, 10), textcoords='offset points', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                       fontsize=9)
        
        # Calculate and plot Pareto frontier
        pareto_points = self._find_pareto_frontier(x_values, y_values, maximize_y=True, minimize_x=True)
        pareto_x = [x_values[i] for i in pareto_points]
        pareto_y = [y_values[i] for i in pareto_points]
        
        # Sort Pareto points for line plotting
        pareto_sorted = sorted(zip(pareto_x, pareto_y))
        pareto_x_sorted, pareto_y_sorted = zip(*pareto_sorted)
        
        ax.plot(pareto_x_sorted, pareto_y_sorted, 'r--', linewidth=2, alpha=0.8, label='Pareto Frontier')
        ax.scatter(pareto_x, pareto_y, s=200, c='red', marker='*', alpha=0.8, label='Pareto Optimal')
        
        ax.set_xlabel(f'{x_metric.replace("_", " ").title()}')
        ax.set_ylabel(f'{y_metric.replace("_", " ").title()}')
        ax.set_title(f'Pareto Frontier: {y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = f"{self.save_path}/{save_filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Pareto frontier plot saved to {save_path}")
        return save_path
    
    def plot_device_comparison(self, device_metrics: Dict[str, Dict], 
                             save_filename: str = "device_comparison.png") -> str:
        """
        Create device-specific performance comparison
        
        Args:
            device_metrics: Dictionary with device-specific metrics
            save_filename: Filename to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        devices = list(device_metrics.keys())
        metrics_to_plot = ['accuracy', 'inference_time_ms', 'model_size_mb', 'energy_consumption_mw']
        metric_titles = ['Accuracy (%)', 'Inference Time (ms)', 'Model Size (MB)', 'Energy Consumption (mW)']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
            ax = axes[idx]
            
            # Extract data for all optimization techniques across devices
            techniques = set()
            for device_data in device_metrics.values():
                techniques.update(device_data.keys())
            techniques = sorted(list(techniques))
            
            x_pos = np.arange(len(techniques))
            width = 0.8 / len(devices)
            
            for i, device in enumerate(devices):
                values = []
                for technique in techniques:
                    if technique in device_metrics[device]:
                        values.append(device_metrics[device][technique]['average_metrics'][metric])
                    else:
                        values.append(0)
                
                ax.bar(x_pos + i * width, values, width, label=device, alpha=0.8)
            
            ax.set_xlabel('Optimization Technique')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Across Devices')
            ax.set_xticks(x_pos + width * (len(devices) - 1) / 2)
            ax.set_xticklabels(techniques, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.save_path}/{save_filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Device comparison plot saved to {save_path}")
        return save_path
    
    def plot_federated_learning_convergence(self, federated_history: List[Dict], 
                                          save_filename: str = "federated_convergence.png") -> str:
        """
        Plot federated learning convergence
        
        Args:
            federated_history: List of round statistics from federated training
            save_filename: Filename to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        rounds = [r['round_number'] for r in federated_history]
        accuracies = [r['global_accuracy'] for r in federated_history]
        losses = [r['global_loss'] for r in federated_history]
        participating_clients = [r['participating_clients'] for r in federated_history]
        round_times = [r['round_time_s'] for r in federated_history]
        
        # Plot 1: Global Accuracy Convergence
        ax1.plot(rounds, accuracies, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Round')
        ax1.set_ylabel('Global Accuracy (%)')
        ax1.set_title('Federated Learning Convergence - Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Global Loss Convergence
        ax2.plot(rounds, losses, marker='s', linewidth=2, markersize=6, color='red')
        ax2.set_xlabel('Federated Round')
        ax2.set_ylabel('Global Loss')
        ax2.set_title('Federated Learning Convergence - Loss')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Client Participation
        ax3.bar(rounds, participating_clients, alpha=0.7, color='green')
        ax3.set_xlabel('Federated Round')
        ax3.set_ylabel('Number of Participating Clients')
        ax3.set_title('Client Participation per Round')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Round Time Analysis
        ax4.plot(rounds, round_times, marker='^', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('Federated Round')
        ax4.set_ylabel('Round Time (seconds)')
        ax4.set_title('Training Time per Round')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.save_path}/{save_filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Federated learning convergence plot saved to {save_path}")
        return save_path
    
    def create_interactive_dashboard(self, all_metrics: Dict[str, Any], 
                                   save_filename: str = "interactive_dashboard.html") -> str:
        """
        Create interactive Plotly dashboard
        
        Args:
            all_metrics: All collected metrics
            save_filename: Filename to save the dashboard
            
        Returns:
            Path to saved dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy vs Model Size', 'Inference Time vs Energy', 
                          'Compression Ratios', 'Performance Radar Chart'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "polar"}]]
        )
        
        techniques = list(all_metrics.keys())
        colors = px.colors.qualitative.Set1[:len(techniques)]
        
        # Extract metrics
        accuracies = [all_metrics[t]['average_metrics']['accuracy'] for t in techniques]
        model_sizes = [all_metrics[t]['average_metrics']['model_size_mb'] for t in techniques]
        inference_times = [all_metrics[t]['average_metrics']['inference_time_ms'] for t in techniques]
        energy_consumptions = [all_metrics[t]['average_metrics']['energy_consumption_mw'] for t in techniques]
        compression_ratios = [all_metrics[t]['average_metrics'].get('compression_ratio', 1.0) for t in techniques]
        
        # Plot 1: Accuracy vs Model Size scatter
        fig.add_trace(
            go.Scatter(x=model_sizes, y=accuracies, mode='markers+text',
                      text=techniques, textposition='top center',
                      marker=dict(size=12, color=colors[:len(techniques)]),
                      name='Techniques'),
            row=1, col=1
        )
        
        # Plot 2: Inference Time vs Energy scatter
        fig.add_trace(
            go.Scatter(x=inference_times, y=energy_consumptions, mode='markers+text',
                      text=techniques, textposition='top center',
                      marker=dict(size=12, color=colors[:len(techniques)]),
                      showlegend=False),
            row=1, col=2
        )
        
        # Plot 3: Compression ratios bar chart
        fig.add_trace(
            go.Bar(x=techniques, y=compression_ratios,
                   marker=dict(color=colors[:len(techniques)]),
                   showlegend=False),
            row=2, col=1
        )
        
        # Plot 4: Radar chart for performance comparison
        # Normalize metrics for radar chart
        norm_accuracy = [a/max(accuracies) for a in accuracies]
        norm_speed = [max(inference_times)/t for t in inference_times]  # Inverse for speed
        norm_size = [max(model_sizes)/s for s in model_sizes]  # Inverse for size
        norm_energy = [max(energy_consumptions)/e for e in energy_consumptions]  # Inverse for energy
        
        categories = ['Accuracy', 'Speed', 'Compactness', 'Efficiency']
        
        for i, technique in enumerate(techniques):
            values = [norm_accuracy[i], norm_speed[i], norm_size[i], norm_energy[i]]
            fig.add_trace(
                go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                               fill='toself', name=technique,
                               line=dict(color=colors[i])),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Edge AI Optimization Interactive Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update subplot titles and axes
        fig.update_xaxes(title_text="Model Size (MB)", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_xaxes(title_text="Inference Time (ms)", row=1, col=2)
        fig.update_yaxes(title_text="Energy Consumption (mW)", row=1, col=2)
        fig.update_xaxes(title_text="Optimization Technique", row=2, col=1)
        fig.update_yaxes(title_text="Compression Ratio", row=2, col=1)
        
        save_path = f"{self.save_path}/{save_filename}"
        fig.write_html(save_path)
        
        self.logger.info(f"Interactive dashboard saved to {save_path}")
        return save_path
    
    def plot_technique_breakdown(self, metrics_data: Dict[str, Any], 
                               save_filename: str = "technique_breakdown.png") -> str:
        """
        Create detailed breakdown of each optimization technique
        
        Args:
            metrics_data: Dictionary containing metrics for different techniques
            save_filename: Filename to save the plot
            
        Returns:
            Path to saved plot
        """
        techniques = list(metrics_data.keys())
        n_techniques = len(techniques)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'inference_time_ms', 'model_size_mb', 
                  'energy_consumption_mw', 'compression_ratio', 'accuracy_drop']
        metric_titles = ['Accuracy (%)', 'Inference Time (ms)', 'Model Size (MB)',
                        'Energy Consumption (mW)', 'Compression Ratio', 'Accuracy Drop (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx]
            
            values = []
            for technique in techniques:
                if metric in metrics_data[technique]['average_metrics']:
                    values.append(metrics_data[technique]['average_metrics'][metric])
                else:
                    values.append(0)
            
            bars = ax.bar(techniques, values, alpha=0.8, 
                         color=plt.cm.Set3(np.linspace(0, 1, n_techniques)))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.save_path}/{save_filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Technique breakdown plot saved to {save_path}")
        return save_path
    
    def _find_pareto_frontier(self, x_values: List[float], y_values: List[float], 
                            maximize_y: bool = True, minimize_x: bool = True) -> List[int]:
        """
        Find Pareto frontier points
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            maximize_y: Whether to maximize y values
            minimize_x: Whether to minimize x values
            
        Returns:
            List of indices of Pareto optimal points
        """
        points = list(zip(x_values, y_values))
        pareto_indices = []
        
        for i, (x1, y1) in enumerate(points):
            is_pareto = True
            for j, (x2, y2) in enumerate(points):
                if i != j:
                    # Check if point j dominates point i
                    x_dominates = (x2 <= x1) if minimize_x else (x2 >= x1)
                    y_dominates = (y2 >= y1) if maximize_y else (y2 <= y1)
                    
                    if x_dominates and y_dominates and (x2 != x1 or y2 != y1):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def generate_summary_report(self, all_metrics: Dict[str, Any], 
                              federated_results: Optional[Dict] = None,
                              save_filename: str = "summary_report.png") -> str:
        """
        Generate comprehensive summary report with multiple visualizations
        
        Args:
            all_metrics: All collected metrics
            federated_results: Federated learning results (optional)
            save_filename: Filename to save the report
            
        Returns:
            Path to saved report
        """
        # Create figure with multiple subplots
        if federated_results:
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Extract data
        techniques = list(all_metrics.keys())
        accuracies = [all_metrics[t]['average_metrics']['accuracy'] for t in techniques]
        model_sizes = [all_metrics[t]['average_metrics']['model_size_mb'] for t in techniques]
        inference_times = [all_metrics[t]['average_metrics']['inference_time_ms'] for t in techniques]
        energy_consumptions = [all_metrics[t]['average_metrics']['energy_consumption_mw'] for t in techniques]
        compression_ratios = [all_metrics[t]['average_metrics'].get('compression_ratio', 1.0) for t in techniques]
        
        # Plot 1: Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(techniques, accuracies, alpha=0.8, color='skyblue')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Model size comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(techniques, model_sizes, alpha=0.8, color='lightcoral')
        ax2.set_title('Model Size Comparison', fontweight='bold')
        ax2.set_ylabel('Model Size (MB)')
        ax2.tick_params(axis='x', rotation=45)
        for bar, size in zip(bars2, model_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(model_sizes)*0.02,
                    f'{size:.1f}', ha='center', va='bottom')
        
        # Plot 3: Inference time comparison
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(techniques, inference_times, alpha=0.8, color='lightgreen')
        ax3.set_title('Inference Time Comparison', fontweight='bold')
        ax3.set_ylabel('Inference Time (ms)')
        ax3.tick_params(axis='x', rotation=45)
        for bar, time in zip(bars3, inference_times):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(inference_times)*0.02,
                    f'{time:.1f}', ha='center', va='bottom')
        
        # Plot 4: Accuracy vs Size scatter
        ax4 = fig.add_subplot(gs[1, 0])
        scatter4 = ax4.scatter(model_sizes, accuracies, s=100, alpha=0.7, c=range(len(techniques)), cmap='viridis')
        ax4.set_xlabel('Model Size (MB)')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy vs Model Size Trade-off', fontweight='bold')
        for i, technique in enumerate(techniques):
            ax4.annotate(technique, (model_sizes[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 5: Energy consumption
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(techniques, energy_consumptions, alpha=0.8, color='gold')
        ax5.set_title('Energy Consumption', fontweight='bold')
        ax5.set_ylabel('Energy (mW)')
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Compression ratios
        ax6 = fig.add_subplot(gs[1, 2])
        bars6 = ax6.bar(techniques, compression_ratios, alpha=0.8, color='mediumpurple')
        ax6.set_title('Compression Ratios', fontweight='bold')
        ax6.set_ylabel('Compression Ratio')
        ax6.tick_params(axis='x', rotation=45)
        for bar, ratio in zip(bars6, compression_ratios):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(compression_ratios)*0.02,
                    f'{ratio:.1f}x', ha='center', va='bottom')
        
        # Plot 7: Performance radar chart (spanning two columns)
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Normalize metrics for radar chart
        norm_accuracy = np.array(accuracies) / max(accuracies)
        norm_speed = np.array([max(inference_times)/t for t in inference_times])
        norm_size = np.array([max(model_sizes)/s for s in model_sizes])
        norm_energy = np.array([max(energy_consumptions)/e for e in energy_consumptions])
        
        categories = ['Accuracy', 'Speed', 'Compactness', 'Efficiency']
        
        # Create radar chart manually
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, technique in enumerate(techniques[:3]):  # Limit to first 3 for clarity
            values = [norm_accuracy[i], norm_speed[i], norm_size[i], norm_energy[i]]
            values += values[:1]  # Complete the circle
            
            ax7.plot(angles, values, 'o-', linewidth=2, label=technique)
            ax7.fill(angles, values, alpha=0.25)
        
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(categories)
        ax7.set_ylim(0, 1)
        ax7.set_title('Performance Radar Chart (Normalized)', fontweight='bold')
        ax7.legend()
        ax7.grid(True)
        
        # Plot 8: Efficiency scores
        ax8 = fig.add_subplot(gs[2, 2])
        efficiency_scores = [all_metrics[t]['efficiency_score'] for t in techniques]
        bars8 = ax8.bar(techniques, efficiency_scores, alpha=0.8, color='orange')
        ax8.set_title('Overall Efficiency Scores', fontweight='bold')
        ax8.set_ylabel('Efficiency Score')
        ax8.tick_params(axis='x', rotation=45)
        for bar, score in zip(bars8, efficiency_scores):
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(efficiency_scores)*0.02,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Add federated learning results if available
        if federated_results:
            ax9 = fig.add_subplot(gs[3, :])
            rounds = list(range(1, len(federated_results) + 1))
            accuracies_fed = [r['global_accuracy'] for r in federated_results]
            ax9.plot(rounds, accuracies_fed, marker='o', linewidth=2, markersize=6)
            ax9.set_xlabel('Federated Round')
            ax9.set_ylabel('Global Accuracy (%)')
            ax9.set_title('Federated Learning Convergence', fontweight='bold')
            ax9.grid(True, alpha=0.3)
        
        plt.suptitle('Edge AI Optimization - Comprehensive Summary Report', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        save_path = f"{self.save_path}/{save_filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary report saved to {save_path}")
        return save_path


if __name__ == "__main__":
    # Example usage
    viz_engine = VisualizationEngine()
    
    # Sample metrics data for testing
    sample_metrics = {
        'baseline': {
            'average_metrics': {
                'accuracy': 85.0,
                'inference_time_ms': 50.0,
                'model_size_mb': 25.0,
                'energy_consumption_mw': 1000.0,
                'compression_ratio': 1.0
            },
            'efficiency_score': 0.6
        },
        'pruning_50': {
            'average_metrics': {
                'accuracy': 82.5,
                'inference_time_ms': 40.0,
                'model_size_mb': 12.5,
                'energy_consumption_mw': 800.0,
                'compression_ratio': 2.0
            },
            'efficiency_score': 0.75
        },
        'quantization_8bit': {
            'average_metrics': {
                'accuracy': 83.0,
                'inference_time_ms': 25.0,
                'model_size_mb': 6.25,
                'energy_consumption_mw': 600.0,
                'compression_ratio': 4.0
            },
            'efficiency_score': 0.8
        }
    }
    
    # Generate sample plots
    print("Generating sample visualizations...")
    
    compression_plot = viz_engine.plot_compression_comparison(sample_metrics)
    print(f"Compression comparison plot saved: {compression_plot}")
    
    pareto_plot = viz_engine.plot_pareto_frontier(sample_metrics)
    print(f"Pareto frontier plot saved: {pareto_plot}")
    
    breakdown_plot = viz_engine.plot_technique_breakdown(sample_metrics)
    print(f"Technique breakdown plot saved: {breakdown_plot}")
    
    summary_report = viz_engine.generate_summary_report(sample_metrics)
    print(f"Summary report saved: {summary_report}")
    
    print("Sample visualizations generated successfully!")
