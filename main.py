"""
Main pipeline orchestrator for Edge AI Optimization
Coordinates all optimization techniques and runs comprehensive experiments
"""

import torch
import torch.nn as nn
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import all pipeline components
from edge_ai_pipeline.models.base_model import create_model, get_model_info
from edge_ai_pipeline.compression.pruning import MagnitudePruning, compare_pruning_methods
from edge_ai_pipeline.compression.quantization import PostTrainingQuantization, compare_quantization_methods
from edge_ai_pipeline.compression.distillation import KnowledgeDistillation, compare_distillation_strategies
from edge_ai_pipeline.federated.client import ClientManager
from edge_ai_pipeline.federated.server import FederatedServer, compare_aggregation_methods
from edge_ai_pipeline.metrics.performance import MetricsCollector, BenchmarkSuite
from edge_ai_pipeline.metrics.visualization import VisualizationEngine
from edge_ai_pipeline.utils.data_loader import DataLoader
from edge_ai_pipeline.utils.device_simulator import EdgeDeviceSimulator
from edge_ai_pipeline.utils.config import *
# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


class EdgeAIOptimizationPipeline:
    """Main pipeline for edge AI optimization experiments"""
    
    def __init__(self, config_overrides: Optional[Dict] = None):
        """
        Initialize the optimization pipeline
        
        Args:
            config_overrides: Optional configuration overrides
        """
        self.logger = logging.getLogger("EdgeAIOptimizationPipeline")
        self.logger.info("Initializing Edge AI Optimization Pipeline")
        
        # Apply configuration overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.metrics_collector = MetricsCollector()
        self.visualization_engine = VisualizationEngine(PLOTS_DIR)
        
        # Load data
        self.train_loader, self.test_loader = self.data_loader.load_centralized_data()
        self.logger.info(f"Loaded CIFAR-10 dataset: {len(self.train_loader.dataset)} train, {len(self.test_loader.dataset)} test")
        
        # Create models
        self.baseline_model = create_model('base')
        self.teacher_model = create_model('teacher')
        self.student_model = create_model('student')
        
        # Log model information
        for name, model in [('Baseline', self.baseline_model), ('Teacher', self.teacher_model), ('Student', self.student_model)]:
            info = get_model_info(model)
            self.logger.info(f"{name} model: {info['total_parameters']:,} parameters, {info['model_size_mb']:.2f} MB")
        
        # Results storage
        self.experiment_results = {}
        self.optimization_models = {}
        
    def _apply_config_overrides(self, overrides: Dict):
        """Apply configuration overrides"""
        for key, value in overrides.items():
            if hasattr(globals(), key):
                globals()[key] = value
                self.logger.info(f"Config override: {key} = {value}")
    
    def run_baseline_evaluation(self) -> Dict[str, Any]:
        """
        Run baseline model evaluation across different devices
        
        Returns:
            Baseline evaluation results
        """
        self.logger.info("Running baseline evaluation...")
        
        baseline_results = {}
        
        for device_type in EDGE_DEVICE_CONFIGS.keys():
            self.logger.info(f"Evaluating baseline on {device_type}")
            
            # Collect comprehensive metrics
            metrics = self.metrics_collector.collect_comprehensive_metrics(
                self.baseline_model, self.test_loader, 'baseline', 'baseline', device_type
            )
            
            baseline_results[device_type] = {
                'metrics': metrics,
                'device_config': EDGE_DEVICE_CONFIGS[device_type]
            }
        
        self.experiment_results['baseline'] = baseline_results
        return baseline_results
    
    def run_pruning_experiments(self) -> Dict[str, Any]:
        """
        Run comprehensive pruning experiments
        
        Returns:
            Pruning experiment results
        """
        self.logger.info("Running pruning experiments...")
        
        pruning_results = {}
        
        for ratio in PRUNING_RATIOS:
            if ratio == 0.0:
                continue  # Skip baseline
            
            self.logger.info(f"Testing pruning with ratio {ratio}")
            
            # Test both unstructured and structured pruning
            for structured in [False, True]:
                method_name = f"{'structured' if structured else 'unstructured'}_pruning_{ratio}"
                
                # Create pruned model
                pruned_model = torch.nn.Module()
                try:
                    pruned_model = self._create_pruned_model(ratio, structured)
                    self.optimization_models[method_name] = pruned_model
                    
                    # Evaluate across devices
                    device_results = {}
                    for device_type in EDGE_DEVICE_CONFIGS.keys():
                        metrics = self.metrics_collector.collect_comprehensive_metrics(
                            pruned_model, self.test_loader, method_name, method_name, 
                            device_type, self.baseline_model
                        )
                        device_results[device_type] = metrics
                    
                    pruning_results[method_name] = {
                        'device_results': device_results,
                        'pruning_ratio': ratio,
                        'structured': structured
                    }
                    
                except Exception as e:
                    self.logger.error(f"Failed to create {method_name}: {e}")
                    continue
        
        self.experiment_results['pruning'] = pruning_results
        return pruning_results
    
    def run_quantization_experiments(self) -> Dict[str, Any]:
        """
        Run comprehensive quantization experiments
        
        Returns:
            Quantization experiment results
        """
        self.logger.info("Running quantization experiments...")
        
        quantization_results = {}
        
        for bit_width in QUANTIZATION_BITS:
            if bit_width == 32:
                continue  # Skip baseline
            
            self.logger.info(f"Testing quantization with {bit_width}-bit")
            
            # Test dynamic and static quantization
            for method in ['dynamic', 'static']:
                method_name = f"{method}_quantization_{bit_width}bit"
                
                try:
                    quantized_model = self._create_quantized_model(bit_width, method)
                    self.optimization_models[method_name] = quantized_model
                    
                    # Evaluate across devices
                    device_results = {}
                    for device_type in EDGE_DEVICE_CONFIGS.keys():
                        metrics = self.metrics_collector.collect_comprehensive_metrics(
                            quantized_model, self.test_loader, method_name, method_name,
                            device_type, self.baseline_model
                        )
                        device_results[device_type] = metrics
                    
                    quantization_results[method_name] = {
                        'device_results': device_results,
                        'bit_width': bit_width,
                        'quantization_method': method
                    }
                    
                except Exception as e:
                    self.logger.error(f"Failed to create {method_name}: {e}")
                    continue
        
        self.experiment_results['quantization'] = quantization_results
        return quantization_results
    
    def run_distillation_experiments(self) -> Dict[str, Any]:
        """
        Run knowledge distillation experiments
        
        Returns:
            Distillation experiment results
        """
        self.logger.info("Running knowledge distillation experiments...")
        
        # First, train teacher model (simplified training for demo)
        self.logger.info("Training teacher model...")
        trained_teacher = self._train_teacher_model()
        
        # Run distillation
        self.logger.info("Training student model with knowledge distillation...")
        distilled_student = self._train_student_with_distillation(trained_teacher)
        
        self.optimization_models['knowledge_distillation'] = distilled_student
        
        # Evaluate distilled model
        device_results = {}
        for device_type in EDGE_DEVICE_CONFIGS.keys():
            metrics = self.metrics_collector.collect_comprehensive_metrics(
                distilled_student, self.test_loader, 'knowledge_distillation', 
                'knowledge_distillation', device_type, self.baseline_model
            )
            device_results[device_type] = metrics
        
        distillation_results = {
            'knowledge_distillation': {
                'device_results': device_results,
                'teacher_student_ratio': self._calculate_model_ratio(trained_teacher, distilled_student)
            }
        }
        
        self.experiment_results['distillation'] = distillation_results
        return distillation_results
    
    def run_combined_optimization_experiments(self) -> Dict[str, Any]:
        """
        Run experiments combining multiple optimization techniques
        
        Returns:
            Combined optimization results
        """
        self.logger.info("Running combined optimization experiments...")
        
        combined_results = {}
        
        # Define combinations to test
        combinations = [
            {'pruning': 0.4, 'quantization': 16, 'distillation': False, 'name': 'pruning_quantization_light'},
            {'pruning': 0.5, 'quantization': 8, 'distillation': False, 'name': 'pruning_quantization_aggressive'},
            {'pruning': 0.4, 'quantization': 32, 'distillation': True, 'name': 'pruning_distillation'},
            {'pruning': 0.5, 'quantization': 8, 'distillation': True, 'name': 'all_techniques_combined'}
        ]
        
        for combo in combinations:
            self.logger.info(f"Testing combination: {combo['name']}")
            
            try:
                combined_model = self._create_combined_optimized_model(combo)
                self.optimization_models[combo['name']] = combined_model
                
                # Evaluate across devices
                device_results = {}
                for device_type in EDGE_DEVICE_CONFIGS.keys():
                    metrics = self.metrics_collector.collect_comprehensive_metrics(
                        combined_model, self.test_loader, combo['name'], combo['name'],
                        device_type, self.baseline_model
                    )
                    device_results[device_type] = metrics
                
                combined_results[combo['name']] = {
                    'device_results': device_results,
                    'optimization_config': combo
                }
                
            except Exception as e:
                self.logger.error(f"Failed to create combined model {combo['name']}: {e}")
                continue
        
        self.experiment_results['combined'] = combined_results
        return combined_results
    
    def run_federated_learning_experiments(self) -> Dict[str, Any]:
        """
        Run federated learning experiments
        
        Returns:
            Federated learning results
        """
        self.logger.info("Running federated learning experiments...")
        
        # Create federated data split
        client_loaders, test_loader = self.data_loader.get_federated_dataloaders(NUM_CLIENTS, alpha=0.5)
        
        # Create client manager
        client_manager = ClientManager(
            num_clients=NUM_CLIENTS,
            model_template=self.baseline_model,
            client_data_loaders=client_loaders
        )
        
        federated_results = {}
        
        # Test different aggregation methods
        for agg_method in ['fedavg', 'fedmedian']:
            self.logger.info(f"Testing federated learning with {agg_method}")
            
            try:
                # Create server
                server = FederatedServer(
                    global_model=torch.nn.Module(),
                    test_loader=test_loader,
                    aggregation_method=agg_method
                )
                
                # Run federated training
                training_results = server.run_federated_training(
                    client_manager=client_manager,
                    num_rounds=min(FEDERATED_ROUNDS, 10),  # Limit for demo
                    client_fraction=CLIENT_FRACTION,
                    local_epochs=LOCAL_EPOCHS
                )
                
                federated_results[agg_method] = {
                    'training_results': training_results,
                    'convergence_analysis': server.analyze_convergence(),
                    'communication_costs': server.get_communication_costs()
                }
                
            except Exception as e:
                self.logger.error(f"Failed federated learning with {agg_method}: {e}")
                continue
        
        self.experiment_results['federated'] = federated_results
        return federated_results
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete edge AI optimization pipeline
        
        Returns:
            Complete pipeline results
        """
        self.logger.info("Starting complete Edge AI optimization pipeline...")
        
        pipeline_start_time = time.time()
        
        # Run all experiments
        experiments = [
            ("Baseline Evaluation", self.run_baseline_evaluation),
            ("Pruning Experiments", self.run_pruning_experiments),
            ("Quantization Experiments", self.run_quantization_experiments),
            ("Distillation Experiments", self.run_distillation_experiments),
            ("Combined Optimization", self.run_combined_optimization_experiments),
            ("Federated Learning", self.run_federated_learning_experiments)
        ]
        
        for experiment_name, experiment_func in experiments:
            try:
                self.logger.info(f"Running {experiment_name}...")
                experiment_start_time = time.time()
                
                results = experiment_func()
                
                experiment_time = time.time() - experiment_start_time
                self.logger.info(f"{experiment_name} completed in {experiment_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to run {experiment_name}: {e}")
                continue
        
        # Generate comprehensive analysis
        self.logger.info("Generating comprehensive analysis...")
        analysis_results = self._generate_comprehensive_analysis()
        
        # Generate visualizations
        self.logger.info("Generating visualizations...")
        visualization_results = self._generate_visualizations()
        
        # Save results
        self.logger.info("Saving results...")
        self._save_results()
        
        total_time = time.time() - pipeline_start_time
        
        pipeline_summary = {
            'experiment_results': self.experiment_results,
            'analysis_results': analysis_results,
            'visualization_results': visualization_results,
            'pipeline_execution_time_s': total_time,
            'models_created': len(self.optimization_models),
            'techniques_tested': len([k for k in self.experiment_results.keys() if k != 'baseline'])
        }
        
        self.logger.info(f"Complete pipeline finished in {total_time:.2f}s")
        self.logger.info(f"Created {len(self.optimization_models)} optimized models")
        
        return pipeline_summary
    
    def _create_pruned_model(self, pruning_ratio: float, structured: bool) -> nn.Module:
        """Create pruned model"""
        import copy
        model = copy.deepcopy(self.baseline_model)
        pruner = MagnitudePruning(model)
        pruner.apply_magnitude_pruning(pruning_ratio, structured)
        return model
    
    def _create_quantized_model(self, bit_width: int, method: str) -> nn.Module:
        """Create quantized model"""
        import copy
        model = copy.deepcopy(self.baseline_model)
        quantizer = PostTrainingQuantization(model)
        
        if method == 'dynamic':
            quantized_model, _ = quantizer.apply_dynamic_quantization(bit_width)
        else:  # static
            quantized_model, _ = quantizer.apply_static_quantization(self.train_loader, bit_width)
        
        return quantized_model
    
    def _train_teacher_model(self) -> nn.Module:
        """Train teacher model (simplified for demo)"""
        import copy
        teacher = copy.deepcopy(self.teacher_model)
        
        # Simplified training - in practice, this would be full training
        teacher.train()
        optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs (limited for demo)
        for epoch in range(2):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if batch_idx >= 50:  # Limit batches for demo
                    break
                
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = teacher(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return teacher
    
    def _train_student_with_distillation(self, teacher_model: nn.Module) -> nn.Module:
        """Train student model with knowledge distillation"""
        import copy
        student = copy.deepcopy(self.student_model)
        
        distiller = KnowledgeDistillation(teacher_model, student)
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        
        # Train for a few epochs (limited for demo)
        student.train()
        for epoch in range(2):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if batch_idx >= 50:  # Limit batches for demo
                    break
                
                data, target = data.to(DEVICE), target.to(DEVICE)
                distiller.train_step(data, target, optimizer)
        
        return student
    
    def _create_combined_optimized_model(self, config: Dict) -> nn.Module:
        """Create model with combined optimizations"""
        import copy
        model = copy.deepcopy(self.baseline_model)
        
        # Apply pruning if specified
        if config['pruning'] > 0:
            pruner = MagnitudePruning(model)
            pruner.apply_magnitude_pruning(config['pruning'], structured=False)
        
        # Apply quantization if specified
        if config['quantization'] < 32:
            quantizer = PostTrainingQuantization(model)
            model, _ = quantizer.apply_dynamic_quantization(config['quantization'])
        
        # Knowledge distillation would require retraining, so we'll use the pre-trained student
        if config['distillation']:
            model = copy.deepcopy(self.student_model)
            # Apply other optimizations to student model
            if config['pruning'] > 0:
                pruner = MagnitudePruning(model)
                pruner.apply_magnitude_pruning(config['pruning'], structured=False)
        
        return model
    
    def _calculate_model_ratio(self, model1: nn.Module, model2: nn.Module) -> float:
        """Calculate size ratio between two models"""
        info1 = get_model_info(model1)
        info2 = get_model_info(model2)
        return info1['model_size_mb'] / info2['model_size_mb']
    
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of all results"""
        # Collect all metrics for trade-off analysis
        all_metrics = {}
        
        # Collect baseline metrics
        if 'baseline' in self.experiment_results:
            baseline_desktop = self.experiment_results['baseline'].get('desktop', {}).get('metrics')
            if baseline_desktop:
                all_metrics['baseline'] = {
                    'average_metrics': {
                        'accuracy': baseline_desktop.accuracy,
                        'inference_time_ms': baseline_desktop.inference_time_ms,
                        'model_size_mb': baseline_desktop.model_size_mb,
                        'energy_consumption_mw': baseline_desktop.energy_consumption_mw,
                        'compression_ratio': 1.0,
                        'accuracy_drop': 0.0
                    },
                    'efficiency_score': 0.6  # Default baseline score
                }
        
        # Collect optimization technique metrics
        for category, experiments in self.experiment_results.items():
            if category == 'baseline':
                continue
            
            for technique_name, technique_data in experiments.items():
                if 'device_results' in technique_data:
                    desktop_metrics = technique_data['device_results'].get('desktop')
                    if desktop_metrics:
                        all_metrics[technique_name] = {
                            'average_metrics': {
                                'accuracy': desktop_metrics.accuracy,
                                'inference_time_ms': desktop_metrics.inference_time_ms,
                                'model_size_mb': desktop_metrics.model_size_mb,
                                'energy_consumption_mw': desktop_metrics.energy_consumption_mw,
                                'compression_ratio': desktop_metrics.compression_ratio,
                                'accuracy_drop': desktop_metrics.accuracy_drop
                            },
                            'efficiency_score': self._calculate_efficiency_score(desktop_metrics)
                        }
        
        # Generate trade-off analysis
        trade_off_analysis = self.metrics_collector.analyze_trade_offs() if hasattr(self.metrics_collector, 'analyze_trade_offs') else {}
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(all_metrics)
        
        return {
            'all_metrics': all_metrics,
            'trade_off_analysis': trade_off_analysis,
            'recommendations': recommendations,
            'summary_statistics': self._calculate_summary_statistics(all_metrics)
        }
    
    def _calculate_efficiency_score(self, metrics) -> float:
        """Calculate efficiency score for a model"""
        # Simple efficiency calculation
        accuracy_score = metrics.accuracy / 100.0
        speed_score = 1.0 / (1.0 + metrics.inference_time_ms / 100.0)
        size_score = 1.0 / (1.0 + metrics.model_size_mb / 10.0)
        energy_score = 1.0 / (1.0 + metrics.energy_consumption_mw / 1000.0)
        
        return 0.4 * accuracy_score + 0.2 * speed_score + 0.2 * size_score + 0.2 * energy_score
    
    def _generate_optimization_recommendations(self, all_metrics: Dict) -> Dict[str, str]:
        """Generate optimization recommendations based on results"""
        if not all_metrics:
            return {'error': 'No metrics available for recommendations'}
        
        recommendations = {}
        
        # Find best technique for different criteria
        try:
            best_accuracy = max(all_metrics.items(), key=lambda x: x[1]['average_metrics']['accuracy'])
            best_speed = min(all_metrics.items(), key=lambda x: x[1]['average_metrics']['inference_time_ms'])
            best_size = min(all_metrics.items(), key=lambda x: x[1]['average_metrics']['model_size_mb'])
            best_efficiency = max(all_metrics.items(), key=lambda x: x[1]['efficiency_score'])
            
            recommendations.update({
                'best_accuracy': f"{best_accuracy[0]} (Accuracy: {best_accuracy[1]['average_metrics']['accuracy']:.2f}%)",
                'best_speed': f"{best_speed[0]} (Inference: {best_speed[1]['average_metrics']['inference_time_ms']:.2f}ms)",
                'best_size': f"{best_size[0]} (Size: {best_size[1]['average_metrics']['model_size_mb']:.2f}MB)",
                'best_overall': f"{best_efficiency[0]} (Efficiency: {best_efficiency[1]['efficiency_score']:.3f})"
            })
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations['error'] = str(e)
        
        return recommendations
    
    def _calculate_summary_statistics(self, all_metrics: Dict) -> Dict[str, float]:
        """Calculate summary statistics across all techniques"""
        if not all_metrics:
            return {}
        
        accuracies = [m['average_metrics']['accuracy'] for m in all_metrics.values()]
        inference_times = [m['average_metrics']['inference_time_ms'] for m in all_metrics.values()]
        model_sizes = [m['average_metrics']['model_size_mb'] for m in all_metrics.values()]
        
        return {
            'accuracy_range': max(accuracies) - min(accuracies),
            'average_accuracy': sum(accuracies) / len(accuracies),
            'speed_improvement_max': max(inference_times) / min(inference_times),
            'size_reduction_max': max(model_sizes) / min(model_sizes),
            'num_techniques_tested': len(all_metrics)
        }
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """Generate all visualizations"""
        visualization_paths = {}
        
        try:
            # Get metrics for visualization
            analysis_results = self._generate_comprehensive_analysis()
            all_metrics = analysis_results.get('all_metrics', {})
            
            if all_metrics:
                # Generate comparison plots
                comp_plot = self.visualization_engine.plot_compression_comparison(all_metrics)
                visualization_paths['compression_comparison'] = comp_plot
                
                # Generate Pareto frontier
                pareto_plot = self.visualization_engine.plot_pareto_frontier(all_metrics)
                visualization_paths['pareto_frontier'] = pareto_plot
                
                # Generate technique breakdown
                breakdown_plot = self.visualization_engine.plot_technique_breakdown(all_metrics)
                visualization_paths['technique_breakdown'] = breakdown_plot
                
                # Generate summary report
                summary_plot = self.visualization_engine.generate_summary_report(all_metrics)
                visualization_paths['summary_report'] = summary_plot
                
                # Generate federated learning plots if available
                if 'federated' in self.experiment_results:
                    for method, fed_results in self.experiment_results['federated'].items():
                        if 'training_results' in fed_results and 'training_history' in fed_results['training_results']:
                            fed_plot = self.visualization_engine.plot_federated_learning_convergence(
                                fed_results['training_results']['training_history'],
                                f"federated_convergence_{method}.png"
                            )
                            visualization_paths[f'federated_{method}'] = fed_plot
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            visualization_paths['error'] = str(e)
        
        return visualization_paths
    
    def _save_results(self):
        """Save all results to files"""
        # Save experiment results
        results_file = os.path.join(RESULTS_DIR, "experiment_results.json")
        with open(results_file, 'w') as f:
            # Convert any torch tensors or complex objects to serializable format
            serializable_results = self._make_serializable(self.experiment_results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save model information
        models_info = {}
        for name, model in self.optimization_models.items():
            models_info[name] = get_model_info(model)
        
        models_file = os.path.join(RESULTS_DIR, "models_info.json")
        with open(models_file, 'w') as f:
            json.dump(models_info, f, indent=2, default=str)
        
        # Save metrics
        metrics_file = os.path.join(RESULTS_DIR, "comprehensive_metrics.json")
        self.metrics_collector.save_metrics_to_json(metrics_file)
        
        self.logger.info(f"Results saved to {RESULTS_DIR}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if hasattr(obj, '_asdict'):  # dataclass
            return obj._asdict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):  # custom objects
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_') and not callable(v)}
        else:
            return obj


def main():
    """Main entry point for the pipeline"""
    logger = logging.getLogger("main")
    logger.info("Starting Edge AI Optimization Pipeline")
    
    try:
        # Initialize pipeline
        pipeline = EdgeAIOptimizationPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EDGE AI OPTIMIZATION PIPELINE SUMMARY")
        logger.info("="*80)
        logger.info(f"Total execution time: {results['pipeline_execution_time_s']:.2f} seconds")
        logger.info(f"Models created: {results['models_created']}")
        logger.info(f"Techniques tested: {results['techniques_tested']}")
        
        if 'analysis_results' in results and 'recommendations' in results['analysis_results']:
            recommendations = results['analysis_results']['recommendations']
            logger.info("\nRECOMMENDATIONS:")
            for key, value in recommendations.items():
                logger.info(f"  {key.replace('_', ' ').title()}: {value}")
        
        logger.info(f"\nResults saved to: {RESULTS_DIR}")
        logger.info(f"Plots saved to: {PLOTS_DIR}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
