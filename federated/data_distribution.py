"""
Advanced Data Distribution for Federated Learning
Handles non-IID data distribution patterns and client data heterogeneity.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader, Subset
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class DataDistribution:
    """
    Handles various non-IID data distribution patterns for federated learning
    """
    
    def __init__(self, dataset: Dataset, num_clients: int, seed: int = 42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        self.client_data_indices = {}
        self.distribution_stats = {}
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    def create_iid_distribution(self) -> Dict[int, List[int]]:
        """
        Create IID (Independent and Identically Distributed) data distribution
        Each client gets a random sample from the entire dataset
        """
        total_samples = len(self.dataset)
        samples_per_client = total_samples // self.num_clients
        
        # Shuffle all indices
        all_indices = list(range(total_samples))
        np.random.shuffle(all_indices)
        
        # Distribute equally among clients
        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            
            # Handle remaining samples for the last client
            if client_id == self.num_clients - 1:
                end_idx = total_samples
            
            self.client_data_indices[client_id] = all_indices[start_idx:end_idx]
        
        self._calculate_distribution_stats("iid")
        logger.info(f"Created IID distribution for {self.num_clients} clients")
        return self.client_data_indices
    
    def create_label_skew_distribution(self, alpha: float = 0.5, min_samples: int = 10) -> Dict[int, List[int]]:
        """
        Create non-IID distribution with label skew using Dirichlet distribution
        
        Args:
            alpha: Concentration parameter (lower = more skewed)
            min_samples: Minimum samples per client per class
        """
        # Get labels for all samples
        labels = self._extract_labels()
        unique_labels = sorted(set(labels))
        num_classes = len(unique_labels)
        
        # Group indices by label
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        # Generate Dirichlet distribution for each class
        client_class_distributions = np.random.dirichlet([alpha] * self.num_clients, num_classes)
        
        # Distribute samples for each class among clients
        for client_id in range(self.num_clients):
            self.client_data_indices[client_id] = []
        
        for class_idx, label in enumerate(unique_labels):
            class_indices = label_indices[label]
            np.random.shuffle(class_indices)
            
            # Calculate number of samples per client for this class
            total_class_samples = len(class_indices)
            client_proportions = client_class_distributions[class_idx]
            
            start_idx = 0
            for client_id in range(self.num_clients):
                if client_id == self.num_clients - 1:
                    # Last client gets remaining samples
                    client_samples = class_indices[start_idx:]
                else:
                    num_samples = max(min_samples, int(total_class_samples * client_proportions[client_id]))
                    client_samples = class_indices[start_idx:start_idx + num_samples]
                    start_idx += num_samples
                
                self.client_data_indices[client_id].extend(client_samples)
        
        # Shuffle each client's data
        for client_id in range(self.num_clients):
            np.random.shuffle(self.client_data_indices[client_id])
        
        self._calculate_distribution_stats("label_skew", {"alpha": alpha})
        logger.info(f"Created label skew distribution (alpha={alpha}) for {self.num_clients} clients")
        return self.client_data_indices
    
    def create_quantity_skew_distribution(self, min_ratio: float = 0.1, max_ratio: float = 0.9) -> Dict[int, List[int]]:
        """
        Create non-IID distribution with quantity skew
        Some clients have much more data than others
        
        Args:
            min_ratio: Minimum proportion of data for smallest client
            max_ratio: Maximum proportion of data for largest client
        """
        total_samples = len(self.dataset)
        
        # Generate random proportions that sum to 1
        proportions = np.random.dirichlet([1.0] * self.num_clients)
        
        # Scale proportions to respect min/max ratios
        proportions = proportions * (max_ratio - min_ratio) + min_ratio
        proportions = proportions / proportions.sum()  # Normalize
        
        # Shuffle all indices
        all_indices = list(range(total_samples))
        np.random.shuffle(all_indices)
        
        # Distribute based on proportions
        start_idx = 0
        for client_id in range(self.num_clients):
            if client_id == self.num_clients - 1:
                # Last client gets remaining samples
                client_samples = all_indices[start_idx:]
            else:
                num_samples = int(total_samples * proportions[client_id])
                client_samples = all_indices[start_idx:start_idx + num_samples]
                start_idx += num_samples
            
            self.client_data_indices[client_id] = client_samples
        
        self._calculate_distribution_stats("quantity_skew", {"min_ratio": min_ratio, "max_ratio": max_ratio})
        logger.info(f"Created quantity skew distribution for {self.num_clients} clients")
        return self.client_data_indices
    
    def create_feature_skew_distribution(self, noise_level: float = 0.1, 
                                       feature_shift: float = 0.2) -> Dict[int, List[int]]:
        """
        Create non-IID distribution with feature distribution skew
        Different clients see different feature distributions
        
        Args:
            noise_level: Level of noise to add to features
            feature_shift: Amount to shift feature distributions
        """
        # Start with IID distribution
        self.create_iid_distribution()
        
        # Apply feature transformations to create skew
        # This is conceptual - actual implementation would modify the dataset
        # For now, we'll track the skew parameters
        
        self._calculate_distribution_stats("feature_skew", {
            "noise_level": noise_level, 
            "feature_shift": feature_shift
        })
        logger.info(f"Created feature skew distribution for {self.num_clients} clients")
        return self.client_data_indices
    
    def create_temporal_skew_distribution(self, time_windows: List[Tuple[float, float]]) -> Dict[int, List[int]]:
        """
        Create non-IID distribution with temporal skew
        Clients have data from different time periods
        
        Args:
            time_windows: List of (start_ratio, end_ratio) for each client
        """
        total_samples = len(self.dataset)
        
        if len(time_windows) != self.num_clients:
            # Generate random time windows
            time_windows = []
            for _ in range(self.num_clients):
                start = np.random.uniform(0, 0.8)
                end = np.random.uniform(start + 0.1, 1.0)
                time_windows.append((start, end))
        
        # Assign samples based on temporal windows
        for client_id in range(self.num_clients):
            start_ratio, end_ratio = time_windows[client_id]
            start_idx = int(total_samples * start_ratio)
            end_idx = int(total_samples * end_ratio)
            
            self.client_data_indices[client_id] = list(range(start_idx, end_idx))
        
        self._calculate_distribution_stats("temporal_skew", {"time_windows": time_windows})
        logger.info(f"Created temporal skew distribution for {self.num_clients} clients")
        return self.client_data_indices
    
    def create_pathological_distribution(self, shards_per_client: int = 2) -> Dict[int, List[int]]:
        """
        Create pathological non-IID distribution
        Each client only sees a few classes (shards)
        
        Args:
            shards_per_client: Number of class shards per client
        """
        # Get labels and group by class
        labels = self._extract_labels()
        unique_labels = sorted(set(labels))
        num_classes = len(unique_labels)
        
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        # Create shards (divide each class into small shards)
        shards_per_class = max(1, (self.num_clients * shards_per_client) // num_classes)
        all_shards = []
        
        for label in unique_labels:
            class_indices = label_indices[label]
            np.random.shuffle(class_indices)
            
            shard_size = len(class_indices) // shards_per_class
            for i in range(shards_per_class):
                start_idx = i * shard_size
                if i == shards_per_class - 1:
                    shard = class_indices[start_idx:]
                else:
                    shard = class_indices[start_idx:start_idx + shard_size]
                
                if shard:  # Only add non-empty shards
                    all_shards.append((label, shard))
        
        # Randomly assign shards to clients
        np.random.shuffle(all_shards)
        
        for client_id in range(self.num_clients):
            self.client_data_indices[client_id] = []
        
        shard_idx = 0
        for client_id in range(self.num_clients):
            for _ in range(min(shards_per_client, len(all_shards) - shard_idx)):
                if shard_idx < len(all_shards):
                    _, shard_indices = all_shards[shard_idx]
                    self.client_data_indices[client_id].extend(shard_indices)
                    shard_idx += 1
        
        # Shuffle each client's data
        for client_id in range(self.num_clients):
            np.random.shuffle(self.client_data_indices[client_id])
        
        self._calculate_distribution_stats("pathological", {"shards_per_client": shards_per_client})
        logger.info(f"Created pathological distribution ({shards_per_client} shards/client) for {self.num_clients} clients")
        return self.client_data_indices
    
    def _extract_labels(self) -> List[int]:
        """Extract labels from the dataset"""
        labels = []
        
        for i in range(len(self.dataset)):
            try:
                if hasattr(self.dataset, 'targets'):
                    # For datasets like CIFAR-10
                    labels.append(self.dataset.targets[i])
                elif hasattr(self.dataset, 'labels'):
                    labels.append(self.dataset.labels[i])
                else:
                    # Try to get label from dataset item
                    _, label = self.dataset[i]
                    labels.append(label)
            except Exception as e:
                logger.warning(f"Could not extract label for sample {i}: {e}")
                labels.append(0)  # Default label
        
        return labels
    
    def _calculate_distribution_stats(self, distribution_type: str, params: Dict = None):
        """Calculate statistics about the data distribution"""
        labels = self._extract_labels()
        unique_labels = sorted(set(labels))
        
        # Calculate per-client statistics
        client_stats = {}
        overall_label_counts = Counter(labels)
        
        for client_id in range(self.num_clients):
            client_indices = self.client_data_indices.get(client_id, [])
            client_labels = [labels[idx] for idx in client_indices]
            client_label_counts = Counter(client_labels)
            
            # Calculate entropy and class distribution
            total_samples = len(client_labels)
            entropy = 0.0
            class_distribution = {}
            
            for label in unique_labels:
                count = client_label_counts.get(label, 0)
                proportion = count / total_samples if total_samples > 0 else 0
                class_distribution[label] = proportion
                
                if proportion > 0:
                    entropy -= proportion * np.log2(proportion)
            
            client_stats[client_id] = {
                'num_samples': total_samples,
                'num_classes': len([c for c in class_distribution.values() if c > 0]),
                'entropy': entropy,
                'class_distribution': class_distribution,
                'label_counts': dict(client_label_counts)
            }
        
        # Calculate global statistics
        sample_sizes = [stats['num_samples'] for stats in client_stats.values()]
        entropies = [stats['entropy'] for stats in client_stats.values()]
        num_classes_per_client = [stats['num_classes'] for stats in client_stats.values()]
        
        self.distribution_stats = {
            'distribution_type': distribution_type,
            'parameters': params or {},
            'num_clients': self.num_clients,
            'total_samples': len(self.dataset),
            'unique_labels': unique_labels,
            'client_stats': client_stats,
            'global_stats': {
                'avg_samples_per_client': np.mean(sample_sizes),
                'std_samples_per_client': np.std(sample_sizes),
                'min_samples_per_client': np.min(sample_sizes),
                'max_samples_per_client': np.max(sample_sizes),
                'avg_entropy': np.mean(entropies),
                'std_entropy': np.std(entropies),
                'avg_classes_per_client': np.mean(num_classes_per_client),
                'non_iid_score': self._calculate_non_iid_score()
            }
        }
    
    def _calculate_non_iid_score(self) -> float:
        """
        Calculate a score indicating how non-IID the distribution is
        Score ranges from 0 (perfectly IID) to 1 (extremely non-IID)
        """
        if not self.client_data_indices:
            return 0.0
        
        labels = self._extract_labels()
        unique_labels = sorted(set(labels))
        num_classes = len(unique_labels)
        
        # Calculate KL divergence between each client's distribution and global distribution
        global_label_counts = Counter(labels)
        global_distribution = np.array([global_label_counts[label] / len(labels) for label in unique_labels])
        
        kl_divergences = []
        
        for client_id in range(self.num_clients):
            client_indices = self.client_data_indices.get(client_id, [])
            client_labels = [labels[idx] for idx in client_indices]
            client_label_counts = Counter(client_labels)
            
            # Create client distribution vector
            client_distribution = np.array([
                client_label_counts.get(label, 0) / len(client_labels) if client_labels else 0
                for label in unique_labels
            ])
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            global_distribution_smooth = global_distribution + epsilon
            client_distribution_smooth = client_distribution + epsilon
            
            # Calculate KL divergence
            kl_div = np.sum(client_distribution_smooth * np.log(client_distribution_smooth / global_distribution_smooth))
            kl_divergences.append(kl_div)
        
        # Normalize score to [0, 1] range
        avg_kl_div = np.mean(kl_divergences)
        max_possible_kl = np.log(num_classes)  # Maximum KL divergence for uniform vs one-hot
        non_iid_score = min(1.0, avg_kl_div / max_possible_kl)
        
        return non_iid_score
    
    def get_client_dataloader(self, client_id: int, batch_size: int = 32, 
                            shuffle: bool = True, **kwargs) -> DataLoader:
        """
        Get DataLoader for a specific client
        
        Args:
            client_id: Client identifier
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            **kwargs: Additional arguments for DataLoader
        
        Returns:
            DataLoader for the client's data
        """
        if client_id not in self.client_data_indices:
            raise ValueError(f"Client {client_id} not found in distribution")
        
        client_indices = self.client_data_indices[client_id]
        client_subset = Subset(self.dataset, client_indices)
        
        return DataLoader(client_subset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def get_distribution_summary(self) -> Dict:
        """Get comprehensive summary of the data distribution"""
        return self.distribution_stats
    
    def visualize_distribution(self, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Visualize the data distribution across clients
        
        Args:
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        if not self.distribution_stats:
            logger.warning("No distribution stats available. Create a distribution first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Data Distribution Analysis - {self.distribution_stats['distribution_type']}", fontsize=16)
        
        # 1. Samples per client
        client_ids = list(range(self.num_clients))
        sample_counts = [self.distribution_stats['client_stats'][cid]['num_samples'] for cid in client_ids]
        
        axes[0, 0].bar(client_ids, sample_counts)
        axes[0, 0].set_title('Samples per Client')
        axes[0, 0].set_xlabel('Client ID')
        axes[0, 0].set_ylabel('Number of Samples')
        
        # 2. Classes per client
        class_counts = [self.distribution_stats['client_stats'][cid]['num_classes'] for cid in client_ids]
        
        axes[0, 1].bar(client_ids, class_counts)
        axes[0, 1].set_title('Classes per Client')
        axes[0, 1].set_xlabel('Client ID')
        axes[0, 1].set_ylabel('Number of Classes')
        
        # 3. Entropy per client
        entropies = [self.distribution_stats['client_stats'][cid]['entropy'] for cid in client_ids]
        
        axes[1, 0].bar(client_ids, entropies)
        axes[1, 0].set_title('Data Entropy per Client')
        axes[1, 0].set_xlabel('Client ID')
        axes[1, 0].set_ylabel('Entropy')
        
        # 4. Class distribution heatmap
        unique_labels = self.distribution_stats['unique_labels']
        distribution_matrix = np.zeros((self.num_clients, len(unique_labels)))
        
        for cid in client_ids:
            for i, label in enumerate(unique_labels):
                distribution_matrix[cid, i] = self.distribution_stats['client_stats'][cid]['class_distribution'].get(label, 0)
        
        sns.heatmap(distribution_matrix, 
                   xticklabels=unique_labels, 
                   yticklabels=[f"Client {i}" for i in client_ids],
                   ax=axes[1, 1],
                   cmap='Blues',
                   cbar_kws={'label': 'Proportion'})
        axes[1, 1].set_title('Class Distribution Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def compare_distributions(self, other_distributions: List['DataDistribution']) -> Dict:
        """
        Compare multiple data distributions
        
        Args:
            other_distributions: List of other DataDistribution objects to compare
        
        Returns:
            Comparison results
        """
        all_distributions = [self] + other_distributions
        comparison = {
            'distributions': [],
            'metrics_comparison': {},
            'rankings': {}
        }
        
        # Collect metrics from all distributions
        metrics = ['avg_samples_per_client', 'avg_entropy', 'non_iid_score', 'avg_classes_per_client']
        
        for i, dist in enumerate(all_distributions):
            if not dist.distribution_stats:
                continue
                
            dist_info = {
                'index': i,
                'type': dist.distribution_stats['distribution_type'],
                'parameters': dist.distribution_stats['parameters'],
                'metrics': {metric: dist.distribution_stats['global_stats'][metric] for metric in metrics}
            }
            comparison['distributions'].append(dist_info)
        
        # Calculate metric comparisons
        for metric in metrics:
            values = [d['metrics'][metric] for d in comparison['distributions']]
            comparison['metrics_comparison'][metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Rank distributions by non-IID score
        sorted_dists = sorted(comparison['distributions'], key=lambda x: x['metrics']['non_iid_score'])
        comparison['rankings']['by_non_iid_score'] = [d['type'] for d in sorted_dists]
        
        return comparison

def create_federated_data_distribution(dataset: Dataset, num_clients: int, 
                                     distribution_type: str = "iid", 
                                     **kwargs) -> DataDistribution:
    """
    Factory function to create federated data distributions
    
    Args:
        dataset: The dataset to distribute
        num_clients: Number of federated clients
        distribution_type: Type of distribution ("iid", "label_skew", "quantity_skew", etc.)
        **kwargs: Additional parameters for specific distribution types
    
    Returns:
        DataDistribution object with client data assigned
    """
    data_dist = DataDistribution(dataset, num_clients, kwargs.get('seed', 42))
    
    if distribution_type == "iid":
        data_dist.create_iid_distribution()
    
    elif distribution_type == "label_skew":
        alpha = kwargs.get('alpha', 0.5)
        min_samples = kwargs.get('min_samples', 10)
        data_dist.create_label_skew_distribution(alpha, min_samples)
    
    elif distribution_type == "quantity_skew":
        min_ratio = kwargs.get('min_ratio', 0.1)
        max_ratio = kwargs.get('max_ratio', 0.9)
        data_dist.create_quantity_skew_distribution(min_ratio, max_ratio)
    
    elif distribution_type == "feature_skew":
        noise_level = kwargs.get('noise_level', 0.1)
        feature_shift = kwargs.get('feature_shift', 0.2)
        data_dist.create_feature_skew_distribution(noise_level, feature_shift)
    
    elif distribution_type == "temporal_skew":
        time_windows = kwargs.get('time_windows', None)
        data_dist.create_temporal_skew_distribution(time_windows)
    
    elif distribution_type == "pathological":
        shards_per_client = kwargs.get('shards_per_client', 2)
        data_dist.create_pathological_distribution(shards_per_client)
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return data_dist
