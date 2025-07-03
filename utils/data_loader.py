"""
Data loading utilities for CIFAR-10 dataset
Handles data loading, preprocessing, and non-IID distribution for federated learning
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from typing import List, Tuple, Dict
import random

from .config import *


class DataLoader:
    """Handles CIFAR-10 data loading and preprocessing"""
    
    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def load_centralized_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-10 data for centralized training"""
        trainset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=True, transform=self.transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        
        testset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, download=True, transform=self.transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        
        return trainloader, testloader
    
    def create_non_iid_split(self, num_clients: int, alpha: float = 0.5) -> List[Subset]:
        """
        Create non-IID data split for federated learning using Dirichlet distribution
        
        Args:
            num_clients: Number of clients
            alpha: Concentration parameter (lower = more non-IID)
        
        Returns:
            List of data subsets for each client
        """
        trainset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=True, transform=self.transform_train
        )
        
        # Get labels
        targets = np.array(trainset.targets)
        num_classes = len(np.unique(targets))
        
        # Create Dirichlet distribution for each client
        client_data_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices)).astype(int)
            
            # Adjust for rounding errors
            proportions[-1] = len(class_indices) - sum(proportions[:-1])
            
            # Distribute indices to clients
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = start_idx + proportions[client_id]
                client_data_indices[client_id].extend(
                    class_indices[start_idx:end_idx].tolist()
                )
                start_idx = end_idx
        
        # Shuffle indices for each client
        for client_id in range(num_clients):
            random.shuffle(client_data_indices[client_id])
        
        # Create subsets
        client_datasets = []
        for client_id in range(num_clients):
            subset = Subset(trainset, client_data_indices[client_id])
            client_datasets.append(subset)
        
        return client_datasets
    
    def get_federated_dataloaders(self, num_clients: int, alpha: float = 0.5) -> Tuple[List[DataLoader], DataLoader]:
        """
        Get federated data loaders for clients and test set
        
        Args:
            num_clients: Number of clients
            alpha: Non-IID concentration parameter
        
        Returns:
            Tuple of (client_dataloaders, test_dataloader)
        """
        client_datasets = self.create_non_iid_split(num_clients, alpha)
        
        client_dataloaders = []
        for dataset in client_datasets:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
            )
            client_dataloaders.append(dataloader)
        
        # Test set remains the same
        testset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, download=True, transform=self.transform_test
        )
        test_dataloader = torch.utils.data.DataLoader(
            testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        
        return client_dataloaders, test_dataloader
    
    def analyze_data_distribution(self, client_datasets: List[Subset]) -> Dict:
        """Analyze the class distribution across clients"""
        distribution_info = {}
        
        for client_id, dataset in enumerate(client_datasets):
            labels = [dataset.dataset.targets[idx] for idx in dataset.indices]
            class_counts = np.bincount(labels, minlength=NUM_CLASSES)
            distribution_info[f'client_{client_id}'] = {
                'total_samples': len(labels),
                'class_distribution': class_counts.tolist(),
                'dominant_classes': np.argsort(class_counts)[-3:].tolist()
            }
        
        return distribution_info


class DistillationDataLoader:
    """Specialized data loader for knowledge distillation"""
    
    def __init__(self, temperature: float = 4.0):
        self.temperature = temperature
        self.data_loader = DataLoader()
    
    def get_distillation_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for teacher-student training"""
        return self.data_loader.load_centralized_data()
    
    def get_unlabeled_data(self, ratio: float = 0.3) -> DataLoader:
        """
        Create unlabeled dataset for distillation
        
        Args:
            ratio: Fraction of training data to treat as unlabeled
        
        Returns:
            DataLoader with unlabeled data
        """
        trainset = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=True, 
            transform=self.data_loader.transform_train
        )
        
        # Split dataset
        total_size = len(trainset)
        unlabeled_size = int(total_size * ratio)
        labeled_size = total_size - unlabeled_size
        
        _, unlabeled_set = random_split(trainset, [labeled_size, unlabeled_size])
        
        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        
        return unlabeled_loader


# CIFAR-10 class names for visualization
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
