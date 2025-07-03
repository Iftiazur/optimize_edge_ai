"""
Configuration file for Edge AI Optimization Pipeline
Contains all hyperparameters and experiment configurations
"""

import torch
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# Data configuration
DATASET = "CIFAR10"
BATCH_SIZE = 128
TEST_BATCH_SIZE = 100
DATA_PATH = "./data"

# Model configuration
NUM_CLASSES = 10
INPUT_CHANNELS = 3
INPUT_SIZE = 32

# Training configuration
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Federated Learning configuration
NUM_CLIENTS = 8
FEDERATED_ROUNDS = 50
LOCAL_EPOCHS = 5
CLIENT_FRACTION = 0.5  # Fraction of clients participating per round

# Compression configuration
PRUNING_RATIOS = [0.0, 0.4, 0.5, 0.6]
QUANTIZATION_BITS = [32, 16, 8]
DISTILLATION_TEMPERATURE = 4.0
DISTILLATION_ALPHA = 0.7

# Energy estimation parameters (in mW)
ENERGY_PARAMS = {
    'cpu_active': 1000,
    'cpu_idle': 100,
    'memory_active': 500,
    'memory_idle': 50,
    'ops_per_watt': 1e9  # Operations per watt
}

# Hardware simulation parameters
EDGE_DEVICE_CONFIGS = {
    'raspberry_pi': {
        'memory_limit': 1024,  # MB
        'compute_power': 0.1,  # Relative to baseline
        'energy_multiplier': 0.5
    },
    'jetson_nano': {
        'memory_limit': 2048,  # MB
        'compute_power': 0.3,
        'energy_multiplier': 0.7
    },
    'mobile_cpu': {
        'memory_limit': 4096,  # MB
        'compute_power': 0.2,
        'energy_multiplier': 0.6
    },
    'desktop': {
        'memory_limit': 8192,  # MB
        'compute_power': 1.0,
        'energy_multiplier': 1.0
    }
}

# Experiment configurations
EXPERIMENTS = {
    'baseline': {
        'pruning': 0.0,
        'quantization': 32,
        'distillation': False,
        'description': 'Original model without optimization'
    },
    'pruning_40': {
        'pruning': 0.4,
        'quantization': 32,
        'distillation': False,
        'description': '40% magnitude-based pruning'
    },
    'pruning_50': {
        'pruning': 0.5,
        'quantization': 32,
        'distillation': False,
        'description': '50% magnitude-based pruning'
    },
    'pruning_60': {
        'pruning': 0.6,
        'quantization': 32,
        'distillation': False,
        'description': '60% magnitude-based pruning'
    },
    'quantization_8bit': {
        'pruning': 0.0,
        'quantization': 8,
        'distillation': False,
        'description': '8-bit quantization'
    },
    'quantization_16bit': {
        'pruning': 0.0,
        'quantization': 16,
        'distillation': False,
        'description': '16-bit quantization'
    },
    'distillation': {
        'pruning': 0.0,
        'quantization': 32,
        'distillation': True,
        'description': 'Knowledge distillation'
    },
    'combined_light': {
        'pruning': 0.4,
        'quantization': 16,
        'distillation': True,
        'description': 'Light combined optimization'
    },
    'combined_aggressive': {
        'pruning': 0.6,
        'quantization': 8,
        'distillation': True,
        'description': 'Aggressive combined optimization'
    }
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "edge_ai_pipeline.log"
RESULTS_DIR = "./results"
MODELS_DIR = "./saved_models"
PLOTS_DIR = "./plots"

# Create directories if they don't exist
for directory in [DATA_PATH, RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Performance thresholds for evaluation
PERFORMANCE_THRESHOLDS = {
    'accuracy_drop_max': 5.0,  # Maximum acceptable accuracy drop (%)
    'inference_time_max': 100,  # Maximum inference time (ms)
    'memory_usage_max': 512,   # Maximum memory usage (MB)
    'energy_consumption_max': 1000  # Maximum energy consumption (mW)
}
