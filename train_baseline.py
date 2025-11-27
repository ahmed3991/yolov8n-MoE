#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8n Baseline Training Script

This script trains a standard YOLOv8n model on the pet dataset for comparison
with the YOLOv8n-MoE architecture. It uses the same dataset, hyperparameters,
and training configuration to ensure fair performance comparison.

Usage:
    # Train with default configuration
    python train_baseline.py
    
    # Train with custom configuration
    python train_baseline.py --config config_baseline.yaml
    
    # Train with command-line overrides
    python train_baseline.py --epochs 50 --batch 8
    
    # Quick test (1 epoch)
    python train_baseline.py --epochs 1 --batch 4
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

def get_device():
    """
    Automatically detect and return the best available device.
    
    Returns:
        tuple: (device_string, device_name) for Ultralytics YOLO
    """
    if torch.cuda.is_available():
        device_str = 0  # Use first CUDA device
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_str = 'mps'
        device_name = "MPS (Apple Silicon)"
    else:
        device_str = 'cpu'
        device_name = "CPU"
    
    print(f"🔧 Detected device: {device_name}")
    return device_str, device_name

def load_config(config_path):
    """
    Load training configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        print(f"⚠️  Configuration file not found: {config_path}")
        print(f"   Using default parameters instead")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded configuration from: {config_path}")
    return config

def check_dataset(data_path):
    """
    Check if dataset exists and is properly configured.
    
    Args:
        data_path (str): Path to dataset YAML file
        
    Returns:
        bool: True if dataset is valid
    """
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset configuration not found at {data_path}")
        print(f"   Please run 'python prepare_dataset.py' first to download and prepare the dataset.")
        return False
    
    # Read dataset config
    with open(data_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Check required fields
    if 'nc' not in dataset_config:
        print(f"❌ Error: Dataset config missing 'nc' (number of classes)")
        return False
    
    if 'names' not in dataset_config:
        print(f"❌ Error: Dataset config missing 'names' (class names)")
        return False
    
    # Get dataset base path
    dataset_base = Path(data_path).parent
    
    # Check if image directories exist
    for split in ['train', 'val']:
        if split in dataset_config:
            img_dir = dataset_base / dataset_config[split]
            if not img_dir.exists():
                print(f"⚠️  Warning: {split} directory not found: {img_dir}")
    
    print(f"✅ Dataset validated: {dataset_config['nc']} classes")
    return True

def train_baseline(config_path=None, **kwargs):
    """
    Train YOLOv8n baseline model.
    
    Args:
        config_path (str, optional): Path to configuration YAML file
        **kwargs: Additional parameters to override config
    """
    print("\n" + "="*60)
    print("  YOLOv8n Baseline Training for MoE Comparison")
    print("="*60 + "\n")
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = {}
    
    # Override with command-line arguments
    config.update({k: v for k, v in kwargs.items() if v is not None})
    
    # Set default values if not in config
    data_path = config.get('data', './pet-dataset-yolo/pet_moe.yaml')
    epochs = config.get('epochs', 100)
    batch = config.get('batch', 16)
    imgsz = config.get('imgsz', 640)
    name = config.get('name', 'yolov8n_baseline_pet')
    project = config.get('project', 'runs/detect')
    
    # Check dataset
    if not check_dataset(data_path):
        sys.exit(1)
    
    # Auto-detect device
    if config.get('device') is None:
        device, device_name = get_device()
    else:
        device = config['device']
        device_name = str(device)
    
    # Display training configuration
    print("\n📋 Training Configuration:")
    print(f"   Model: YOLOv8n (baseline)")
    print(f"   Dataset: {data_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch}")
    print(f"   Image Size: {imgsz}")
    print(f"   Device: {device_name}")
    print(f"   Output: {project}/{name}")
    print()
    
    # Initialize YOLOv8n model with pretrained weights
    print("🚀 Initializing YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Prepare training arguments
    train_args = {
        'data': data_path,
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'name': name,
        'project': project,
        'device': device,
        'verbose': config.get('verbose', True),
        'patience': config.get('patience', 50),
        'save': config.get('save', True),
        'plots': config.get('plots', True),
        'val': config.get('val', True),
    }
    
    # Add optional parameters if present
    optional_params = [
        'lr0', 'lrf', 'momentum', 'weight_decay', 
        'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
        'save_period', 'cache', 'workers',
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 
        'scale', 'shear', 'perspective', 'flipud', 'fliplr',
        'mosaic', 'mixup', 'copy_paste'
    ]
    
    for param in optional_params:
        if param in config:
            train_args[param] = config[param]
    
    # Start training
    print("🏋️  Starting training...\n")
    results = model.train(**train_args)
    
    # Display results
    print("\n" + "="*60)
    print("✅ Baseline training completed!")
    print("="*60)
    print(f"\n📊 Results saved to: {project}/{name}")
    print(f"   - Best weights: {project}/{name}/weights/best.pt")
    print(f"   - Last weights: {project}/{name}/weights/last.pt")
    print(f"   - Training plots: {project}/{name}/*.png")
    print(f"   - Metrics: {project}/{name}/results.csv")
    
    # Validation results
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\n📈 Final Validation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
    
    print("\n💡 Next Steps:")
    print("   1. Compare results with YOLOv8n-MoE training")
    print("   2. Analyze mAP, precision, recall, and loss curves")
    print("   3. Check model size and inference speed")
    print("="*60 + "\n")

def main():
    """Main entry point for baseline training."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8n baseline model for MoE comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python train_baseline.py
  
  # Train with custom config file
  python train_baseline.py --config config_baseline.yaml
  
  # Override specific parameters
  python train_baseline.py --epochs 50 --batch 8
  
  # Quick test with 1 epoch
  python train_baseline.py --epochs 1 --batch 4
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config_baseline.yaml',
        help='Path to training configuration YAML file (default: config_baseline.yaml)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to dataset YAML file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Device to use (e.g., 0, cpu, mps)'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name'
    )
    parser.add_argument(
        '--project',
        type=str,
        help='Project directory'
    )
    
    args = parser.parse_args()
    
    # Prepare kwargs from command-line arguments
    kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'name': args.name,
        'project': args.project,
    }
    
    # Train baseline model
    train_baseline(config_path=args.config, **kwargs)

if __name__ == "__main__":
    main()
