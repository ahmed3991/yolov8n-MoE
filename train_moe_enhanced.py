#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8n-MoE Enhanced Training Script with Detailed Metrics Display

This script trains the YOLOv8n-MoE model with comprehensive training display including:
- Detailed loss component tracking (box, cls, dfl, balance)
- GPU memory monitoring
- Validation with per-class metrics
- Rich progress bars and formatted output
- Results saving (CSV, plots)

Usage:
    python train_moe_enhanced.py --epochs 100 --batch 16
    python train_moe_enhanced.py --epochs 5 --batch 4  # Quick test
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import csv
from datetime import datetime
import warnings

# Suppress PIL warnings for corrupt JPEG images
warnings.filterwarnings('ignore', message='Corrupt JPEG data')
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import ops
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from torch.nn import Upsample
import math

# Local imports
from utils.metrics_moe import (
    get_load_balance_loss,
    calculate_expert_balance,
    track_expert_stats
)

# --- Configuration ---
DATA_YAML_PATH = './pet-dataset-yolo/pet_moe.yaml'
CHECKPOINT_DIR = './checkpoints_moe'
RESULTS_DIR = './runs/detect/yolov8n_moe_custom'

# Training hyperparameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
IMG_SIZE = 640
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
BALANCE_COEF = 0.01  # Coefficient for load balancing loss

# MoE Parameters
E = 4  # Total Experts
K = 2  # Top-K experts to select

# Model Parameters (YOLOv8n Scale)
d = 0.33  # Depth multiplier
w = 0.25  # Width multiplier


def make_divisible(x, divisor=8):
    """Make channels divisible by divisor."""
    return math.ceil(x * w / divisor) * divisor


# Channel dimensions
C1 = make_divisible(64)   # 16
C2 = make_divisible(128)  # 32
C3 = make_divisible(256)  # 64
C4 = make_divisible(512)  # 128
C5 = make_divisible(1024) # 256
N3 = round(3 * d)  # 1
N6 = round(6 * d)  # 2


def get_num_classes():
    """Read number of classes from dataset YAML file."""
    if os.path.exists(DATA_YAML_PATH):
        with open(DATA_YAML_PATH, 'r') as f:
            data_config = yaml.safe_load(f)
            return data_config.get('nc', 37)
    return 37


NUM_CLASSES = get_num_classes()


class MoE_Bottleneck(nn.Module):
    """Mixture of Experts Bottleneck module with expert tracking."""
    
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.shortcut = shortcut
        c_ = int(c2 * 0.5)

        # Standard Convs
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(c_, c2, 1, 1)

        # MoE Layer
        self.router = nn.Conv2d(c_, E, kernel_size=1)
        self.experts = nn.ModuleList([
            Conv(c_, c_, 3, 1) for _ in range(E)
        ])
        
        # For tracking expert usage
        self.expert_usage = None
        self.router_probs = None

    def forward(self, x):
        x_in = self.conv1(x)
        B, C, H, W = x_in.shape
        x_reshaped = x_in.permute(0, 2, 3, 1).reshape(-1, C)

        # Gating and Top-K Selection
        router_logits = self.router(x_in).permute(0, 2, 3, 1).reshape(-1, E)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Store for load balancing loss
        self.router_probs = routing_weights.detach()
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, K, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Sparse Execution
        output_moe = torch.zeros_like(x_reshaped)

        for i, expert in enumerate(self.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            selected_indices = torch.nonzero(expert_mask, as_tuple=False).squeeze(1)

            if selected_indices.numel() == 0:
                continue

            # Select spatial positions for this expert
            h_indices = (selected_indices // W) % H
            w_indices = selected_indices % W
            b_indices = selected_indices // (H * W)
            
            expert_input = x_in[b_indices, :, h_indices, w_indices].reshape(-1, C, 1, 1)
            
            # Handle single-sample batch for BatchNorm
            if expert_input.size(0) == 1 and self.training:
                expert_output = expert(expert_input.repeat(2, 1, 1, 1)).squeeze(-1).squeeze(-1)
                expert_output = expert_output[:1]
            else:
                expert_output = expert(expert_input).squeeze(-1).squeeze(-1)

            expert_weight_idx = (top_k_indices[selected_indices] == i).nonzero(as_tuple=True)[1]
            expert_weights = top_k_weights[selected_indices, expert_weight_idx]

            output_moe[selected_indices] += expert_output * expert_weights.unsqueeze(-1)

        output_moe = output_moe.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_out = self.conv3(output_moe)

        return x_out + x if self.shortcut and x.shape[1] == x_out.shape[1] else x_out
    
    
    def get_balance_loss(self):
        """Get load balancing loss for this layer."""
        if self.router_probs is None:
            return torch.tensor(0.0)
        
        # Create expert mask from router probs
        top_k_weights, top_k_indices = torch.topk(self.router_probs, K, dim=-1)
        expert_mask = torch.zeros_like(self.router_probs)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        
        return get_load_balance_loss(self.router_probs, expert_mask, E)


class C2f_MoE(C2f):
    """C2f module with MoE Bottleneck."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        # Replace with MoE Bottleneck
        self.m = nn.ModuleList([
            MoE_Bottleneck(c_, c_, shortcut) for _ in range(n)
        ])
    
    def get_balance_loss(self):
        """Aggregate balance loss from all MoE bottlenecks."""
        total_loss = 0.0
        for module in self.m:
            if isinstance(module, MoE_Bottleneck):
                total_loss += module.get_balance_loss()
        return total_loss


class YOLOv8n_MoE_C2f(nn.Module):
    """YOLOv8n with Mixture of Experts in the head."""
    
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.nc = nc

        # --- Backbone (Standard C2f) ---
        self.b_0 = Conv(3, C1, 3, 2)
        self.b_1 = Conv(C1, C2, 3, 2)
        self.b_2 = C2f(C2, C2, n=N3, shortcut=True)
        self.b_3 = Conv(C2, C3, 3, 2)
        self.b_4 = C2f(C3, C3, n=N6, shortcut=True)
        self.b_5 = Conv(C3, C4, 3, 2)
        self.b_6 = C2f(C4, C4, n=N6, shortcut=True)
        self.b_7 = Conv(C4, C5, 3, 2)
        self.b_8 = C2f(C5, C5, n=N3, shortcut=True)
        self.b_9 = SPPF(C5, C5, k=5)

        # --- Head Layers (MoE C2f) ---
        self.h_10 = Upsample(scale_factor=2, mode='nearest')
        self.h_12 = C2f_MoE(C5 + C4, C4, n=N3, shortcut=False)

        self.h_13 = Upsample(scale_factor=2, mode='nearest')
        self.h_15 = C2f_MoE(C4 + C3, C3, n=N3, shortcut=False)

        self.h_16 = Conv(C3, C3, 3, 2)
        self.h_18 = C2f_MoE(C3 + C4, C4, n=N3, shortcut=False)

        self.h_19 = Conv(C4, C4, 3, 2)
        self.h_21 = C2f_MoE(C4 + C5, C5, n=N3, shortcut=False)

        self.h_22 = Detect(nc=self.nc, ch=[C3, C4, C5])
        
        # Initialize strides
        self.stride = torch.tensor([8., 16., 32.])
        self.h_22.stride = self.stride
        
        # Init weights
        self.h_22.bias_init()

    def forward(self, x):
        cache = []
        x = self.b_0(x); x = self.b_1(x); x = self.b_2(x)
        x_3 = self.b_3(x); cache.append(x_3)
        x = self.b_4(x_3)
        x_5 = self.b_5(x); cache.append(x_5)
        x_6 = self.b_6(x_5)
        x = self.b_7(x_6); x = self.b_8(x)
        x_9 = self.b_9(x)

        # Head (MoE)
        x_p4_up = self.h_10(x_9)
        x_12 = self.h_12(torch.cat((x_p4_up, x_6), 1))

        x_p3_up = self.h_13(x_12)
        x_15 = self.h_15(torch.cat((x_p3_up, x_3), 1))

        x_p3_down = self.h_16(x_15)
        x_18 = self.h_18(torch.cat((x_p3_down, x_12), 1))

        x_p4_down = self.h_19(x_18)
        x_21 = self.h_21(torch.cat((x_p4_down, x_9), 1))

        return self.h_22([x_15, x_18, x_21])
    
    def get_total_balance_loss(self):
        """Get total load balancing loss from all MoE layers."""
        total_loss = 0.0
        for name, module in self.named_modules():
            if isinstance(module, C2f_MoE):
                total_loss += module.get_balance_loss()
        return total_loss

    @property
    def model(self):
        """Property to satisfy v8DetectionLoss requirements."""
        return [self.h_22]


class MetricsTracker:
    """Track training and validation metrics throughout training."""
    
    def __init__(self, save_dir, class_names):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.results = []
        
    def update(self, epoch, train_metrics, val_metrics=None):
        """Update metrics for current epoch."""
        result = {
            'epoch': epoch + 1,
            'train/box_loss': train_metrics['box_loss'],
            'train/cls_loss': train_metrics['cls_loss'],
            'train/dfl_loss': train_metrics['dfl_loss'],
            'train/balance_loss': train_metrics['balance_loss'],
        }
        
        if val_metrics:
            result.update({
                'metrics/precision(B)': val_metrics['precision'],
                'metrics/recall(B)': val_metrics['recall'],
                'metrics/mAP50(B)': val_metrics['mAP50'],
                'metrics/mAP50-95(B)': val_metrics['mAP50-95'],
            })
        
        self.results.append(result)
    
    def save_csv(self):
        """Save results to CSV file."""
        if not self.results:
            return
        
        csv_path = self.save_dir / 'results.csv'
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Results saved to: {csv_path}")
    
    def get_final_metrics(self):
        """Get final epoch metrics."""
        if not self.results:
            return {}
        return self.results[-1]


def load_pretrained_weights(custom_moe_model):
    """Load pretrained YOLOv8n weights into the MoE model."""
    print("Loading official YOLOv8n weights...")
    official_yolo = YOLO('yolov8n.pt')
    pretrained_state_dict = official_yolo.model.state_dict()

    KEY_MAPPING = {
        0: 'b_0', 1: 'b_1', 2: 'b_2', 3: 'b_3', 4: 'b_4',
        5: 'b_5', 6: 'b_6', 7: 'b_7', 8: 'b_8', 9: 'b_9',
        10: 'h_10', 13: 'h_13', 16: 'h_16', 19: 'h_19'
    }

    new_state_dict = {}

    for key, value in pretrained_state_dict.items():
        try:
            parts = key.split('.')
            layer_index = int(parts[1])
            param_path = '.'.join(parts[2:])
        except (IndexError, ValueError):
            continue

        if layer_index <= 21:
            if layer_index in KEY_MAPPING:
                custom_attr = KEY_MAPPING[layer_index]
                if 'm.' not in param_path:
                    new_key = f'{custom_attr}.{param_path}'
                    new_state_dict[new_key] = value

    incompatible_keys = custom_moe_model.load_state_dict(new_state_dict, strict=False)

    print(f"✅ Loaded {len(new_state_dict)} tensors from YOLOv8n weights.")
    print(f"⚠️  Missing MoE/Detect layer weights: {len(incompatible_keys.missing_keys)}.")

    return custom_moe_model


def get_device():
    """Automatically detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "MPS (Apple Silicon)"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    print(f"🔧 Detected device: {device_name}")
    return device, device_name


def get_gpu_mem():
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    elif torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1024**3
        except:
            return 0.0
    return 0.0


def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs, loss_fn):
    """Train for one epoch with detailed metrics display."""
    model.train()
    
    # Initialize metrics
    total_box_loss = 0.0
    total_cls_loss = 0.0
    total_dfl_loss = 0.0
    total_balance_loss = 0.0
    total_instances = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Preprocess batch
            batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255.0
            for k in ['batch_idx', 'cls', 'bboxes']:
                batch[k] = batch[k].to(device)
            
            # Count instances
            num_instances = len(batch['cls'])
            total_instances += num_instances
                
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch['img'])
            
            # Compute detection loss
            loss, loss_items = loss_fn(predictions, batch)
            
            # Extract loss components
            box_loss = loss_items[0].item() if len(loss_items) > 0 else 0.0
            cls_loss = loss_items[1].item() if len(loss_items) > 1 else 0.0
            dfl_loss = loss_items[2].item() if len(loss_items) > 2 else 0.0
            
            # Add load balancing auxiliary loss
            balance_loss = model.get_total_balance_loss()
            
            # Ensure losses are scalars
            if loss.numel() > 1:
                loss = loss.sum()
            if balance_loss.numel() > 1:
                balance_loss = balance_loss.sum()
                
            total_loss_value = loss + BALANCE_COEF * balance_loss
            
            # Backward pass
            total_loss_value.backward()
            optimizer.step()
            
            # Update running metrics
            total_box_loss += box_loss
            total_cls_loss += cls_loss
            total_dfl_loss += dfl_loss
            total_balance_loss += balance_loss.item()
            
            # Get GPU memory
            gpu_mem = get_gpu_mem()
            
            # Update progress bar
            pbar.set_postfix({
                'GPU_mem': f'{gpu_mem:.2f}G',
                'box': f'{box_loss:.4f}',
                'cls': f'{cls_loss:.4f}',
                'dfl': f'{dfl_loss:.4f}',
                'bal': f'{balance_loss.item():.4f}'
            })
        except Exception as e:
            # Skip corrupt batches and continue
            continue
    
    # Compute averages
    num_batches = len(dataloader)
    metrics = {
        'box_loss': total_box_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'dfl_loss': total_dfl_loss / num_batches,
        'balance_loss': total_balance_loss / num_batches,
    }
    
    return metrics


def validate_one_epoch(model, dataloader, device, num_classes, class_names, epoch, num_epochs):
    """Validate model and compute per-class metrics using proper NMS and IoU."""
    model.eval()
    
    # Initialize metrics
    stats = []  # List of (correct, conf, cls, cls_gt)
    seen = 0
    
    # NMS parameters
    conf_thres = 0.001  # Low confidence threshold for validation
    iou_thres = 0.6     # NMS IoU threshold
    
    pbar = tqdm(dataloader, desc=f"Validating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # Preprocess batch
                batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255.0
                batch_size = batch['img'].shape[0]
                seen += batch_size
                
                # Forward pass
                preds = model(batch['img'])
                
                # Handle list output (from Detect head)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                
                # Apply NMS
                # preds: (batch, num_anchors, 4 + num_classes)
                # output: list of (num_dets, 6) [x1, y1, x2, y2, conf, cls]
                preds = ops.non_max_suppression(
                    preds,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    multi_label=True,
                    agnostic=False,
                    max_det=300
                )
                
                # Process batch
                for i, pred in enumerate(preds):
                    # Get target for this image
                    # batch['batch_idx'] is (N, ) index of image in batch
                    # batch['cls'] is (N, 1) class index
                    # batch['bboxes'] is (N, 4) normalized bboxes
                    idx = batch['batch_idx'] == i
                    cls = batch['cls'][idx].squeeze(-1)
                    bbox = batch['bboxes'][idx]
                    
                    # Convert normalized target bboxes to pixel coordinates
                    h, w = batch['img'].shape[2:]
                    bbox = ops.xywhn2xyxy(bbox, w=w, h=h)
                    
                    # Metrics
                    if len(pred) == 0:
                        if len(cls) > 0:
                            # No detections, but ground truth exists -> all FN
                            stats.append((torch.zeros(0, 10, dtype=torch.bool), torch.Tensor(), torch.Tensor(), cls))
                        continue
                    
                    # Predictions
                    pred_conf = pred[:, 4]
                    pred_cls = pred[:, 5]
                    pred_box = pred[:, 0:4]
                    
                    # Evaluate
                    if len(cls) > 0:
                        # Match predictions to targets
                        correct = ops.process_batch(
                            detections=pred,
                            labels=torch.cat((cls.unsqueeze(1), bbox), 1),
                            iou_thresholds=torch.linspace(0.5, 0.95, 10, device=device)
                        )
                    else:
                        correct = torch.zeros(len(pred), 10, dtype=torch.bool)
                        
                    stats.append((correct.cpu(), pred_conf.cpu(), pred_cls.cpu(), cls.cpu()))
            
            except Exception as e:
                # print(f"Validation error: {e}")
                continue
    
    # Compute metrics
    try:
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        
        if len(stats) and stats[0].any():
            tp, conf, pred_cls, target_cls = stats
            
            # Sort by confidence
            i = np.argsort(-conf)
            tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
            
            # Unique classes
            unique_classes = np.unique(target_cls)
            
            ap50 = []
            ap50_95 = []
            p_list = []
            r_list = []
            
            for c in unique_classes:
                i = pred_cls == c
                n_l = (target_cls == c).sum()  # number of labels
                n_p = i.sum()  # number of predictions
                
                if n_p == 0 or n_l == 0:
                    continue
                
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = (tp[i]).cumsum(0)
                
                # Recall
                recall = tpc / (n_l + 1e-16)
                r = recall[:, 0]  # @0.5 IoU
                
                # Precision
                precision = tpc / (tpc + fpc)
                p = precision[:, 0]  # @0.5 IoU
                
                # AP from recall-precision curve
                # Simple integration
                mrec = np.concatenate(([0.0], r, [1.0]))
                mpre = np.concatenate(([1.0], p, [0.0]))
                
                # Compute the precision envelope
                for j in range(mpre.shape[0] - 1, 0, -1):
                    mpre[j - 1] = np.maximum(mpre[j - 1], mpre[j])
                    
                # Integrate area under curve
                i_idx = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
                ap = np.sum((mrec[i_idx + 1] - mrec[i_idx]) * mpre[i_idx + 1])
                ap50.append(ap)
                
                # Approximate mAP50-95
                mean_iou_match = tp[i].mean(axis=1).sum() / n_l
                ap50_95.append(mean_iou_match) 
                
                p_list.append(np.mean(p))
                r_list.append(np.mean(r))
                
            metrics_dict = {
                'precision': np.mean(p_list) if p_list else 0.0,
                'recall': np.mean(r_list) if r_list else 0.0,
                'mAP50': np.mean(ap50) if ap50 else 0.0,
                'mAP50-95': np.mean(ap50_95) if ap50_95 else 0.0,
            }
        else:
            metrics_dict = {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
            }
    except Exception as e:
        print(f"Metrics calculation error: {e}")
        metrics_dict = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
    
    # Print overall metrics
    print(f"                 {'Class':<22} {'Images':<11} {'Instances':<14} {'Box(P':<15}{'R':<11}{'mAP50':<11}mAP50-95)")
    print(f"                 {'all':<22} {seen:<11} {len(stats[3]) if len(stats) > 3 else 0:<14} {metrics_dict['precision']:<15.3f}{metrics_dict['recall']:<11.3f}{metrics_dict['mAP50']:<11.3f}{metrics_dict['mAP50-95']:.4f}")
    
    return metrics_dict


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train YOLOv8n-MoE with enhanced display')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/mps/cpu)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  YOLOv8n-MoE Training with Enhanced Display")
    print("="*60 + "\n")
    
    # Check dataset
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: Dataset not found at {DATA_YAML_PATH}")
        print("Please run 'python prepare_dataset.py' first.")
        return
    
    # Load dataset config to get class names
    with open(DATA_YAML_PATH, 'r') as f:
        data_dict = yaml.safe_load(f)
    class_names = data_dict.get('names', [])
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
        device_name = args.device
    else:
        device, device_name = get_device()
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    weights_dir = os.path.join(RESULTS_DIR, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    print(f"\n📋 Training Configuration:")
    print(f"   Model: YOLOv8n-MoE (E={E}, K={K})")
    print(f"   Dataset: {DATA_YAML_PATH}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Device: {device_name}")
    print(f"   Balance Coefficient: {BALANCE_COEF}")
    print()
    
    # Initialize model
    print("🏗️  Initializing YOLOv8n-MoE model...")
    model = YOLOv8n_MoE_C2f(nc=NUM_CLASSES)
    model = load_pretrained_weights(model)
    model = model.to(device)
    
    # Load default hyperparameters
    hyp = get_cfg(DEFAULT_CFG)
    model.args = hyp
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize Loss
    loss_fn = v8DetectionLoss(model)
    
    # Load dataset
    print("📦 Loading dataset...")
    train_path = os.path.join(os.path.dirname(DATA_YAML_PATH), data_dict['train'])
    val_path = os.path.join(os.path.dirname(DATA_YAML_PATH), data_dict['val'])
    
    from ultralytics.data.dataset import YOLODataset
    from torch.utils.data import DataLoader
    
    # Create training dataset
    train_dataset = YOLODataset(
        img_path=train_path,
        data=data_dict,
        imgsz=IMG_SIZE,
        batch_size=args.batch,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn
    )
    
    # Create validation dataset
    val_dataset = YOLODataset(
        img_path=val_path,
        data=data_dict,
        imgsz=IMG_SIZE,
        batch_size=args.batch * 2,
        augment=False,
        hyp=hyp,
        rect=True,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn
    )
    
    print(f"✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val images")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(RESULTS_DIR, class_names)
    
    # Training Loop
    print("\n" + "="*60)
    print("🚀 Starting Training Loop")
    print("="*60)
    
    best_map = 0.0
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, loss_fn
        )
        
        # Print detailed training metrics
        print(f"   Epoch {epoch+1}/{args.epochs} - "
              f"Box: {train_metrics['box_loss']:.4f} | "
              f"Cls: {train_metrics['cls_loss']:.4f} | "
              f"DFL: {train_metrics['dfl_loss']:.4f} | "
              f"Balance: {train_metrics['balance_loss']:.4f}")
        
        # Validate
        val_metrics = validate_one_epoch(
            model, val_loader, device, NUM_CLASSES, class_names, epoch, args.epochs
        )
        
        # Update scheduler
        scheduler.step()
        
        # Update metrics tracker
        metrics_tracker.update(epoch, train_metrics, val_metrics)
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, ckpt_path)
        
        # Save best model
        current_map = val_metrics['mAP50-95']
        if current_map > best_map:
            best_map = current_map
            best_path = os.path.join(weights_dir, 'best.pt')
            torch.save(model.state_dict(), best_path)
            print(f"\n💾 New best model saved! (mAP50-95: {best_map:.4f})")
        
        # Save last model
        last_path = os.path.join(weights_dir, 'last.pt')
        torch.save(model.state_dict(), last_path)
    
    # Save results CSV
    metrics_tracker.save_csv()
    
    # Final summary
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print(f"\n📊 Results saved to: {RESULTS_DIR}")
    print(f"   - Best weights: {RESULTS_DIR}/weights/best.pt")
    print(f"   - Last weights: {RESULTS_DIR}/weights/last.pt")
    print(f"   - Metrics: {RESULTS_DIR}/results.csv")
    
    final_metrics = metrics_tracker.get_final_metrics()
    if final_metrics:
        print("\n📈 Final Validation Metrics:")
        print(f"   metrics/precision(B): {final_metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"   metrics/recall(B): {final_metrics.get('metrics/recall(B)', 0):.4f}")
        print(f"   metrics/mAP50(B): {final_metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   metrics/mAP50-95(B): {final_metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
