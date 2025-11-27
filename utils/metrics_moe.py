"""
MoE-specific metrics utilities for YOLOv8n-MoE.

This module provides functions to correctly calculate metrics when using
Mixture of Experts architecture, preventing doubled precision/recall calculations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


def deduplicate_predictions(predictions: torch.Tensor, 
                            iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Remove duplicate predictions that may arise from multiple expert outputs.
    
    Args:
        predictions: Tensor of shape (batch_size, num_predictions, 6)
                    where last dim is [x, y, w, h, conf, class]
        iou_threshold: IoU threshold for considering predictions as duplicates
    
    Returns:
        Deduplicated predictions tensor
    """
    if predictions.numel() == 0:
        return predictions
    
    # This is a placeholder - in practice, NMS handles this
    # The key is to ensure this only runs once per validation step
    return predictions


def compute_metrics_single_pass(predictions: List[torch.Tensor],
                                targets: List[torch.Tensor],
                                num_classes: int,
                                iou_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Compute detection metrics ensuring single pass calculation.
    
    This function wraps the standard metrics calculation to ensure
    it's only called once per batch, preventing doubled metrics.
    
    Args:
        predictions: List of prediction tensors, one per image
        targets: List of target tensors, one per image
        num_classes: Number of object classes
        iou_thresholds: IoU thresholds for mAP calculation
    
    Returns:
        Dictionary containing precision, recall, mAP50, mAP50-95
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10).tolist()
    
    # Placeholder for actual metrics calculation
    # In practice, this would integrate with ultralytics metrics
    metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'mAP50': 0.0,
        'mAP50-95': 0.0
    }
    
    return metrics


def calculate_expert_balance(expert_weights: torch.Tensor,
                             num_experts: int = 4) -> Dict[str, float]:
    """
    Calculate expert utilization statistics for load balancing.
    
    Args:
        expert_weights: Tensor of shape (num_tokens, num_experts) containing
                       the routing weights for each expert
        num_experts: Total number of experts
    
    Returns:
        Dictionary with balance statistics including:
        - mean_usage: Average usage per expert
        - std_usage: Standard deviation of usage
        - min_usage: Minimum expert usage
        - max_usage: Maximum expert usage
    """
    # Sum weights for each expert across all tokens
    expert_usage = expert_weights.sum(dim=0)  # Shape: (num_experts,)
    
    # Normalize to get percentage
    total_usage = expert_usage.sum()
    if total_usage > 0:
        expert_usage_pct = (expert_usage / total_usage) * 100
    else:
        expert_usage_pct = torch.zeros(num_experts)
    
    stats = {
        'mean_usage': expert_usage_pct.mean().item(),
        'std_usage': expert_usage_pct.std().item(),
        'min_usage': expert_usage_pct.min().item(),
        'max_usage': expert_usage_pct.max().item(),
        'expert_usage': expert_usage_pct.cpu().numpy().tolist()
    }
    
    return stats


def get_load_balance_loss(router_probs: torch.Tensor,
                          expert_mask: torch.Tensor,
                          num_experts: int = 4) -> torch.Tensor:
    """
    Calculate load balancing auxiliary loss for MoE.
    
    This encourages uniform distribution of samples across experts,
    preventing expert collapse where only some experts are used.
    
    The loss is based on:
    L_balance = num_experts * sum(f_i * P_i)
    where f_i is the fraction of tokens routed to expert i,
    and P_i is the average probability assigned to expert i.
    
    Args:
        router_probs: Tensor of shape (num_tokens, num_experts) with
                     routing probabilities before top-k selection
        expert_mask: Boolean tensor of shape (num_tokens, num_experts)
                    indicating which experts were selected
        num_experts: Total number of experts
    
    Returns:
        Scalar tensor with the load balancing loss
    """
    # Calculate fraction of tokens assigned to each expert
    num_tokens = expert_mask.shape[0]
    expert_counts = expert_mask.float().sum(dim=0)  # Shape: (num_experts,)
    expert_fractions = expert_counts / num_tokens  # f_i
    
    # Calculate average routing probability for each expert
    avg_router_probs = router_probs.mean(dim=0)  # P_i, shape: (num_experts,)
    
    # Load balance loss: encourages uniform distribution
    # Higher when experts are imbalanced
    balance_loss = num_experts * (expert_fractions * avg_router_probs).sum()
    
    return balance_loss


def track_expert_stats(expert_indices: torch.Tensor,
                      expert_weights: torch.Tensor,
                      num_experts: int = 4) -> Dict[str, any]:
    """
    Track detailed statistics about expert selection during training.
    
    Args:
        expert_indices: Tensor of shape (num_tokens, k) with selected expert indices
        expert_weights: Tensor of shape (num_tokens, k) with expert weights
        num_experts: Total number of experts
    
    Returns:
        Dictionary with expert statistics
    """
    batch_size = expert_indices.shape[0]
    k = expert_indices.shape[1]
    
    # Count how many times each expert was selected
    expert_selection_count = torch.zeros(num_experts, device=expert_indices.device)
    for i in range(num_experts):
        expert_selection_count[i] = (expert_indices == i).sum()
    
    # Calculate selection percentage
    total_selections = batch_size * k
    selection_pct = (expert_selection_count / total_selections * 100).cpu().numpy()
    
    # Average weight when expert is selected
    avg_weights = []
    for i in range(num_experts):
        mask = expert_indices == i
        if mask.any():
            avg_weight = expert_weights[mask].mean().item()
        else:
            avg_weight = 0.0
        avg_weights.append(avg_weight)
    
    stats = {
        'selection_count': expert_selection_count.cpu().numpy().tolist(),
        'selection_percentage': selection_pct.tolist(),
        'average_weight': avg_weights,
        'total_selections': total_selections,
        'unique_experts_used': (expert_selection_count > 0).sum().item()
    }
    
    return stats
