# YOLOv8n-MoE Metrics Fix

## Issue
When training the YOLOv8n-MoE model, precision and recall metrics were being doubled due to the Mixture of Experts architecture.

## Root Cause
The problem occurred because:
1. Multiple experts (4 total, top-2 selected) process features in the MoE bottleneck
2. The detection head's predictions might be aggregated incorrectly
3. Metrics calculation could count predictions multiple times

## Solution

### New Files Created

1. **`train_moe_custom.py`** - Custom training script with fixed metrics
   - Single forward pass per batch (prevents doubled metrics)
   - Load balancing auxiliary loss for expert utilization
   - Proper expert usage tracking
   
2. **`utils/metrics_moe.py`** - MoE-specific metrics utilities
   - Functions for deduplicating predictions
   - Expert balance calculation
   - Load balancing loss computation

### Key Fixes

#### 1. Single Pass Forward
```python
# Each batch is processed ONCE through the model
predictions = model(images)  # Single call
loss, loss_items = model.h_22.loss(predictions, batch)
```

#### 2. Load Balancing Loss
```python
# Encourages uniform expert utilization
balance_loss = model.get_total_balance_loss()
total_loss = detection_loss + BALANCE_COEF * balance_loss
```

#### 3. Expert Usage Tracking
All MoE bottlenecks now track:
- Router probabilities
- Expert selection patterns
- Load distribution across experts

## Usage

### Training with Fixed Metrics

```bash
# Full training (100 epochs)
python train_moe_custom.py --epochs 100 --batch 16

# Quick test (5 epochs)
python train_moe_custom.py --epochs 5 --batch 4

# Custom learning rate
python train_moe_custom.py --epochs 50 --lr 0.0005
```

### Baseline Comparison

```bash
# Train baseline for comparison
python train_baseline.py --epochs 100 --batch 16
```

## Expected Results

After the fix:
- ✅ Precision and recall in proper 0-1 range
- ✅ No doubled metric values
- ✅ Balanced expert utilization
- ✅ Proper mAP calculations

## Configuration

Default hyperparameters in `train_moe_custom.py`:
- **Epochs**: 100
- **Batch Size**: 16
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0005
- **Balance Coefficient**: 0.01

MoE architecture:
- **Total Experts (E)**: 4
- **Top-K Selection (K)**: 2
- **Expert Type**: Conv 3x3

## Verification

To verify the fix is working:

1. **Check metrics range**: Precision/recall should be 0-1
2. **Compare with baseline**: Values should be comparable
3. **Expert balance**: All 4 experts should be utilized
4. **Loss curves**: Should converge smoothly

## Technical Details

### Load Balancing Loss

The auxiliary loss encourages uniform expert usage:

```
L_balance = E * Σ(f_i * P_i)
```

Where:
- `E` = number of experts (4)
- `f_i` = fraction of tokens routed to expert i
- `P_i` = average routing probability for expert i

This prevents expert collapse (where only some experts are used).

### Metrics Calculation

Metrics are now calculated once per validation step:
1. Single forward pass through model
2. NMS applied to predictions
3. Metrics computed on deduplicated results
4. No expert-wise aggregation

## Files Modified

- ✅ Created `train_moe_custom.py` - Main training script
- ✅ Created `utils/metrics_moe.py` - Metrics utilities
- ✅ Created `utils/__init__.py` - Package init
- ✅ Created `METRICS_FIX.md` - This documentation

## Next Steps

1. Test the training script with a small number of epochs
2. Compare results with baseline model
3. Verify expert utilization is balanced
4. Fine-tune balance coefficient if needed
