# YOLOv8n Baseline vs YOLOv8n-MoE Comparison

This directory contains scripts to train and compare the standard YOLOv8n model with the YOLOv8n-MoE (Mixture of Experts) variant.

## Files Overview

### Training Scripts

- **`train_baseline.py`** - Clean baseline YOLOv8n training script for comparison
- **`train.py`** - YOLOv8n-MoE architecture and training script
- **`prepare_dataset.py`** - Dataset preparation and download script

### Configuration Files

- **`config_baseline.yaml`** - Training configuration for baseline YOLOv8n
- **`pet-dataset-yolo/pet_moe.yaml`** - Dataset configuration (37 pet classes)

### Model Weights

- **`yolov8n.pt`** - Pretrained YOLOv8n weights
- **`yolov8n_moe_init.pt`** - Initialized YOLOv8n-MoE weights
- **`best.pt`** - Best trained model checkpoint

## Dataset

The pet dataset contains **37 classes** (12 cat breeds + 25 dog breeds):

**Cats:** Abyssinian, Bengal, Birman, Bombay, British_Shorthair, Egyptian_Mau, Maine_Coon, Persian, Ragdoll, Russian_Blue, Siamese, Sphynx

**Dogs:** american_bulldog, american_pit_bull_terrier, basset_hound, beagle, boxer, chihuahua, english_cocker_spaniel, english_setter, german_shorthaired, great_pyrenees, havanese, japanese_chin, keeshond, leonberger, miniature_pinscher, newfoundland, pomeranian, pug, saint_bernard, samoyed, scottish_terrier, shiba_inu, staffordshire_bull_terrier, wheaten_terrier, yorkshire_terrier

## Quick Start

### 1. Prepare Dataset

```bash
python prepare_dataset.py
```

This will download and prepare the pet dataset in YOLO format.

### 2. Train Baseline YOLOv8n

```bash
# Full training (100 epochs)
python train_baseline.py

# Custom configuration
python train_baseline.py --config config_baseline.yaml

# Quick test (1 epoch)
python train_baseline.py --epochs 1 --batch 4

# Custom parameters
python train_baseline.py --epochs 50 --batch 8 --imgsz 640
```

### 3. Train YOLOv8n-MoE

```bash
# Using the MoE training script
python train.py
```

## Training Configuration

Both models use the same hyperparameters for fair comparison:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 100 | Total training epochs |
| Batch Size | 16 | Batch size (adjust for GPU memory) |
| Image Size | 640 | Input image resolution |
| Learning Rate | 0.01 | Initial learning rate |
| Weight Decay | 0.0005 | L2 regularization |
| Patience | 50 | Early stopping patience |
| Device | Auto | Auto-detect CUDA/MPS/CPU |

## Results Location

After training, results are saved in:

```
runs/detect/
├── yolov8n_baseline_pet/     # Baseline YOLOv8n results
│   ├── weights/
│   │   ├── best.pt           # Best model checkpoint
│   │   └── last.pt           # Last epoch checkpoint
│   ├── results.csv            # Training metrics
│   └── *.png                  # Training plots
│
└── [moe_experiment_name]/     # MoE model results
    └── ...
```

## Comparison Metrics

To compare model performance, analyze:

1. **Detection Metrics:**
   - mAP@0.5 (mean Average Precision at IoU 0.5)
   - mAP@0.5:0.95 (mean Average Precision at IoU 0.5-0.95)
   - Precision
   - Recall

2. **Training Metrics:**
   - Training loss curves
   - Validation loss curves
   - Convergence speed

3. **Model Efficiency:**
   - Model size (file size in MB)
   - Inference speed (ms per image)
   - FLOPs (floating point operations)
   - Parameter count

4. **Resource Usage:**
   - Training time per epoch
   - GPU memory usage
   - Total training time

## Command-Line Options

### train_baseline.py

```
usage: train_baseline.py [--config CONFIG] [--data DATA] [--epochs EPOCHS]
                         [--batch BATCH] [--imgsz IMGSZ] [--device DEVICE]
                         [--name NAME] [--project PROJECT]

Arguments:
  --config    Path to config YAML file (default: config_baseline.yaml)
  --data      Path to dataset YAML file
  --epochs    Number of training epochs
  --batch     Batch size
  --imgsz     Input image size
  --device    Device (0 for GPU, mps, cpu)
  --name      Experiment name
  --project   Project directory
```

## Architecture Comparison

### YOLOv8n (Baseline)
- Standard C2f modules throughout
- Channel multiplier: 0.25
- Depth multiplier: 0.33
- Parameters: ~3.0M

### YOLOv8n-MoE
- Standard C2f in backbone
- **MoE-enhanced C2f in head** (layers 12, 15, 18, 21)
- Mixture of Experts with:
  - E = 4 total experts
  - K = 2 top experts selected
  - Sparse gating mechanism
- Parameters: ~4.5M (estimated)

## Expected Differences

**Baseline YOLOv8n:**
- ✅ Faster training
- ✅ Smaller model size
- ✅ Lower memory usage
- ✅ Proven performance

**YOLOv8n-MoE:**
- ✅ Potentially better accuracy (if MoE is beneficial)
- ✅ More capacity for complex patterns
- ❌ Larger model size
- ❌ Slower training
- ❌ Higher memory usage

## Troubleshooting

### Dataset Not Found
```bash
# Run dataset preparation first
python prepare_dataset.py
```

### CUDA Out of Memory
```bash
# Reduce batch size
python train_baseline.py --batch 8
# or
python train_baseline.py --batch 4
```

### MPS (Apple Silicon) Issues
```bash
# Force CPU if MPS has issues
python train_baseline.py --device cpu
```

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Mixture of Experts Papers](https://arxiv.org/abs/1701.06538)
- [Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
