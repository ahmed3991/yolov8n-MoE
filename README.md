# YOLOv8n-MoE: YOLOv8 Nano with Mixture of Experts

This project implements a custom YOLOv8n (nano) architecture enhanced with **Mixture of Experts (MoE)** layers in the detection head for improved object detection performance on the Oxford-IIIT Pet Dataset.

## 🎯 Overview

- **Architecture**: YOLOv8n with MoE-enhanced head
- **Dataset**: Oxford-IIIT Pet Dataset (37 pet breeds)
- **Key Features**:
  - Sparse MoE layers with 4 experts, top-2 selection
  - Transfer learning from pretrained YOLOv8n
  - Automatic device detection (CUDA/MPS/CPU)
  - Optimized for Apple Silicon (M1/M2) with MPS support

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download and prepare the Oxford-IIIT Pet Dataset:

```bash
python prepare_dataset.py
```

This will:
- Download images and annotations from Oxford VGG
- Convert Pascal VOC annotations to YOLO format
- Split data into train (80%), validation (10%), test (10%)
- Create `pet_moe.yaml` configuration file

### 3. Train the Model

```bash
python train.py
```

Training configuration:
- **Epochs**: 100
- **Batch size**: 32
- **Image size**: 640×640
- **Device**: Auto-detected (CUDA/MPS/CPU)

## 📁 Project Structure

```
yolov8n-MoE/
├── prepare_dataset.py      # Dataset download and preparation
├── train.py                 # Model training script
├── yolon8_moe.py           # Original combined script (legacy)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── pet-dataset-yolo/       # Dataset directory (created after preparation)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── pet_moe.yaml           # Dataset configuration (created after preparation)
```

## 🏗️ Architecture Details

### MoE Integration

The Mixture of Experts (MoE) layers are integrated into the YOLOv8 head at layers 12, 15, 18, and 21:

- **4 Expert Networks**: Each expert is a 3×3 convolutional layer
- **Top-2 Selection**: Dynamic routing selects the 2 most relevant experts per input
- **Sparse Execution**: Only activated experts compute, reducing overhead

### Model Parameters

- **Width multiplier**: 0.25 (YOLOv8n scale)
- **Depth multiplier**: 0.33
- **Number of classes**: 37 (pet breeds)
- **Total experts**: 4
- **Top-K**: 2

## 💻 Hardware Support

The code automatically detects and uses the best available device:

- **NVIDIA GPUs**: CUDA acceleration
- **Apple Silicon (M1/M2/M3)**: MPS (Metal Performance Shaders) acceleration
- **CPU**: Fallback for systems without GPU

## 📊 Results

After training, results are saved to:
```
runs/detect/yolov8n_moe_pet_run/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
└── confusion_matrix.png
```

## 🔧 Customization

### Modify Training Parameters

Edit `train.py` and adjust:

```python
results = yolo_trainer.train(
    data=DATA_YAML_PATH,
    epochs=100,        # Change number of epochs
    batch=32,          # Change batch size
    imgsz=640,         # Change image size
    name='yolov8n_moe_pet_run',
    device=selected_device,
)
```

### Adjust MoE Configuration

In `train.py`, modify:

```python
E = 4     # Total number of experts
K = 2     # Top-K experts to select
```

## 📝 Requirements

- Python 3.8+
- PyTorch 2.1.0+ (with MPS support for Apple Silicon)
- Ultralytics 8.1.0+
- See `requirements.txt` for full list

## 🎓 Citation

This project uses:
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## 📄 License

This project is for educational and research purposes.
