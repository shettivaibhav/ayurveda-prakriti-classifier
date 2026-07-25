# Wrist Image Dosha Classification - Project Architecture

## 📁 Project Structure

```
mini/
├── training_pipeline.ipynb    # Main training notebook
├── inference.py               # Standalone inference script
├── app.py                     # Streamlit web application
├── model.pth                  # Trained model checkpoint
├── requirements.txt           # Python dependencies
├── model_decisions.md         # Design decisions documentation
├── training_pipeline.html     # Exported training results
│
├── Vata/                      # Training data
│   ├── 1005_female/
│   │   ├── palm_open_back.jpg
│   │   ├── palm_open_front.jpg
│   │   ├── palm_close_back.jpg
│   │   ├── palm_close_front.jpg
│   │   └── metadata.json
│   ├── 1008_female/
│   └── ...
│
├── Pitta/                     # Training data
│   ├── 1002_female/
│   └── ...
│
├── Kapha/                     # Training data
│   ├── 1001_male/
│   └── ...
│
└── wrist_classification_env/  # Virtual environment
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
├─────────────────────────────────────────────────────────────┤
│  Dataset: ~80 subjects × 3 classes (Vata/Pitta/Kapha)      │
│  • Each subject: 4 wrist images + metadata                  │
│  • Selected view: palm_open_back.jpg                        │
│  • Split: 80% train / 20% validation (stratified)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING & AUGMENTATION                   │
├─────────────────────────────────────────────────────────────┤
│  Training:                                                   │
│  • Resize to 224×224                                        │
│  • Random rotation (±15°)                                   │
│  • Color jitter (brightness, contrast, saturation, hue)     │
│  • Random horizontal flip                                   │
│  • Normalize (ImageNet mean/std)                            │
│                                                              │
│  Validation:                                                 │
│  • Resize to 224×224                                        │
│  • Normalize (ImageNet mean/std)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODEL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────┤
│  MobileNetV2 (Pre-trained on ImageNet)                      │
│  │                                                           │
│  ├─► Backbone: Feature Extraction                           │
│  │    • Depthwise separable convolutions                    │
│  │    • Lightweight: ~3.5M parameters                       │
│  │                                                           │
│  └─► Custom Classifier Head:                                │
│       • Dropout(0.2)                                         │
│       • Linear(1280 → 256)                                  │
│       • ReLU                                                 │
│       • Dropout(0.3)                                         │
│       • Linear(256 → 3)  [Vata/Pitta/Kapha]                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────┤
│  Loss Function: CrossEntropyLoss                            │
│  Optimizer: Adam (lr=1e-4, weight_decay=1e-5)               │
│  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)      │
│                                                              │
│  Early Stopping:                                             │
│  • Monitor: Validation Accuracy                             │
│  • Patience: 10 epochs                                      │
│  • Max Epochs: 50                                           │
│                                                              │
│  Metrics:                                                    │
│  • Accuracy (overall & per-class)                           │
│  • F1 Score (weighted)                                      │
│  • Confusion Matrix                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODEL CHECKPOINT                          │
├─────────────────────────────────────────────────────────────┤
│  model.pth contains:                                         │
│  • model_state_dict (trained weights)                       │
│  • class_to_idx mapping                                     │
│  • idx_to_class mapping                                     │
│  • model_architecture info                                  │
│  • input_size (224×224)                                     │
│  • normalization parameters                                 │
│  • best_val_acc, best_val_f1                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Option 1: Command Line (inference.py)                      │
│  ┌──────────────────────────────────────┐                  │
│  │ $ python inference.py image.jpg      │                  │
│  │                                       │                  │
│  │ Output:                               │                  │
│  │ • Predicted Class                     │                  │
│  │ • Confidence Score                    │                  │
│  │ • All Class Probabilities             │                  │
│  └──────────────────────────────────────┘                  │
│                                                              │
│  Option 2: Web App (app.py - Streamlit)                    │
│  ┌──────────────────────────────────────┐                  │
│  │  Upload Image → Classify → Results   │                  │
│  │                                       │                  │
│  │  Smart Classification Logic:          │                  │
│  │  • Single-Dosha (dominant)            │                  │
│  │  • Dual-Dosha (2 similar)             │                  │
│  │  • Tri-Dosha (all balanced)           │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Training Pipeline
```
Raw Images → WristDataset → DataLoader → Model → Loss → 
Backprop → Optimizer → Update Weights → Validation → 
Early Stopping Check → Save Best Model
```

### Inference Pipeline
```
Input Image → Resize & Normalize → Model Forward Pass → 
Softmax → Probabilities → Dosha Classification → 
Display Results
```

---

## 🧩 Key Components

### 1. **WristDataset (Custom PyTorch Dataset)**
```python
Input:  DataFrame with image paths & labels
Output: Preprocessed image tensors + labels
Role:   Bridge between raw data and model
```

### 2. **WristClassifier (Neural Network)**
```python
Input:  224×224×3 RGB image tensor
Output: 3-class logits [Vata, Pitta, Kapha]
Role:   Feature extraction + classification
```

### 3. **Training Functions**
```python
train_epoch():    Train for one epoch, return metrics
validate_epoch(): Validate without gradients, return predictions
Role:             Orchestrate training process
```

### 4. **Dosha Classifier Logic (Streamlit)**
```python
classify_dosha(): 
  - Analyzes probability distribution
  - Threshold: 0.15 (15% difference)
  - Returns: Single/Dual/Tri-Dosha type
```

---

## 📊 Model Performance Tracking

```
┌─────────────────────────────────────────┐
│   METRICS DASHBOARD (per epoch)         │
├─────────────────────────────────────────┤
│  Training:                               │
│  • Loss                                  │
│  • Accuracy                              │
│  • F1 Score                              │
│                                          │
│  Validation:                             │
│  • Loss                                  │
│  • Accuracy ← (Early Stopping Monitor)   │
│  • F1 Score                              │
│  • Per-Class Accuracy                    │
│  • Confusion Matrix                      │
└─────────────────────────────────────────┘
```

---

## 🚀 Deployment Options

### Development
```bash
# Terminal inference
python inference.py <image_path>

# Web interface
streamlit run app.py
```

### Production (Future)
```
Model Optimization:
├─ ONNX Export
├─ TensorRT Conversion
└─ INT8 Quantization

Deployment:
├─ Docker Container
├─ Cloud API (FastAPI/Flask)
└─ Mobile (TensorFlow Lite)
```

---

## 🔧 Technology Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | PyTorch 2.9 |
| **Model** | MobileNetV2 (torchvision) |
| **Data Processing** | NumPy, Pandas, PIL |
| **Visualization** | Matplotlib, Seaborn |
| **Metrics** | scikit-learn |
| **Web App** | Streamlit |
| **Environment** | Python 3.13 + venv |

---

## 📈 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 16 | Small dataset, stable gradients |
| **Learning Rate** | 1e-4 | Conservative for fine-tuning |
| **Weight Decay** | 1e-5 | Light L2 regularization |
| **Max Epochs** | 50 | Sufficient for convergence |
| **Patience** | 10 | Prevent premature stopping |
| **Image Size** | 224×224 | MobileNetV2 standard input |

---

## 🎯 Model Design Philosophy

1. **Lightweight**: MobileNetV2 chosen for efficiency (~3.5M params)
2. **Transfer Learning**: Pre-trained on ImageNet for feature extraction
3. **Regularization**: Dropout (0.2, 0.3) to prevent overfitting
4. **Data Augmentation**: Realistic transforms (rotation, color jitter, flip)
5. **Early Stopping**: Automatic training termination based on validation
6. **Comprehensive Checkpointing**: Save all metadata for reproducibility

---

## 🔍 Classification Logic

### Single-Dosha
```
If max_prob - min_prob > threshold:
  → Return dominant dosha (Vata/Pitta/Kapha)
```

### Dual-Dosha
```
If top1_prob - top2_prob < threshold:
  → Return combination (VP/PK/VK)
```

### Tri-Dosha
```
If top1_prob - top3_prob < threshold:
  → Return VPK (Sama Prakriti)
```

Threshold: **15%** (0.15 probability difference)

---

## 📦 Deliverables

| File | Purpose |
|------|---------|
| `training_pipeline.ipynb` | Complete training workflow |
| `inference.py` | CLI prediction tool |
| `app.py` | Web-based classifier |
| `model.pth` | Trained model weights |
| `model_decisions.md` | Architecture rationale |
| `requirements.txt` | Python dependencies |

---

**Built with PyTorch • Transfer Learning • MobileNetV2**
