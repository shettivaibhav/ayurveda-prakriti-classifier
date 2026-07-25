# Model Architecture and Design Decisions

## Project Overview
This document explains the key decisions made in building a 3-class image classification system for wrist images (Vata, Pitta, Kapha) using PyTorch and transfer learning.

---

## 1. Architecture Choice: MobileNetV2

### Why MobileNetV2 over ResNet18?

**MobileNetV2 was chosen as the primary architecture for several reasons:**

#### ✅ Advantages of MobileNetV2:
- **Lightweight and Efficient**: Only ~3.5M parameters vs ResNet18's ~11M parameters
- **Faster Inference**: Optimized for real-time applications and edge devices
- **Lower Memory Footprint**: Critical for deployment on resource-constrained environments
- **Comparable Accuracy**: Achieves similar performance to ResNet18 on many vision tasks
- **Mobile-First Design**: Uses depthwise separable convolutions for efficiency
- **Good Transfer Learning Base**: Pre-trained on ImageNet, generalizes well to medical/anatomical images

#### ⚖️ Trade-offs:
- **ResNet18** has stronger performance on very complex datasets but is heavier
- For wrist image classification with ~80 samples, the lighter MobileNetV2 provides:
  - Less risk of overfitting due to fewer parameters
  - Faster training and iteration cycles
  - Better deployment characteristics

### Model Modifications:
- **Replaced final classifier**: Added custom head with dropout for regularization
- **Custom head architecture**:
  ```
  Dropout(0.2) → Linear(1280 → 256) → ReLU → Dropout(0.3) → Linear(256 → 3)
  ```
- **Dropout rates**: 0.2 and 0.3 to prevent overfitting on small dataset
- **Intermediate layer**: 256 units to learn task-specific features before final classification

---

## 2. Image Selection: palm_open_back.jpg

### Why Only One View?

**Using a single view simplifies the problem while maintaining effectiveness:**

#### ✅ Reasons for palm_open_back.jpg:
- **Anatomical Consistency**: Open palm provides standardized hand positioning
- **Back View Advantages**: 
  - Less variation from palm lines (unique to each individual)
  - More consistent skin tone and texture visibility
  - Better vein/pattern visualization
- **Reduced Complexity**: One view = cleaner dataset, easier debugging
- **Baseline Establishment**: Start simple, expand later if needed
- **Data Efficiency**: 80 samples across 3 classes is manageable for single-view learning

#### 🔄 Future Extensions:
- Multi-view fusion could be explored if performance plateaus
- Ensemble of different views could improve robustness
- Currently not needed given the small dataset size

---

## 3. Image Preprocessing and Augmentation

### Input Size: 224×224

**Standard size for ImageNet pre-trained models:**
- Matches pre-training resolution of MobileNetV2
- Balances computational cost with feature preservation
- Industry standard for transfer learning

### Normalization: ImageNet Statistics

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

**Why ImageNet normalization?**
- Required for transfer learning from pre-trained weights
- Model expects inputs in this distribution
- Ensures feature extractors work optimally

### Augmentation Strategy

#### Training Augmentations:
1. **Random Rotation (±15°)**
   - Accounts for natural hand positioning variations
   - Small angle to preserve anatomical validity
   
2. **ColorJitter**
   - Brightness: ±20%
   - Contrast: ±20%
   - Saturation: ±10%
   - Hue: ±5%
   - Simulates different lighting conditions and camera settings
   
3. **Random Horizontal Flip (p=0.5)**
   - **Anatomically valid** for wrists (bilateral symmetry)
   - Left/right wrist are mirror images
   - Doubles effective dataset size

#### ❌ Avoided Augmentations:
- **Vertical Flip**: Not anatomically valid
- **Heavy Rotation**: Would distort natural hand position
- **Random Cropping**: Could lose important wrist features
- **Cutout/Mixup**: Too aggressive for small dataset

#### Validation/Test:
- **No augmentation**: Only resize and normalize
- Ensures fair evaluation on clean data

---

## 4. Dataset Split: 80/20 Stratified

### Train/Validation Split:

```python
train_test_split(test_size=0.2, stratify=df['class'])
```

**Why stratified 80/20?**

#### ✅ Stratified Split:
- **Maintains class distribution** in both train and validation sets
- Critical with small dataset (~80 samples)
- Prevents class imbalance in validation
- Each class gets proportional representation

#### ✅ 80/20 Ratio:
- Standard split for small datasets
- ~64 training samples + ~16 validation samples
- Validation set large enough for reliable metrics
- Training set large enough to learn patterns

#### No Separate Test Set:
- With only 80 samples, holding out a test set would leave too little for training
- Validation set serves as test set for final evaluation
- Early stopping on validation prevents overfitting

---

## 5. Training Configuration

### Loss Function: CrossEntropyLoss

**Why CrossEntropyLoss?**
- Standard for multi-class classification
- Combines LogSoftmax + NLLLoss
- Handles class probabilities naturally
- Well-suited for 3-class problem

### Optimizer: Adam

**Why Adam over SGD?**

#### ✅ Adam Advantages:
- **Adaptive learning rates**: Different rates for different parameters
- **Faster convergence**: Especially important with small datasets
- **Less hyperparameter tuning**: Works well with default settings
- **Handles sparse gradients**: Better for transfer learning scenarios

#### Configuration:
```python
Adam(lr=1e-4, weight_decay=1e-5)
```

- **Learning rate 1e-4**: Conservative for fine-tuning pre-trained model
- **Weight decay 1e-5**: Light L2 regularization to prevent overfitting

### Learning Rate Scheduler: ReduceLROnPlateau

```python
ReduceLROnPlateau(mode='max', factor=0.5, patience=5)
```

**Why this scheduler?**
- **Monitors validation accuracy**: Reduces LR when accuracy plateaus
- **Factor 0.5**: Halves learning rate when triggered
- **Patience 5**: Waits 5 epochs before reducing (allows for natural fluctuations)
- **Adaptive**: Responds to training dynamics rather than fixed schedule

---

## 6. Training Strategy

### Epochs and Early Stopping

```python
NUM_EPOCHS = 50
PATIENCE = 10
```

**Why these values?**

#### ✅ Max 50 Epochs:
- Sufficient for convergence on small dataset
- Pre-trained model converges faster than training from scratch
- Early stopping usually triggers before reaching 50

#### ✅ Patience 10:
- Prevents premature stopping from random fluctuations
- Allows model to explore different learning rate regimes
- Balances training time vs overfitting risk

### Batch Size: 16

**Why batch size 16?**
- **Small dataset consideration**: ~64 training samples / 16 = 4 batches per epoch
- **Gradient stability**: Larger than minimum (4-8) for stable gradients
- **Memory efficiency**: Fits easily in GPU memory
- **Not too large**: Preserves some stochasticity for regularization

### Fine-Tuning Strategy: Full Model Training

**All layers are trainable from the start:**
- Pre-trained backbone provides good initialization
- Small dataset benefits from adapting all features
- Alternative (freezing backbone) tested but performed worse
- Dropout and early stopping prevent overfitting

---

## 7. Evaluation Metrics

### Primary Metrics:

1. **Accuracy**
   - Overall classification correctness
   - Easy to interpret
   - Suitable for balanced classes

2. **F1 Score (Weighted)**
   - Balances precision and recall
   - Handles class imbalance if it exists
   - More robust than accuracy alone

3. **Confusion Matrix**
   - Shows per-class performance
   - Identifies systematic misclassifications
   - Visual representation of errors

4. **Per-Class Accuracy**
   - Ensures no single class is failing
   - Important for 3-class problem
   - Detects class-specific issues

### Why These Metrics?

- **Multi-class nature**: Single metric insufficient
- **Balanced evaluation**: Combination of micro and macro perspectives
- **Clinical relevance**: Understanding where model fails matters
- **Development guidance**: Metrics guide architecture decisions

---

## 8. Model Saving and Deployment

### Checkpoint Contents:

```python
{
    'model_state_dict': model.state_dict(),
    'class_to_idx': {'Vata': 0, 'Pitta': 1, 'Kapha': 2},
    'idx_to_class': {0: 'Vata', 1: 'Pitta', 2: 'Kapha'},
    'model_architecture': 'MobileNetV2',
    'num_classes': 3,
    'input_size': (224, 224),
    'best_val_acc': <float>,
    'best_val_f1': <float>,
    'normalization': {'mean': [...], 'std': [...]}
}
```

**Why save all this information?**

✅ **Complete reproducibility**: All preprocessing details included
✅ **Self-contained**: No need for external configuration files
✅ **Version tracking**: Model architecture and performance recorded
✅ **Easy deployment**: Inference script loads everything automatically

---

## 9. Inference Design

### Standalone Script (inference.py)

**Design principles:**

1. **No external dependencies on training code**: Fully self-contained
2. **Command-line interface**: Easy integration with other systems
3. **Flexible device selection**: Auto-detects CUDA, allows manual override
4. **Verbose/quiet modes**: Suitable for both interactive use and scripting
5. **Error handling**: Graceful failures with informative messages
6. **Probability outputs**: Full softmax probabilities, not just top class

### Usage Examples:

```bash
# Basic usage
python inference.py image.jpg

# Custom model path
python inference.py image.jpg --model custom_model.pth

# Force CPU
python inference.py image.jpg --device cpu

# Quiet mode (only output class name)
python inference.py image.jpg --quiet
```

---

## 10. Alternative Approaches Considered

### What Was NOT Chosen and Why:

#### ❌ ResNet18
- **Reason**: More parameters, higher risk of overfitting on small dataset
- **When to use**: If dataset grows to 500+ samples

#### ❌ EfficientNet
- **Reason**: More complex, requires careful hyperparameter tuning
- **When to use**: Production system with larger dataset

#### ❌ Custom CNN from Scratch
- **Reason**: Requires much more data, reinvents the wheel
- **When to use**: Never for this task (transfer learning superior)

#### ❌ Multi-View Fusion
- **Reason**: Added complexity without clear benefit for small dataset
- **When to use**: If single-view performance plateaus

#### ❌ K-Fold Cross-Validation
- **Reason**: Time-consuming, simple hold-out sufficient for initial model
- **When to use**: Final performance estimation before deployment

#### ❌ Class Weighting
- **Reason**: Classes are relatively balanced (~25-30 samples each)
- **When to use**: If significant class imbalance emerges

---

## 11. Potential Improvements

### If Dataset Grows (200+ samples):

1. **More aggressive augmentation**: Random erasing, mixup, cutmix
2. **Separate test set**: 70/15/15 split for train/val/test
3. **Heavier model**: ResNet34 or EfficientNet-B0
4. **Learning rate warmup**: Gradual LR increase in first epochs
5. **Multi-view ensemble**: Combine predictions from all 4 views

### For Production Deployment:

1. **Model quantization**: INT8 for faster inference
2. **ONNX export**: Framework-agnostic deployment
3. **TensorRT optimization**: GPU acceleration
4. **Uncertainty estimation**: Monte Carlo dropout for confidence
5. **Explainability**: Grad-CAM to visualize important regions

### For Better Performance:

1. **External data**: Pre-train on larger hand/wrist dataset
2. **Semi-supervised learning**: Leverage unlabeled wrist images
3. **Architecture search**: AutoML to find optimal architecture
4. **Test-time augmentation**: Average predictions over multiple augmentations

---

## 12. Summary

This pipeline was designed with **pragmatism and efficiency** in mind:

✅ **Right-sized for the data**: ~80 samples demands lightweight, regularized approach  
✅ **Production-ready**: Clean code, modular design, easy deployment  
✅ **Interpretable**: Clear metrics, visualizations, and decision transparency  
✅ **Extensible**: Easy to add more views, classes, or augmentations  
✅ **Reproducible**: Fixed seeds, saved configurations, documented choices  

The choices prioritize:
1. **Avoiding overfitting** (small dataset)
2. **Fast iteration** (lightweight model)
3. **Deployment efficiency** (MobileNetV2)
4. **Code quality** (clean, documented, modular)

**This is a baseline model** - a starting point for iteration, not a final product. Performance should be monitored on real-world data, and the architecture evolved based on actual deployment requirements.
