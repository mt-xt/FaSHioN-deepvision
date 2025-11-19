# Fashion-MNIST Image Classification

A Convolutional Neural Network (CNN) implementation for classifying [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset images into 10 clothing categories using TensorFlow/Keras.

## Overview

This project implements a CNN to classify grayscale 28×28 images of clothing items from the Fashion-MNIST dataset. The model achieves ~91% test accuracy through a multi-layer convolutional architecture.

## Dataset

**Fashion-MNIST** contains 70,000 grayscale images across 10 categories:

| Label | Category |
|-------|----------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

**Data Split:**
- Training: 48,000 images
- Validation: 12,000 images
- Test: 10,000 images

## Model Architecture

```
Input (28×28×1 grayscale image)
    ↓
Conv2D (28 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (56 filters, 3×3, ReLU)
    ↓
Flatten
    ↓
Dense (56 units, ReLU)
    ↓
Dense (10 units, Softmax)
```

**Total Trainable Parameters:** 394,530

## Requirements

```bash
pip install numpy matplotlib tensorflow
```

## Usage

Run the training script:

```bash
python fashion_mnist.py
```

When prompted, enter the number of epochs to train (recommended: 5-10).

The script will:
1. Train the model on 48,000 training images
2. Validate on 12,000 validation images
3. Evaluate on 10,000 test images
4. Generate visualizations of training history and misclassifications

## Results

### Performance Metrics

**Final Test Accuracy:** 90.95%

### Training vs Validation Accuracy

![Training History](imgs/training_validation_accuracy_loss.png)

The graph reveals important training dynamics:
- **Training Accuracy:** Steadily increases from ~85% to ~97.5%
- **Validation Accuracy:** Plateaus around 92% after epoch 3

### Misclassification Examples

![Misclassification Examples](imgs/misclassified_examples.png)

Common misclassifications include:
- Sandals → Ankle boots
- Shirts → T-shirts/Coats
- Pullovers → Coats/Shirts

## Analysis & Observations

### Overfitting Detection

The ~5% gap between training (97.5%) and validation accuracy (92%) indicates **overfitting** after epoch 2-3:

- The model memorizes training data rather than learning generalizable patterns
- Validation accuracy plateaus while training accuracy continues climbing
- Additional training beyond epoch 5-6 doesn't improve generalization

### Misclassification Patterns

Analysis of misclassified images reveals the model struggles with:

1. **Similar silhouettes**: Sandals vs. ankle boots, shirts vs. coats
2. **Fine-grained details**: Missing subtle features like sandal straps, coat length
3. **Class boundaries**: Items with overlapping visual characteristics

### Proposed Improvements

**1. Regularization Techniques**
- Add dropout layers (0.3-0.5) after dense layers
- Implement L2 regularization in convolutional layers
- Use early stopping to halt training when validation plateaus

**2. Data Augmentation**
- Random rotations (±15°)
- Random shifts and zooms
- Brightness/contrast adjustments
- This forces the model to learn robust features rather than memorizing specific pixel patterns

**3. Architecture Enhancements**
- Add additional convolutional layers before pooling
- Increase filter depth for richer feature extraction
- Experiment with batch normalization

**4. Training Optimization**
- Reduce learning rate when validation plateaus
- Use learning rate scheduling
- Stop training at epoch 5-6 when validation accuracy peaks

## Experimental Results

Additional architecture tested:
- **2 extra Conv2D layers + 1 MaxPooling layer**
  - Test Accuracy @ 5 epochs: 90.41%
  - Test Accuracy @ 20 epochs: 90.41%
  - *Note: No improvement with extended training, confirming overfitting*

## Project Structure

```
.
├── fashion_mnist.py                    # Main training script
├── imgs/
│   ├── training_validation_accuracy_loss.png
│   └── misclassified_examples.png
└── README.md
```

## Key Takeaways

1. **Validation is crucial**: Monitoring validation accuracy reveals overfitting early
2. **More training ≠ better results**: The model plateaus after epoch 3-4
3. **Architecture matters**: Model capacity (394K parameters) is sufficient, but regularization is needed
4. **Data augmentation helps**: Introducing variation prevents memorization of training-specific patterns

## References

1. [Keras Sequential Model Guide](https://keras.io/guides/sequential_model)
2. [Understanding Train vs Validation Accuracy](https://www.reddit.com/r/learnmachinelearning/comments/fjfr3y/can_someone_explain_train_and_validation_accuracy/)
3. [Improving Image Classification Performance](https://medium.com/accredian/increase-the-performance-of-image-classification-models-b466e1ae3101)
4. [AWS: What is Overfitting?](https://aws.amazon.com/what-is/overfitting/)
5. [Keras Prediction Documentation](https://stackoverflow.com/questions/37891954/keras-how-do-i-predict-after-i-trained-a-model/)
6. [Keras Input Shape Explanation](https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc)

## License

This project uses the Fashion-MNIST dataset, which is available under the MIT License.

---

**Author:** Matthew  
**Course:** ECS 170 - Machine Learning  
**Project:** Fashion-MNIST Classification
