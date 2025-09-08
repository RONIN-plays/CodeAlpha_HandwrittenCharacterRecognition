Report – Handwritten Digit Recognition (MNIST)

 1. Objective
Build and train a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset.

 2. Dataset
- **Source:** `tensorflow.keras.datasets` (MNIST)
- **Images:** 60,000 training + 10,000 test
- **Resolution:** 28 × 28 pixels, grayscale
- **Preprocessing:**
  - Normalize pixel values to [0, 1]
  - Reshape to (samples, 28, 28, 1)
  - Labels one-hot encoded to 10 classes

## 3. Methodology
### Model Architecture
| Layer | Details |
|-------|---------|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.3 |
| Dense | 10 units, Softmax |

### Training Configuration
- Optimizer: **Adam**
- Loss: **Categorical Cross-Entropy**
- Epochs: **5**
- Batch size: **128**
- Validation split: **10%**

 4. Results
 Training Progress
| Epoch | Accuracy | Loss  | Val Accuracy | Val Loss |
|-------|----------|-------|--------------|----------|
| 1     | 82.16%   | 0.5829| 98.10%       | 0.0608   |
| 2     | 97.62%   | 0.0796| 98.58%       | 0.0457   |
| 3     | 98.16%   | 0.0591| 98.90%       | 0.0369   |
| 4     | 98.52%   | 0.0462| 98.83%       | 0.0372   |
| 5     | **98.89%** | 0.0349| **99.07%**   | 0.0331   |

- Final Validation Accuracy: ~99.1%
- Final Validation Loss: ~0.033

Confusion Matrix
A confusion matrix heatmap (see `confusion_matrix.png`) confirms very low misclassification rates across all digits.

 5. Discussion
- Accuracy reaches >98% after just 2 epochs, stabilizing at ~99% by epoch 5.
- Very low loss indicates strong generalization.
- Few confusions occur between visually similar digits (e.g., 4 vs 9).
- Further improvements (data augmentation, more epochs) could push accuracy slightly higher.

 6. Conclusion
The CNN successfully learns discriminative features for handwritten digits, achieving ~99% accuracy on validation. This matches common benchmarks for MNIST and confirms CNNs are highly effective for small-scale image classification.

7. References
- LeCun, Y. et al., “Gradient-based Learning Applied to Document Recognition,” Proc. IEEE, 1998.
- TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs
