 Handwritten Digit Recognition (MNIST)

A Convolutional Neural Network (CNN) for classifying handwritten digits (0–9) using the MNIST dataset.

 Objective
Predict the correct digit from grayscale images (28×28 pixels).

 Dataset
MNIST — 60,000 training, 10,000 test samples.  
Loaded directly via `tensorflow.keras.datasets.mnist`.

 Approach
- Normalize image data to [0,1]
- CNN: Conv2D → MaxPooling → Flatten → Dense layers
- Train with Adam optimizer, categorical crossentropy
- Evaluate on test set, generate confusion matrix

 Results
- Test accuracy: ~99%
- Confusion matrix saved as `confusion_matrix.png`

 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/mnist-handwritten-cnn.git
