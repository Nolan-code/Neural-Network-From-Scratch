# Neural-Network-From-Scratch

## Project description

First implementation of NN from scratch (without any Deep learning framework such as Pytorch) to predict the Iris variety based on physical aspects features. This project includes model training, data cleaning and comprehensive visualization. The goal of the manual implementation is to deepen my understanding of NN.
It was also made to demonstrate my understanding of:
- Fordward propagation
- Backward propagation
- gradient derivation
- softmax and cross-entropy loss for multi-classification
- training with gradient descent
- features normalization and data spliting

## Model Architecture
Input (n_features)
      ↓
Linear layer (W1, b1)
      ↓
ReLU activation
      ↓
Linear layer (W2, b2)
      ↓
Softmax
      ↓
Cross-entropy Loss

Parameters dimensions:
- W1 : (n_features,64)
- b1 : (1,64)
- W2 : (64,3)
- b1 : (1,3)

## Key Features
- Fully manual implementation(forward + backward propagation)
- gradient calculation using matrix calculus
- Multi-class softmax + cross entropy
- Features normalization
- Train/Validation/Test data spliting
- Accuracy computation
- Only numpy framework

## Results
- Get an accuracy of 96.67% at the end of the optimization of the weight and bias.
![Loss Curve](Figures/NN_manual_implementation_loss.png)

## Required package
- Numpy
- Matplotlib
- Pandas

## Instalation
git clone https://github.com/<username>/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -r requirements.txt

## Usage example
from src.neural_network import NN
import pandas as pd

df = pd.read_csv("data/dataset.csv")

val_acc, test_acc, losses = NN(
    df,
    n_iters=2000,
    lr=0.01,
    target_name="label"
)

print("Validation accuracy:", val_acc)
print("Test accuracy:", test_acc)

## The math
# Forward propagation
- 1st layer:
- Z1 = X W1 + b1
- A1 = ReLU(Z1)

- 2nd layer:
- Z2 = A1 W2 + b2
- Ŷ  = softmax(Z2)

# Backward propagation
- dZ2 = (A2 - Y_onehot) / N
- dW2 = A1.T @ dZ2
- db2 = sum(dZ2)

- dA1 = dZ2 @ W2.T
- dZ1 = dA1 * (Z1 > 0)

- dW1 = X.T @ dZ1
- db1 = sum(dZ1)


