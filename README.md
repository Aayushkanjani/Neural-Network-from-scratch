# Neural Network from Scratch Without Using Pytorch, TensorFlow, or Keras
Implemented Neural Network from scratch on MNIST dataset

Dataset link: https://www.kaggle.com/c/digit-recognizer/data <br>

This project demonstrates the implementation of a neural network from scratch using only NumPy for numerical computations and basic Python libraries for data handling and visualization. The neural network is trained on the MNIST dataset, which contains 60,000 images of handwritten digits, each having a dimension of 28x28 pixels.

## Introduction

Neural networks are at the core of deep learning. With intricate layers of artificial neurons, these networks emulate the workings of the human brain. The most complex tasks in artificial intelligence are usually handled by Artificial Neural Networks (ANNs), and many libraries abstract the creation of Neural Networks into extremely few lines of code. However, this project aims to build a neural network from scratch without using any high-level libraries like TensorFlow, Keras, or PyTorch, relying solely on NumPy and mathematical computations.

## Features

- Implementation of a neural network from scratch using NumPy.
- Training on the MNIST dataset of handwritten digits.
- Forward and backward propagation with ReLU and Softmax activations.
- One-hot encoding for labels.
- Visualization of predictions and comparison with actual labels.

## Dataset

The dataset used in this project is the MNIST dataset, which is available in CSV format. The training set consists of 60,000 images of handwritten digits, each with a dimension of 28x28 pixels.

## Implementation

The neural network consists of:
- An input layer with 784 neurons (for the 28x28 pixel input images).
- One hidden layer with 10 neurons.
- An output layer with 10 neurons (for the 10 possible digit classes).

### Functions and Components

1. **Initialization**:
   - `init_params()`: Initializes weights and biases for the layers.

2. **Activation Functions**:
   - `ReLU(x)`: ReLU activation function.
   - `Softmax(Z)`: Softmax activation function.
   - `ReLU_derivative(x)`: Derivative of the ReLU function.

3. **Forward Propagation**:
   - `forward_prop(W1, b1, W2, b2, X)`: Computes the forward propagation through the network.

4. **Backward Propagation**:
   - `backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)`: Computes the gradients for backpropagation.

5. **Parameter Updates**:
   - `update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)`: Updates the network parameters using the computed gradients.

6. **Helper Functions**:
   - `one_hot(Y)`: Converts labels to one-hot encoded vectors.
   - `get_predictions(A2)`: Gets the predictions from the output layer.
   - `get_accuracy(predictions, Y)`: Computes the accuracy of the predictions.
   - `gradient_descent(X, Y, alpha, iterations)`: Trains the network using gradient descent.
   - `make_predictions(X, W1, b1, W2, b2)`: Makes predictions on new data.
   - `test_prediction(index, W1, b1, W2, b2)`: Tests the network on a specific example.

## Usage

1. **Load the Dataset**:
   ```python
   data = pd.read_csv('train.csv').
   test_data = pd.read_csv('test.csv').

2. **Preprocess the Data**:
   data = np.array(data)
  np.random.shuffle(data)
  data_train = data.T
  Y_train = data_train[0]
  X_train = data_train[1:]
  X_train = X_train / 255
  Y_train = Y_train / 255 

3. **Train the Neural Network**:
   W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

4. **Test the Network**:
  test_prediction(0, W1, b1, W2, b2)
  test_prediction(24, W1, b1, W2, b2)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgements

This project was inspired by the need to understand the inner workings of neural networks and deep learning by building a neural network from scratch. Special thanks to the open-source community for providing valuable resources and tools that made this project possible.

---
Feel free to adjust any sections or details to better match your project.
