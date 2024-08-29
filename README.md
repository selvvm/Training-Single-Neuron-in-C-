# TinyTorch

This repository contains a simple implementation of a neural network designed to solve the XOR problem. The neural network is implemented in C++ and uses a feedforward architecture with backpropagation for training.

## Overview

The neural network consists of three layers:
1. Input layer
2. Hidden layer
3. Output layer

The network uses the sigmoid activation function for both the hidden and output layers and is trained using the mean squared error (MSE) as the cost function.

## Files

- `main.cpp`: The main C++ file that contains the implementation of the neural network, training logic, and the XOR problem example.

## Dependencies

- Standard C++ libraries (iostream, vector, cmath, ctime)

## How to Run

1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/neural-network-xor.git
    cd neural-network-xor
    ```

2. Compile the code:
    ```sh
    g++ main.cpp -o neural_network
    ```

3. Run the executable:
    ```sh
    ./neural_network
    ```

## Code Structure

### NeuralNetwork Class

The `NeuralNetwork` class encapsulates the neural network's structure and methods:

- **Constructor**: Initializes the network with the specified number of input, hidden, and output nodes, and a learning rate. Randomly initializes weights and biases.

- **feedForward**: Performs the feedforward pass, computing the outputs of the hidden and output layers.

- **train**: Trains the network using backpropagation, adjusting weights and biases based on the error between predicted and target outputs.

### Helper Functions

- **sigmoid**: Activation function used in the network.
- **sigmoidDerivative**: Derivative of the sigmoid function used in backpropagation.
- **meanSquaredError**: Computes the mean squared error between predictions and targets.

### Example Usage

The main function demonstrates how to create and train the neural network using the XOR problem dataset.

## Example Output

During training, the program will output the current cost and the network's predictions for the XOR problem at each iteration. Here's an example snippet of the output:

