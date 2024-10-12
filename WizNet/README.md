# WizNet: A Mini Neural Network

WizNet is a lightweight implementation of a neural network framework, enabling easy experimentation with neural network parameters such as weights, biases, and activations. It features basic operations and backpropagation to compute gradients.

## Features

- **Parameter Class**: Core class representing the parameters of the neural network, with support for addition and multiplication.
- **Activation Functions**: Implements commonly used activation functions: Sigmoid, ReLU, and Tanh.
- **Backpropagation**: Calculates gradients using chain rule for effective learning.
- **Visualization**: Visualizes the computation graph of parameters using Graphviz.

## Installation

To use WizNet, ensure you have Python installed and install the required packages:

```bash
pip install numpy graphviz
