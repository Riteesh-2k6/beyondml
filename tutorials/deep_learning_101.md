# Deep Learning 101: A Beginner's Guide

Welcome to the world of Deep Learning! BeyondML now supports a simplified way to train Neural Networks on image datasets. This guide will walk you through the basics.

## What is Deep Learning?

Deep Learning is a subset of Machine Learning inspired by the structure and function of the human brain. It uses **Neural Networks** with many layers (hence "deep") to learn complex patterns in data, especially images, text, and sound.

## The Basic Building Block: CNN

For images, we use a special type of network called a **Convolutional Neural Network (CNN)**.

1.  **Convolutional Layer**: Finds features like edges, corners, or textures.
2.  **Pooling Layer**: Reduces the size of the image, keeping only the most important information.
3.  **Fully Connected Layer**: Takes the detected features and classifies the image (e.g., "This is a cat").

## Training Concepts

### 1. Epochs
One "Epoch" is when the network has seen the entire dataset once. We usually train for multiple epochs (e.g., 5 to 10) so the network can refine its understanding.

### 2. Learning Rate
This controls how much the network adjusts its weights based on errors. A rate too high might skip over the best solution; a rate too low will take forever to learn.

### 3. Loss Function
The "scorecard" for the network. It measures how far the prediction was from the actual label. The network tries to minimize this score.

## Get Started with MNIST

The "Hello World" of Deep Learning is the **MNIST** dataset, which consists of 70,000 images of handwritten digits (0-9).

Check out `examples/mnist_demo.py` to see how BeyondML solves it in just a few lines of code!
