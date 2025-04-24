# Sentiment Analysis with Feedforward Neural Networks

This project implements sentiment analysis on IMDb movie reviews using feedforward neural networks (FNNs) in PyTorch. The goal is to build a robust binary classifier that determines whether a given review expresses a positive or negative sentiment. The model is trained on TF-IDF features derived from preprocessed text data and evaluated using multiple training strategies, including hyperparameter tuning, k-fold cross-validation, dropout regularization, and ensemble learning.

## Overview

This project explores the application of deep learning techniques to natural language processing tasks, specifically sentiment classification. It emphasizes end-to-end model development—from data preprocessing and vectorization to model training and performance evaluation.

## Features

- TF-IDF feature extraction using `scikit-learn`
- Feedforward neural network implementation with PyTorch
- Hyperparameter tuning (layers, hidden units, learning rate, weight decay)
- Manual k-fold cross-validation training pipeline
- Dropout regularization applied at multiple layers
- Ensemble learning using bagged dropout models
- Performance comparison with logistic regression as a baseline

## Technologies Used

- Python 3.x
- PyTorch
- scikit-learn
- NumPy
- pandas
- matplotlib (for result visualization)


## Methodology

### Data Preparation

- Raw IMDb reviews are cleaned and normalized (punctuation removal, lowercasing, tokenization).
- Reviews are transformed into TF-IDF vectors using `TfidfVectorizer`.
- The dataset is split into 70% training and 30% testing data.

### Model Architecture

- The classifier is a multi-layer feedforward neural network.
- Hidden layers use ReLU activation.
- Output layer uses sigmoid activation for binary classification.
- Model is trained using binary cross-entropy loss and optimized via Adam.

### Hyperparameter Tuning

- The number of layers, hidden units, learning rate, and weight decay are tuned empirically.
- Accuracy is compared to a baseline logistic regression model trained on the same data.

### Cross-Validation

- A custom implementation of k-fold cross-validation is used due to lack of native PyTorch support.
- The model is trained multiple times with different folds, and performance metrics are averaged.

### Dropout and Regularization

- Dropout layers are inserted between fully connected layers to prevent overfitting.
- Dropout probabilities are tuned per layer.
- Model performance with dropout is compared to baseline.

### Ensemble Learning

- Multiple FNN models are trained with different dropout configurations.
- Predictions are aggregated using majority voting (bagging) to form an ensemble classifier.
- Ensemble performance is compared to individual models in terms of accuracy and training time.

## How to Run

### Virtual Environment (recommended if dependencies dont install properly)

In the project directory (Linux):

Create the virtual environment: `python3 -m venv venv`

Start it: `source venv/bin/activate`

### Installation

Install dependencies:
`pip install -r meta/requirements.txt`

