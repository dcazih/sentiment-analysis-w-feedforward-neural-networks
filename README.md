# Sentiment Analysis with Feedforward Neural Networks

This project applies feedforward neural networks (FNNs) to binary sentiment classification of IMDb movie reviews using PyTorch. Reviews are vectorized using TF-IDF, and multiple training strategies—such as dropout regularization, k-fold cross-validation, and ensemble learning—are used to assess and improve model performance. Accuracy is compared against a logistic regression baseline.

## Overview

This project demonstrates a full deep learning pipeline for natural language sentiment classification:
- Data preprocessing and vectorization
- Model design and training
- Evaluation via cross-validation and regularization
- Comparison against traditional ML models

## Features

- TF-IDF vectorization using `scikit-learn`
- Deep feedforward network (3 hidden layers)
- ReLU activations, softmax output
- AdamW optimizer with L2 regularization
- Manual k-fold cross-validation
- Dropout for regularization
- Bagging ensemble of dropout-trained models
- Baseline comparison with logistic regression

## Technologies

- Python 3.x  
- PyTorch  
- scikit-learn  
- NumPy  
- pandas  
- matplotlib  

## Methodology

### Data Preprocessing
- IMDb reviews are cleaned (lowercased, tokenized, punctuation removed).
- Text is vectorized using `TfidfVectorizer` with a max feature limit (5000 terms).
- Dataset split: 70% train / 30% test.

### Model Architecture
- Fully connected feedforward neural network:
  - Input: 5000-dim TF-IDF vector
  - Hidden Layer 1: 512 units, ReLU
  - Hidden Layer 2: 128 units, ReLU
  - Hidden Layer 3: 64 units, ReLU
  - Output Layer: 2 units, softmax
- Loss function: CrossEntropyLoss
- Optimizer: AdamW (learning rate = 0.0001, weight decay = 0.01)
- Batch size: 512

### Baseline Comparison
- A logistic regression model (from textbook) is trained with GridSearchCV.
- Accuracy comparison:
  - **FNN:** 89.0%
  - **Logistic Regression:** 89.9%

### Cross-Validation
- 5-fold cross-validation is manually implemented.
- Each fold trains and evaluates the model on a different subset.
- Final average accuracy across folds: **88.2%**
- Training time remains under one minute per fold.

### Dropout Regularization
- Dropout layers added between hidden layers.
- Dropout model slightly outperforms baseline (89.1% vs 89.0%).
- Prevents overfitting by improving generalization.

### Ensemble Learning
- An ensemble of five dropout-regularized models is trained.
- Bagging (majority voting) used to combine predictions.
- Ensemble accuracy: **88.2%**
- Slight decrease in accuracy likely due to over-regularization and averaging.

## How to Run

### Installation

Install dependencies:
`pip install -r requirements.txt`

### (Optional) Set up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
```

## Notes

- FNN performs competitively with logistic regression although the regression is simpler the FNN trains faster due to fewer grid search steps.



