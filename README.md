# Iris Classification Model Comparison

Comparative analysis of classification algorithms on the Iris dataset, achieving 100% accuracy with optimized K-Nearest Neighbors.

## Project Overview

This project implements and compares multiple classification algorithms to identify iris species based on botanical measurements. The analysis demonstrates algorithm selection impact on classification performance.

## Key Results

- **K-Nearest Neighbors:** 100% accuracy (perfect classification)
- **Logistic Regression:** 97% accuracy
- **Key Finding:** Strategic use of stratified sampling improved model stability
- **Comprehensive evaluation:** Precision, recall, F1-score, and confusion matrices

## Technologies

- Python (scikit-learn, pandas, numpy)
- Classification algorithms (KNN, Logistic Regression)
- Model evaluation and comparison framework
- Modular architecture with reusable components

## Project Structure
```
Iris-Classification/
├── notebooks/ # Jupyter notebooks
│ └── model_comparison_analysis.ipynb
├── src/ # Source code
│ ├── models.py # Model definitions
│ └── trainer.py # Training and evaluation logic
├── .gitignore # Git ignore rules
└── README.md # Project documentation
```


## Features

- Modular model architecture for easy algorithm swapping
- Comprehensive model evaluation pipeline
- Classification report with detailed metrics
- Reusable trainer module for multiple algorithms
- Confusion matrix analysis

## Key Insights

The perfect accuracy of KNN can be attributed to:
- Small, well-separated dataset characteristics
- Optimal hyperparameter selection (n_neighbors=5)
- Stratified train-test split preserving class distribution

## Usage

```python
from src.models import get_logistic_model, get_knn_model
from src.trainer import train_model, model_evaluation

# Train KNN model
knn = get_knn_model(n_neighbors=5)
model, X_train, X_test, y_train, y_test = train_model(knn, X, y)

# Evaluate performance
acc, cm, cr = model_evaluation(model, X_test, y_test)
