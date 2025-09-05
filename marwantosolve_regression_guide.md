# Complete Guide to Machine Learning Regression

A comprehensive guide covering all types of regression algorithms in machine learning with practical Python examples and real-world use cases.

## Table of Contents

- [Overview](#overview)
- [Linear Regression](#1-linear-regression)
- [Polynomial Regression](#2-polynomial-regression)
- [Ridge Regression](#3-ridge-regression-l2-regularization)
- [Lasso Regression](#4-lasso-regression-l1-regularization)
- [Elastic Net Regression](#5-elastic-net-regression)
- [Logistic Regression](#6-logistic-regression)
- [Support Vector Regression](#7-support-vector-regression-svr)
- [Decision Tree Regression](#8-decision-tree-regression)
- [Random Forest Regression](#9-random-forest-regression)
- [Comparison & When to Use](#comparison-and-when-to-use)
- [Performance Metrics](#performance-metrics)
- [Hyperparameter Tuning](#hyperparameter-tuning)

## Overview

Regression is a fundamental machine learning technique used to predict continuous numerical values. This guide covers 9 essential regression algorithms with their mathematical foundations, use cases, and Python implementations.

## 1. Linear Regression

### What is it?
Models a linear relationship between input features and target variable using a straight line.

### Mathematical Formula
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Use Cases
- House price prediction
- Sales forecasting
- Temperature prediction
- Stock price trends
- Simple baseline models

### Python Implementation

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Coefficient: {lr.coef_[0]:.4f}")
print(f"Intercept: {lr.intercept_:.4f}")
```

### Pros & Cons
✅ **Pros:** Simple, fast, interpretable, good baseline
❌ **Cons:** Only captures linear relationships, sensitive to outliers

---

## 2. Polynomial Regression

### What is it?
Extends linear regression by adding polynomial terms to capture non-linear relationships.

### Mathematical Formula
```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

### Use Cases
- Population growth modeling
- Parabolic trajectories
- Economic curves
- Biological growth patterns
- Any curved relationship

### Python Implementation

```python
from sklearn.preprocessing import PolynomialFeatures

# Generate data with non-linear pattern
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.ravel() ** 3 + 2 * X.ravel() ** 2 + np.random.normal(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Train polynomial regression
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Predict and evaluate
y_pred = poly_reg.predict(X_test_poly)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

### Pros & Cons
✅ **Pros:** Captures non-linear patterns, still interpretable
❌ **Cons:** Prone to overfitting with high degrees, computationally expensive

---

## 3. Ridge Regression (L2 Regularization)

### What is it?
Linear regression with L2 penalty that shrinks coefficients to prevent overfitting.

### Mathematical Formula
```
Cost = MSE + α∑βᵢ²
```

### Use Cases
- High-dimensional datasets
- Multicollinearity problems
- Preventing overfitting
- Gene expression analysis
- Text classification

### Python Implementation

```python
from sklearn.linear_model import Ridge

# Create dataset with multiple features
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regression with different alpha values
alphas = [0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    print(f"Alpha: {alpha}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print("-" * 30)
```

### Pros & Cons
✅ **Pros:** Handles multicollinearity, prevents overfitting, stable
❌ **Cons:** Doesn't perform feature selection, requires tuning α

---

## 4. Lasso Regression (L1 Regularization)

### What is it?
Linear regression with L1 penalty that can shrink coefficients to zero, performing automatic feature selection.

### Mathematical Formula
```
Cost = MSE + α∑|βᵢ|
```

### Use Cases
- Feature selection
- Sparse models
- High-dimensional data
- Genomics and bioinformatics
- Text mining

### Python Implementation

```python
from sklearn.linear_model import Lasso

# Create dataset with many irrelevant features
X, y = make_regression(n_samples=1000, n_features=50, n_informative=10, 
                      noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# Check feature selection
selected_features = np.sum(lasso.coef_ != 0)
print(f"Selected features: {selected_features} out of {X.shape[1]}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Plot coefficient values
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(lasso.coef_, 'bo-')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

### Pros & Cons
✅ **Pros:** Automatic feature selection, creates sparse models, interpretable
❌ **Cons:** May remove correlated important features, unstable selection

---

## 5. Elastic Net Regression

### What is it?
Combines Ridge (L2) and Lasso (L1) regularization to get benefits of both methods.

### Mathematical Formula
```
Cost = MSE + α₁∑|βᵢ| + α₂∑βᵢ²
```

### Use Cases
- When you have grouped features
- High-dimensional data with correlation
- Balanced feature selection and regularization
- Gene selection problems
- Image processing

### Python Implementation

```python
from sklearn.linear_model import ElasticNet

# Create dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Elastic Net with different l1_ratio values
l1_ratios = [0.1, 0.5, 0.7, 0.9]
for l1_ratio in l1_ratios:
    elastic = ElasticNet(alpha=0.1, l1_ratio=l1_ratio)
    elastic.fit(X_train, y_train)
    y_pred = elastic.predict(X_test)
    
    selected_features = np.sum(elastic.coef_ != 0)
    print(f"L1 ratio: {l1_ratio}")
    print(f"Selected features: {selected_features}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print("-" * 30)
```

### Pros & Cons
✅ **Pros:** Combines benefits of Ridge and Lasso, handles grouped features well
❌ **Cons:** More hyperparameters to tune, computationally more expensive

---

## 6. Logistic Regression

### What is it?
Classification algorithm that uses the logistic function to model the probability of class membership.

### Mathematical Formula
```
p = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))
```

### Use Cases
- Email spam detection
- Medical diagnosis
- Marketing response prediction
- Credit approval
- Binary/multiclass classification

### Python Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Sample probabilities: {y_pred_proba[0]}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Pros & Cons
✅ **Pros:** Outputs probabilities, fast, no assumptions about feature distributions
❌ **Cons:** Assumes linear decision boundary, sensitive to outliers

---

## 7. Support Vector Regression (SVR)

### What is it?
Uses Support Vector Machine principles for regression, finding a function that deviates from targets by at most ε.

### Mathematical Formula
Uses kernel trick to map data into higher-dimensional space:
```
f(x) = ∑αᵢK(xᵢ, x) + b
```

### Use Cases
- Non-linear pattern recognition
- Time series forecasting
- Financial modeling
- Image processing
- High-dimensional data

### Python Implementation

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Generate non-linear dataset
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svr = SVR(kernel=kernel, C=100, gamma='scale')
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)
    
    print(f"Kernel: {kernel}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"Support vectors: {len(svr.support_)}")
    print("-" * 30)
```

### Pros & Cons
✅ **Pros:** Handles non-linear patterns well, robust to outliers, memory efficient
❌ **Cons:** Hard to interpret, slow on large datasets, requires feature scaling

---

## 8. Decision Tree Regression

### What is it?
Creates a tree-like model of decisions to predict continuous values by splitting data based on feature values.

### How it Works
1. Split data based on feature that minimizes variance
2. Repeat recursively for each subset
3. Stop when stopping criteria are met
4. Predict mean of leaf node values

### Use Cases
- Non-linear relationships
- Mixed data types (numerical + categorical)
- Interpretable models
- Feature importance analysis
- Medical diagnosis trees

### Python Implementation

```python
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Generate dataset with non-linear pattern
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.25, X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree with different max_depth values
depths = [3, 5, 10, None]
for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    print(f"Max depth: {depth}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"Tree depth: {dt.get_depth()}")
    print(f"Number of leaves: {dt.get_n_leaves()}")
    print("-" * 30)

# Feature importance (for multi-feature dataset)
X_multi, y_multi = make_regression(n_samples=1000, n_features=10, random_state=42)
dt_multi = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_multi.fit(X_multi, y_multi)

print("Feature Importances:")
for i, importance in enumerate(dt_multi.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")
```

### Pros & Cons
✅ **Pros:** Highly interpretable, handles non-linear patterns, no preprocessing needed
❌ **Cons:** Prone to overfitting, unstable (small data changes = different tree)

---

## 9. Random Forest Regression

### What is it?
Ensemble method that combines multiple decision trees and averages their predictions to reduce overfitting.

### How it Works
1. Create multiple bootstrap samples of training data
2. Train decision tree on each sample using random feature subsets
3. Average predictions from all trees
4. Use out-of-bag samples for validation

### Use Cases
- Robust predictions
- Feature importance ranking
- Large datasets
- Mixed data types
- When interpretability is less critical

### Python Implementation

```python
from sklearn.ensemble import RandomForestRegressor

# Generate dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with different n_estimators
n_estimators_list = [10, 50, 100, 200]
for n_est in n_estimators_list:
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print(f"N_estimators: {n_est}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print("-" * 30)

# Feature importance analysis
rf_best = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_best.fit(X_train, y_train)

print("Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': range(X.shape[1]),
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(rf_best.feature_importances_)), rf_best.feature_importances_)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

### Pros & Cons
✅ **Pros:** Reduces overfitting, handles large datasets, provides feature importance
❌ **Cons:** Less interpretable than single tree, can overfit with very noisy data

---

## Comparison and When to Use

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Linear Regression** | Simple relationships, baseline | Fast, interpretable | Only linear patterns |
| **Polynomial Regression** | Curved relationships | Captures non-linearity | Overfitting risk |
| **Ridge Regression** | Many correlated features | Handles multicollinearity | No feature selection |
| **Lasso Regression** | Feature selection needed | Automatic selection | May remove important features |
| **Elastic Net** | Balanced regularization | Best of both worlds | More hyperparameters |
| **Logistic Regression** | Classification problems | Probability outputs | Linear boundaries only |
| **SVR** | Non-linear patterns | Robust, flexible | Hard to interpret |
| **Decision Tree** | Interpretability needed | Easy to understand | Prone to overfitting |
| **Random Forest** | Robust performance | Reduces overfitting | Less interpretable |

## Performance Metrics

### For Regression
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
```

### For Classification
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```

## Hyperparameter Tuning

### Grid Search Example
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Fit and find best parameters
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Test the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Test R² score: {r2_score(y_test, y_pred_best):.4f}")
```

### Random Search Example
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Random search
random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
```

## Quick Start Template

```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load your data
# X, y = load_your_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose and train model
model = LinearRegression()  # Replace with your chosen algorithm
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

## Installation Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Contributing

Feel free to contribute by:
- Adding new regression algorithms
- Improving code examples
- Adding more use cases
- Fixing bugs or typos

## License

This guide is available under the MIT License.
