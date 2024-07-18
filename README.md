# Logistic Regression Implementation for Iris Dataset

This repository contains an implementation of logistic regression using different optimization methods for the Iris dataset. The task involves classifying Iris Versicolor and Iris Virginica species. The logistic regression model is implemented from scratch without using the `LogisticRegression` class from `scikit-learn`.

## Task Description

The task requires the application of optimization theory and machine learning to implement logistic regression. The following steps are performed:

1. **Load the Iris dataset** and retain only two classes: Iris Versicolor and Iris Virginica.
2. **Implement logistic regression** from scratch using `pandas`, `numpy`, and `math` libraries.
3. **Implement gradient descent** to train the logistic regression model.
4. **Implement RMSProp optimization** and train the logistic regression model.
5. **Implement Nadam optimization** and train the logistic regression model.
6. **Compare the performance** of the implemented optimization methods using a selected metric (accuracy).

## Solution

The solution involves the following steps:

### Data Loading and Preprocessing

1. Load the Iris dataset.
2. Retain only two classes: Iris Versicolor and Iris Virginica.
3. Split the dataset into training and testing sets.

### Logistic Regression Implementation

1. Implement logistic regression with a sigmoid activation function.
2. Implement gradient descent, RMSProp, and Nadam optimization methods.
3. Train the logistic regression model using each optimization method.
4. Evaluate the model's accuracy on the test set.

### Code

The code implementation is as follows:

```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Retain only two classes: Iris Versicolor and Iris Virginica
mask = y != 0
X = X[mask]
y = y[mask]

# Redefine class labels to 0 and 1
y = y - 1

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, optimizer='gd', beta=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def compute_cost(self, y_true, y_pred):
        m = y_true.shape[0]
        cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.initialize_weights(n)

        if self.optimizer == 'gd':
            self.gradient_descent(X, y)
        elif self.optimizer == 'rmsprop':
            self.rmsprop(X, y)
        elif self.optimizer == 'nadam':
            self.nadam(X, y)

    def gradient_descent(self, X, y):
        m = X.shape[0]
        for i in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                cost = self.compute_cost(y, y_pred)
                print(f'Epoch {i}, Cost: {cost}')

    def rmsprop(self, X, y):
        m = X.shape[0]
        v_dw = np.zeros_like(self.weights)
        v_db = 0

        for i in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            v_dw = self.beta * v_dw + (1 - self.beta) * np.square(dw)
            v_db = self.beta * v_db + (1 - self.beta) * np.square(db)

            self.weights -= self.learning_rate * (dw / (np.sqrt(v_dw) + self.epsilon))
            self.bias -= self.learning_rate * (db / (np.sqrt(v_db) + self.epsilon))

            if i % 100 == 0:
                cost = self.compute_cost(y, y_pred)
                print(f'Epoch {i}, Cost: {cost}')

    def nadam(self, X, y):
        m = X.shape[0]
        v_dw = np.zeros_like(self.weights)
        v_db = 0
        m_dw = np.zeros_like(self.weights)
        m_db = 0

        for i in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            m_dw = self.beta * m_dw + (1 - self.beta) * dw
            m_db = self.beta * m_db + (1 - self.beta) * db
            v_dw = self.beta2 * v_dw + (1 - self.beta2) * np.square(dw)
            v_db = self.beta2 * v_db + (1 - self.beta2) * np.square(db)

            m_dw_hat = m_dw / (1 - self.beta ** (i + 1))
            m_db_hat = m_db / (1 - self.beta ** (i + 1))
            v_dw_hat = v_dw / (1 - self.beta2 ** (i + 1))
            v_db_hat = v_db / (1 - self.beta2 ** (i + 1))

            self.weights -= self.learning_rate * (self.beta * m_dw_hat + (1 - self.beta) * dw) / (np.sqrt(v_dw_hat) + self.epsilon)
            self.bias -= self.learning_rate * (self.beta * m_db_hat + (1 - self.beta) * db) / (np.sqrt(v_db_hat) + self.epsilon)

            if i % 100 == 0:
                cost = self.compute_cost(y, y_pred)
                print(f'Epoch {i}, Cost: {cost}')

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

# Training and evaluation using Gradient Descent
print ('-'*50)
model_gd = LogisticRegression(learning_rate=0.01, epochs=1000, optimizer='gd')
model_gd.fit(X_train, y_train)
predictions_gd = model_gd.predict(X_test)
accuracy_gd = accuracy_score(y_test, predictions_gd)
print(f'Accuracy (Gradient Descent): {accuracy_gd}')

# Training and evaluation using RMSProp
print ('-'*50)
model_rmsprop = LogisticRegression(learning_rate=0.01, epochs=1000, optimizer='rmsprop')
model_rmsprop.fit(X_train, y_train)
predictions_rmsprop = model_rmsprop.predict(X_test)
accuracy_rmsprop = accuracy_score(y_test, predictions_rmsprop)
print(f'Accuracy (RMSProp): {accuracy_rmsprop}')

# Training and evaluation using Nadam
print ('-'*50)
model_nadam = LogisticRegression(learning_rate=0.01, epochs=1000, optimizer='nadam')
model_nadam.fit(X_train, y_train)
predictions_nadam = model_nadam.predict(X_test)
accuracy_nadam = accuracy_score(y_test, predictions_nadam)
print(f'Accuracy (Nadam): {accuracy_nadam}')

# Comparison of results
print ('-'*50)
results = pd.DataFrame({
    'Method': ['Gradient Descent', 'RMSProp', 'Nadam'],
    'Accuracy': [accuracy_gd, accuracy_rmsprop, accuracy_nadam]
})

print(results)
```

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib

You can install the required packages using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

To run the code, simply execute the script in your Python environment.

## Results

The accuracy of the logistic regression model trained using different optimization methods will be printed in the console output and displayed in a comparison table.
