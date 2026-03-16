# Logistic Regression with NumPy and Python

A from-scratch implementation of **Logistic Regression** using **NumPy** to classify whether a student passes or fails a DMV written test based on two exam scores.

This project demonstrates the **mathematical foundations of logistic regression**, including:
- Logistic (sigmoid) function
- Cost function
- Gradient computation
- Gradient descent optimization
- Decision boundary visualization

The entire model is implemented **without using machine learning libraries like Scikit-Learn** focusing purely on **NumPy-based numerical computation**.

## Project Overview
This project builds a **binary classification model** that learns the relationship between two exam scores, and the probability of passing the DMV written test.

The workflow includes:
1. Loading and exploring the dataset
2. Visualizing the dataset
3. Implementing the sigmoid function
4. Defining the logistic regression cost function
5. Computing gradients
6. Implementing gradient descent from scratch
7. Plotting convergence of the cost function
8. Visualizing the decision boundary
9. Making predictions using trained parameters

## Dataset
**File:** `DMV_Written_Tests.csv`

The dataset contains **100 training examples** with two input features.

| Feature    | Description                       |
| ---------- | --------------------------------- |
| DMV_Test_1 | Score in DMV Written Test 1       |
| DMV_Test_2 | Score in DMV Written Test 2       |
| Results    | Binary label (1 = Pass, 0 = Fail) |

The goal is to predict the **probability of passing** based on the two test scores.

## Implementation Details

### 1. Importing Libraries
Essential libraries used in the project:
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```
Visualization styles and plotting settings are also configured.

### 2. Loading and Exploring the Dataset
The dataset is loaded using Pandas.
```
data = pd.read_csv("DMV_Written_Tests.csv")
```
Initial inspection is done using:
- `data.head()`
- `data.info()`

The input features and labels are separated into:
```
scores = data[['DMV_Test_1', 'DMV_Test_2']].values
results = data['Results'].values
```

### 3. Data Visualization

Before training the model, the dataset is visualized using **Seaborn scatter plots**.
- Green triangles represent students who passed
- Red crosses represent students who failed

This helps visualize whether the dataset is **linearly separable**.

### 4. Logistic Sigmoid Function
Logistic regression uses the sigmoid function to map predictions to probabilities.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

#### Implementation:
```
def logistic_function(x):
    return 1 / (1 + np.exp(-x))
```
#### Properties:
- Output range: **0 to 1**
- Interpreted as **probability**
- Threshold of **0.5** used for classification

### 5. Cost Function and Gradient
The cost function for logistic regression is defined as:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))\right]
$$

#### Implementation:
```
def compute_cost(theta, x, y):
    m = len(y)
    y_pred = logistic_function(np.dot(x, theta))
    error = (y * np.log(y_pred)) + (1 - y) * np.log(1 - y_pred)
    cost = (-1/m) * sum(error)
    gradient = (1/m) * np.dot(x.transpose(), (y_pred - y))
    return cost[0], gradient
```
This function returns:
- Current **cost value**
- **Gradient vector**

### 6. Feature Scaling
Feature scaling is applied to normalize the dataset.
```
scores = (scores - mean_scores) / std_scores
```
This improves **gradient descent convergence speed**.

A bias term is then added:
```
X = np.append(np.ones((rows, 1)), scores, axis = 1)
```

### 7. Gradient Descent Implementation
Gradient descent is implemented from scratch to optimize parameters.
```
def gradient_descent(x, y, theta, alpha, iterations):
    costs = []
    
    for i in range(iterations):
        cost, gradient = compute_cost(theta, x, y)
        theta -= (alpha * gradient)
        costs.append(cost)
        
    return theta, costs
```
Parameters used:
- **Learning rate (α)** = 1
- **Iterations** = 500

### 8. Convergence of Cost Function
The change in cost over iterations is visualized to confirm optimization.
```
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("J(Θ)")
```
A **monotonically decreasing cost curve** indicates proper convergence.

### 9. Decision Boundary Visualization
The trained logistic regression model produces a **linear decision boundary**.

The boundary equation is derived from:

$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0
$$

#### Implementation:
```
y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]
```
This boundary is plotted on top of the scatter plot to visualize classification separation.

### 10. Prediction Function
Predictions are made using the optimized parameters.
```
def predict(theta, x):
    results = x.dot(theta)
    return results > 0
```
The model predicts:
- **1 → Pass**
- **0 → Fail**

### 11. Model Accuracy
Training accuracy is computed as:
```
p = predict(theta, X)
```
The number of correct predictions is compared with the actual labels to evaluate the training accuracy of the model.
```
Training Accuracy: ~89%
```

### 12. Example Prediction
Example: Predict the probability of passing for a student with scores 50 and 79.
```
test = np.array([50, 79])
probability = logistic_function(test.dot(theta))
```
Output:
```
Predicted Probability of Passing ≈ 0.74
```

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Key Learning Outcomes
This project demonstrates:
- Logistic regression **without ML libraries**.
- Numerical optimization using **gradient descent**.
- Implementation of **cost functions** and **gradients**.
- Feature scaling.
- Decision boundary visualization.
- Binary classification from scratch.

---
