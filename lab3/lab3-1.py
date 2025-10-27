import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the dataset
iris = load_iris ()
X = iris.data [:, [2, 3]] # Using petal length and petal width
y = iris.target

# --- Case 1: Not Linearly Separable ( Versicolor vs. Virginica ) ---
X_nls_full = X [ y != 0]
y_nls_full = y [ y != 0]
y_nls_full [ y_nls_full == 1] = 0 # Versicolor
y_nls_full [ y_nls_full == 2] = 1 # Virginica
X_train_nls, X_test_nls, y_train_nls, y_test_nls = train_test_split (X_nls_full, y_nls_full, test_size =0.3, random_state =42)

# --- Case 2: Linearly Separable ( Setosa vs. The Rest ) ---
X_ls_full = X
y_ls_full = np.copy ( y )
y_ls_full [ y_ls_full != 0] = 1 # Versicolor and Virginica are class 1
X_train_ls, X_test_ls, y_train_ls, y_test_ls = train_test_split (X_ls_full, y_ls_full, test_size =0.3, random_state =42)

# Helper function to add the bias term (x0 =1)
def add_bias ( X ) :
    return np.c_[np.ones(X.shape[0]), X]

# Take a scalar or numpy array ‘a‘ and return the sigmoid activation
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Compute the Negative Log Likelihood loss
def nll_loss(y_true, y_pred):
    # NLL is defined as NLL = −[y⋅log(pred)+(1−y)⋅log(1−pred)]
    return -np.mean((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))

# Calculate gradient of NLL loss with respect to weights W
def calculate_gradient(X, y_true, weights):
    n = X.shape[0]
    # Gradient is deriviative i.e. 1/n . xt . (y^ - y_true) where X is feature matrix, y^ is sigmoid(X . w) and y_true is vector
    y_pred = sigmoid(X @ weights)
    gradient = (X.T @ (y_pred - y_true)) / n
    return gradient

# Perform Gradient Descent. Initialize weights, and then iteratively update them. 
# The function returns the final weights and a history of the loss and gradient norm.
def train_logistic_regression(X, y_true, alpha, iterations):
    # Initialise weights.
    X_bias = add_bias(X)
    current_weights = np.zeros(X_bias.shape[1])
    losses = []
    gradients = []
    weights = []
   # Iterate
    for i in range(iterations):
        gradient = calculate_gradient(X_bias, y_true, current_weights)
        gradients.append(np.linalg.norm(gradient))
        losses.append(nll_loss(y_true, sigmoid(X_bias @ current_weights)))
        current_weights -= alpha * gradient
        weights.append(np.linalg.norm(current_weights))
    return weights, gradients, losses

# Task 1: Non-Linearly Separable Case Train your model on the non-linearly-separable dataset. 
# Plot the loss and gradient norm over iterations. You should see them both converge.
# weights_nls, gradients_nls, losses_nls = train_logistic_regression(X_train_nls, y_train_nls, 0.05, 10000)
# plt.title("Non Linearly-Separable, 10000 iterations, learning rate 0.05")
# plt.xlabel("Iteration")
# plt.ylabel("Gradient / Loss")
# plt.plot(gradients_nls, label="Gradient")
# plt.plot(losses_nls, label="Loss")
# plt.legend()
# plt.twinx()
# plt.ylabel("Weight")
# plt.plot(weights_nls, "r", label="Weight")
# plt.legend()
# plt.show()

# Task 2: Linearly Separable Case Train your model on the linearly-separable dataset
# for a large number of iterations (e.g., 5000). Plot the loss and gradient norm.
# weights_ls, gradients_ls, losses_ls = train_logistic_regression(X_train_ls, y_train_ls, 0.05, 10000)
# plt.title("Linearly-Separable, 10000 iterations, learning rate 0.05")
# plt.xlabel("Iteration")
# plt.ylabel("Gradient / Loss")
# plt.plot(gradients_ls, label="Gradient")
# plt.plot(losses_ls, label="Loss")
# plt.legend()
# plt.twinx()
# plt.ylabel("Weight")
# plt.plot(weights_ls, "r", label="Weight")
# plt.legend()
# plt.show()

# # Analyze: What do you observe about the magnitude of the final weight vector w? What
# # happens to the loss and gradient norm over time? Explain why this behavior occurs.

# # # Compute the Negative Log Likelihood loss
# def nll_loss(y_true, y_pred):
#     # NLL is defined as NLL = −[y⋅log(pred)+(1−y)⋅log(1−pred)]
#     return -np.mean((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))

# # # Calculate gradient of NLL loss with respect to weights W
# def calculate_gradient(X, y_true, weights):
#     n = X.shape[0]
#     # Gradient is deriviative i.e. 1/n . xt . (y^ - y_true) where X is feature matrix, y^ is sigmoid(X . w) and y_true is vector
#     y_pred = sigmoid(X @ weights)
#     gradient = (X.T @ (y_pred - y_true)) / n
#     return gradient

# # Compute the Negative Log Likelihood loss
def nll_loss_with_penalty(y_true, y_pred, weights, lmbda):
    # J(w) = NLL(w) + λ||w||^2
    penalty = lmbda * (np.linalg.norm(weights)**2)
    return nll_loss(y_true, y_pred) + penalty

# # Calculate gradient of NLL loss with respect to weights W
def calculate_gradient_with_penalty(X, y_true, weights, lmbda):
    # Penalty gradient is derivative of penalty i.e. 2.lmbda.weights
    penalty = 2 * lmbda * weights
    return calculate_gradient(X, y_true, weights) + penalty

# # # Perform Gradient Descent. Initialize weights, and then iteratively update them. 
# # # The function returns the final weights and a history of the loss and gradient norm.
def train_logistic_regression_with_penalty(X, y_true, alpha, iterations, lmbda):
    # Initialise weights.
    X_bias = add_bias(X)
    current_weights = np.zeros(X_bias.shape[1])
    losses = []
    gradients = []
    weights = []
   # Iterate
    for i in range(iterations):
        gradient = calculate_gradient_with_penalty(X_bias, y_true, current_weights, lmbda)
        gradients.append(np.linalg.norm(gradient))
        losses.append(nll_loss_with_penalty(y_true, sigmoid(X_bias @ current_weights), current_weights, lmbda))
        current_weights -= alpha * gradient
        weights.append(np.linalg.norm(current_weights))
    return weights, gradients, losses, current_weights

# weights_lsp, gradients_lsp, losses_lsp = train_logistic_regression_with_penalty(X_train_ls, y_train_ls, 0.05, 1000, 0.1)

# plt.title("Linearly-Separable, 1000 iterations, learning rate 0.05, penalty 0.1")
# plt.xlabel("Iteration")
# plt.ylabel("Gradient / Loss")
# plt.plot(gradients_lsp, label="Gradient")
# plt.plot(losses_lsp, label="Loss")
# plt.plot(weights_lsp, label="Weight")
# plt.legend()
# plt.show()

weights_nlsp, gradients_nlsp, losses_nlsp, end_weight = train_logistic_regression_with_penalty(X_train_nls, y_train_nls, 0.05, 1000, 0.1)

# plt.title("Non-Linearly-Separable, 1000 iterations, learning rate 0.05, penalty 0.1")
# plt.xlabel("Iteration")
# plt.ylabel("Gradient / Loss")
# plt.plot(gradients_nlsp, label="Gradient")
# plt.plot(losses_nlsp, label="Loss")
# plt.legend()
# plt.twinx()
# plt.ylabel("Weight")
# plt.plot(weights_nlsp, "r", label="Weight")
# plt.legend()
# plt.show()

from sklearn . metrics import classification_report, confusion_matrix, roc_auc_score ,RocCurveDisplay

# --- Your code here ---
# 1. Train your regularized model on the NLS training data to get final_w
# 2. Make predictions on the test data X_test_nls
X_test_nls_bias = add_bias (X_test_nls)

probabilities = sigmoid(X_test_nls_bias @ end_weight)
y_pred = ( probabilities >= 0.5).astype(int)

print(probabilities)
print(y_pred)

# --- Evaluation Code ---
print (" - - - Classification Report - - -")
print ( classification_report ( y_test_nls , y_pred ))

print ("\n- - - Confusion Matrix - - -")
print ( confusion_matrix ( y_test_nls , y_pred ))

print (f"\ nAUC Score : { roc_auc_score ( y_test_nls , probabilities ):.4f}")

RocCurveDisplay . from_predictions ( y_test_nls , probabilities )
plt.title (" ROC Curve for Custom Logistic Regression ")
plt.show ()
