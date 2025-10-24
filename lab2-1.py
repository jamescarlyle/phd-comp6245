import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt

# Generate consistent data for regression tests.
X, y = make_regression (n_samples=100, n_features=10, noise=10, random_state=42)

# Lambda value to use for all evaluations. Determines penalty size of all weights.
lmbda = 0.1

#  Calculate final weights and costs of gradient descent.
def ridge_regression_gradient_descent(X, y, alpha, lmbda, iterations):
    samples, features = X.shape
    # Initialise weights and costs to zero.
    weights = np.zeros(features)
    costs = np.zeros(iterations)


    for i in range(iterations):
        # Compute prediction.
        y_delta = y - X @ weights
        costs[i] = np.sum(np.square(y_delta))
        # Compute gradient based on the derivative of cost function (sum of squares of error).
        gradient = (-2 / samples) * X.T @ y_delta + 2 * lmbda * weights
        # Update weights.
        weights -= alpha * gradient

    return weights, costs

# Generate weights using analytical formula.
n, d = X.shape
I = np.eye(d)
ridge_analytical = np.linalg.inv(X.T @ X + lmbda * I) @ X.T @ y

# Generate weights by gradient descent calculation with a learning rate of 0.01 and 1000 iterations.
ridge_gradient_weights, ridge_gradient_costs = ridge_regression_gradient_descent(X, y, 0.05, lmbda, 100)

# Generate weights by sklearn Ridge class, without intercept for consistency.
ridge_model = Ridge (alpha = lmbda , fit_intercept = False)
ridge_model.fit (X , y)

# Check if the results are numerically very close.
print ( f" Weights from analytical solution :\n{ridge_analytical.round (4)}")
print ( f" Weights from gradient descent :\n{ridge_gradient_weights.round(4)}")
print ( f" Weights from scikit - learn :\n{ridge_model.coef_.round(4)}")
print ( f" Solutions are close : {np.allclose (ridge_analytical, ridge_model.coef_ )}")

# Generate cost by number of iterations and plot.
fig, ax = plt.subplots()
ax.plot(np.log(ridge_gradient_costs))
ax.set(xlabel='iterations', ylabel='logn cost',
       title='Effect of iterations on residual cost for ridge regression gradient descent')
ax.grid()
plt.show()

# Iterate through a range of lambda values (penalty coefficients).
lambda_upper = 100
step_size = 0.2
n_iterations = int(lambda_upper / step_size)
# Matrices to store coefficients for different lambda values, with each lambda having a row of coefficients.
ridge_coefficients = np.zeros((n_iterations, 10))
lasso_coefficients = np.zeros((n_iterations, 10))

for lmbda in range(n_iterations):
    ridge_model = Ridge (alpha = lmbda , fit_intercept = False)
    ridge_coefficients[lmbda] = ridge_model.fit(X , y).coef_
    lasso_model = Lasso (alpha = lmbda, fit_intercept = False)
    lasso_coefficients[lmbda] = lasso_model.fit(X , y).coef_

# Generate cost by number of iterations and plot.
fig, ax = plt.subplots()
# Transpose coefficients so each of 10 coefficients becomes a row.
ridge_coefficients = ridge_coefficients.T
lasso_coefficients = lasso_coefficients.T

ax.set(xlabel='log alpha value', ylabel='coefficient value',
       title='Lasso coefficients versus alpha')
ax.grid()
ax.set_xscale('log')
for i in range(10):
    ax.plot(ridge_coefficients[i])
plt.show()
