from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

import scipy.io as sio
import numpy as np

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

A = warmUpExercise()
print(A)

input('Program paused. Press enter to continue.\n')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')

data = np.loadtxt('./ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = y.size # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
plotData(X, y)

input('Program paused. Press enter to continue.\n')

# =================== Part 3: Cost and Gradient descent ===================
X = np.hstack((np.ones((m,1)), data[:,0].reshape(m,1))) # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f\n', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J)
print('Expected cost value (approx) 54.24\n')

input('Program paused. Press enter to continue.\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('%f\n', theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
#hold on; # keep previous plot visible
#plot(X(:,2), X*theta, '-')
#legend('Training data', 'Linear regression')
#hold off # don't overlay any more plots on this figure

# Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta
print('For population = 35,000, we predict a profit of %f\n', predict1*10000)
predict2 = [1, 7] * theta
print('For population = 70,000, we predict a profit of %f\n', predict2*10000)

input('Program paused. Press enter to continue.\n')

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros(theta0_vals.size, theta1_vals.size)

# Fill out J_vals
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i,j] = computeCost(X, y, t)