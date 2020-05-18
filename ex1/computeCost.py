import numpy as np

#COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
def computeCost(X, y, theta):
    # Initialize some useful values
    m = y.size # number of training examples

    # You need to return the following variables correctly 
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    hypothesis = X @ theta
    diff = np.subtract(hypothesis, y)
    diff_sq = diff**2
    J = diff_sq.sum() / (2 * m)



    # =========================================================================
    return J