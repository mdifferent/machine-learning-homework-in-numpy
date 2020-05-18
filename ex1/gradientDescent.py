from computeCost import computeCost
import numpy as np

# GRADIENTDESCENT Performs gradient descent to learn theta
def gradientDescent(X, y, theta, alpha, num_iters):
        
    #   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = y.size # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(1, num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        hypothesis = X @ theta
        diff = np.subtract(hypothesis, y)
        theta = np.subtract(theta, (alpha / m) * X.transpose() @ diff)





        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)
 
    return theta, J_history