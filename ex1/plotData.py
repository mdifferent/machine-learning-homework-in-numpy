import matplotlib.pyplot as plt

#PLOTDATA Plots the data points x and y into a new figure 
#   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
#   population and profit.
def plotData(X, y):
    
    plt.ion()
    fig, ax = plt.subplots()    # open a new figure window
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the 
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the 
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    # Hint: You can use the 'rx' option with plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
    plt.plot(X, y, 'rx', markersize=10)  # Plot some data on the axes.

    plt.xlabel('Population of City in 10,000s') # Set the y􀀀axis label
    plt.ylabel('Profit in $10,000s') # Set the x􀀀axis label
    plt.draw()
    return



