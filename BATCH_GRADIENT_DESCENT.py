import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#BATCH GRADIENT DESCENT

#LOADING THE DATA
data = pd.read_csv('C:/Users/faris/Python_project_1/DATASETS/simulated_data_multiple_linear_regression_for_ML.csv')
x = data[["age","BMI","BP","blood_sugar","Gender"]]
y= data["disease_score_fluct"]

print(y.shape)
print(x.shape)
#check for null values
print(data.isnull().sum())

#defining hypothesis function

def hypothesis_func(theta,x):
    hypothesis = x @ theta
    return hypothesis

#defining cost function

def cost_func(hypothesis,actual_value):
    m=len(actual_value)
    error = hypothesis - actual_value
    sse = np.sum(error ** 2)/(2 * m)
    return sse

#GRADIENT DESCENT FUNCTION
#n-iterations, alpha- learning rate are the hyperparameters that can be tuned
def gradient_descent(x,y,alpha,n):
    actual_value = y
    #appending cost at each iteration
    cost = []
    #initialising the parametrs to zeroes
    theta = np.zeros(x.shape[1])

    for i in range(n):
        hypothesis = hypothesis_func(theta,x)
        cost.append(cost_func(hypothesis,actual_value))
        error = hypothesis - actual_value
        gradient = np.dot(x.T,error)
        #estimating optimal parameters
        theta = theta - (alpha * gradient)
    return theta, cost

def main():
    alpha = 0.000001
    n = 1000
    theta, cost = gradient_descent(x, y, alpha, n)
    print(theta, cost)
    #plotting the cost function to check its descent
    plt.plot(np.arange(1000), cost)
    plt.show()


if __name__ == '__main__':
    main()






