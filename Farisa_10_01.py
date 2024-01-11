#Name  : Farisa Fathima
#Objective  :  Machine Learning
#Date : 11-01-2024

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



#QUESTION 1
#loading the data using pandas
data= pd.read_csv('C:/Users/faris/Python_project_1/SEM_2_FILES/simulated_data_multiple_linear_regression_for_ML.csv')
#creating feature matrix
x=data[['age','BMI',"BP","blood_sugar","Gender"]]
y=data['disease_score']

#splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,train_size=0.7)

#defining the model
model=LinearRegression()

#training the model
model.fit(x_train,y_train)

#testing the model
y_pred=model.predict(x_test)

#Coefficient of determination also called as R2 score is used to evaluate the performance of a linear regression model.
r2_score_y1 = r2_score(y_pred,y_test)
print(r2_score)

y2=data['disease_score_fluct']
x_train,x_test,y2_train,y2_test=train_test_split(x,y2,test_size=0.3,train_size=0.7)

model=LinearRegression()

model.fit(x_train,y2_train)

y2_pred=model.predict(x_test)

r2_score_y2=r2_score(y_test,y_pred)
print(r2_score_y2)

#QUESTION 2

#A_Transpose . A will return a square matrix - way to convert rectangular matrix to square
A=np.array([(1,2,3),(4,5,6)])
A_transpose = A.T
print(A_transpose.dot(A))

#QUESTION 3

#implement h(x) = theta0 . x0 + theta1. x1,    where theta0 = 3 , x0 = 1, theta1 = 2
# plot x, h(x) for [ -100, 100 , num =100 ]

###in ML, this kind of linear function is used in linear regression( predict the continuous / dependent var based on features/independent var
###during training process the parameters--theta0... are adjusted to minimise the diff b/w predicted and actual values


theta0 = 3
x0 = 1
theta1 = 2

#range of x values
x= np.linspace(-100,100,num=100)

h_value = theta0 * x0 + theta1 * x

plt.plot(x,h_value)
plt.xlabel('x')
plt.ylabel('h(x)')
plt.title('x vs h(x)')
plt.grid(True)
plt.show()

#QUESTION 4
##Implement h(x) = theta0 . x0 + theta1 . x1 + theta2 . x2 ^2 ,   where theta0 = 4, theta1 = 3 , theta 2 = 2
#and plot h(x) in the range (start=-10,stop=10, num=100)

theta0 = 4
theta1 = 3
theta2 = 2

x= np.linspace(-10,10,num=100)

h_value = theta0 * x + theta1 * x + theta2 * x**2

plt.plot(x,h_value)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs h(x)--quadratic')
plt.show()

#QUESTION 5
##Implement Gaussian PDF mean=0, sigma=15 in the range(-100,100,100)

#Gaussian probability density function (PDF), also known as the normal distribution, is a continuous probability distribution that is symmetric and bell-shaped.
#used in statistics, probability theory, and various fields to model natural phenomena where randomness and variability are present
#
mean=0
sigma=15
x=np.linspace(-100,100,num=100)

pdf=norm.pdf(x,loc=mean,scale=sigma) #loc:represents the mean
#scale:represents the standard deviation

plt.plot(x,pdf)
plt.xlabel('x')
plt.ylabel('Gaussian PDF')
plt.show()

#QUESTION 6
#Implement y=x^2, its derivative and plot both function and derivative in the range(-100,100,100)

x=np.linspace(-100,100,num=100)
func = x **2
func_der = np.gradient(func,x)
plt.plot(x,func)
plt.plot(x,func_der)
plt.xlabel('x')
plt.ylabel("f(x)/f'(x)")
plt.title("plot of f(x) & f'(x)")
plt.show()