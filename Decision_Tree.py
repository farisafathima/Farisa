#Decision tree classifier and Regressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree

#DECISION TREE REGRESSOR

#loading the data set
data=pd.read_csv('/home/ibab/PycharmProjects/pythonProject/BDB_FILES/simulated_data_multiple_linear_regression_for_ML.csv')
#extracting the features and target
x = data[["age","BMI","BP","blood_sugar","Gender"]]
y= data["disease_score_fluct"]

#split the data
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#define the model
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("RSE:",np.sqrt(mean_squared_error(y_test,y_pred)))

#output file
export_graphviz(reg,feature_names=["age","BMI","BP","blood_sugar","Gender"],out_file='dec_reg.out')

#visualisation
plt.figure(figsize=(10,8), dpi=150)
plot_tree(reg,feature_names=["age","BMI","BP","blood_sugar","Gender"])
plt.show()


#DECISION TREE CLASSIFIER
data=pd.read_csv('/home/ibab/PycharmProjects/pythonProject/BDB_FILES/binary_logistic_regression_data_simulated_for_ML.csv')
#extracting the features and target
x = data[["gender","age","blood_pressure","LDL_cholesterol"]]
y= data["disease_status"]

#split the data
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#define the model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred)*100)

#visualisation
plt.figure(figsize=(15,10))
plot_tree(clf,feature_names=["gender","age","blood_pressure","LDL_cholesterol"])
plt.show()

