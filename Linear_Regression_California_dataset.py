#A simple model for California dataset 


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_data():
    #loading data
    [x,y]=fetch_california_housing(return_X_y=True)

    #splitting the data into 70% and 30%
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)

    # defining the model
    model=LinearRegression()
    #predicting the data
    # learning the parameters using fit
    model.fit(x_train,y_train)

    #validating the model-predicting whether the predicted y(y hat) is same as actual y(ground truth value)
    y_pred=model.predict(x_test)

    #testing the error
    r2=r2_score(y_test,y_pred)
    print(r2)


def main():
    load_data()

if __name__ == '__main__':
    main()
