# Simple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data
dataset = pd.read_csv('Salary_Data.csv')
print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into a Training data and Testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, train_size= 2/3,
                                                    random_state= 0)

print('This is the testing dataset for x,  \n',x_test,'\n')
print('This is the training dataset for y,  \n',y_train,'\n')
print('This is the training dataset for x,  \n',x_train, '\n')
print('This is the testing dataset for y,  \n',y_test, '\n')

##### y_test contains the real salaries #############

# Fitting the simple Linear Progression Model to the training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set Results, x_pred contains the PREDICTED SALARIES
x_pred = regressor.predict(x_test)
for predictions in x_pred:
    print('This is the prediction of the model:', predictions)


# Visualizing the Training set results
plt.scatter(x_train, y_train, color = 'red') ## The observation points will be colored in red ##
plt.plot(x_train, regressor.predict(x_train), color = 'blue')

plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.show()
