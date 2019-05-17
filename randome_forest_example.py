import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""Step 1: Pick at random K data points from the Training Set.
   
   Step 2: Build the Decision Tree associated with K data points.
   
   Step 3: Choose the number Ntree of trees you want to build and repeat steps 1 & 2.
   
   Step 4: For a new data point, make each one of your Ntree predict the value of Y
   for the data point in question, and assign the new data point the average across all of the predicted Y values.
   
"""

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 3000, random_state= 0)
regressor.fit(X, y)




y_pred = regressor.predict(np.array([[6.5]]))
y_pred2 = regressor.predict(np.array([[2]]))

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color ='red')
plt.plot(X_grid, regressor.predict(X_grid), color ='blue')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.title('Truth or Bluff? (Random Forest Example)')
plt.plot()
