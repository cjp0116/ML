# Importing the Libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Out of the R&D Spends and Marketing spends, which yields better profit?
   With the assumption that the location of the startup hold now significance.
   
   A Caveat to Linear Regressions:
   1. Linearity
   2. Homoscedasticity
   3. Multivariate normality
   4. Independence of errors
   5. Lack of multi-collinearity
   
multiple linear equation is.. {y = b0 + x1*b1(R&D) + b2*x2(Admin) + b3+x3(Marketing) + b4*D1 (state)}
as for the state variables, we need to create Dummy Variables
   
   
   5 methods of building models:
   1.) All-in - Throw in all the cases
        i. Prior knowledge; OR
        ii. You have to; OR
        iii. Preparing for Backward Elimination.
   
   2.) Backward Elimination (Stepwise Regression)
        i. Select a significance level to stay in the model (e.g. SL(significance level) =0.05)
        ii. Fit the full model with all possible predictors.
        iii. Consider the predictor with the highest P-value. If P > Significance Level, go to step 4, otherwise go to FIN
        ix. Remove the predictor.
        X. Fit model without this variable*.
   
   3.) Forward Selection (Stepwise Regression)
        i. Select a significance level to enter the model (eg. SL = 0.05)
        ii. Fit all simple regression models y + Xn. Select the one with the lowest P-value.
        iii. Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have.
        Xi. Consider the predictor with the LOWEST P-value. If P < SL, go to STEP3, otherwise go to FIN.
        
   4.) Bidirectional Elimination (Stepwise Regression)
        i. Select a significance level to enter and to stay in the model e.g: SLENTER = 0.05, SLSTAY = 0.05
        ii. Perform the next step of Forward Selection (new variables must have: P < SLENTER to enter)
        iii. Perform ALL steps of Backward Elimination (old variables must have: P < SLSTAY to stay)
        iX. No new variables can enter and no old variables can exit.
   
   5.) All Possible Models
        i. Select a criterion of goodness of fit (e.g. Akaike criterion).
        ii. Construct all possible Regression Models (2**n)-1 total combinations.
        iii. Select the one with best criterion.
        
   
"""

# Opening the csv to work with
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # Dependant variables aka profits
y = dataset.iloc[:, 4].values # Independant variables aka: (R&D, Administration, Marketing, State)

# Encoding the state variable (categorical data)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_y = LabelEncoder()
X[:,3] = labelencoder_y.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting Test Data and Training Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2, random_state= 0)
