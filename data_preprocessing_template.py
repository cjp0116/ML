# Data Pre-Processing Template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------importing the dataset------------------------------------
dataset = pd.read_csv('Data.csv')
print(dataset)
print('-' * 99)

# ----------------------------Independant Variable-------------------------------------
x = dataset.iloc[:,:-1].values



# ----------------------------Dependant Variable---------------------------------------
y = dataset.iloc[:,3].values
print(x)
print(y)
print("-"*99)
np.set_printoptions(threshold= np.nan)
#
#
# -----------------------------Taking care of the missing data---------------------------
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy= 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
#
#
# ----------------Encoding Categorical Data-------------------------------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features= [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y[:, -1] = labelencoder_y.fit_transform(y[: -1])
y = onehotencoder.fit_transform(y).toarray(y)
#
#
#
# ----------Splitting the Data into Training set and Testing set--------------------
from sklearn.cross_decomposition import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, train_size = 0.75,
                                                    random_state = 0)
#
#
# -----------------Feature Scaling------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_test)
x_test = sc_x.transform(x_test)
#
#
#
# ---------------Data Pre-processing Template---------------------------------------------

