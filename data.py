#pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
# fetch dataset 
combined_cycle_power_plant = fetch_ucirepo(id=294) 
  
# data (as pandas dataframes) 
X = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets 
  
# metadata 
print(combined_cycle_power_plant.metadata) 
  
# variable information 
print(combined_cycle_power_plant.variables)
print(X.head())
print(y.head())


# EDA
print("shape of X", X.shape) 
print("shape of y",y.shape)
print(X.describe())
print(y.describe())

# check for missing values
print("missing values in X", X.isnull().sum())
print("missing values in y", y.isnull().sum())

#relationship between features and target variable
df_relation = pd.concat([X, y], axis=1)
plt.figure(figsize=(10, 8))
correlation_matrix = df_relation.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("correlation heatmap")
plt.savefig("correlatin.png")
plt.show()
# train date
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("shape of x train", X_train.shape)
print("shape of x test", X_test.shape)
print("shape of y train", y_train.shape)
print("shape of y test", y_test.shape)

# build random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt')
rf.fit(X_train, y_train.values.ravel())
# predict the model
pred = rf.predict(X_test)
# evaluate the model
mse = mean_squared_error(y_test, pred)
print("mean squared error", mse)
joblib.dump(rf, "power_plant_model.pkl")
