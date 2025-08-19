# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy import stats

from google.colab import drive
drive.mount('/content/drive')

filepath = "/content/drive/MyDrive/Vedant Ranka/data/DrivAerNet_ParametricData.csv"

data = pd.read_csv(filepath)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure to hold the subplots
plt.figure(figsize=(20, 10))

# Histogram of Average Cd
plt.subplot(2, 2, 1)
sns.histplot(data['Average Cd'], kde=True)
plt.title('Histogram of Average Drag Coefficient (Cd)')

# Histogram of Average Cl
plt.subplot(2, 2, 2)
sns.histplot(data['Average Cl'], kde=True)
plt.title('Histogram of Average Lift Coefficient (Cl)')

# Scatter plot of Average Cd vs. Average Cl
plt.subplot(2, 2, 3)
sns.scatterplot(x='Average Cd', y='B_Diffusor_Angle', data=data)
plt.title('Average Drag Coefficient (Cd) vs. Average Lift Coefficient (Cl)')

# Box plot of all aerodynamic coefficients
plt.subplot(2, 2, 4)
melted_data = data.melt(value_vars=['Average Cd', 'Average Cl', 'Average Cl_f', 'Average Cl_r'], var_name='Coefficient',
                        value_name='Value')
sns.boxplot(x='Coefficient', y='Value', data=melted_data)
plt.title('Box Plot of Aerodynamic Coefficients')

plt.tight_layout()
plt.show()

data['ld'] = data['Average Cl_f'] / data['Average Cd']

X = data[['B_Diffusor_Angle', 'B_Ramp_Angle']]
y = data['ld']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

forest_model = RandomForestRegressor(max_depth= 20, min_samples_leaf= 20, min_samples_split= 2, n_estimators= 200, random_state =42)
forest_model.fit(X_train, y_train)

tree_model = DecisionTreeRegressor(splitter="best", max_depth= 15, min_samples_leaf= 4, min_weight_fraction_leaf= 0.1, max_features= "sqrt", max_leaf_nodes= None, random_state=42)
tree_model.fit(X_train, y_train)

knn_model = KNeighborsRegressor(n_neighbors=40, algorithm = "ball_tree", weights = "distance", leaf_size = 100, metric = "minkowski", p = 1)
knn_model.fit(X_train, y_train)

gbr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, loss = "squared_error")
gbr_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X)
forest_pred = forest_model.predict(X)
knn_pred = knn_model.predict(X)
gbr_pred = gbr_model.predict(X)

tree_mse = mean_squared_error(y, tree_pred)
tree_r2 = r2_score(y, tree_pred)

forest_mse = mean_squared_error(y, forest_pred)
forest_r2 = r2_score(y, forest_pred)

knn_mse = mean_squared_error(y, knn_pred)
knn_r2 = r2_score(y, knn_pred)

gbr_mse = mean_squared_error(y, gbr_pred)
gbr_r2 = r2_score(y, gbr_pred)

print("Decision Tree Model - Mean Squared Error:", tree_mse)
print("Decision Tree Model - r2:", tree_r2)

print("Random Forest Model - Mean Squared Error:", forest_mse)
print("Random Forest Model - r2:", forest_r2)

print("KNN Model - Mean Squared Error:", knn_mse)
print("KNN Model - r2:", knn_r2)

print("GBR Model - Mean Squared Error:", gbr_mse)
print("GBR Model - r2:", gbr_r2)

forest_parameters = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"]
}

tree_parameters = {"max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"]}

knn_parameters = {'n_neighbors':[10, 15, 20, 25, 30, 35, 40],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'leaf_size': [20, 30, 50, 100],
                  'p': [1, 2],
                  'metric': ['minkowski', 'euclidean', 'manhattan']}

gbr_parameters = {'n_estimators': [50, 100, 200],
                  'learning_rate': [0.01, 0.1, 1],
                  'max_depth': [1, 2, 3],
                  'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']}

rf = DecisionTreeRegressor(random_state = 42)
clf = GridSearchCV(rf, tree_parameters)
clf.fit(X, y)

rf = KNeighborsRegressor(random_state = 42)
clf = GridSearchCV(rf, knn_parameters)
clf.fit(X, y)

rf = GradientBoostingRegressor(random_state = 42)
clf = GridSearchCV(rf, gbr_parameters)
clf.fit(X, y)

rf = RandomForestRegressor(random_state = 42)
clf = GridSearchCV(rf, forest_parameters)
clf.fit(X, y)

print("Best Parameters:", clf.best_params_)

from scipy.optimize import minimize, rosen, rosen_der

model = KNeighborsRegressor(n_neighbors=20, algorithm = "ball_tree", weights = "distance", leaf_size = 100, metric = "minkowski", p = 1)
model.fit(X_train, y_train)

target_lift_to_drag = -0.2

def objective(variables):
    diffusor_angle, ramp_angle = variables
    prediction = model.predict([[diffusor_angle, ramp_angle]])[0]
    return (prediction - target_lift_to_drag) ** 2

initial_guess = [5.0, 10.0]
bounds = [(0, 20), (0, 20)]

result = minimize(objective, initial_guess, bounds=bounds)

optimal_variables = result.x
predicted_ratio = model.predict([optimal_variables])[0]

print(f"Optimal Diffusor Angle: {optimal_variables[0]:.2f} degrees")
print(f"Optimal Ramp Angle: {optimal_variables[1]:.2f} degrees")
print(f"Predicted Lift-to-Drag Ratio: {predicted_ratio:.4f}")
