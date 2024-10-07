# 850_project_1

#importing all the packages 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV


#step 1: Data processing 
#data is read from the file and converted into a datafrqamce using Panadas. 
#data is printed to verify that the file has been converted properly

df = pd.read_csv("Project_1_data.csv")
print("First few rows of the DataFrame:")
print(df.head())


#step 2: Data Visualization 

#Extract the columns for plotting
x = df['X']
y = df['Y']
z = df['Z']
steps = df['Step']

# Create a 3D scatter plot
fig = plt.figure()
plot1 = fig.add_subplot(111, projection='3d')

#Plot the data
scatter = plot1.scatter(x, y, z, c=steps, cmap='inferno', marker='o')

# Add labels and title
plot1.set_xlabel('X Coordinate')
plot1.set_ylabel('Y Coordinate')
plot1.set_zlabel('Z Coordinate')
plot1.set_title('3D Plot of Coordinates vs Step')

#Add color bar to show the steps
cbar = fig.colorbar(scatter, ax=plot1, label='Step')

# Display the plot
plt.show()

#step 3: Correlation Analysis

df["coordinate categories"] = pd.cut(df['X'],
                          bins=[-np.inf, 3, 6, 9.5, np.inf],
                          labels=[1, 2, 3, 4])
my_splitter = StratifiedShuffleSplit(n_splits = 1,
                                test_size = 0.2,
                                random_state = 42)

for train_index, test_index in my_splitter.split(df, df["coordinate categories"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
strat_df_train = strat_df_train.drop(columns=["coordinate categories"], axis = 1)
strat_df_test = strat_df_test.drop(columns=["coordinate categories"], axis = 1)




X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]




# Initialize the scaler
my_scaler = StandardScaler()

my_scaler.fit(X_train)
scaled_data_train = my_scaler.transform(X_train)
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=X_train.columns)

scaled_data_test = my_scaler.transform(X_test)
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=X_test.columns)

corr_matrix = scaled_data_train_df.corr()
sns.heatmap(np.abs(corr_matrix), cmap='rocket')
plt.title('Correlation Heatmap')


corr1 = y_train.corr(X_train['X'])
print(corr1)
corr2 = y_train.corr(X_train['Y'])
print(corr2)
corr3 = y_train.corr(X_train['Z'])
print(corr3)

 

# step 4: Classification Model Development/Engineering
# ML model 1 linear Regression

linear_reg = LinearRegression()
param_grid_lr = {}  # No hyperparameters to tune for plain linear regression, but you still apply GridSearchCV.
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Linear Regression Model:", best_model_lr)



#Support Vector Machine (SVM)
svr = SVR()
param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)
best_model_svr = grid_search_svr.best_estimator_
print("Best SVM Model:", best_model_svr)



# Decision Tree
decision_tree = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)



# Random Forest
random_forest = RandomForestRegressor(random_state=42)
param_distributions_rf = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search_rf = RandomizedSearchCV(
    random_forest,
    param_distributions=param_distributions_rf,
    n_iter=50,                    # Number of parameter combinations to sample
    cv=5,                         # 5-fold cross-validation
    scoring='neg_mean_absolute_error',
    n_jobs=-1,                    # Use all available cores
    random_state=42               # For reproducibility
)

random_search_rf.fit(X_train, y_train)
best_model_rf = random_search_rf.best_estimator_
best_params_rf = random_search_rf.best_params_
print("Best Random Forest Model:", best_model_rf)
print("Best Hyperparameters:", best_params_rf)


# Training and testing error for Linear Regression
y_train_pred_lr = best_model_lr.predict(X_train)
y_test_pred_lr = best_model_lr.predict(X_test)
mae_train_lr = mean_absolute_error(y_train, y_train_pred_lr)
mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)
print(f"Linear Regression - MAE (Train): {mae_train_lr}, MAE (Test): {mae_test_lr}")

for i in range(5):
     print("Predictions:", y_train_pred_lr[i], "Actual values:", y_train[i])




# Training and testing error for SVM
y_train_pred_svr = best_model_svr.predict(X_train)
y_test_pred_svr = best_model_svr.predict(X_test)
mae_train_svr = mean_absolute_error(y_train, y_train_pred_svr)
mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
print(f"SVM - MAE (Train): {mae_train_svr}, MAE (Test): {mae_test_svr}")

for i in range(5):
     print("Predictions:", y_train_pred_svr[i], "Actual values:", y_train[i])

# Training and testing error for Decision Tree
y_train_pred_dt = best_model_dt.predict(X_train)
y_test_pred_dt = best_model_dt.predict(X_test)
mae_train_dt = mean_absolute_error(y_train, y_train_pred_dt)
mae_test_dt = mean_absolute_error(y_test, y_test_pred_dt)
print(f"Decision Tree - MAE (Train): {mae_train_dt}, MAE (Test): {mae_test_dt}")

for i in range(5):
     print("Predictions:", y_train_pred_dt[i], "Actual values:", y_train[i])

# Training and testing error for Random Forest
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)
print(f"Random Forest - MAE (Train): {mae_train_rf}, MAE (Test): {mae_test_rf}")

for i in range(5):
     print("Predictions:", y_train_pred_rf[i], "Actual values:", y_train[i])
     
     
     
    
# Step % Model Performance Analysis

