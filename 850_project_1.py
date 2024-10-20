# 850_project_1_1
#importing all the packages
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import joblib as jb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#step 1: Data processing 
#data is read from the file and converted into a datafrqamce using Panadas. 
#data is read from the file and converted into a datafrqamce using Pandas. 
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

cbar = fig.colorbar(scatter, ax=plot1, pad=0.1) 
cbar.set_label('Step')

plt.tight_layout()
# Display the plot
plt.show()


print(df.describe())

# Check class distribution of the target variable 'Step'
print(df['Step'].value_counts())

#step 3: Correlation Analysis



df["coordinate categories"] = pd.cut(df['X'],
                                bins=[-np.inf, 3, 6, 9.5, np.inf], 
                                      labels=[1, 2, 3, 4])

my_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)



for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True) 
    strat_df_test = df.loc[test_index].reset_index(drop=True)    

X_train = strat_df_train.drop("Step", axis = 1)
strat_df_train = strat_df_train.drop(columns=["coordinate categories"], axis=1)
strat_df_test = strat_df_test.drop(columns=["coordinate categories"], axis=1)


X_train = strat_df_train.drop(["Step"], axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
X_test = strat_df_test.drop(["Step"], axis = 1)
y_test = strat_df_test["Step"]


print(X_train.head())

# Initialize the scaler
my_scaler = StandardScaler()
scaler = StandardScaler()
scaler.fit(X_train)

# Apply scaling to both training and test data
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train_with_step = X_train.copy()
X_train_with_step['Step'] = y_train

# Compute the new correlation matrix including 'Step'
corr_matrix_with_step = X_train_with_step.corr()

# Plot the new correlation matrix including 'Step'
plt.figure(figsize=(10, 8))
sns.heatmap(np.abs(corr_matrix_with_step), annot=True, cmap='coolwarm', 
            xticklabels=corr_matrix_with_step.columns, 
            yticklabels=corr_matrix_with_step.columns, 
            fmt=".2f", linewidths=.5)

plt.title('Correlation Matrix Including Step')
plt.show()

print(corr_matrix_with_step)

corr1 = y_train.corr(X_train['X'])
print("Correlation between X and Step:", corr1)

corr2 = y_train.corr(X_train['Y'])
print("Correlation between Y and Step:", corr2)

corr3 = y_train.corr(X_train['Z'])
print("Correlation between Z and Step:",corr3)

 
# step 4: Classification Model Development/Engineering
#logregression
logistic_reg = LogisticRegression(class_weight=None, max_iter=2000, random_state=42)

linear_reg = LinearRegression()
param_grid_lr = {}  # No hyperparameters to tune for plain linear regression, but you still apply GridSearchCV.
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Linear Regression Model:", best_model_lr)
param_grid_logistic = {
    'penalty': ['l2'],  
    'C': [0.1, 1, 10, 100],
    'max_iter': [2000] 
}

grid_search_logistic = GridSearchCV(logistic_reg, param_grid_logistic, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_logistic.fit(X_train, y_train)
best_model_logistic = grid_search_logistic.best_estimator_

print("Best Logistic Regression Model:", best_model_logistic)
print("Best Hyperparameters for Logistic Regression:", grid_search_logistic.best_params_)


#Support Vector Machine (SVM)
svm = SVC(class_weight='balanced', random_state=42)
param_grid_svm = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3, 4]
}

grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_model_svm = grid_search_svm.best_estimator_

print("Best SVM Model:", best_model_svm)
print("Best Hyperparameters for SVR:", grid_search_svm.best_params_)


# Decision Tree
decision_tree = DecisionTreeClassifier(class_weight='balanced',random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_model_dt = grid_search_dt.best_estimator_

print("Best Decision Tree Model:", best_model_dt)
print("Best Hyperparameters for Decision Tree:", grid_search_dt.best_params_)



# Random Forest
random_forest = RandomForestRegressor(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
param_distributions_rf = {
    'n_estimators': [10, 50, 100, 200, 500], 
    'max_depth': [None, 10, 20, 30, 40, 50], 
    'min_samples_split': [2, 5, 10, 20],  
    'min_samples_leaf': [1, 2, 4, 8], 
    'max_features': [None, 'sqrt', 'log2'],  
    'bootstrap': [True, False]  
}

# Setup RandomizedSearchCV
random_search_rf = RandomizedSearchCV(
    random_forest,
    param_distributions=param_distributions_rf,
    n_iter=50,                    # Number of parameter combinations to sample
    cv=5,                         # 5-fold cross-validation
    scoring='neg_mean_absolute_error',
    n_jobs=-1,                    # Use all available cores
    random_state=42               # For reproducibility
    n_iter=50,                
    cv=5,                     
    scoring='accuracy',      
    n_jobs=-1,                
    random_state=42
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
y_train_pred_logistic = best_model_logistic.predict(X_train)
y_test_pred_logistic = best_model_logistic.predict(X_test)

print("Results for Logistic Regression:")
for i in range(5):
     print("Predictions:", y_train_pred_lr[i], "Actual values:", y_train[i])

     print("Predictions:", y_train_pred_logistic[i], "Actual values:", y_train[i])


y_train_pred_svm = best_model_svm.predict(X_train)
y_test_pred_svm = best_model_svm.predict(X_test)

# Training and testing error for SVM
y_train_pred_svr = best_model_svr.predict(X_train)
y_test_pred_svr = best_model_svr.predict(X_test)
mae_train_svr = mean_absolute_error(y_train, y_train_pred_svr)
mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
print(f"SVM - MAE (Train): {mae_train_svr}, MAE (Test): {mae_test_svr}")
print("Results for Support Vector Machine")

for i in range(5):
     print("Predictions:", y_train_pred_svr[i], "Actual values:", y_train[i])
      print("Predictions:", y_train_pred_svm[i], "Actual values:", y_train[i])

# Training and testing error for Decision Tree
y_train_pred_dt = best_model_dt.predict(X_train)
y_test_pred_dt = best_model_dt.predict(X_test)
mae_train_dt = mean_absolute_error(y_train, y_train_pred_dt)
mae_test_dt = mean_absolute_error(y_test, y_test_pred_dt)
print(f"Decision Tree - MAE (Train): {mae_train_dt}, MAE (Test): {mae_test_dt}")

print("Results for Decision Tree")

for i in range(5):
     print("Predictions:", y_train_pred_dt[i], "Actual values:", y_train[i])
      print("Predictions:", y_train_pred_dt[i], "Actual values:", y_train[i])

# Training and testing error for Random Forest
y_train_pred_rf = best_model_rf.predict(X_train)
y_test_pred_rf = best_model_rf.predict(X_test)
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
print(f"Random Forest - MAE (Train): {mae_train_rf}, MAE (Test): {mae_test_rf}")

print("Results for Random Forest")

for i in range(5):
     print("Predictions:", y_train_pred_rf[i], "Actual values:", y_train[i])
      print("Predictions:", y_train_pred_rf[i], "Actual values:", y_train[i])
     
     
     
    
# Step % Model Performance Analysis
#Step 5 Model Performance Analysis


print("Logistic Regression Evaluation Metrics:")
accuracy_logistic = accuracy_score(y_test_pred_logistic, y_test)
precision_logistic = precision_score(y_test, y_test_pred_logistic, average='macro', zero_division=1)  
f1_logistic = f1_score(y_test, y_test_pred_logistic, average='macro')  # or 'weighted'

print(f"Accuracy: {accuracy_logistic}")
print(f"Precision: {precision_logistic}")
print(f"F1 Score: {f1_logistic}")


print("\nSVM Evaluation Metrics:")
accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
precision_svm = precision_score(y_test, y_test_pred_svm, average='macro', zero_division=1)
f1_svm = f1_score(y_test, y_test_pred_svm, average='macro')

print(f"Accuracy: {accuracy_svm}")
print(f"Precision: {precision_svm}")
print(f"F1 Score: {f1_svm}")

print("\nDecision Tree Evaluation Metrics:")
accuracy_dt = accuracy_score(y_test_pred_dt, y_test)
precision_dt = precision_score(y_test, y_test_pred_dt, average='macro', zero_division=1)
f1_dt = f1_score(y_test, y_test_pred_dt, average='macro')

print(f"Accuracy: {accuracy_dt}")
print(f"Precision: {precision_dt}")
print(f"F1 Score: {f1_dt}")

print("\nRandom Forest Evaluation Metrics:")
accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
precision_rf = precision_score(y_test, y_test_pred_rf, average='macro', zero_division=1)
f1_rf = f1_score(y_test, y_test_pred_rf, average='macro')

print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"F1 Score: {f1_rf}")

conf_mx = confusion_matrix(y_test, y_test_pred_svm)

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mx, annot=True, fmt='d', cmap='Accent', 
            xticklabels=np.unique(y_test), 
            yticklabels=np.unique(y_test))

plt.title('Confusion Matrix for Support Vector Machine')
plt.show()



#Step 6 Stacked model Performance

# Choosing two models for stacking
est = [('svm', svm), ('rf',random_forest)]
# Logistic regression as the final estimator
stacked_model = StackingClassifier(estimators = est, final_estimator= LogisticRegression(max_iter=1000))
#Fitting the stacked model
stacked_model.fit(X_train, y_train)
y_train_pred_stacked = stacked_model.predict(X_train)
y_test_pred_stacked = stacked_model.predict(X_test)


print("\nStacked Model Evaluation Metrics:")
accuracy_stacked = accuracy_score(y_test, y_test_pred_stacked)
precision_stacked = precision_score(y_test, y_test_pred_stacked, average='macro', zero_division=1)
f1_stacked = f1_score(y_test, y_test_pred_stacked, average='macro')

print(f"Accuracy: {accuracy_stacked}")
print(f"Precision: {precision_stacked}")
print(f"F1 Score: {f1_stacked}")



conf_mx = confusion_matrix(y_test, y_test_pred_stacked)

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mx, annot=True, fmt='d', cmap='Accent', 
            xticklabels=np.unique(y_test), 
            yticklabels=np.unique(y_test))

plt.title('Confusion Matrix for Stacked model')
plt.show()



#Step 7 Prediction

jb.dump(best_model_svm, 'best_model_svr.joblib')
print("Model saved as 'sbest_model_svr.joblib'")

# Load the model
loaded_model = jb.load('best_model_svr.joblib')
print("Model loaded successfully.")

# Define the coordinates for prediction
coordinates = np.array([[9.375, 3.0625, 1.51],
                        [6.995, 5.125, 0.3875],
                        [0, 3.0625, 1.93],
                        [9.4, 3, 1.8],
                        [9.4, 3, 1.3]])

# Convert coordinates to DataFrame with the same columns as training data
coordinates_df = pd.DataFrame(coordinates, columns=X_train.columns)

# Scale the input coordinates using the same scaler
scaled_coordinates = scaler.transform(coordinates_df)

scaled_coordinates_df = pd.DataFrame(scaled_coordinates, columns=X_train.columns)

predictions = loaded_model.predict(scaled_coordinates_df)

print("Predicted maintenance steps:", predictions)



