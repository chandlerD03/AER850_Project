# 850_project_1

#importing all the packages 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#step 3: Data Visualization 
#first data must be organized in x and y since we are doing supervised learning
#step 3: Correlation Analysis

X_columns = df[['X', 'Y', 'Z']]  # Features (make sure these are columns in your DataFrame)
y_column = df['Step']  # Target (make sure this is a column in your DataFrame)

X_train, X_test, y_train, y_test = train_test_split(X_columns, y_column ,
                                                    test_size = 0.2,
                                                    random_state = 42);



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



