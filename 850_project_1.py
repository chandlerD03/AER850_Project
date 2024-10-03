# 850_project_1

#importing all the packages 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

sns.heatmap(np.abs(corr_matrix))




