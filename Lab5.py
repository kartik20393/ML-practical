import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_iris 
import pandas as pd 
# Load the Iris dataset 
iris = load_iris() 
# Convert the dataset to a pandas DataFrame 
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
iris_df['target'] = iris.target 
# Scatter plot using Matplotlib 
plt.figure(figsize=(8, 6)) 
plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], 
c=iris_df['target'], cmap='viridis', s=80, alpha=0.7) 
plt.xlabel('Sepal Length (cm)') 
plt.ylabel('Sepal Width (cm)') 
plt.title('Scatter Plot of Sepal Length vs Sepal Width') 
plt.colorbar(label='Species') 
plt.show() 
# Bar chart using Seaborn 
plt.figure(figsize=(8, 6)) 
sns.countplot(x='target', data=iris_df, palette='viridis') 
plt.xlabel('Species') 
plt.ylabel('Count') 
plt.title('Bar Chart: Count of Each Species') 
plt.show()
