import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 
 
# Generate a synthetic dataset 
X,y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42) 
 
# Split the dataset into training and testing sets 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42) 
 
# Initialize the Linear Regression model 
linear_reg = LinearRegression() 
 
# Train the model on the training data 
linear_reg.fit(X_train, y_train) 
 
# Make predictions on the testing data 
predictions = linear_reg.predict(X_test) 
 
# Evaluate the model's performance 
mse = mean_squared_error(y_test, predictions) 
r2 = r2_score(y_test, predictions) 
print("Mean Squared Error (MSE):\n",mse) 
print("R-squared:\n",r2) 
# Plotting the regression line (optional) 
plt.scatter(X_test, y_test, color='blue') 
plt.plot(X_test, predictions, color='red', linewidth=3) 
plt.xlabel('X') 
plt.ylabel('y') 
plt.title('Linear Regression') 
plt.show()
