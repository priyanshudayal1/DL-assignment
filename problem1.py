import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection (Assume or create data)
# Assuming a dataset with Price, Popularity, and Demand
data = {
    'Price': [10, 20, 15, 30, 25, 35, 40, 50, 45, 60],
    'Popularity': [50, 70, 65, 80, 75, 85, 90, 95, 85, 100],
    'Demand': [200, 180, 210, 160, 170, 150, 140, 130, 135, 120]
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Visualize the data
sns.pairplot(df)
plt.show()

# Step 3: Model Creation
# Define the features (Price and Popularity) and target (Demand)
X = df[['Price', 'Popularity']]
y = df['Demand']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Step 4: Prediction & Analysis
# Predict on the test set
y_pred = model.predict(X_test)

# Compare predicted and actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df)

# Visualize the comparison
comparison_df.plot(kind='bar')
plt.title('Actual vs Predicted Demand')
plt.xlabel('Test Data Points')
plt.ylabel('Demand')
plt.show()

# Analysis of coefficients
print("Coefficients of the model:", model.coef_)
print("Intercept of the model:", model.intercept_)

# Making predictions for custom input
price = 55
popularity = 85
predicted_demand = model.predict([[price, popularity]])
print(f"Predicted Demand for Price = {price} and Popularity = {popularity}: {predicted_demand[0]}")

