import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression


# Load Data
df = pd.read_csv("../Data/energy_load.csv")


# Features
X = df[['hour','temperature','humidity','holiday']]

# Target
y = df['load']


# Best Model
model = LinearRegression()


# Train
model.fit(X,y)


# Save Model
joblib.dump(
    model,
    "../Model/best_model.pkl"
)


print("Best Model Saved Successfully")