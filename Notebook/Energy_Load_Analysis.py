import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df=pd.read_csv('../Data/energy_load.csv')
X=df[['hour','temperature','humidity','holiday']]
y=df['load']

model=RandomForestRegressor(random_state=42)
model.fit(X,y)

joblib.dump(model,'../Model/best_model.pkl')
print('Model saved')
