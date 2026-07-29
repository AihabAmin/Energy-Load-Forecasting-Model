import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor


# Load Data
df = pd.read_csv("../Data/energy_load.csv")


# Features
X = df[['hour','temperature','humidity','holiday']]

# Target
y = df['load']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


models = {

"Linear Regression": LinearRegression(),

"Random Forest": RandomForestRegressor(
    random_state=42
),

"Gradient Boosting": GradientBoostingRegressor(
    random_state=42
),

"XGBoost": XGBRegressor(
    random_state=42
)

}


results = {}


for name, model in models.items():

    model.fit(X_train,y_train)

    prediction = model.predict(X_test)

    mape = mean_absolute_percentage_error(
        y_test,
        prediction
    )

    rmse = mean_squared_error(
        y_test,
        prediction
    ) ** 0.5


    results[name] = {
        "MAPE": round(mape*100,2),
        "RMSE": round(rmse,2)
    }


print("\nModel Comparison\n")

for model,result in results.items():

    print(
        model,
        "MAPE:",
        result["MAPE"],
        "%",
        "RMSE:",
        result["RMSE"]
    )


best = min(
    results,
    key=lambda x: results[x]["MAPE"]
)


print("\nBest Model:", best)