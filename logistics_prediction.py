"""
Week 4: Predictive Modeling and Optimization in Logistics Systems

Run:
    pip install -r requirements.txt
    python logistics_prediction.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/logistics_delivery_data.csv"


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return (
        mean_absolute_error(y_test, pred),
        np.sqrt(mean_squared_error(y_test, pred)),
        r2_score(y_test, pred),
    )


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="delivery_time_minutes")
    y = df["delivery_time_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    categorical = ["traffic_level", "weather", "vehicle_type"]
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    linear_model = Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression()),
    ])

    rf_model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )),
    ])

    print("\n=== Model Evaluation ===")
    for name, model in [
        ("Linear Regression", linear_model),
        ("Random Forest", rf_model),
    ]:
        model.fit(X_train, y_train)
        mae, rmse, r2 = evaluate(model, X_test, y_test)
        print(f"\n{name}")
        print(f"MAE  : {mae:.2f} minutes")
        print(f"RMSE : {rmse:.2f} minutes")
        print(f"R2   : {r2:.3f}")

    param_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 1.0],
    }

    search = RandomizedSearchCV(
        rf_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    mae, rmse, r2 = evaluate(best_model, X_test, y_test)

    print("\n=== Tuned Random Forest ===")
    print("Best parameters:", search.best_params_)
    print(f"Test MAE  : {mae:.2f} minutes")
    print(f"Test RMSE : {rmse:.2f} minutes")
    print(f"Test R2   : {r2:.3f}")

    feature_names = best_model.named_steps["prep"].get_feature_names_out()
    importance = pd.Series(
        best_model.named_steps["model"].feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print("\n=== Top 10 Feature Importances ===")
    print(importance.head(10))


if __name__ == "__main__":
    main()
