"""Assignment-6: Weather Condition Classification using SVM and Open-Meteo API.

This script fetches hourly weather observations from Open-Meteo, prepares a
classification dataset, trains an RBF-kernel SVM model, and evaluates it.
"""

from __future__ import annotations

import json
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


API_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE = 28.6139
LONGITUDE = 77.2090
HOURLY_PARAMS = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
FORECAST_DAYS = 7
RANDOM_STATE = 42
LOCATIONS = [
    {"name": "Delhi", "latitude": 28.6139, "longitude": 77.2090},
    {"name": "London", "latitude": 51.5072, "longitude": -0.1276},
    {"name": "Reykjavik", "latitude": 64.1466, "longitude": -21.9426},
    {"name": "Sydney", "latitude": -33.8688, "longitude": 151.2093},
]


def fetch_weather_data() -> pd.DataFrame:
    """Fetch weather observations from Open-Meteo and return a DataFrame."""

    frames: list[pd.DataFrame] = []

    for location in LOCATIONS:
        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "hourly": HOURLY_PARAMS,
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
        }

        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        hourly_data = pd.DataFrame(payload["hourly"])
        hourly_data["location"] = location["name"]
        frames.append(hourly_data)

    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and label the dataset for SVM classification."""

    required_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
    ]
    cleaned = df[required_columns].copy()

    print("Missing values before cleaning:\n", cleaned.isnull().sum())
    cleaned = cleaned.dropna().reset_index(drop=True)

    cleaned["Weather_Class"] = cleaned["temperature_2m"].apply(
        lambda value: "Warm" if value >= 25 else "Cool"
    )
    cleaned["Weather_Class_Encoded"] = LabelEncoder().fit_transform(
        cleaned["Weather_Class"]
    )

    return cleaned


def train_and_evaluate(df: pd.DataFrame) -> dict[str, float]:
    """Train an RBF SVM model and return the evaluation metrics."""

    feature_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
    ]

    x = df[feature_columns]
    y = df["Weather_Class_Encoded"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = SVC(kernel="rbf", random_state=RANDOM_STATE)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    print("\nEvaluation Metrics")
    for name, value in metrics.items():
        print(f"{name.title()}: {value:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    return metrics


def main() -> None:
    weather_df = fetch_weather_data()

    print("First five records:")
    print(pd.DataFrame(weather_df).head())

    print("\nInput Features:")
    print("temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m")
    print("Target Variable: Weather_Class")

    prepared_df = prepare_dataset(weather_df)

    print("\nPrepared dataset preview:")
    print(prepared_df.head())
    print("\nClass distribution:")
    print(prepared_df["Weather_Class"].value_counts())

    metrics = train_and_evaluate(prepared_df)

    print("\nThree observations based on model performance:")
    print("1. The SVM model can separate Cool and Warm weather well when temperature is strongly informative.")
    print("2. Standardization is essential because SVM depends on feature distance in a transformed space.")
    print("3. Misclassifications usually occur when temperatures are near the 25°C threshold or when other features overlap.")

    print("\nSummary JSON:")
    print(json.dumps(metrics, indent=2))

    conclusion = (
        "This experiment showed that Open-Meteo data can be used to classify weather conditions into Cool and Warm "
        "categories with an SVM classifier. Temperature was the most important feature because it directly defined the "
        "target label, while humidity, pressure, and wind speed helped shape the decision boundary. Feature scaling was "
        "critical for SVM because the algorithm is sensitive to differences in feature magnitude; without scaling, larger "
        "valued variables can dominate the model. One advantage of SVM is that it performs well on small to medium-sized "
        "datasets with clear class separation. One limitation is that it can become less interpretable and less efficient "
        "on large datasets or highly overlapping classes."
    )
    print("\nConclusion (100-150 words):")
    print(conclusion)


if __name__ == "__main__":
    main()
