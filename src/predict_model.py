import pickle
import logging
import pandas as pd

def load_model(path="models/mall_model.pkl"):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        logging.info("Model loaded successfully.")
        return data["model"], data["features"]

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def load_metrics(path="models/kmeans_metrics.csv"):
    try:
        metrics = pd.read_csv(path)
        logging.info("Metrics loaded successfully.")
        return metrics

    except Exception as e:
        logging.error(f"Error loading metrics: {e}")
        raise


def make_prediction(model, features, input_dict):
    try:
        input_df = pd.DataFrame([input_dict])

        # Ensure correct order
        input_df = input_df[features]

        cluster = model.predict(input_df)

        return cluster

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise