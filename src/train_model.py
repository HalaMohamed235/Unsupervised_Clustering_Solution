import pandas as pd
import pickle
import logging
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- LOGGING ----------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- MAIN PIPELINE ----------------
def train_pipeline():
    try:
        # Load data
        df = pd.read_csv("data/mall_customers.csv")
        logging.info("Data loaded successfully.")

        # Select features
        X = df[['Age', 'Annual_Income', 'Spending_Score']]
        logging.info("Features selected.")

        # ---------------- ADD THIS ----------------
        K_values = list(range(3, 9))
        WCSS = []
        silhouette_scores = []

        best_k = 0
        best_score = -1

        for k in K_values:
            model = KMeans(n_clusters=k, random_state=42)

            labels = model.fit_predict(X)

            # Silhouette
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)

            # Elbow (WCSS)
            wcss = model.inertia_
            WCSS.append(wcss)

            # Logging BOTH
            logging.info(f"k={k} | WCSS={wcss} | Silhouette={sil_score}")

            # Best k using silhouette
            if sil_score > best_score:
                best_score = sil_score
                best_k = k

        logging.info(f"Best k found: {best_k}")

        # Save metrics for Streamlit
        metrics_df = pd.DataFrame({
            "k": K_values,
            "WCSS": WCSS,
            "Silhouette": silhouette_scores
        })

        os.makedirs("models", exist_ok=True)
        metrics_df.to_csv("models/kmeans_metrics.csv", index=False)

        logging.info("Elbow and Silhouette metrics saved.")

        # ---------------- FINAL MODEL ----------------
        final_model = KMeans(n_clusters=best_k, random_state=42)
        final_model.fit(X)

        with open("models/mall_model.pkl", "wb") as f:
            pickle.dump({
                "model": final_model,
                "features": ['Age', 'Annual_Income', 'Spending_Score']
            }, f)

        logging.info("Model saved successfully.")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    train_pipeline()