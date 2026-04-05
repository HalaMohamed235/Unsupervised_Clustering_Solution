import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import logging
from src.predict_model import load_model, make_prediction, load_metrics


# ---------------- LOGGING ----------------
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- LOAD ----------------
model, features = load_model()
metrics = load_metrics()

# ---------------- UI ----------------
st.title("Mall Customer Segmentation")

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Customer Details")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0, value=50)
spending = st.number_input("Spending Score", min_value=0, max_value=100, value=50)

# ---------------- PREDICTION ----------------
if st.button("Predict Cluster"):
    try:
        input_data = {
            "Age": age,
            "Annual_Income": income,
            "Spending_Score": spending
        }

        cluster = make_prediction(model, features, input_data)

        st.success(f"Customer belongs to Cluster {cluster[0]}")
        logging.info(f"Prediction successful: Cluster {cluster[0]}")

    except Exception as e:
        st.error("Prediction failed. Check inputs.")
        logging.error(f"Prediction error: {e}")

# ---------------- METRICS VISUALIZATION ----------------
if st.button("Show Analysis"):
    st.subheader("Clustering Evaluation")

    st.write("### Elbow Method (WCSS)")
    st.line_chart(metrics.set_index("k")["WCSS"])

    
    # Elbow → approximate (largest drop in WCSS)
    wcss_diff = metrics["WCSS"].diff().abs()
    
    best_k_elbow = metrics.loc[wcss_diff.idxmax(), "k"]
    st.success(f"According to Elbow Method → Best k ≈ {best_k_elbow}")


    st.write("### Silhouette Scores")
    st.line_chart(metrics.set_index("k")["Silhouette"])

    # Silhouette → best (MAX)
    best_k_silhouette = metrics.loc[metrics["Silhouette"].idxmax(), "k"]
    best_sil_score = metrics["Silhouette"].max()

  

    st.success(f"According to Silhouette Method → Best k = {best_k_silhouette}")
    st.info(f"Silhouette Score = {best_sil_score:.3f}")

    
