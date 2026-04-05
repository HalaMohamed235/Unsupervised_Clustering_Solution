# Mall Customer Segmentation (Unsupervised Clustering)

## Project Overview
In the competitive retail environment, malls aim to increase sales by understanding their customers' behavior. 
This project uses Unsupervised Machine Learning to segment mall customers into distinct groups based on their spending patterns and income levels.

By identifying these clusters, a supermarket or mall owner can develop targeted marketing strategies for the right audience. 
This solution has been modularized from a Jupyter Notebook and deployed as an interactive Streamlit application.

## Live Demo
[Insert your Streamlit Cloud Link Here]

## Features
- **Unsupervised Learning**: Implements the K-Means Clustering algorithm to find natural groupings in customer data.
- **Data-Driven Insights**: Helps businesses identify "Big Spenders," "Budget Conscious," and "Target" customer segments.
- **Metric Persistence**: Stores clustering metrics (K-Means inertia/scores) in `kmeans_metrics.csv` for analysis.
- **Interactive Visualization**: The Streamlit UI allows users to see which cluster a new customer falls into based on their profile.
- **Modular Architecture**: Professional code organization with separate modules for training, inference, and logging.

## Folder Structure
```text
Unsupervised_Clustering_Solution/
├── app/
│   └── app.py              # Streamlit dashboard for segmenting customers
├── data/
│   └── mall_customers.csv  # Dataset containing hypothetical customer profiles
├── logs/
│   └── app.log             # Tracks model loading and user sessions
├── models/
│   ├── mall_model.pkl      # Serialized K-Means model
│   └── kmeans_metrics.csv  # Metrics used to validate the clustering
├── src/
│   ├── __init__.py
│   ├── train_model.py      # Logic for finding optimal clusters (Elbow Method/K-Means)
│   └── predict_model.py    # Functions to assign new data points to a cluster
├── requirements.txt        # Python dependencies (Scikit-learn, Pandas, Streamlit)
└── README.md               # Project documentation
```
## Input Variables
The model segments customers based on the following key metrics:
- **Age**: The age of the customer.
- **Annual Income (k$)**: The yearly income of the customer.
- **Spending Score (1-100)**: A score assigned by the mall based on customer behavior and spending nature.

## Installation & Usage

1. Clone the Repository:
   git clone https://github.com/YOUR_USERNAME/Unsupervised_Clustering_Solution.git
   cd Unsupervised_Clustering_Solution

2. Install Dependencies:
   pip install -r requirements.txt

3. Run the App Locally:
   streamlit run app/app.py

## Implementation Details
- **Algorithm**: The project utilizes K-Means Clustering. The optimal number of clusters (K) was determined during the training phase to ensure meaningful segmentation.
- **Modularization**: The training script (`train_model.py`) is designed to be re-run if new data is collected, automatically updating the serialized model in the `models/` folder.
- **Business Logic**: Unlike supervised learning, this model provides descriptive insights, allowing the mall owner to label clusters (e.g., "High Income, High Spending").