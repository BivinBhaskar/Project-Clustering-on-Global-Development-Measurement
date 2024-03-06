import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load your data here
data = pd.read_excel(r"C:\Users\Rehaman shaik\data_new.xlsx")

def preprocess_data(data):
    # Perform any necessary data preprocessing
    return data

def run_kmeans(data, num_clusters, features):
    # Assuming features is a list of selected features
    X = data[features].values

    # Standardize the features
    scaler = StandardScaler()
    data_X = scaler.fit_transform(X)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_X)
    return data

def plot_clusters(data, x_feature, y_feature):
    fig, ax = plt.subplots()

    # Generate distinct colors based on the number of clusters
    num_clusters = len(data['Cluster'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    for i, cluster in enumerate(data['Cluster'].unique()):
        clustered_data = data[data['Cluster'] == cluster]
        ax.scatter(
            clustered_data[x_feature],
            clustered_data[y_feature],
            color=colors[i],
            label=f'Cluster {cluster}'
        )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title('Clusters of Measurements')
    ax.legend()
    return fig

def main():
    # Load your data here
    # data = pd.read_excel("your_file.xlsx")
    st.title("Clustering Model Deployment")

    # Assuming data preprocessing function is defined
    # data = preprocess_data(data)

    st.sidebar.header("Settings")
    features = st.sidebar.multiselect("Select Features", data.columns)
    
    if len(features) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()

    num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

    st.write(f"Running KMeans with {num_clusters} clusters...")

    # Assuming the run_kmeans function is defined
    clustered_data = run_kmeans(data.copy(), num_clusters, features)

    # Assuming the plot_clusters function is defined
    fig = plot_clusters(clustered_data, features[0], features[1])

    st.pyplot(fig)

    silhouette_avg = silhouette_score(data[features], clustered_data['Cluster'])
    st.write(f"Silhouette Score: {silhouette_avg}")

if __name__ == "__main__":
    main()