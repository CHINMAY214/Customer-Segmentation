import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Custom HTML & CSS Styling
st.markdown("""
    <style>
        body {
            background-color: #F5F5F5;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
        }
        .title {
            color: #4CAF50;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .sidebar {
            background-color: #333;
            padding: 20px;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Title
st.markdown('<p class="title">📊 Customer Segmentation Dashboard</p>', unsafe_allow_html=True)

# Sidebar Design
st.sidebar.markdown('<p class="sidebar">🔍 Upload Your Data & Choose Settings</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### 📂 Uploaded Dataset", df.head())

    # Selecting numeric features
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        st.error("Dataset must have at least two numeric columns for clustering.")
    else:
        # Feature selection
        selected_features = st.sidebar.multiselect("🎯 Select Features for Clustering", numeric_columns, default=numeric_columns[:2])

        if len(selected_features) >= 2:
            X = df[selected_features]

            # Normalize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Select clustering algorithm
            algorithm = st.sidebar.radio("⚙️ Choose Clustering Algorithm", ["K-Means", "Hierarchical", "DBSCAN"])

            if algorithm == "K-Means":
                # Elbow Method to suggest best K
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)

                fig = plt.figure(figsize=(6, 4))
                plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='blue')
                plt.xlabel("Number of Clusters")
                plt.ylabel("WCSS")
                plt.title("📈 Elbow Method to Find Optimal K")
                st.pyplot(fig)

                # User selects K
                k = st.sidebar.slider("🔢 Select Number of Clusters (K)", min_value=2, max_value=10, value=3)

                # Apply K-Means
                kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
                df["Cluster"] = kmeans.fit_predict(X_scaled)

            elif algorithm == "Hierarchical":
                # Display Dendrogram
                st.write("### 🛠 Dendrogram for Hierarchical Clustering")
                plt.figure(figsize=(8, 5))
                dendrogram(linkage(X_scaled, method="ward"))
                plt.title("Dendrogram")
                plt.xlabel("Data Points")
                plt.ylabel("Euclidean Distance")
                st.pyplot(plt)

                # User selects number of clusters
                k = st.sidebar.slider("🔢 Select Number of Clusters", min_value=2, max_value=10, value=3)

                # Apply Hierarchical Clustering
                hc = AgglomerativeClustering(n_clusters=k, affinity="euclidean", linkage="ward")
                df["Cluster"] = hc.fit_predict(X_scaled)

            elif algorithm == "DBSCAN":
                # User selects parameters
                eps = st.sidebar.slider("🌎 Select EPS (Neighborhood Distance)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
                min_samples = st.sidebar.slider("👥 Select Min Samples", min_value=2, max_value=10, value=5)

                # Apply DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                df["Cluster"] = dbscan.fit_predict(X_scaled)

            # Display clustered data
            st.write("### 📊 Clustered Data", df.head())

            # Cluster visualization using Plotly
            if len(selected_features) == 2:
                fig = px.scatter(df, x=selected_features[0], y=selected_features[1], color=df["Cluster"].astype(str),
                                 title=f"{algorithm} - Customer Segmentation", 
                                 color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig)

            # Download segmented data
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Clustered Data", data=csv, file_name="segmented_data.csv", mime="text/csv")

else:
    st.sidebar.info("📂 Upload a CSV file to begin.")
