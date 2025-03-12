import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit app title
st.title("📊 Customer Segmentation with K-Means")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    # Selecting numeric features
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.error("Dataset must have at least two numeric columns for clustering.")
    else:
        # Feature selection
        selected_features = st.multiselect("Select features for clustering", numeric_columns, default=numeric_columns[:2])
        
        if len(selected_features) >= 2:
            X = df[selected_features]
            
            # Normalize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Choosing number of clusters
            k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3)

            # Apply K-Means
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
            df["Cluster"] = kmeans.fit_predict(X_scaled)

            # Display results
            st.write("### Clustered Data", df.head())

            # Visualize Clusters
            if len(selected_features) == 2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df["Cluster"], palette="viridis", s=100)
                plt.title("Customer Segmentation")
                plt.xlabel(selected_features[0])
                plt.ylabel(selected_features[1])
                st.pyplot(fig)
            
            # Show cluster counts
            st.write("### Cluster Distribution")
            st.bar_chart(df["Cluster"].value_counts())

else:
    st.info("Upload a CSV file to begin.")
