"""
Iris Dataset Explorer
A Streamlit app for exploring and visualizing the Iris dataset.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

# Configure page
st.set_page_config(page_title="Iris Dataset Explorer", layout="wide")

# Main title
st.title("🌸 Iris Dataset Explorer")

# Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(
    data=iris_data.data,
    columns=iris_data.feature_names
)
iris_df['Species'] = iris_data.target_names[iris_data.target]

# Display section: First rows of the dataset
st.header("Dataset Overview")
st.subheader("First 5 Rows")
st.dataframe(iris_df.head())

# Display section: Summary statistics
st.subheader("Summary Statistics")
st.dataframe(iris_df.describe())

# Display section: Dataset shape and info
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Rows", len(iris_df))
with col2:
    st.metric("Total Columns", len(iris_df.columns))

# Get numeric column names (exclude species)
numeric_columns = iris_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# User selection: Choose columns for visualization
st.header("Data Visualization")
selected_columns = st.multiselect(
    "Select numeric columns to visualize:",
    numeric_columns,
    default=numeric_columns[:2]
)

# Only proceed with visualization if columns are selected
if selected_columns:
    # Histogram section
    st.subheader("Histogram")
    histogram_column = st.selectbox(
        "Choose a column for histogram:",
        selected_columns
    )
    
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    ax_hist.hist(iris_df[histogram_column], bins=20, color='skyblue', edgecolor='black')
    ax_hist.set_xlabel(histogram_column)
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(f"Distribution of {histogram_column}")
    st.pyplot(fig_hist)
    
    # Scatter plot section
    st.subheader("Scatter Plot")
    col1_scatter, col2_scatter = st.columns(2)
    
    with col1_scatter:
        x_column = st.selectbox(
            "Choose X-axis column:",
            selected_columns,
            key="x_scatter"
        )
    
    with col2_scatter:
        y_column = st.selectbox(
            "Choose Y-axis column:",
            selected_columns,
            index=1 if len(selected_columns) > 1 else 0,
            key="y_scatter"
        )
    
    # Create scatter plot with species color coding
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    
    # Plot each species with a different color
    for species in iris_df['Species'].unique():
        species_data = iris_df[iris_df['Species'] == species]
        ax_scatter.scatter(
            species_data[x_column],
            species_data[y_column],
            label=species,
            s=100,
            alpha=0.7
        )
    
    ax_scatter.set_xlabel(x_column)
    ax_scatter.set_ylabel(y_column)
    ax_scatter.set_title(f"{x_column} vs {y_column}")
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    
    st.pyplot(fig_scatter)
else:
    st.warning("Please select at least one numeric column to visualize.")

# Footer
st.divider()
st.caption("Built with Streamlit | Dataset: Iris from scikit-learn")
