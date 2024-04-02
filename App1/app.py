# Step 1: Setup
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Set page configuration
st.set_page_config(layout="wide")

# Step 2: Data Loading
@st.cache
def load_data():
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df["WineType"] = [wine.target_names[t] for t in wine.target]
    return wine_df

# Load the data
wine_df = load_data()

# Step 3: Widgets
# Title
st.title("Wine Dataset Analysis Dashboard")

# Widgets for scatter plot
st.sidebar.header("Scatter Plot Configuration")
x_axis = st.sidebar.selectbox("X-Axis", wine_df.columns[:-1])
y_axis = st.sidebar.selectbox("Y-Axis", wine_df.columns[:-1])
color_encode = st.sidebar.checkbox("Color Encode by WineType")

# Widgets for bar chart
st.sidebar.header("Bar Chart Configuration")
bar_features = st.sidebar.multiselect("Select features for Bar Chart", wine_df.columns[:-1], default=["alcohol"])

# Step 4: Visualization
# Scatter plot
st.subheader("Scatter Plot")
scatter_fig = px.scatter(wine_df, x=x_axis, y=y_axis, color="WineType" if color_encode else None,
                         width=800, height=600, title=f"{x_axis.capitalize()} vs {y_axis.capitalize()}")
st.plotly_chart(scatter_fig)

# Bar chart
st.subheader("Bar Chart")
bar_fig, ax = plt.subplots(figsize=(10, 6))
wine_df.groupby("WineType")[bar_features].mean().plot(kind="bar", ax=ax)
plt.title("Average Ingredients Per Wine Type")
plt.xlabel("WineType")
plt.ylabel("Average Value")
st.pyplot(bar_fig)

# Histogram
st.subheader("Histogram")
hist_fig, ax = plt.subplots(figsize=(10, 6))
for feature in wine_df.columns[:-1]:
    sns.histplot(wine_df, x=feature, hue="WineType", element="step", kde=True, ax=ax, alpha=0.7, legend=True)
plt.title("Histogram of Features")
plt.xlabel("Value")
plt.ylabel("Frequency")
st.pyplot(hist_fig)


# Step 5: Run the App
if __name__ == "__main__":
    st.sidebar.title("Wine Dataset Analysis Dashboard")
    st.sidebar.write("Explore the Wine dataset with various visualizations.")
