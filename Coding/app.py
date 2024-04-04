# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and description
st.title('Streamlit Tableau-like Dashboard')
st.write("""
    Welcome to the Streamlit Tableau-like Dashboard! 
    Upload your CSV file to visualize and explore your data interactively.
""")

# Upload CSV data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Sidebar
st.sidebar.header("Dashboard Options")

if uploaded_file is not None:
    # Read CSV data into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write('## Raw Data')
    st.write(df)

    # Sidebar for selecting visualization type
    visualization_type = st.sidebar.radio("Choose Visualization Type", ["Table", "Bar Chart", "Line Chart", "Scatter Plot", "Histogram"])

    if visualization_type == "Table":
        # Table view
        st.subheader('Table View')
        st.write("This table displays the raw data. Use it to view all the details in tabular format.")

    elif visualization_type == "Bar Chart":
        # Bar chart
        st.subheader('Bar Chart')
        selected_column = st.sidebar.selectbox('Select column for X-axis', df.columns)
        st.write("""
            This bar chart represents the frequency/count of values in the selected column.
            Use the sidebar to choose the column for the X-axis.
        """)
        barplot_data = df[selected_column].value_counts()
        st.bar_chart(barplot_data)

    elif visualization_type == "Line Chart":
        # Line chart
        st.subheader('Line Chart')
        selected_column_x = st.sidebar.selectbox('Select column for X-axis', df.columns)
        selected_column_y = st.sidebar.selectbox('Select column for Y-axis', df.columns)
        st.write("""
            This line chart visualizes the relationship between two numerical columns.
            Use the sidebar to choose the columns for the X-axis and Y-axis.
        """)
        st.line_chart(df[[selected_column_x, selected_column_y]])

    elif visualization_type == "Scatter Plot":
        # Scatter plot
        st.subheader('Scatter Plot')
        selected_column_x = st.sidebar.selectbox('Select column for X-axis', df.columns)
        selected_column_y = st.sidebar.selectbox('Select column for Y-axis', df.columns)
        st.write("""
            This scatter plot shows the relationship between two numerical columns.
            Use the sidebar to choose the columns for the X-axis and Y-axis.
        """)
        st.scatter_chart(df[[selected_column_x, selected_column_y]])

    elif visualization_type == "Histogram":
        # Histogram
        st.subheader('Histogram')
        selected_column = st.sidebar.selectbox('Select column', df.columns)
        st.write("""
            This histogram displays the distribution of values in the selected column.
            Use the sidebar to choose the column.
        """)
        plt.figure(figsize=(8, 6))
        sns.histplot(df[selected_column], bins=20, kde=True)
        st.pyplot()

# Conclusion
st.write("""
    ## Conclusion
    Explore your data effectively using this Streamlit Tableau-like Dashboard.
    Choose different visualization types from the sidebar and customize them as per your requirements.
""")
