# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Disable the warning about calling st.pyplot() without arguments
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title
st.title('🚀 Streamlit Funky Data Explorer 🎨')

# Main Description
st.write("""
    Welcome to the Streamlit Funky Data Explorer! 
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

    # Interactive Filtering
    st.sidebar.subheader("Interactive Filtering")
    selected_column = st.sidebar.selectbox("Select column for filtering", df.columns)
    filter_min, filter_max = st.sidebar.slider(f"Filter {selected_column}", float(df[selected_column].min()), float(df[selected_column].max()), (float(df[selected_column].min()), float(df[selected_column].max())))
    filtered_df = df[(df[selected_column] >= filter_min) & (df[selected_column] <= filter_max)]
    st.write('## Filtered Data')
    st.write(filtered_df)
    st.write("""
        Use the sidebar to select a column and adjust the slider to filter the data based on the selected column's values.
        The filtered data will be displayed below.
    """)

    # Basic data statistics
    st.write('## Summary Statistics')
    st.write(df.describe())
    st.write("""
        This section provides basic statistical information about the dataset such as mean, standard deviation, minimum, maximum, etc.
    """)

    # Visualize missing values
    st.write('## Missing Values')
    st.write(df.isnull().sum())
    st.write("""
        This section shows the count of missing values for each column in the dataset.
    """)

    # Data types
    st.write('## Data Types')
    st.write(df.dtypes)
    st.write("""
        This section displays the data types of each column in the dataset.
    """)

    # Outlier Detection and Removal
    st.sidebar.subheader("Outlier Detection and Removal")
    outlier_columns = st.sidebar.multiselect("Select columns for outlier detection", df.select_dtypes(include=np.number).columns.tolist())
    for col in outlier_columns:
        outlier_threshold = st.sidebar.slider(f"Outlier threshold for {col}", float(df[col].min()), float(df[col].max()), (float(df[col].min()), float(df[col].max())))
        filtered_df = filtered_df[(filtered_df[col] >= outlier_threshold[0]) & (filtered_df[col] <= outlier_threshold[1])]
    st.write('## Data After Outlier Removal')
    st.write(filtered_df)
    st.write("""
        Use the sidebar to select numerical columns and adjust the sliders to set outlier removal thresholds.
        The data after outlier removal will be displayed below.
    """)

    # Data Sampling
    st.sidebar.subheader("Data Sampling")
    sample_size = st.sidebar.number_input("Enter sample size", min_value=1, max_value=len(filtered_df))
    sampled_data = filtered_df.sample(n=sample_size, random_state=42)
    st.write('## Sampled Data')
    st.write(sampled_data)
    st.write("""
        Use the sidebar to specify the sample size, and the sampled data will be displayed below.
    """)

    # Correlation heatmap
    st.write('## Correlation Heatmap')
    numeric_df = filtered_df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns found for correlation calculation.")
    st.write("""
        This section displays a correlation heatmap for numerical columns.
        Stronger correlations are indicated by colors closer to 1 or -1.
    """)

    # Distribution of numerical features
    st.write('## Distribution of Numerical Features')
    num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        st.write(f"### {col}")
        plt.figure(figsize=(8, 6))
        sns.histplot(filtered_df[col], bins=20, kde=True)
        st.pyplot()
    st.write("""
        This section displays the distribution of numerical features using histograms.
        Adjust the bin size using the slider to control the level of detail in the histograms.
    """)

    # Pairplot for numerical features
    st.write('## Pairplot for Numerical Features')
    plt.figure(figsize=(10, 8))
    sns.pairplot(filtered_df[num_cols])
    st.pyplot()
    st.write("""
        This section displays pair plots for numerical features, allowing you to visualize relationships between pairs of variables.
    """)

    # Boxplot for categorical features
    cat_cols = filtered_df.select_dtypes(include='object').columns.tolist()
    st.write('## Boxplot for Categorical Features')
    for col in cat_cols:
        st.write(f"### {col}")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=col, y=num_cols[0], data=filtered_df)
        st.pyplot()
    st.write("""
        This section displays box plots for categorical features against a selected numerical feature.
    """)

    # Data Export
    st.write('## Data Export')
    export_format = st.selectbox("Select export format", ["CSV", "Excel"])
    if st.button("Export Data"):
        if export_format == "CSV":
            filtered_df.to_csv("exported_data.csv", index=False)
            st.success("Data exported successfully as CSV.")
        elif export_format == "Excel":
            filtered_df.to_excel("exported_data.xlsx", index=False)
            st.success("Data exported successfully as Excel.")
    st.write("""
        Use the dropdown menu to select the export format, and click the button to export the filtered data.
    """)

    # Machine Learning Integration (Dummy)
    st.write('## Machine Learning Integration (Dummy)')
    st.sidebar.subheader("Dummy Classifier")
    target_column = st.sidebar.selectbox("Select target column", filtered_df.columns)
    if st.sidebar.button("Train Dummy Classifier"):
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X = filtered_df.drop(columns=[target_column])
        y = filtered_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dummy_classifier = DummyClassifier(strategy="most_frequent")
        dummy_classifier.fit(X_train, y_train)
        y_pred = dummy_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f"Accuracy: {accuracy:.2f}")

# Conclusion
st.write("""
    ## Conclusion
    Explore and analyze your data effectively using the Streamlit Funky Data Explorer!
    Upload your CSV file, apply interactive filtering, visualize data distributions, detect outliers, export data, and even train a dummy classifier.
    Enjoy exploring your data with Streamlit!
""")
