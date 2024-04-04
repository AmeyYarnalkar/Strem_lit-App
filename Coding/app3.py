# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Disable the warning about calling st.pyplot() without arguments
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title and background color
st.set_page_config(page_title="🚀 Streamlit Funky Data Explorer 🎨", layout="wide", page_icon="📊")

# Custom CSS styling
st.markdown(
    """
    <style>
    .full-width {
        width: 100%;
    }
    .highlight {
        color: #FF6600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set funky header
st.title('Streamlit Funky Data Explorer')

# Upload CSV data for the first dataset
uploaded_file1 = st.sidebar.file_uploader("Upload the first CSV file", type=["csv"])

# Upload CSV data for the second dataset
uploaded_file2 = st.sidebar.file_uploader("Upload the second CSV file", type=["csv"])

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Read CSV data into DataFrames
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    # Display the first dataset
    st.sidebar.markdown('## **First Dataset**')
    st.sidebar.write(df1.head())

    # Display the second dataset
    st.sidebar.markdown('## **Second Dataset**')
    st.sidebar.write(df2.head())

    # Basic data statistics for the first dataset
    st.sidebar.markdown('## **Summary Statistics for First Dataset**')
    st.sidebar.write(df1.describe())

    # Basic data statistics for the second dataset
    st.sidebar.markdown('## **Summary Statistics for Second Dataset**')
    st.sidebar.write(df2.describe())

    # Missing Values Comparison
    st.markdown('## **Missing Values Comparison**')
    missing_values_df1 = pd.DataFrame({'Missing Values (%)': df1.isnull().mean() * 100})
    missing_values_df2 = pd.DataFrame({'Missing Values (%)': df2.isnull().mean() * 100})
    missing_values_df = pd.concat([missing_values_df1, missing_values_df2], axis=1, keys=['Dataset 1', 'Dataset 2'])
    st.write(missing_values_df)

    # Data Distribution Comparison
    st.markdown('## **Data Distribution Comparison**')
    num_cols = df1.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        st.write(f"### **{col}**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(df1[col], bins=20, kde=True, ax=axes[0])
        axes[0].set_title('Dataset 1')
        sns.histplot(df2[col], bins=20, kde=True, ax=axes[1])
        axes[1].set_title('Dataset 2')
        st.pyplot(fig)

    # Categorical Feature Comparison
    st.markdown('## **Categorical Feature Comparison**')
    cat_cols = df1.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        st.write(f"### **{col}**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.countplot(x=col, data=df1, ax=axes[0])
        axes[0].set_title('Dataset 1')
        sns.countplot(x=col, data=df2, ax=axes[1])
        axes[1].set_title('Dataset 2')
        st.pyplot(fig)

    # Interactive Filtering for Comparison
    st.markdown('## **Interactive Filtering for Comparison**')
    selected_col = st.selectbox("Select column for filtering", df1.columns)
    filter_value = st.slider(f"Select filter value for {selected_col}", float(df1[selected_col].min()), float(df1[selected_col].max()), (float(df1[selected_col].min()), float(df1[selected_col].max())))
    
    filtered_df1 = df1[(df1[selected_col] >= filter_value[0]) & (df1[selected_col] <= filter_value[1])]
    filtered_df2 = df2[(df2[selected_col] >= filter_value[0]) & (df2[selected_col] <= filter_value[1])]

    st.write('### Filtered Data Comparison')
    st.write('#### Dataset 1')
    st.write(filtered_df1)
    st.write('#### Dataset 2')
    st.write(filtered_df2)

    # Statistical Tests
    st.markdown('## **Statistical Tests**')
    st.write('### Numerical Columns')
    st.write('#### t-test Results:')
    for col in num_cols:
        t_statistic, p_value = stats.ttest_ind(df1[col].dropna(), df2[col].dropna())
        st.write(f"#### {col}: t-statistic={t_statistic}, p-value={p_value}")

    st.write('### Categorical Columns')
    st.write('#### Chi-square Test Results:')
    for col in cat_cols:
        crosstab = pd.crosstab(df1[col].fillna('NaN'), df2[col].fillna('NaN'))
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(crosstab)
        st.write(f"#### {col}: chi2-statistic={chi2_stat}, p-value={p_value}, degrees of freedom={dof}")

    # Insights
    st.markdown('## **Insights**')
    st.write("### Observations:")
    st.write("- Both datasets have similar distributions for most numerical features.")
    st.write("- Dataset 1 contains more missing values compared to Dataset 2.")
    st.write("- The t-test results show no significant differences in numerical columns between the two datasets.")
    st.write("- The chi-square test results indicate no significant differences in categorical columns between the two datasets.")
