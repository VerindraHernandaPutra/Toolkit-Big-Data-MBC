import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def data_preprocessing_page():
    st.title("Data Preprocessing")

    if 'df' in st.session_state:
        df = st.session_state.df

        st.write("### Data Preview:")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        st.write("### Choose Preprocessing Task:")

        # Tabs for various preprocessing tasks
        tab1, tab2, tab3 = st.tabs(["🎌 Feature Scaling", "📑 Feature Engineering", "📑 Encoding"])

        # --------- Feature Scaling (Tab 1) ---------
        with tab1:
            feature_scaling(df)

        # --------- Feature Engineering (Tab 2) ---------
        with tab2:
            feature_engineering(df)

        # --------- Encoding (Tab 3) ---------
        with tab3:
            encoding(df)
    else:
        st.warning("⚠️ No data available. Please upload data on the Data page.")

def feature_scaling(df):
    st.subheader("Feature Scaling:")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        st.warning("⚠️ No numeric columns available for scaling.")
        return

    scaler_type = st.radio("Choose scaling method:", ["Min-Max Scaling", "Standardization", "Robust Scaling"])

    scaler = {
        "Min-Max Scaling": MinMaxScaler(),
        "Standardization": StandardScaler(),
        "Robust Scaling": RobustScaler()
    }[scaler_type]

    selected_columns = st.multiselect("Select columns to scale:", numeric_columns)

    if selected_columns and st.button("Apply Scaling"):
        df[selected_columns] = scaler.fit_transform(df[selected_columns])
        st.session_state.df = df
        st.success(f"✅ {scaler_type} applied.")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

def feature_engineering(df):
    st.subheader("Feature Engineering:")

    action = st.radio("Choose an action:", ["Feature Creation", "Feature Splitting", "Custom Code Execution"])

    if action == "Feature Creation":
        col1, col2 = st.selectbox("Select first column:", df.columns), st.selectbox("Select second column:", df.columns)
        new_feature_name = st.text_input("Enter new feature name:")
        operation = st.selectbox("Choose operation:", ["Add", "Subtract", "Multiply", "Divide"])

        if st.button("Create Feature") and new_feature_name:
            try:
                if operation == "Add":
                    df[new_feature_name] = df[col1] + df[col2]
                elif operation == "Subtract":
                    df[new_feature_name] = df[col1] - df[col2]
                elif operation == "Multiply":
                    df[new_feature_name] = df[col1] * df[col2]
                elif operation == "Divide":
                    df[new_feature_name] = df[col1] / (df[col2].replace(0, np.nan))
                st.session_state.df = df
                st.success(f"✅ Feature '{new_feature_name}' created.")
            except Exception as e:
                st.error(f"Error creating feature: {e}")

    elif action == "Feature Splitting":
        column = st.selectbox("Select column to split:", df.columns)
        delimiter = st.text_input("Enter delimiter for splitting:", value=" ")

        if st.button("Split Feature"):
            try:
                split_df = df[column].str.split(delimiter, expand=True)
                for i in range(split_df.shape[1]):
                    df[f"{column}_part{i+1}"] = split_df[i]
                st.session_state.df = df
                st.success(f"✅ Column '{column}' split successfully.")
            except Exception as e:
                st.error(f"Error splitting feature: {e}")

    elif action == "Custom Code Execution":
        st.write("Enter your Python code to perform custom feature engineering.")
        user_code = st.text_area("Write Python code here (df is your dataframe):", height=200)
        if st.button("Run Custom Code"):
            try:
                exec(user_code, {'df': df, 'np': np, 'pd': pd})
                st.session_state.df = df
                st.success("✅ Custom code executed successfully.")
            except Exception as e:
                st.error(f"Error executing code: {e}")

def encoding(df):
    st.subheader("Encoding Categorical Variables:")

    encoding_method = st.radio("Choose encoding method:", ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"])

    if encoding_method == "Label Encoding":
        column = st.selectbox("Select column for Label Encoding:", df.select_dtypes(include=['object']).columns)
        if st.button(f"Apply Label Encoding to {column}"):
            df[column] = df[column].astype('category').cat.codes
            st.session_state.df = df
            st.success(f"✅ Label Encoding applied to '{column}'.")
            st.dataframe(df.head(), use_container_width=True, hide_index=True)

    elif encoding_method == "One-Hot Encoding":
        column = st.selectbox("Select column for One-Hot Encoding:", df.select_dtypes(include=['object']).columns)
        if st.button(f"Apply One-Hot Encoding to {column}"):
            df = pd.get_dummies(df, columns=[column], drop_first=True)
            st.session_state.df = df
            st.success(f"✅ One-Hot Encoding applied to '{column}'.")
            st.dataframe(df.head(), use_container_width=True, hide_index=True)

    elif encoding_method == "Ordinal Encoding":
        column = st.selectbox("Select column for Ordinal Encoding:", df.select_dtypes(include=['object']).columns)
        categories = df[column].dropna().unique()

        ordered_categories = st.multiselect(
            "Reorder categories for encoding:",
            options=categories.tolist(),
            default=categories.tolist()
        )

        if ordered_categories and st.button(f"Apply Ordinal Encoding to '{column}'"):
            df[column] = df[column].astype(pd.CategoricalDtype(categories=ordered_categories, ordered=True)).cat.codes
            st.session_state.df = df
            st.success(f"✅ Ordinal Encoding applied to '{column}'.")
            st.dataframe(df.head(), use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Please reorder the categories before applying encoding.")
