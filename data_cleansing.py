import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import io

def handle_missing_values(df):
    tab1, tab2 = st.tabs(["📈 Deletion", "☣️ Imputation"])
    missing_columns = df.columns[df.isnull().any()].tolist()

    if missing_columns:
        tab1.warning(f"⚠️ Columns with missing values: {', '.join(missing_columns)}")
        tab2.warning(f"⚠️ Columns with missing values: {', '.join(missing_columns)}")
    else:
        tab1.success("✅ No missing values detected.")
        tab2.success("✅ No missing values detected.")
        return

    selected_columns = tab1.multiselect("Select columns to remove missing values:", missing_columns, key="remove_missing_select")
    if tab1.button("Remove Selected Missing Values", key="remove_missing"):
        df_cleaned = df.dropna(subset=selected_columns)
        st.session_state.df = df_cleaned
        tab1.success("✅ Selected Missing Values Removed.")
        tab1.write(df_cleaned.head())

    tab2.write("### Choose Columns & Imputation Method:")
    selected_columns = tab2.multiselect("Select columns to clean:", missing_columns, key="column_select")
    impute_method = tab2.radio("Select an Imputation Method:", ["Mean", "Median", "Mode", "Custom Value"], horizontal=True, key="impute_radio")
    
    custom_value = None
    if impute_method == "Custom Value":
        custom_value = tab2.text_input("Enter Custom Value:", key="custom_value_input")
    
    if tab2.button("Apply Imputation", key="apply_missing"):
        df_cleaned = df.copy()
        for column in selected_columns:
            if impute_method == "Mean" and df[column].dtype in ['float64', 'int64']:
                df_cleaned[column] = df[column].fillna(df[column].mean())
            elif impute_method == "Median" and df[column].dtype in ['float64', 'int64']:
                df_cleaned[column] = df[column].fillna(df[column].median())
            elif impute_method == "Mode":
                df_cleaned[column] = df[column].fillna(df[column].mode()[0])
            elif impute_method == "Custom Value" and custom_value is not None:
                df_cleaned[column] = df[column].fillna(custom_value)
        
        st.session_state.df = df_cleaned
        tab2.success(f"✅ Missing Values Imputed using {impute_method}.")
        tab2.write(df_cleaned.head())


def detect_outliers_iqr(df):
    outlier_columns = {}
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        if not outliers.empty:
            outlier_columns[col] = outliers.index.tolist()
    
    return outlier_columns

def handle_outliers(df):
    tab1, tab2, tab3, tab4 = st.tabs(["🗑 Remove", "🔄 Transform", "📏 Cap/Floor", "🛠 Imputation"])

    # Deteksi Outlier
    outlier_columns = detect_outliers_iqr(df)

    if outlier_columns:
        tab1.warning(f"⚠️ Columns with outliers: {', '.join(outlier_columns.keys())}")
        tab2.warning(f"⚠️ Columns with outliers: {', '.join(outlier_columns.keys())}")
        tab3.warning(f"⚠️ Columns with outliers: {', '.join(outlier_columns.keys())}")
        tab4.warning(f"⚠️ Columns with outliers: {', '.join(outlier_columns.keys())}")
    else:
        tab1.success("✅ No outliers detected.")
        tab2.success("✅ No outliers detected.")
        tab3.success("✅ No outliers detected.")
        tab4.success("✅ No outliers detected.")
        return

    # Visualisasi Boxplot
    st.write("### Outlier Visualization (Boxplots)")
    for col in outlier_columns.keys():
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(y=df[col], ax=ax)
        st.pyplot(fig)

    # --- Method 1: Remove Outliers ---
    with tab1:
        selected_columns = st.multiselect("Select columns to remove outliers:", list(outlier_columns.keys()), key="remove_outlier_select")
        
        if st.button(f"Remove Outliers", key="remove_outlier"):
            df_cleaned = df.copy()
            for col in selected_columns:
                df_cleaned = df_cleaned.drop(index=outlier_columns[col])
            st.session_state.df = df_cleaned
            tab1.success("✅ Outliers removed.")
            tab1.write(df_cleaned.head())

    # --- Method 2: Transform Data ---
    with tab2:
        selected_column = st.selectbox("Select a column to transform:", list(outlier_columns.keys()), key="transform_outlier_select")
        transformation_type = st.radio("Choose Transformation:", ["Log", "Box-Cox"], key="transformation_radio")

        if st.button(f"Apply {transformation_type} Transformation to {selected_column}", key=f"transform_outlier_{selected_column}"):
            df_cleaned = df.copy()
            if transformation_type == "Log":
                df_cleaned[selected_column] = np.log1p(df_cleaned[selected_column])
            elif transformation_type == "Box-Cox":
                df_cleaned[selected_column], _ = boxcox(df_cleaned[selected_column] + 1)  # Box-Cox requires positive values
            st.session_state.df = df_cleaned
            tab2.success(f"✅ {transformation_type} Transformation Applied to '{selected_column}'.")
            tab2.write(df_cleaned.head())

    # --- Method 3: Cap/Floor ---
    with tab3:
        selected_columns = st.multiselect("Select columns to cap/floor:", list(outlier_columns.keys()), key="cap_floor_select")
        
        if st.button("Cap/Floor Outliers", key="cap_floor"):
            df_cleaned = df.copy()
            for col in selected_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df_cleaned[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            st.session_state.df = df_cleaned
            tab3.success("✅ Outliers Capped/Floored.")
            tab3.write(df_cleaned.head())

    # --- Method 4: Imputation ---
    with tab4:
        selected_column = st.selectbox("Select a column to impute:", list(outlier_columns.keys()), key="impute_outlier_select")
        impute_method = st.radio("Choose Imputation Method:", ["Mean", "Median", "Mode", "Custom Value"], key="outlier_impute_radio")

        custom_value = None
        if impute_method == "Custom Value":
            custom_value = st.number_input(f"Enter Custom Value for {selected_column}:", key=f"outlier_custom_value_{selected_column}")

        if st.button(f"Apply {impute_method} Imputation to {selected_column}", key=f"apply_outlier_{selected_column}"):
            df_cleaned = df.copy()
            if impute_method == "Mean":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = df[selected_column].mean()
            elif impute_method == "Median":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = df[selected_column].median()
            elif impute_method == "Mode":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = df[selected_column].mode()[0]
            elif impute_method == "Custom Value":
                df_cleaned[selected_column].iloc[outlier_columns[selected_column]] = custom_value

            st.session_state.df = df_cleaned
            tab4.success(f"✅ Outliers in '{selected_column}' Imputed using {impute_method}.")
            tab4.write(df_cleaned.head())

def handle_change_dtype(df):
    st.subheader("Change Data Type:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Pilih kolom
    selected_column = st.selectbox("Select a column:", df.columns, key="change_dtype_column")

    # Pilih tipe data
    new_dtype = st.selectbox("Select new data type:", ["int", "float", "string", "category"], key="change_dtype_type")

    # Apply
    if st.button(f"Convert '{selected_column}' to {new_dtype}", key=f"convert_dtype_{selected_column}"):
        df_cleaned = df.copy()
        try:
            if new_dtype == "int":
                df_cleaned[selected_column] = df_cleaned[selected_column].astype(int)
            elif new_dtype == "float":
                df_cleaned[selected_column] = df_cleaned[selected_column].astype(float)
            elif new_dtype == "string":
                df_cleaned[selected_column] = df_cleaned[selected_column].astype(str)
            elif new_dtype == "category":
                df_cleaned[selected_column] = df_cleaned[selected_column].astype("category")
            elif new_dtype == "bool":
                df_cleaned[selected_column] = df_cleaned[selected_column].astype(bool)
            elif new_dtype == "datetime":
                df_cleaned[selected_column] = pd.to_datetime(df_cleaned[selected_column], errors='coerce')
            elif new_dtype == "object":
                df_cleaned[selected_column] = df_cleaned[selected_column].astype(object)

            st.session_state.df = df_cleaned
            st.success(f"✅ Column '{selected_column}' successfully converted to {new_dtype}.")
            st.write(df_cleaned.head())
        except Exception as e:
            st.error(f"❌ Failed to convert column '{selected_column}' to {new_dtype}. Error: {e}")


def handle_drop_columns(df):
    st.subheader("Drop Column:")

    selected_columns = st.multiselect("Select columns to drop:", df.columns, key="drop_column_select")
    if st.button(f"Drop Selected Columns", key="drop_columns"):
        df_cleaned = df.drop(columns=selected_columns)
        st.session_state.df = df_cleaned
        st.success("✅ Selected columns have been dropped.")
        st.write(df_cleaned.head())


def data_cleansing_page():
    
    st.title("Data Cleansing")

    if 'df' in st.session_state:
        df = st.session_state.df

        st.write("### Data Preview:")
        st.write(df.head(10))

        st.write("### What cleansing you're gonna do huh?")

        # Mau tambahkan tab baru, sini aj bre
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Missing Values", "☣️ Duplicated Values", "🎌 Outliers", "📑 Drop Column", "📑 Change Data Type"])

        # --------- Missing Values (Tab 1) ---------
        with tab1:
            handle_missing_values(df)

        # --------- Duplicated Values (Tab 2) ---------
        with tab2:
            tab2.subheader("Duplicated Values: ")
            if tab2.button("Remove Duplicates"):
                df_cleaned = df.drop_duplicates()
                st.session_state.df = df_cleaned
                tab2.success("✅ Duplicates removed.")
                tab2.write(df_cleaned.head())

        # --------- Outliers (Tab 3) ---------
        with tab3:
            handle_outliers(df)

        # --------- Drop Column (Tab 4) ---------
        with tab4:
            handle_drop_columns(df)

        # --------- Change Data Type (Tab 5) ---------
        with tab5:
            handle_change_dtype(df)

    else:
        st.warning("⚠️ No data available. Please upload data on the Data page.")
