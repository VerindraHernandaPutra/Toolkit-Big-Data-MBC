import streamlit as st
import pandas as pd
import io

def data_page():
    st.title("Data")
    st.header("Upload Data")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader")

    if uploaded_file is not None:
        try:
            # Read CSV with index column handling
            st.session_state.df = pd.read_csv(uploaded_file).reset_index(drop=True)
            st.session_state.file_uploaded = True

            # Clean any existing index columns
            if 'Unnamed: 0' in st.session_state.df.columns:
                st.session_state.df = st.session_state.df.drop(columns=['Unnamed: 0'])

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except pd.errors.ParserError:
            st.error("Error parsing CSV file. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    if 'df' in st.session_state and st.session_state.file_uploaded:
        df = st.session_state.df

        # Data Preview
        st.write("Data Preview:")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Complete Table", "☣️ Null Values", "🎌 Duplicated Values", "📑 Data Types"])

        # Tab 1 - Complete Table
        tab1.subheader("Complete Table:")
        tab1.dataframe(df, use_container_width=True, hide_index=True)

        # Tab 2 - Null Values
        tab2.subheader("Null Values:")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ["Column name", "Missing value count"]
        tab2.dataframe(null_counts, use_container_width=True, hide_index=True)

        # Tab 3 - Duplicated Values
        dup_count = df.duplicated().sum()
        tab3.subheader("Duplicated Values: ")
        tab3.write(df.duplicated().sum())
        tab3.dataframe(df[df.duplicated()], 
                      use_container_width=True,
                      hide_index=True)

        # Tab 4 - Data Types
        tab4.subheader("Data Types Summary:")
        
        # Create improved type summary
        type_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values
        }).reset_index(drop=True)
        
        tab4.dataframe(type_info, use_container_width=True, hide_index=True)
