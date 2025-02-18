import streamlit as st
import pandas as pd
import io

def data_page():
    st.title("Data")
    st.header("Upload Data")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader")

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.file_uploaded = True

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
        st.write(df.head(10))

        # Mau tambahkan tab baru, sini aj bre
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Complete Table", "☣️ Null Values", "🎌 Duplicated Values", "📑 Data Types"])

        # Tab 1 - Complete Tabel, untuk menampilkan semua tabel secara lengkap
        tab1.subheader("Complete Table:")
        tab1.write(df)

        # Tab 2 - Null Values, untuk menampilkan semua null values beserta detailnya
        tab2.subheader("Null Values:")
        tab2.write(df.isnull().sum())
        tab2.write("More Detail: ")
        tab2.write(df.isnull())

        # Tab 3 - Duplicated Value, untuk menampilkan apakah data terdeteksi adanya nilai duplikasi atau tidak
        tab3.subheader("Duplicated Values: ")
        tab3.write(df.duplicated().sum())
        tab3.write("More Detail: ")
        tab3.write(df.duplicated())

        # Tab 4 - Data Type, untuk mengecek tipe data pada tabel
        tab4.subheader("Data Type: ")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        tab4.text(s)