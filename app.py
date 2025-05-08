import streamlit as st
import pandas as pd

import input_data as data
import data_visualization as dvz
import data_cleansing as dcsg
import data_preprocessing as dppc
import data_modeling as dm
import competition_page as comp

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
footer > div:first-of-type {visibility: hidden;}
body {display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; flex-direction: column;}
input[type='text'] {width: 300px; padding: 10px; font-size: 18px;}
.stAlert {font-size: 20px;}
a {color: blue; text-decoration: underline;}
</style>
"""

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown(hide_st_style, unsafe_allow_html=True)

VALID_USERS = {
    "user1": "password1",
    "user2": "password2"
}

def login_page():
    st.title("Login Page")

    username = "user1"
    password = "password1"

    if st.button("Login"):
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "main"
        else:
            st.error("Invalid username or password")

def modeling_page():
    st.title("Modeling")
    st.write("This is where modeling functionalities will be implemented.")

def main_page():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Data", "Data Visualization", "Data Cleansing", "Data Preprocessing", "Modeling", "Competition"], key="navbar")

    if selection == "Data":
        data.data_page()
    elif selection == "Data Visualization":
        dvz.data_visualization_page()
    elif selection == "Data Cleansing":
        dcsg.data_cleansing_page()
    elif selection == "Data Preprocessing":
        dppc.data_preprocessing_page()
    elif selection == "Modeling":
        dm.modeling_page()
    elif selection == "Competition":
        comp.competition()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'page' not in st.session_state:
    st.session_state.page = "login"

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

if st.session_state.page == "main":
    main_page()
else:
    login_page()
