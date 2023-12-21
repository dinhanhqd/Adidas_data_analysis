import streamlit as st
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as components

# Load your data
df = pd.read_csv('Adidas.csv', encoding='latin-1')

st.set_page_config(
    page_title="uuuuuuuuuuu",
    layout="wide"  # Use lowercase "layout" here
)
st.title("sádasdasdasd")

pyg_html = pyg.walk(df, return_html=True)

components.html(pyg_html, height=1000, scrolling=True)