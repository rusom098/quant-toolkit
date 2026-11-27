"""
Entry point: streamlit run app.py

The main page is the Options-Implied Distribution Lab (lab.py). The older
dashboard tabs live on a second page, "Other tools" (other_tools.py).
"""
import streamlit as st

from theme import theme_picker, apply_theme

st.set_page_config(page_title="Options-Implied Distribution Lab", layout="wide")

pages = st.navigation([
    st.Page("lab.py", title="Implied Distribution Lab", default=True),
    st.Page("other_tools.py", title="Other tools"),
])
theme_picker()   # sidebar theme switch, shared by both pages
apply_theme()
pages.run()
