import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os

# Set page configuration
st.set_page_config(page_title="Exjobb Project Analysis", layout="wide", page_icon="📊")

# Custom CSS styles
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.medium-font {
    font-size:20px !important;
}
.button-container {
    display: flex;
    justify-content: center;
    gap: 20px;
}
.stButton>button {
    width: 200px;
    height: 60px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">Exjobb Project Data Analysis and Visualization</p>', unsafe_allow_html=True)

# Welcome message
st.markdown('<p class="medium-font">Welcome to the Exjobb Project Analysis App!</p>', unsafe_allow_html=True)

# Introduction text
st.markdown("""
This application provides comprehensive data analysis and visualizations of exjobb projects 
for Linköping University and KTH (Royal Institute of Technology).

Choose a university to explore its project data:
""")

# Create button container
st.markdown('<div class="button-container">', unsafe_allow_html=True)

# Linköping University button
if st.button("Linköping University 🎓"):
    switch_page("liu")

# KTH button
if st.button("KTH (Royal Institute of Technology) 🏛️"):
    switch_page("kth")
    
# CTH button
if st.button("CTH (Chalmers University of Technology) 🏫"):
	switch_page("cth")

st.markdown('</div>', unsafe_allow_html=True)

# Add some extra information or images
st.image("./logo.png", width=500)

# Visit counter
if 'visit_count' not in st.session_state:
    st.session_state.visit_count = 0

st.session_state.visit_count += 1

st.markdown(f"<p style='text-align: center;'>Total Visits: {st.session_state.visit_count}</p>", unsafe_allow_html=True)

# Custom sidebar name
def sidebar():
    with st.sidebar:
        st.markdown("## Navigation")
        st.page_link("streamlit_app.py", label="Home", icon="🏠")
        st.page_link("pages/liu.py", label="Linköping University", icon="🎓")
        st.page_link("pages/kth.py", label="KTH", icon="🏛️")
        st.page_link("pages/cth.py", label="Chalmers", icon="🏫")

sidebar()