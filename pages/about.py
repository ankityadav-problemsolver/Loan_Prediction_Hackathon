import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animation from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Page config
st.set_page_config(page_title="About | Loan Prediction App", layout="centered")

# Title and subtitle
st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: bold;
    color: #00f5a0;
}
.subtitle {
    font-size: 20px;
    color: #cccccc;
}
.section-header {
    font-size: 28px;
    color: #00f5a0;
    margin-top: 2rem;
}
.team-member {
    font-size: 18px;
    color: #dddddd;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💸 Loan Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering smarter lending decisions with machine learning</div>', unsafe_allow_html=True)
st.markdown("---")



# Project Description
st.markdown('<div class="section-header">🔍 Project Overview</div>', unsafe_allow_html=True)
st.write("""
This application helps banks assess **loan eligibility** using machine learning.
It processes applicant data and predicts the likelihood of loan approval using trained models.
""")

# Features
st.markdown('<div class="section-header">🚀 Key Features</div>', unsafe_allow_html=True)
st.markdown("""
- ✅ Predicts loan approval based on applicant information  
- 🤖 Trained with 5 different machine learning models  
- 🔧 Hyperparameter tuning to improve performance  
- 📊 Interactive dashboard for analysis and insights  
- 🌐 Built using Python, Streamlit, and Scikit-learn  
""")

# Team Section
st.markdown('<div class="section-header">👥 Meet the Team</div>', unsafe_allow_html=True)
st.markdown("""
- 👨‍🏫 <span class="team-member">**Ankit Yadav** – Team Leader, Data Engineer, Model Optimization,Data Scientist</span>  
- 🧑‍💻 <span class="team-member">**Shiva** – Data Engineer</span>  
- 👩‍💻 <span class="team-member">**Sriharini** – UI/UX Designer, Feature Engineer</span>  
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("✨ *Built with love, logic, and lots of coffee ☕️*")
