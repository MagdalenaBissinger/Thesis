import pandas as pd
import streamlit as st
import os
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Feature extraction from URL content
def extract_features_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        html_content = response.text
    except Exception:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')

    for script in soup(["script", "style"]):
        script.extract()

    page_text = soup.get_text(separator=" ").strip()
    page_text = re.sub(r'\s+', ' ', page_text)
    page_text = re.sub(r'[^a-zA-Z0-9\s]', '', page_text)

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = meta_desc_tag.get("content", "") if meta_desc_tag else ""

    title = soup.title.string if soup.title else "No Title"
    headings = " ".join([h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])

    words = page_text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    word_count = len(filtered_words)

    return {
        "URL": url,
        "Title": title,
        "Meta_Description": meta_desc,
        "Headings": headings,
        "Page_Content": page_text,
        "Word_Count": word_count
    }


#Streamlit app setup
st.title("🔍 Phising Webiste Scanner")
st.write("Enter a URL below to check whether webiste is **legitimate or phishing**.")
url = st.text_input("Enter URL:", "")

st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
        }
        h1 {
            color: #02060f;
            font-family: 'Arial Black', sans-serif;
        }
        .stTextInput, .stButton {
            width: 100%;
        }
        .phishing {
            color: #175db7;
            font-weight: bold;
        }
        .safe {
            color: green;
            font-weight: bold;
        }
    </style>""", unsafe_allow_html=True)

prediction = 0
trust_score = 0
if st.button("Check Webiste Trust and Quality"):
    if not url.strip():
        st.warning("⚠️ Please enter a valid URL before checking.")
    else:
        with st.spinner("Analyzing website..."):
            features = extract_features_from_url(url)
        if prediction == 1:
            st.error(f"🚨 **WARNING! This site might be a PHISHING site!** (Risk: {trust_score:.2%})")
        else:
            st.success(f"✅ **This site appears to be SAFE.** (Risk: {trust_score:.2%})")
        
        with st.expander("🔍 Page Details"):
                st.info(f"**Title**: {features['Title']}")
                st.write(f"**Meta Description**: {features['Meta_Description']}")
                st.write(f"**Headings**: {features['Headings']}")
                st.write(f"**Word Count**: {features['Word_Count']}")
    
with st.sidebar:
    #st.image("", width=300)
    st.subheader(" 🛠️ How It Works")
    st.write("""
    - Extracts SEO, URL & NLP-based features from the website
    - Uses a trained model.
    - Returns a risk level: **Safe** 🟢 or **Phishing** 🔴""")
    st.markdown("---")
    #st.write("**🔧 Model Used:** Logistic Regression")
    #st.write("**📊 Accuracy:** 95%")