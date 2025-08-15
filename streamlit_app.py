# Importing necessary libraries
import pandas as pd
import streamlit as st
import os
import re
import requests
import socket
import streamlit as st
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from urllib.parse import urlparse
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import time
from io import BytesIO

# Download NLTK stopwords 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Funtion for trying to access the webiste three times
def get_url_with_retry(url, retries=3, delay=2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            return response.text, response
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                st.error(f"🔌 Failed to retrieve website: {e}")
                return None, None


# Function to extract features from a given URL
def extract_features_from_url(url):
    html_content, response = get_url_with_retry(url)
    if html_content is None:
        return None
        
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

    # Handle potential parsing issues
    except Exception:
        st.error("❌ Failed to parse HTML content.")
        return None
        
    try:
        start_time = time.time()
        response = requests.get(url, timeout=5)
        load_time = time.time() - start_time
        response.raise_for_status()
        html_content = response.text

    # Handle potential network issues
    except requests.exceptions.Timeout:
            st.error("⏱️ Network timeout while trying to reach the website.")
            return None

    try:
        ip_check = socket.inet_aton(domain.split(':')[0]) 
        is_ip = 1

    # Handle cases where domain is not an IP address
    except socket.error:
        is_ip = 0

    for script in soup(["script", "style"]):
        script.extract()

    try:
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

        text_lines = page_text.splitlines()
        largest_line_length = max((len(line.strip()) for line in text_lines), default=0)
        line_of_code = len(text_lines)


        # NoOfObfuscatedChar = count of non-alphanumeric chars in URL
        no_of_obfuscated_char = len(re.findall(r'[^a-zA-Z0-9]', url))

        # CharContinuationRate = ratio of repeated characters sequence length / URL length
        char_continuations = re.findall(r'(.)\1{1,}', url)  # sequences of 2 or more repeated chars
        cont_length = sum(len(match)*2 for match in char_continuations)  # approx length (each char repeated at least twice)
        char_continuation_rate = cont_length / len(url) if len(url) > 0 else 0

        # LetterRatioInURL, DegitRatioInURL, SpacialCharRatioInURL
        letters = re.findall(r'[a-zA-Z]', url)
        digits = re.findall(r'[0-9]', url)
        special_chars = re.findall(r'[^a-zA-Z0-9]', url)

        letter_ratio_in_url = len(letters) / len(url) if len(url) > 0 else 0
        digit_ratio_in_url = len(digits) / len(url) if len(url) > 0 else 0
        spacial_char_ratio_in_url = len(special_chars) / len(url) if len(url) > 0 else 0

        # NoOfPopup - count occurrences of popup-related scripts in HTML
        no_of_popup = len(re.findall(r'popup', html_content, re.IGNORECASE))

        # HasExternalFormSubmit - any <form> whose action domain != current domain
        forms = soup.find_all('form')
        has_external_form_submit = 0
        for form in forms:
            action = form.get('action')
            if action:
                action_domain = urlparse(action).netloc
                if action_domain and action_domain != '' and action_domain != domain:
                    has_external_form_submit = 1
                    break

        # NoOfImage - number of <img> tags
        no_of_image = len(soup.find_all('img'))

        # HasSubmitButton - presence of submit input/button in any form
        has_submit_button = 0
        for form in forms:
            if form.find('input', {'type': 'submit'}) or form.find('button'):
                has_submit_button = 1
                break

        # HasHiddenFields - presence of hidden input fields
        has_hidden_fields = 0
        for form in forms:
            if form.find('input', {'type': 'hidden'}):
                has_hidden_fields = 1
                break

        # HasPasswordField - presence of password input fields
        has_password_field = 0
        for form in forms:
            if form.find('input', {'type': 'password'}):
                has_password_field = 1
                break

    # Error handling for feature extraction
    except Exception:
        st.error("⚠️ Error during content processing.")
        return None

    return {
        "URL": url,
        "URLLength": len(url),
        "DomainLength": len(domain),
        "TLDLength": len(domain.split('.')[-1]) if '.' in domain else 0,
        "IsDomainIP": is_ip,
        "NoOfSubDomain": domain.count('.') - 1,
        "IsHTTPS": int(parsed.scheme == "https"),
        "HasTitle": int(bool(title)),
        "LargestLineLength": largest_line_length,
        "HasFavicon": int(bool(soup.find("link", rel=lambda x: x and "icon" in x.lower()))),
        "HasDescription": int(bool(meta_desc)),
        "HasCopyrightInfo": int("copyright" in page_text.lower()),
        "NoOfJS": len(soup.find_all("script")),
        "NoOfURLRedirect": len(response.history),
        "Title": title,
        "Meta_Description": meta_desc,
        "Headings": headings,
        "Page_Content": page_text,
        "Word_Count": word_count,
        "Load_Time": round(load_time, 2),
        ##################################
        "NoOfCSS": len(soup.find_all("link", rel="stylesheet")),
        "LetterRatioInURL": letter_ratio_in_url,
        "DegitRatioInURL": digit_ratio_in_url,
        "SpacialCharRatioInURL": spacial_char_ratio_in_url,
        "NoOfPopup": no_of_popup,
        "HasExternalFormSubmit": has_external_form_submit,
        "NoOfImage": no_of_image,
        "HasSubmitButton": has_submit_button,
        "HasHiddenFields": has_hidden_fields,
        "HasPasswordField": has_password_field,
        "LineOfCode": line_of_code

    }

# Cleaning the catche folder if exists (errors without doing that)
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/transformers")

def clear_transformers_cache():
    if os.path.exists(CACHE_DIR):
        print(f"Clearing Huggingface cache at {CACHE_DIR} ...")
        shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")
    else:
        print("Cache folder not found, skipping clear.")

#Load the pre-trained BERT model
def load_bert_model():
    try:
        print("Loading tokenizer and model...")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Check if model params are still meta tensors
        is_meta = all(param.device.type == 'meta' for param in model.parameters())
        if is_meta:
            device = torch.device("cpu")
            model.to_empty(device)
            raise RuntimeError("Model parameters are meta tensors (weights not loaded)")
        
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        print("Model loaded and moved to CPU successfully.")
        return tokenizer, model

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Clearing cache and retrying download...")
        clear_transformers_cache()
        
        # Retry once after cache clear
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        print("Model reloaded after cache clear.")
        return tokenizer, model

# Usage in your main code
tokenizer, bert_model = load_bert_model()

# Function to ensure URL starts with http:// or https://
def add_https_if_missing(url):
    if not url.startswith(('http://', 'https://')):
        return 'https://' + url
    return url

# Function to get BERT embeddings for the input features
def get_bert_embedding(features):
    title = features.get("Title") or ""
    meta_desc = features.get("Meta_Description") or ""
    headings = features.get("Headings") or ""
    page_content = features.get("Page_Content") or ""
   
    combined_text = title + " " + meta_desc + " " + headings + " " + page_content
    inputs = tokenizer([combined_text], padding=True, truncation=True, return_tensors="pt")
    
    device = next(bert_model.parameters()).device
    inputs = {key: value.to("cpu") for key, value in inputs.items()}  

    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  

    return cls_embeddings.cpu().numpy()[0]

# Load pre-trained models and scaler
clf = joblib.load('random_forest_model.pkl')
xgb = joblib.load('xgboost_model.pkl')
clf_scaler = joblib.load('clf_scaler.pkl')
xgb_scaler = joblib.load('xgb_scaler.pkl')

#Streamlit app setup
st.title("🔍 Phising Webiste Scanner")
st.write("Enter a URL below to check whether webiste is **legitimate or phishing**.")

url = st.text_input("Enter URL:", "")
if url:
    url = add_https_if_missing(url)

# Custom CSS for styling
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


model_choice = st.selectbox(
    "Choose Prediction Model:",
    ("Random Forest", "XGBoost")
)


if st.button("Check Webiste Trust and Quality"):
    if not url.strip():
        st.warning("⚠️ Please enter a valid URL before checking.")
    else:
        with st.spinner("Analyzing website..."):
            features = extract_features_from_url(url)
        if features:
            selected_num_cols = [
                            "URLLength", "DomainLength", "TLDLength", "IsDomainIP", "NoOfSubDomain",
                            "IsHTTPS", "HasTitle", "LargestLineLength", "HasFavicon",
                            "HasDescription", "HasCopyrightInfo", "NoOfJS", "NoOfURLRedirect", "Word_Count", "NoOfCSS", "LetterRatioInURL", "DegitRatioInURL", "SpacialCharRatioInURL",
                            "NoOfPopup", "HasExternalFormSubmit", "NoOfImage",
                            "HasSubmitButton", "HasHiddenFields", "HasPasswordField", "LineOfCode"
                        ]
            
            # Load BERT embeddings for input URL 
            bert_embedding = get_bert_embedding(features) 
            X_input_num = np.array([[features[col] for col in selected_num_cols]])
            
            if model_choice == "Random Forest":
                X_input_num_scaled = scaler_clf.transform(X_input_num)
                X_combined = np.hstack([bert_embedding.reshape(1, -1), X_input_num_scaled])
                prediction = clf.predict(X_combined)[0]
                prob_phishing = clf.predict_proba(X_combined)[0][1] 
                trust_score = 10 - int(prob_phishing * 10)

                if prob_phishing <= 0.5:
                    st.error(f"🚨 **WARNING! This site might be a PHISHING site!** (Risk: {trust_score}/10)")
                else:
                    st.success(f"✅ **This site appears to be SAFE.** (Risk: {trust_score}/10)")
            
            else:  # XGBoost
                X_input_num_scaled = scaler_xgb.transform(X_input_num)
                X_combined = np.hstack([bert_embedding.reshape(1,-1), X_input_num_scaled])
                prediction = xgb.predict(X_combined)[0]
                prob_phishing_xgb = xgb.predict_proba(X_combined)[0][1]
                if prob_phishing_xgb > 0.1:
                    trust_score_xgb = abs(int(prob_phishing_xgb * 10))
                else:
                    trust_score_xgb = abs(10 - (int(prob_phishing_xgb * 100)))
    
                if prob_phishing_xgb >= 0.1:
                    st.success(f"✅ **This site appears to be SAFE.** (Risk: {trust_score_xgb}/10)")
                else:
                     st.error(f"🚨 **WARNING! This site might be a PHISHING site!** (Risk: {trust_score_xgb}/10)")
            
            with st.expander("🔍 Page Details"):
                    st.info(f"**Title**: {features['Title']}")
                    if features['IsHTTPS'] == 1:
                        st.markdown('<span style="color: green; font-weight: bold;">HTTPS: Yes ✅</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span style="color: red; font-weight: bold;">HTTPS: No ❌</span>', unsafe_allow_html=True)
                    st.write(f"**Meta Description**: {features['Meta_Description']}")
                    st.write(f"**Headings**: {features['Headings']}")
                    #st.write(f"**Word Count**: {features['Word_Count']}")
                    st.write(f"**Load Time**: {features['Load_Time']} seconds")
                    
                    # Just for checking:
                    #st.write(
                    #f"**URL Length**: {features['URLLength']} | "
                    #f"**Domain Length**: {features['DomainLength']} | "
                    #f"**TLD Length**: {features['TLDLength']} | "
                    #f"**Is Domain IP**: {features['IsDomainIP']} | "
                    #f"**Subdomains**: {features['NoOfSubDomain']} | "
                    #f"**HTTPS**: {'Yes ✅' if features['IsHTTPS'] == 1 else 'No ❌'} "
                    #f"**Title Present**: {features['HasTitle']} | "
                    #f"**Largest Line Length**: {features['LargestLineLength']} | "
                    #f"**Favicon**: {features['HasFavicon']} | "
                    #f"**Redirects**: {features['NoOfURLRedirect']} | "
                    #f"**Meta Description Present**: {features['HasDescription']} | "
                    #f"**Copyright Info**: {features['HasCopyrightInfo']} | "
                    #f"**JavaScript Files**: {features['NoOfJS']}")

                    #st.write(
                    #f"**No of CSS Files**: {features['NoOfCSS']} | "
                    #f"**Letter Ratio in URL**: {features['LetterRatioInURL']:.2f} | "
                    #f"**Digit Ratio in URL**: {features['DegitRatioInURL']:.2f} | "
                    #f"**Special Char Ratio in URL**: {features['SpacialCharRatioInURL']:.2f} | "
                    #f"**Popups**: {features['NoOfPopup']} | "
                    #f"**External Form Submit**: {features['HasExternalFormSubmit']} | "
                    #f"**Images**: {features['NoOfImage']} | "
                    #f"**Submit Button**: {features['HasSubmitButton']} | "
                    #f"**Hidden Fields**: {features['HasHiddenFields']} | "
                    #f"**Password Field**: {features['HasPasswordField']}")

        else:
            st.warning("Unable to analyze this URL due to an error.")

# Sidebar content - instructions and model info
with st.sidebar:
    st.subheader(" 🛠️ How It Works")
    st.write("""
    - Extracts SEO, URL & NLP-based features from the website
    - Uses a trained model Random Forest or XGBoost combined with BERT embeddings.
    - Returns a risk level: **Safe** 🟢 or **Phishing** 🔴""")
    st.markdown("---")

    st.sidebar.markdown(f"**📌 Model Used:** {model_choice}")


