import streamlit as st
import pandas as pd
import re
import torch
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.corpus import stopwords

# --- INITIAL SETUP ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Teams AI Analyst V2", page_icon="🤖", layout="wide")

# --- CORE FUNCTIONS ---
def clean_text(text):
    text = text.replace("Participant_", "P_").lower()
    text = re.sub(r'[^a-z0-9\s:]', '', text)
    fillers = {'um', 'uh', 'mmhmm', 'okay', 'yeah', 'ah', 'oh', 'like'}
    all_stop_words = (stop_words.union(fillers)) - {'not', 'no', 'dont'}
    return " ".join([w for w in text.split() if w not in all_stop_words])

def parse_vtt(content):
    content = content.replace("WEBVTT", "")
    pattern = r'<v (.*?)>(.*?)</v>'
    matches = re.findall(pattern, content, re.DOTALL)
    return " ".join([f"{speaker}: {text.strip()}" for speaker, text in matches])

@st.cache_resource
def load_model(folder_path):
    model = T5ForConditionalGeneration.from_pretrained(folder_path)
    tokenizer = T5Tokenizer.from_pretrained(folder_path)
    return model, tokenizer

def run_inference(model, tokenizer, snippet):
    outputs = {}
    for task, prefix in {"Summary": "produce summary: ", "Actions": "list actions: "}.items():
        input_ids = tokenizer(prefix + snippet, return_tensors="pt", truncation=True).input_ids
        gen_tokens = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=3)
        outputs[task] = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    return outputs

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Model Settings")
mode = st.sidebar.radio(
    "Choose View Mode:",
    ("T5-Small Only", "T5-Base Only", "Compare Side-by-Side")
)

# Paths based on your folder names
SMALL_PATH = "./AI_574_NLP_Project_Model_T5_Small"
BASE_PATH = "./AI_574_NLP_Project_Model_T5_Base"

# --- UI LAYOUT ---
st.title("Teams Meeting AI Analyst")
st.info("Upload a .vtt transcript to generate a summary and action items.")

uploaded_file = st.file_uploader("Drop your Teams VTT here", type="vtt")

if uploaded_file:
    vtt_content = uploaded_file.getvalue().decode("utf-8")
    raw_script = parse_vtt(vtt_content)
    cleaned = clean_text(raw_script).split()
    
    # Bookend Strategy
    if len(cleaned) > 500:
        snippet = " ".join(cleaned[:300] + ["..."] + cleaned[-200:])
    else:
        snippet = " ".join(cleaned)

    if mode == "T5-Small Only":
        with st.spinner("Processing with T5-Small..."):
            m, t = load_model(SMALL_PATH)
            res = run_inference(m, t, snippet)
            st.subheader("T5-Small Results")
            st.markdown(f"**Summary:** {res['Summary']}")
            st.success(f"**Actions:** {res['Actions']}")

    elif mode == "T5-Base Only":
        with st.spinner("Processing with T5-Base..."):
            m, t = load_model(BASE_PATH)
            res = run_inference(m, t, snippet)
            st.subheader("T5-Base Results")
            st.markdown(f"**Summary:** {res['Summary']}")
            st.success(f"**Actions:** {res['Actions']}")

    elif mode == "Compare Side-by-Side":
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("T5-Small")
            with st.spinner("Calculating Small..."):
                m_s, t_s = load_model(SMALL_PATH)
                res_s = run_inference(m_s, t_s, snippet)
                st.markdown(f"**Summary:**\n{res_s['Summary']}")
                st.success(f"**Actions:**\n{res_s['Actions']}")
                
        with col2:
            st.header("T5-Base")
            with st.spinner("Calculating Base..."):
                m_b, t_b = load_model(BASE_PATH)
                res_b = run_inference(m_b, t_b, snippet)
                st.markdown(f"**Summary:**\n{res_b['Summary']}")
                st.info(f"**Actions:**\n{res_b['Actions']}")
