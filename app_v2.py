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

st.set_page_config(
    page_title="MS Teams AI - Comparison", 
    page_icon="https://is1-ssl.mzstatic.com/image/thumb/PurpleSource221/v4/c7/bd/2f/c7bd2f1f-f892-13ba-d8df-813d18a7c503/Placeholder.mill/400x400bb-75.webp",
    layout="wide"
)

# --- CORE FUNCTIONS ---
def clean_text(text):
    #text = text.replace("Participant_", "P_").lower()
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s:]', '', text)
    fillers = {'um', 'uh', 'mmhmm', 'okay', 'yeah', 'ah', 'oh', 'like'}
    # Matches your training script exactly:
    essential_words = {'not', 'no', 'dont', 'doesnt', 'isnt', 'wont'}
    all_stop_words = (stop_words.union(fillers)) - essential_words
    return " ".join([w for w in text.split() if w not in all_stop_words])

def parse_vtt(content):
    content = content.replace("WEBVTT", "")
    pattern = r'<v (.*?)>(.*?)</v>'
    matches = re.findall(pattern, content, re.DOTALL)
    return " ".join([f"{speaker}: {text.strip()}" for speaker, text in matches])

@st.cache_resource
def load_model(path):
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    return model, tokenizer

def run_inference(model, tokenizer, snippet, params):
    outputs = {}
    for task, prefix in {"Summary": "produce summary: ", "Actions": "list actions: "}.items():
        input_text = f"{prefix}{snippet}"
        # Ensure we use the same max_length as training (512)
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
        
        gen_tokens = model.generate(
            input_ids, 
            max_length=256, 
            num_beams=params['num_beams'],
            # These two are CRITICAL - they were hardcoded in your training eval
            no_repeat_ngram_size=params['no_repeat'], 
            repetition_penalty=params['rep_penalty'],
            temperature=params['temp'],
            do_sample=True if params['temp'] > 0 else False
        )
        decoded = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        
        # Security check: if model fails, don't return 'False'
        if not decoded or decoded.lower() == 'false':
            outputs[task] = "Model could not generate a clear result. Try adjusting 'Beams' or 'Penalty'."
        else:
            outputs[task] = decoded
    return outputs

# --- SIDEBAR: VIEW SELECTION ---
st.sidebar.header("View Configuration")
view_mode = st.sidebar.selectbox(
    "Choose View Mode:",
    ("T5-Small Only", "T5-Base Only", "Compare Side-by-Side")
)

# --- SIDEBAR: MODEL PARAMETERS ---
def get_params(label):
    with st.sidebar.expander(f"Settings: {label}", expanded=True):
        p = {
            'num_beams': st.slider(f"Beams ({label})", 1, 10, 2, key=f"beams_{label}"),
            'no_repeat': st.slider(f"No Repeat ({label})", 1, 5, 2, key=f"rep_{label}"),
            'temp': st.slider(f"Temperature ({label})", 0.0, 1.5, 0.5, key=f"temp_{label}"),
            'rep_penalty': st.slider(f"Penalty ({label})", 1.0, 3.0, 1.5, key=f"penalty_{label}")
        }
    return p

# Logic to show sliders based on mode
params_small = None
params_base = None

if view_mode == "T5-Small Only" or view_mode == "Compare Side-by-Side":
    params_small = get_params("T5-Small")

if view_mode == "T5-Base Only" or view_mode == "Compare Side-by-Side":
    params_base = get_params("T5-Base")

# Paths for your models
SMALL_PATH = "./AI_574_NLP_Project_Model_T5_Small"
BASE_PATH = "./AI_574_NLP_Project_Model_T5_Base"

# --- UI LAYOUT ---
st.title("MS Teams Meeting AI Analyst")
st.caption("AI 574 NLP, Great Valley, PSU | Abdulaziz Albaiz")

# --- FILE INPUTS ---
uploaded_file = st.file_uploader("Drop your Teams VTT here", type="vtt")
use_example = st.button("Use Example Test Meeting")

vtt_content = None
if uploaded_file:
    vtt_content = uploaded_file.getvalue().decode("utf-8")
elif use_example:
    try:
        with open("test_meeting.vtt", "r", encoding="utf-8") as f:
            vtt_content = f.read()
    except FileNotFoundError:
        st.error("Example file 'test_meeting.vtt' not found!")

if vtt_content:
    with st.expander("Inspect Raw Transcript"):
        st.code(vtt_content[:1000] + "\n... [truncated]", language="text")
    
    raw_script = parse_vtt(vtt_content)
    cleaned = clean_text(raw_script).split()
    
    if len(cleaned) > 500:
        snippet = " ".join(cleaned[:300] + ["..."] + cleaned[-200:])
    else:
        snippet = " ".join(cleaned)

    # --- EXECUTION LOGIC ---
    if view_mode == "T5-Small Only":
        with st.spinner("Processing with T5-Small..."):
            m, t = load_model(SMALL_PATH)
            out = run_inference(m, t, snippet, params_small)
            st.subheader("T5-Small Results")
            st.markdown(f"> {out['Summary']}")
            st.success(out['Actions'])

    elif view_mode == "T5-Base Only":
        with st.spinner("Processing with T5-Base..."):
            m, t = load_model(BASE_PATH)
            
            # --- START HARD-CODED TEST ---
            test_input = "produce summary: The team met today to discuss the new logo. They decided to use blue."
            test_ids = t(test_input, return_tensors="pt").input_ids
            test_gen = m.generate(test_ids, num_beams=2, repetition_penalty=2.5)
            test_res = t.decode(test_gen[0], skip_special_tokens=True)
            st.warning(f"Model Sanity Check: {test_res}")
            # --- END HARD-CODED TEST ---
            
            out = run_inference(m, t, snippet, params_base)
            st.subheader("T5-Base Results")
            st.markdown(f"> {out['Summary']}")
            st.success(out['Actions'])

    elif view_mode == "Compare Side-by-Side":
        col_s, col_b = st.columns(2)
        
        with col_s:
            st.header("T5-Small")
            with st.spinner("Running Small..."):
                m_s, t_s = load_model(SMALL_PATH)
                res_s = run_inference(m_s, t_s, snippet, params_small)
                st.markdown(f"**Summary:**\n{res_s['Summary']}")
                st.success(f"**Actions:**\n{res_s['Actions']}")
                
        with col_b:
            st.header("T5-Base")
            with st.spinner("Running Base..."):
                m_b, t_b = load_model(BASE_PATH)
                res_b = run_inference(m_b, t_b, snippet, params_base)
                st.markdown(f"**Summary:**\n{res_b['Summary']}")
                st.info(f"**Actions:**\n{res_b['Actions']}")
