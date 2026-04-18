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

st.set_page_config(page_title="MS Teams AI", page_icon="https://is1-ssl.mzstatic.com/image/thumb/PurpleSource221/v4/c7/bd/2f/c7bd2f1f-f892-13ba-d8df-813d18a7c503/Placeholder.mill/400x400bb-75.webp")

# --- CORE FUNCTIONS ---
# def clean_text(text):
#     text = text.replace("Participant_", "P_").lower()
#     text = re.sub(r'[^a-z0-9\s:]', '', text)
#     fillers = {'um', 'uh', 'mmhmm', 'okay', 'yeah', 'ah', 'oh', 'like'}
#     all_stop_words = (stop_words.union(fillers)) - {'not', 'no', 'dont'}
#     return " ".join([w for w in text.split() if w not in all_stop_words])

# def clean_text(text):
#     # 1. Standardize speaker labels (Good idea, keep this)
#     text = text.replace("Participant_", "P_")
    
#     # 2. DO NOT lowercase. Models use Capital Letters to find Names/Entities.
#     # 3. DO NOT remove stop words. Transformers need them for grammar.
    
#     # 4. Selective Filler Removal (Optional)
#     # Only remove the most disruptive ones, but keep "okay" or "no" 
#     # as they often signal agreement or disagreement.
#     disruptive_fillers = r'\b(um|uh|mmhmm|ah)\b'
#     text = re.sub(disruptive_fillers, '', text, flags=re.IGNORECASE)
    
#     # 5. Fix whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

def clean_text(text):
    text = text.replace("Participant_", "P_").lower()
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
def load_model():
    path = "./AI_574_NLP_Project_Model_T5_Small"
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    return model, tokenizer

# --- UI LAYOUT ---
st.title("MS Teams Meeting AI Analyst")
st.caption("AI 574 NLP, Great Valley, PSU | Abdulaziz Albaiz")
st.info("Upload a .vtt transcript or use the example below to generate a summary.")

# --- SIDEBAR DEV TOOLS ---
st.sidebar.header("Model Dev Tools")
num_beams = st.sidebar.slider("Beams", 1, 10, 2)
no_repeat = st.sidebar.slider("No Repeat N-Gram", 1, 5, 2)
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.5)
rep_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 3.0, 1.5)

# --- FILE INPUTS ---
uploaded_file = st.file_uploader("Drop your Teams VTT here", type="vtt")
use_example = st.button("Use Example Test Meeting")

vtt_content = None

# Handle the data source
if uploaded_file:
    vtt_content = uploaded_file.getvalue().decode("utf-8")
elif use_example:
    try:
        with open("test_meeting.vtt", "r", encoding="utf-8") as f:
            vtt_content = f.read()
    except FileNotFoundError:
        st.error("Example file 'test_meeting.vtt' not found!")

# --- NEW: UNIFIED INSPECTION BLOCK ---
if vtt_content:
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Inspect Raw Transcript"):
            # Shows a preview of the content (first 1000 chars)
            st.code(vtt_content[:1000] + "\n... [truncated]", language="text")
    with col2:
        # Dynamic filename based on source
        fname = uploaded_file.name if uploaded_file else "test_meeting.vtt"
        st.download_button("Download This VTT", vtt_content, fname)
    st.divider()

# --- PROCESSING & DISPLAY ---
if vtt_content:
    with st.spinner("Processing with T5-Small..."):
        model, tokenizer = load_model()
        
        raw_script = parse_vtt(vtt_content)
        cleaned = clean_text(raw_script).split()
        
        if len(cleaned) > 500:
            snippet = " ".join(cleaned[:300] + ["..."] + cleaned[-200:])
        else:
            snippet = " ".join(cleaned)

        outputs = {}
        for task, prefix in {"Summary": "produce summary: ", "Actions": "list actions: "}.items():
            input_ids = tokenizer(prefix + snippet, return_tensors="pt", truncation=True).input_ids
            
            gen_tokens = model.generate(
                input_ids, 
                max_length=256, 
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat,
                repetition_penalty=rep_penalty,
                temperature=temp,
                do_sample=True if temp > 0 else False
            )
            outputs[task] = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    # --- RESULTS ---
    st.subheader("Meeting Summary")
    st.markdown(f"> {outputs['Summary']}")
    
    st.subheader("Action Items")
    st.success(outputs["Actions"])
    
    report_text = f"SUMMARY:\n{outputs['Summary']}\n\nACTION ITEMS:\n{outputs['Actions']}"
    st.download_button("Download Analysis", report_text, "meeting_analysis.txt")
