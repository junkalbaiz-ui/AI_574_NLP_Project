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

st.set_page_config(page_title="Teams AI Analyst", page_icon="🤖")

# --- CORE FUNCTIONS (Your Training Logic) ---
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
def load_model():
    # If the folder is in your GitHub repo, path is just the folder name
    path = "." 
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    return model, tokenizer

# --- UI LAYOUT ---
st.title("Teams Meeting AI Analyst")
st.info("Upload a .vtt transcript to generate a summary and action items using your fine-tuned T5 model.")

uploaded_file = st.file_uploader("Drop your Teams VTT here", type="vtt")

if uploaded_file:
    # Read the file content
    vtt_content = uploaded_file.getvalue().decode("utf-8")
    
    with st.spinner("Processing with T5-Small..."):
        model, tokenizer = load_model()
        
        # Process text
        raw_script = parse_vtt(vtt_content)
        cleaned = clean_text(raw_script).split()
        
        # Bookend Strategy
        if len(cleaned) > 500:
            snippet = " ".join(cleaned[:300] + ["..."] + cleaned[-200:])
        else:
            snippet = " ".join(cleaned)

        # Generate
        outputs = {}
        for task, prefix in {"Summary": "produce summary: ", "Actions": "list actions: "}.items():
            input_ids = tokenizer(prefix + snippet, return_tensors="pt", truncation=True).input_ids
            gen_tokens = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=3)
            outputs[task] = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    # Display results in pretty boxes
    st.subheader("Meeting Summary")
    st.markdown(f"> {outputs['Summary']}")
    
    st.subheader("Action Items")
    st.success(outputs["Actions"])
