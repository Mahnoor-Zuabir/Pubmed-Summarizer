import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer

model, tokenizer = load_model()

def summarize_text(text, length):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=length, min_length=length//2, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("PubMed Article Summarizer")

uploaded_file = st.file_uploader("Upload a PubMed article", type=["txt"])
if uploaded_file is not None:
    article = uploaded_file.read().decode("utf-8")
    
    st.subheader("Select Summary Length")
    detailed_summary = st.checkbox("Detailed Summary")
    brief_summary = st.checkbox("Brief Summary")

    if detailed_summary and brief_summary:
        st.error("Please select only one option: Detailed or Brief summary.")
    elif not detailed_summary and not brief_summary:
        st.error("Please select at least one option: Detailed or Brief summary.")
    else:
        if detailed_summary:
            summary_length = 400
        else:
            summary_length = 200

        if st.button("Generate Summary"):
            summary = summarize_text(article, length=summary_length)
            
            # Creating two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Article")
                st.markdown(f"<div style='color: red;'>{article}</div>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Summarized Article")
                st.markdown(f"<div style='color: green;'>{summary}</div>", unsafe_allow_html=True)
