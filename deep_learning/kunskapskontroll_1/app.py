import streamlit as st
import keras

from src.analyzer import Analyzer

keras.config.disable_interactive_logging()

a = Analyzer()

if "done" not in st.session_state:
    st.session_state.done = False

if "results" not in st.session_state:
    st.session_state.results = None

def to_csv(df):
    return df.to_csv(index=False)

def progress(p):
    st.progress(p)

@st.cache_resource
def load_model():
    return keras.models.load_model("modelv1.keras")

def analyze():
    with st.spinner("Analyzing..."):
        model = load_model()
        st.session_state.done, st.session_state.results = a.analyze(file=file, model=model, skip=skip, confidence=confidence)

st.title("Audience emotion analyzer")

st.header("What do I do?")
st.write("Upload a video file for analysis. Change any settings you want changed, and press the Analyze button.")

file = st.file_uploader('Select file to upload...')
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
skip = col1.number_input('Frames to skip', 1, None, 100)
col2.info("Skip frames to speed up analysis.  \n1 means all frames will be analyzed.")
confidence = col3.number_input('Emotion analyzer confidence', 0., 1., 0.5)
col4.info("Return predictions above this probability.")

st.button("Analyze", on_click=analyze)

if st.session_state.done:
    st.download_button("Download report", to_csv(st.session_state.results), f'report/{file.name}_results.csv')