import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os, random

# Set the environment variable before anything else if you like
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 1) MUST be the first Streamlit command
st.set_page_config(layout="wide")

# 2) Then you can inject custom CSS, etc.
st.markdown(
    """
    <style>
    /* Force a white background on the main elements */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Continue with the rest of your code:
this_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(this_dir, "joblib", "cnn_model_aug.keras")
model = tf.keras.models.load_model(model_path)
labels = list("0123456789") + ["+", "-"]

def center_symbol(img, size=28, pad=20):
    ...
def segment_and_center(img):
    ...
def predict_expr(pil_img):
    ...

st.title("Handwritten Math Solver")
st.write("Draw digits and + or - signs clearly below:")

canvas = st_canvas(
    fill_color="white",
    stroke_width=14,
    stroke_color="black",
    background_color="white",
    width=900,
    height=200,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Solve"):
    if canvas.image_data is not None:
        data = canvas.image_data.astype("uint8")
        pil_img = Image.fromarray(data, "RGBA")
        pil_img = pil_img.convert("L")
        expr, sol = predict_expr(pil_img)
        emo = random.choice(["😎", "😊", "🤔", "🫡", "👍", "😉", "🙂"])
        st.success(f"**Expression:** {expr}\n\n**Solution:** {sol} {emo}")
    else:
        st.warning("Please draw something first!")