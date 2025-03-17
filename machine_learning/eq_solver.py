import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os, random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
st.set_page_config(layout="wide")

# Custom CSS for mobile responsiveness and to keep buttons side by side
st.markdown(
    """
    <style>
    /* Hide the canvas toolbar */
    div[data-testid="stCanvas"] .toolbar {
        display: none;
    }
    /* Responsive canvas for mobile */
    @media (max-width: 768px) {
        div[data-testid="stCanvas"] > canvas {
            width: 100% !important;
            height: auto !important;
        }
    }
    /* Force columns (the horizontal block) to remain in a row even on mobile */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# Maintain a canvas key in session_state for resetting the canvas
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

this_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(this_dir, "joblib", "cnn_model_aug.keras")
model = tf.keras.models.load_model(model_path)
labels = list("0123456789") + ["+", "-"]

def center_symbol(img, size=28, pad=20):
    h, w = img.shape
    scale = min(pad / h, pad / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=resized.dtype)
    y, x = (size - new_h) // 2, (size - new_w) // 2
    out[y:y+new_h, x:x+new_w] = resized
    return out

def segment_and_center(img):
    _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 25]
    boxes.sort(key=lambda b: b[0])
    out = []
    for x, y, w, h in boxes:
        roi = img[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
        out.append(center_symbol(roi))
    return out

def predict_expr(pil_img):
    gray = pil_img.convert("L")
    gray = np.array(gray)
    gray = cv2.bitwise_not(gray)
    chunks = segment_and_center(gray)
    preds = []
    for c in chunks:
        c = c.astype(np.float32) / 255.0
        c = c.reshape((1, 28, 28, 1))
        p = model.predict(c).argmax()
        preds.append(labels[p])
    expr = "".join(preds)
    try:
        ans = eval(expr)
    except Exception:
        ans = "Could not evaluate"
    return expr, ans

st.title("Handwritten Math Solver 🖊️")
st.write("Draw digits and + or - signs clearly below:")

# Create the drawable canvas without its built-in toolbar
canvas = st_canvas(
    fill_color="white",
    stroke_width=16,
    stroke_color="black",
    background_color="white",
    width=400,
    height=300,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key,
    display_toolbar=False  # Hide the built-in toolbar
)

# Place Solve and Clear buttons side by side using columns.
col1, col2 = st.columns(2)
with col1:
    solve_clicked = st.button("Solve")
with col2:
    clear_clicked = st.button("Clear")

if solve_clicked:
    if canvas.image_data is not None:
        data = canvas.image_data.astype("uint8")
        pil_img = Image.fromarray(data, "RGBA")
        pil_img = pil_img.convert("L")
        expr, sol = predict_expr(pil_img)
        emo = random.choice(["😎", "😊", "🤔", "🫡", "👍", "😉", "🙂"])
        st.success(f"**Expression:** {expr}\n\n**Solution:** {sol} {emo}")
    else:
        st.warning("Please draw something first!")

if clear_clicked:
    # Increment the canvas key to force reinitialization (clearing the canvas)
    st.session_state.canvas_key += 1
    st.experimental_rerun()
