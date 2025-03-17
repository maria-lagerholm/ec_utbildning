import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os, random

# Your existing imports and model loading code remains unchanged
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

this_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(this_dir, "joblib", "cnn_model_aug.keras")
model = tf.keras.models.load_model(model_path)
labels = list("0123456789") + ["+", "-"]

# Your existing functions remain unchanged
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
    except:
        ans = "Could not evaluate"
    return expr, ans

# Add custom CSS for responsive canvas
st.markdown("""
    <style>
        @media (max-width: 600px) {
            .canvas-container {
                width: 400px !important;
                margin: 0 auto;
            }
        }
        @media (min-width: 601px) {
            .canvas-container {
                width: 900px !important;
                margin: 0 auto;
            }
        }
        /* Ensure canvas scales properly */
        .canvas-container canvas {
            width: 100% !important;
            height: auto !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Handwritten Math Solver 🖊️")
st.markdown('<style>h1{font-size:14px;}</style>', unsafe_allow_html=True)
st.write("Draw digits and + or - signs clearly below:")

# Wrap canvas in a div with our responsive class
with st.container():
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    
    # Calculate height based on width to maintain aspect ratio
    # Original was 400x300, so we'll maintain 3:4 ratio
    canvas = st_canvas(
        fill_color="white",
        stroke_width=16,
        stroke_color="black",
        background_color="white",
        width=900,  # This will be overridden by CSS
        height=675, # 900 * (3/4) for desktop
        drawing_mode="freedraw",
        key="canvas"
    )
    st.markdown('</div>', unsafe_allow_html=True)

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