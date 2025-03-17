import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os, random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
st.set_page_config(layout="wide")

# CSS for responsive design with smaller text and canvas on mobile
st.markdown("""
<style>
:root, body, .stApp {
    background: #fff !important;
    color: #000 !important;
    color-scheme: light !important;
}
/* Responsive text size */
@media (max-width: 768px) {
    body, .stApp {
        font-size: 14px !important;
    }
}
/* Make the st_canvas element auto-size to 100% width */
[data-testid="stCanvas"] {
    width: 100% !important;
    max-width: 100% !important;
}
/* Responsive canvas size */
@media (max-width: 768px) {
    [data-testid="stCanvas"] > canvas {
        width: 50% !important;
        height: auto !important;
    }
}
[data-testid="stCanvas"] > canvas {
    width: 100% !important;
    height: auto !important;
}
/* Style our 'Solve' button green */
div.stButton > button:first-child {
    background-color: #4CAF50 !important;
    color: white !important;
    border: none !important;
    padding: 10px 24px !important;
    font-size: 16px !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

this_dir = os.path.dirname(__file__)
model_path = os.path.join(this_dir, "joblib", "cnn_model_aug.keras")
model = tf.keras.models.load_model(model_path)
labels = list("0123456789") + ["+", "-"]

def segment_and_center(img):
    _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 25]
    boxes.sort(key=lambda b: b[0])
    out = []
    for x, y, w, h in boxes:
        roi = img[max(0,y-10):y+h+10, max(0,x-10):x+w+10]
        roi = center_symbol(roi)
        out.append(roi)
    return out

def center_symbol(symbol, size=28, pad=20):
    h, w = symbol.shape
    scale = min(pad / h, pad / w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(symbol, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=resized.dtype)
    sy, sx = (size - nh)//2, (size - nw)//2
    out[sy:sy+nh, sx:sx+nw] = resized
    return out

def predict_expr(pil_img):
    gray = pil_img.convert("L")
    inv = ImageOps.invert(gray)
    arr = np.array(inv)
    chunks = segment_and_center(arr)
    preds = []
    for c in chunks:
        c = c.astype(np.float32)/255.0
        c = c.reshape((1,28,28,1))
        p = model.predict(c).argmax()
        preds.append(labels[p])
    expr = "".join(preds)
    try:
        ans = eval(expr)
    except:
        ans = "Could not evaluate"
    return expr, ans

st.title("Handwritten Subtraction/Addition Solver")
st.write("Draw digits and + or - signs below, then click Solve.")

# We only set a height; width is 100% from our CSS
canvas = st_canvas(
    fill_color="white",
    stroke_width=16,
    stroke_color="black",
    background_color="white",
    height=300,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Solve"):
    if canvas.image_data is not None:
        data = canvas.image_data.astype("uint8")
        pil = Image.fromarray(data, "RGBA")
        if pil.mode == "RGBA":
            tmp = Image.new("RGB", pil.size, (255,255,255))
            tmp.paste(pil, mask=pil.split()[3])
            pil = tmp
        expr, sol = predict_expr(pil)
        emo = random.choice(["😎","😊","🤔","🫡","👍","😉","🙂"])
        st.write(f"**Recognized:** {expr}")
        st.write(f"**Solution:** {sol} {emo}")
    else:
        st.warning("Please draw something first.")
