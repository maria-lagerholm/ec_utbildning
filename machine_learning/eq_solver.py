import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os, random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
st.set_page_config(layout="wide")

# Custom CSS: Responsive canvas; force canvas buttons and clickable elements to grey
st.markdown("""
<style>
/* Canvas container should fill available width */
[data-testid="stCanvas"] {
    width: 100% !important;
    max-width: 100% !important;
}

/* Canvas itself: default 100% width, but on smaller screens use 50% */
[data-testid="stCanvas"] > canvas {
    width: 100% !important;
    height: auto !important;
}
@media (max-width: 1024px) {
    [data-testid="stCanvas"] > canvas {
        width: 50% !important;
        height: auto !important;
    }
    h1 {
        font-size: 2rem !important;
    }
}

/* Style all buttons within the canvas to appear grey */
[data-testid="stCanvas"] button,
[data-testid="stCanvas"] [role="button"] {
    background-color: #808080 !important;
    color: white !important;
    border: none !important;
    padding: 6px 12px !important;
    font-size: 14px !important;
    border-radius: 8px !important;
}

/* Style the external Solve button as well */
div.stButton > button:first-child {
    background-color: #808080 !important;
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
    inv = ImageOps.invert(gray)
    arr = np.array(inv)
    chunks = segment_and_center(arr)
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

st.title("Handwritten Subtraction/Addition Solver")
st.write("Draw digits and + or - signs below, then click Solve.")

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
        pil_img = Image.fromarray(data, "RGBA")
        if pil_img.mode == "RGBA":
            tmp = Image.new("RGB", pil_img.size, (255, 255, 255))
            tmp.paste(pil_img, mask=pil_img.split()[3])
            pil_img = tmp
        expr, sol = predict_expr(pil_img)
        emo = random.choice(["😎", "😊", "🤔", "🫡", "👍", "😉", "🙂"])
        st.write(f"**Recognized:** {expr}")
        st.write(f"**Solution:** {sol} {emo}")
    else:
        st.warning("Please draw something first.")
