import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for Streamlit Cloud

# Inject JavaScript for mobile detection
st.markdown(
    """
    <script>
    function isMobile() {
        return (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent));
    }
    window.isMobile = isMobile();
    </script>
    """,
    unsafe_allow_html=True
)

# Hidden checkbox for storing mobile detection status
is_mobile = st.checkbox("Mobile Device", value=False, key="mobile_checkbox", help="Auto-detected device type")

# Automatically set the checkbox value based on JavaScript detection
st.markdown(
    """
    <script>
    document.getElementById('mobile_checkbox').checked = window.isMobile;
    </script>
    """,
    unsafe_allow_html=True
)

# Cache model loading
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)  
    model_path = os.path.join(base_dir, "joblib", "cnn_model_aug.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = list("0123456789") + ["+", "-"]

def center_symbol(img, size=28, box=20):
    h, w = img.shape
    scale = min(box / h, box / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=resized.dtype)
    y, x = (size - new_h)//2, (size - new_w)//2
    out[y:y+new_h, x:x+new_w] = resized
    return out

def segment_symbols(arr, margin=10):
    _, bin_ = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])

    symbols = []
    for (x, y, w, h) in boxes:
        xm = max(0, x - margin)
        ym = max(0, y - margin)
        wm = min(arr.shape[1], x + w + margin) - xm
        hm = min(arr.shape[0], y + h + margin) - ym
        cropped = arr[ym:ym+hm, xm:xm+wm]
        symbols.append(center_symbol(cropped))
    return symbols

def predict_expression(img_pil):
    gray = img_pil.convert("L")
    inv = ImageOps.invert(gray)
    arr = np.array(inv)
    symbol_imgs = segment_symbols(arr)

    result = []
    for s in symbol_imgs:
        s = (s.astype(np.float32) / 255.0).reshape((1, 28, 28, 1))
        out = model.predict(s)
        label = class_names[out.argmax()]
        result.append(label)

    expression = "".join(result)
    try:
        answer = eval(expression)
    except:
        answer = "Could not evaluate"
    return expression, answer

# ------------------ Streamlit UI ------------------
st.title("Handwritten \n Subtraction/Addition Solver")
st.markdown("<p style='font-size:20px;'>Draw digits and + or - signs below, then click <strong>Solve</strong>.</p>", unsafe_allow_html=True)

# Adjust canvas size dynamically based on mobile detection
canvas_width = 300 if is_mobile else 900
canvas_height = 300 if is_mobile else 900

canvas = st_canvas(
    fill_color="white",
    stroke_width=16,
    stroke_color="black",
    background_color="white",
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Solve"):
    if canvas.image_data is not None:
        data = canvas.image_data.astype("uint8")
        pil_img = Image.fromarray(data, "RGBA")

        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg

        recognized, solution = predict_expression(pil_img)
        emojis = ["😎", "😊", "🙂‍↔️", "🫡", "👍", "😉", "🙂"]
        chosen_emoji = random.choice(emojis)

        st.markdown(
            f"<p style='font-size:26px;'>Recognized: {recognized}</p>", 
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-size:26px;'>Solution: {solution} {chosen_emoji}</p>", 
            unsafe_allow_html=True
        )
    else:
        st.warning("No drawing found. Please draw something!")
