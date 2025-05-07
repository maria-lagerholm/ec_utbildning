import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import skimage as ski
import av
import os
from pathlib import Path
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

keras.config.disable_interactive_logging()

emotion_labels = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

# ─────────────────────────────────── utility ──────────────────────────────────
def _expand_box(x, y, w, h, img_w, img_h, margin=0.15):
    """Pad Haar box by a % margin, clamp to image bounds."""
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)
    return x1, y1, x2 - x1, y2 - y1

# ────────────────────────────────── Analyzer ──────────────────────────────────
class Analyzer:
    """ Audience Emotion Analyzer """
    def __init__(self) -> None:
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.results = pd.DataFrame(
            columns=["frame"] + emotion_labels + ["x", "y", "width", "height"]
        )
        self.saved = 0  # debug‐crop counter

    def analyze(
        self,
        model=None,
        file: UploadedFile | None = None,
        skip: int = 1,
        confidence: float = 0.5,
        debug_crops: int = 10,        # how many crops to save
    ) -> tuple[bool, pd.DataFrame]:

        if file is None:
            raise ValueError("Must provide a video file.")

        dbg_dir = Path("debug_faces")
        if debug_crops:
            dbg_dir.mkdir(exist_ok=True)

        container = av.open(file, mode="r")
        stream = container.streams.video[0]

        for i, frame in enumerate(container.decode(stream)):
            if i % skip:               # temporal down‑sampling
                continue

            # --- proper decode & grayscale ---
            bgr  = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            img_h, img_w = gray.shape

            # --- Haar face detection ---
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                # pad box so crop isn’t too tight
                x, y, w, h = _expand_box(x, y, w, h, img_w, img_h, margin=0.15)
                face = gray[y:y + h, x:x + w]
                if face.size == 0:
                    continue

                # resize to model input
                face48 = cv2.resize(face, (48, 48)).astype("float32") / 255.0
                face48 = face48.reshape(1, 48, 48, 1)

                # save a few crops for manual inspection
                if self.saved < debug_crops:
                    Image.fromarray((face48[0, :, :, 0] * 255).astype("uint8")).save(
                        dbg_dir / f"frame_{i}_x{x}_y{y}.png"
                    )
                    self.saved += 1

                # predict
                probs = model.predict(face48, verbose=0)[0]
                if probs.max() < confidence:
                    continue  # below threshold

                row = pd.Series(
                    np.concatenate([[i], probs, [x, y, w, h]]),
                    index=self.results.columns,
                )
                self.results = pd.concat(
                    [self.results, row.to_frame().T], ignore_index=True
                )

        container.close()
        return True, self.results