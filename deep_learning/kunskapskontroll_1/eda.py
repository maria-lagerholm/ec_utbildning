import av
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model("modelv1.keras")

# Open video
container = av.open("IMG_4123.mov")  
stream = next(s for s in container.streams if s.type == "video")

# Face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

frame_count = 0
faces_found = 0
probs_logged = 0

for frame in container.decode(stream):
    frame_count += 1
    if frame_count % 10 != 0:
        continue  # skip most frames

    bgr = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"Frame {frame_count}: no faces")
        continue

    print(f"Frame {frame_count}: {len(faces)} face(s)")
    faces_found += len(faces)

    for i, (x, y, w, h) in enumerate(faces):
        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (48, 48)).astype("float32") / 255.
        face_input = face_resized.reshape(1, 48, 48, 1)

        probs = model.predict(face_input, verbose=0)[0]
        label = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        dominant = label[np.argmax(probs)]

        print(f" → Face {i+1}: {dict(zip(label, probs.round(3)))}  → {dominant.upper()}")
        probs_logged += 1

        # Visualize one face
        if probs_logged < 3:
            import PIL.Image as Image
            img = Image.fromarray((face_resized * 255).astype("uint8"))
            img.show()

    if faces_found > 5:
        break

print("\n✅ Finished analysis.")