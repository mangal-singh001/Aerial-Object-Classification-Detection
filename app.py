import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import time

st.set_page_config(page_title="Aerial Object Classification", layout="centered")
st.title("Aerial Object Classification (Bird vs Drone)")


MODEL_FILENAME = r"deploy_models\bird_vs_drone_model.keras"  # <- change if needed
CLASS_NAMES = ["Bird", "Drone"]  # adjust if your label ordering differs
DEFAULT_IMAGE_SIZE = (224, 224)  # fallback; will try to infer from model if possible

# === Model loading ===
@st.cache_resource
def load_classifier(path: str):
    if not os.path.exists(path):
        return None, f"Model file/folder not found: {path}"
    try:
        model = tf.keras.models.load_model(path)
        # attempt to infer expected input size from model if available
        input_shape = None
        try:
            # model.input_shape may be like (None, h, w, c) or (None, h, w)
            ish = model.input_shape
            if isinstance(ish, (list, tuple)) and len(ish) >= 4:
                h, w = ish[1], ish[2]
                if all(isinstance(x, int) and x > 0 for x in (h, w)):
                    input_shape = (int(h), int(w))
        except Exception:
            input_shape = None
        return (model, input_shape, f"Loaded model: {path}")
    except Exception as e:
        return (None, None, f"Error loading model: {e}")

model, inferred_size, model_status = load_classifier(MODEL_FILENAME)

st.markdown("**Model status:**")
if model is None:
    st.error(model_status)
else:
    size_info = f" (inferred input size: {inferred_size[0]}x{inferred_size[1]})" if inferred_size else ""
    st.success(model_status + (size_info if inferred_size else ""))

# === Helpers ===
def get_image_size():
    if inferred_size:
        return inferred_size
    return DEFAULT_IMAGE_SIZE

def preprocess_pil(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_from_model(model, img_array):
    preds = model.predict(img_array)
    # Binary (sigmoid) case: shape (1,1)
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0][0])
        # convention: prob > 0.5 -> CLASS_NAMES[1]
        if len(CLASS_NAMES) >= 2:
            if prob > 0.5:
                return CLASS_NAMES[1], prob
            else:
                return CLASS_NAMES[0], 1.0 - prob
        else:
            return ("Positive" if prob > 0.5 else "Negative"), prob
    # Multi-class / softmax
    elif preds.ndim >= 2:
        p = preds[0]
        idx = int(np.argmax(p))
        confidence = float(p[idx])
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        return label, confidence
    else:
        return "Unknown", 0.0

# === UI ===
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        img = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    st.image(img, use_container_width=True)

    if st.button("Predict"):
        if model is None:
            st.error("No model loaded. Update MODEL_FILENAME or place the model in the project folder.")
        else:
            target_size = get_image_size()
            start = time.time()
            try:
                x = preprocess_pil(img, target_size=target_size)
                label, conf = predict_from_model(model, x)
                elapsed = time.time() - start
                st.success(f"Prediction: **{label}** — Confidence: **{conf:.2%}**")
                st.write(f"Inference time: {elapsed:.2f}s")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("Upload an image to run prediction.")
