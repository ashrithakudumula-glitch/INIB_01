import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Page config
st.set_page_config(page_title="Image Classifier", page_icon="🧠", layout="centered")

# Custom CSS for clean white design
st.markdown("""
    <style>
        body { background-color: #ffffff; }
        .main { background-color: #ffffff; }
        h1 { color: #222222; font-size: 32px; font-weight: 700; }
        .subtitle { color: #888888; font-size: 16px; margin-bottom: 32px; }
        .result-box { background: #f9f9f9; border-radius: 12px; padding: 24px; margin-top: 24px; text-align: center; }
        .predicted { font-size: 36px; font-weight: 800; color: #222222; }
        .confidence { font-size: 18px; color: #4CAF50; margin-top: 8px; }
        .stProgress > div > div { background-color: #4CAF50; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🧠 Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image and the AI will classify it instantly.</p>", unsafe_allow_html=True)

st.divider()

# Load model
@st.cache_resource
def load_my_model():
    return load_model('cifar10_model.keras')

model = load_my_model()

# Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # Show image nicely centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.divider()

    # Preprocess
    img = image.resize((32, 32)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analysing..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Result
    st.markdown(f"""
        <div class='result-box'>
            <div class='predicted'>🎯 {predicted_class.upper()}</div>
            <div class='confidence'>Confidence: {confidence:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Probabilities
    st.markdown("#### All Class Probabilities")
    for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.write(f"{name} {prob*100:.1f}%")