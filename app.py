import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Page config
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🧠")

# Title
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image and the AI will classify it into one of 10 categories!")

# Load model
@st.cache_resource
def load_my_model():
    return load_model('cifar10_model.keras')

model = load_my_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize((32, 32)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analysing image..."):
        predictions = model.predict(img_array, verbose=0)

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.success(f"### Prediction: {predicted_class.upper()}")
    st.info(f"### Confidence: {confidence:.2f}%")

    # Show all probabilities as a bar chart
    st.write("### All Class Probabilities:")
    prob_dict = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    st.bar_chart(prob_dict)