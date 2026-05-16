import sys
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the saved model
print("Loading model...")
model = load_model('cifar10_model.keras')
print("Model loaded! ✅")

# Get image path from command line
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Load and preprocess the image
print(f"Processing image: {image_path}")
img = Image.open(image_path)
img = img.resize((32, 32))         # Resize to 32x32 like CIFAR-10
img = img.convert('RGB')           # Make sure it's RGB
img_array = np.array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(img_array, verbose=0)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

# Show results
print("\n===== PREDICTION RESULTS =====")
print(f"Predicted Class : {predicted_class.upper()}")
print(f"Confidence      : {confidence:.2f}%")
print("\nAll class probabilities:")
for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
    bar = '█' * int(prob * 30)
    print(f"  {name:<12}: {prob*100:5.2f}% {bar}")
print("==============================")