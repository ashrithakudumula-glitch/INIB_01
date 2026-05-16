# CIFAR-10 Image Classifier

## Overview
An image classification AI model built with TensorFlow/Keras trained on the CIFAR-10 dataset.

## Dataset
- CIFAR-10: 60,000 images across 10 classes
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Tools Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Model Architecture
- Convolutional Neural Network (CNN)
- 2 Convolutional blocks with MaxPooling and Dropout
- Dense layers for final classification

## Results
- Trained on 50,000 images
- Tested on 10,000 images

## How to Run
1. Clone this repository.
2. Install dependencies
   pip install tensorflow streamlit matplotlib numpy Pillow
3. Run the app
   streamlit run app.py
4. Open your browser.