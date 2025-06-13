import streamlit as st
import cv2
import numpy as np
import pickle

# Load your trained classifier
model = pickle.load(open('model.p', 'rb'))

# Load your label encoder (if you used one)
label_encoder = pickle.load(open('labels.p', 'rb'))  # replace with your actual label encoder file if different

st.title("Sign Language to Speech Converter")

# Upload image
uploaded_file = st.file_uploader("Upload a sign language image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Preprocess as per your training (adjust size as per your model)
    img_resized = cv2.resize(img, (224, 224))  # change 224x224 if your model used other size
    img_flatten = img_resized.flatten().reshape(1, -1)

    # Show image
    st.image(img_resized, caption="Uploaded Image", channels="BGR")

    # Predict
    pred_class = model.predict(img_flatten)[0]

    # Get label
    label = label_encoder.inverse_transform([pred_class])[0]

    st.success(f"Predicted Sign: {label}")

    # ⚠ Skip pyttsx3 in Streamlit Cloud — local speech not supported
    # If running locally, uncomment below:
    # import pyttsx3
    # engine = pyttsx3.init()
    # engine.say(label)
    # engine.runAndWait()
