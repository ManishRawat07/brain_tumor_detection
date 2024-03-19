import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("effnet.h5")

# Define the labels for tumor types
labels = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

def predict_tumor_type(image):
    # Preprocess the image
    image = cv2.resize(image, (150, 150))

    # Make predictions
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    return labels[predicted_class_index]

def get_class_color(class_name):
    if class_name == 'No Tumor':
        return 'green'
    else:
        return 'red'

def main():
    st.markdown('<h1 style="font-size: 28px;">Brain Tumor Detection</h1>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an MRI image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Display the uploaded MRI image with reduced width
        st.image(image, caption='Uploaded MRI Image', width=250)

        # Make predictions
        predicted_tumor_type = predict_tumor_type(image)
        color = get_class_color(predicted_tumor_type)
        
        # Display results with adjusted styling
        st.markdown(
            f'<div style="font-size: 20px;">Results: Predicted tumor type - <span style="color:{color};">{predicted_tumor_type}</span></div>',
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()

# streamlit run brain_tumor_detector.py 
# /home/adminuser/venv/bin/python -m pip install --upgrade pip