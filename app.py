import streamlit as st
import cv2
import numpy as np
import pydicom
import pickle

def load_model():
    with open('classifier_dcm.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def dicom_to_image(dicom_file):
    dicom_data = pydicom.dcmread(dicom_file)
    img_array = dicom_data.pixel_array
    return cv2.convertScaleAbs(img_array)  

def classify_image(image):
    model = load_model()

    img_resized = cv2.resize(image, (128, 128))

    img_flattened = img_resized.reshape(1, -1)

    cluster_label = model.predict(img_flattened)
    return cluster_label[0]

def main():
    st.title("Image Classification with SVM")
    st.write("Upload a `.jpg` or `.dcm` image to classify it.")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "dcm"])

    if uploaded_file is not None:
        image = None
        
        if uploaded_file.name.lower().endswith(".dcm"):
            image = dicom_to_image(uploaded_file)
            st.image(image, caption="Converted DICOM Image", use_column_width=True)  # Fixed here
        else:
            image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            st.image(image, caption="Uploaded Image", use_column_width=True)  # Fixed here

        if image is not None:
            label = classify_image(image)
            st.write(f"Predicted cluster: {label}")
        else:
            st.error("Failed to process the image.")

if __name__ == "__main__":
    main()

