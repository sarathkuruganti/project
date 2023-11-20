import streamlit as st
import requests
from PIL import Image
import io

st.title("Brain Tumor Classifier")

uploaded_file = st.file_uploader("Choose a brain MRI image...", type="jpg")

if uploaded_file is not None:
    try:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to bytes before sending
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        files = {'file': ('image.jpg', image_bytes.getvalue(), 'image/jpeg')}

        # Send the image to Flask app for prediction
        response = requests.post("http://localhost:5000/predict", files=files)

        if response.status_code == 200:
            prediction = response.json().get('prediction')
            if prediction is not None:
                st.success(f"Prediction: {prediction}")
            else:
                st.error("Error: Prediction not received from Flask app.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error predicting the image: {str(e)}")
