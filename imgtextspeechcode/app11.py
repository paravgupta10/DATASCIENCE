!pip install streamlit pyngrok pytesseract pyttsx3 opencv-python tensorflow pillow

%%writefile app11.py
import streamlit as st
import numpy as np
import cv2
import pytesseract
import pyttsx3
import tensorflow as tf
from PIL import Image

# Set page title and description
st.title("AI Utility App")
st.subheader("Choose an action below")

# Radio button for selecting an option
option = st.radio("Select an option:", ["Image Classification", "Make Image Readable", "Image to Text", "Text to Speech"])

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Function to classify an image
def classify_image(image):
    model = tf.keras.models.load_model('cifar10_cnn_1.h5')  # Load your trained model
    img = Image.open(image).resize((32, 32))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    if len(img_array.shape) == 3:  # Ensure the image has 3 channels (RGB)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    return f"Classified as {class_names[class_index]}"

# Function to make an image readable
def make_image_readable(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    _, img_processed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(img_processed)

# Function to extract text from an image
def extract_text(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(img)
    return text

# Function to convert text to speech
def convert_text_to_speech(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, "output.mp3")
    engine.runAndWait()
    return "output.mp3"

# Execute button
if st.button("Execute"):
    if uploaded_file is not None:
        if option == "Image Classification":
            result = classify_image(uploaded_file)
            st.write(f"Prediction: {result}")

        elif option == "Make Image Readable":
            processed_image = make_image_readable(uploaded_file)
            st.image(processed_image, caption="Processed Image")

        elif option == "Image to Text":
            extracted_text = extract_text(uploaded_file)
            st.write(f"Extracted Text: {extracted_text}")

        elif option == "Text to Speech":
            text = st.text_area("Enter text for speech conversion:")
            if text:
                audio_file = convert_text_to_speech(text)
                st.audio(audio_file, format="audio/mp3")
            else:
                st.write("Please enter text to convert.")
    else:
        st.write("Please upload an image first.")


from pyngrok import ngrok

!ngrok authtoken *****  

!streamlit run app.py &>/dev/null&

public_url = ngrok.connect(addr=8501, proto="http")  
print("Streamlit app is live at:", public_url)
