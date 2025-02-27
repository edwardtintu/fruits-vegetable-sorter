import streamlit as st
import numpy as np

import tensorflow.lite as tflite

# Load the TFLite model with caching
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="vegetable_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Load model
model = load_tflite_model()

# Class labels
class_labels = ['Apple', 'Banana', 'Carrot', 'Orange', 'Potato', 'Radish', 'Tomato']

# Function to process image and predict using TFLite model
def process_and_predict(uploaded_file):
    try:
        # Read file and convert to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ Error: Unable to read the uploaded image.")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display Uploaded Image
        st.image(img_rgb, caption="📷 Uploaded Image", use_container_width=True)

        # Resize and Normalize
        img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)

        if model is None:
            st.error("❌ Model is not loaded properly.")
            return

        # Get input/output details
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Run inference
        model.set_tensor(input_details[0]['index'], img_array)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])

        # Get predicted class
        predicted_class = np.argmax(predictions)

        # Display Result
        st.success(f"🟢 Predicted Class: **{class_labels[predicted_class]}**")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# Streamlit UI
st.title("🥦 Vegetable Classifier 🌽")
st.write("📌 Upload an image to classify the vegetable!")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    process_and_predict(uploaded_file)
