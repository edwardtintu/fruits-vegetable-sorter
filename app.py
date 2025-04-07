import streamlit as st
import numpy as np
import cv2
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

# Custom CSS for better UI with black text
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: black;
            text-align: center;
            font-size: 2.5em;
        }
        .uploadedImage {
            border-radius: 10px;
            border: 2px solid #ddd;
            padding: 10px;
        }
        .prediction {
            font-size: 1.5em;
            color: black;
            background-color: #d4edda;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Image preprocessing function
def preprocess_image(img):
    try:
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Image Enhancement (Histogram Equalization)
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        # Image Compression (Resizing to 150x150)
        img_resized = cv2.resize(img_enhanced, (150, 150))

        # Image Segmentation (Grayscale + Threshold)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Normalize and reshape
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Streamlit UI
st.title("Vegetable/Fruit Classifier 🍎🥕🍌")

uploaded_file = st.file_uploader("Upload an image of a vegetable or fruit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    if processed_image is not None:
        # Run inference
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.set_tensor(input_details[0]['index'], processed_image)
        model.invoke()

        # Get prediction
        output_data = model.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        confidence = np.max(output_data) * 100
        prediction = class_labels[predicted_index]

        # Show prediction
        st.markdown(f'<div class="prediction">Prediction: {prediction} ({confidence:.2f}%)</div>', unsafe_allow_html=True)
