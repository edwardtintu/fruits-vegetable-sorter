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

# Function for image preprocessing
def preprocess_image(img):
    try:
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Image Enhancement (Histogram Equalization)
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        # Image Compression (Resizing to lower size)
        img_compressed = cv2.resize(img_rgb, (150, 150))

        # Image Segmentation (Thresholding)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, img_segmented = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Image Restoration (Denoising)
        img_restored = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)

        return img_rgb, img_enhanced, img_compressed, img_segmented, img_restored
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {e}")
        return None, None, None, None, None

# Function to process image and predict using TFLite model
def process_and_predict(uploaded_file):
    try:
        # Read file and convert to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ Error: Unable to read the uploaded image.")
            return

        # Preprocess Image
        img_original, img_enhanced, img_compressed, img_segmented, img_restored = preprocess_image(img)

        # Display All Preprocessed Images
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, caption="📷 Original Image", use_column_width=True)
            st.image(img_enhanced, caption="✨ Enhanced Image", use_column_width=True)
            st.image(img_compressed, caption="📦 Compressed Image", use_column_width=True)
        with col2:
            st.image(img_segmented, caption="🎭 Segmented Image", use_column_width=True, channels="GRAY")
            st.image(img_restored, caption="🔄 Restored Image", use_column_width=True)

        # Resize and Normalize for Model
        img_resized = cv2.resize(img_original, (224, 224)) / 255.0
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

        # Display Final Prediction
        st.success(f"🟢 Predicted Class: **{class_labels[predicted_class]}**")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# Streamlit UI
st.title("🥦 Vegetable & Fruit Classifier 🍎")
st.write("📌 Upload an image to classify the vegetable or fruit!")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    process_and_predict(uploaded_file)
