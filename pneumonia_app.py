import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import BytesIO

# Load the pre-trained model
model = tf.keras.models.load_model('p_model.h5')

# Define convolutional layers for Grad-CAM
conv_layers = [
    "block1_conv1",
    "block1_conv2",
    "block1_pool",
    "block2_conv1",
    "block2_conv2",
    "block2_pool",
    "block3_conv1",
    "block3_conv2",
    "block3_conv3",
    "block3_pool"
]

# Helper functions
def preprocess_image(image):
    """Preprocess the input image for the model."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input is not a valid PIL Image.")
    image = image.resize((256, 256))  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = np.stack([image] * 3, axis=-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def classify_image(model, image):
    """Classify the input image using the loaded model."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Assuming binary classification
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return int(prediction >= 0.5), confidence

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for the input image."""
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """Superimpose Grad-CAM heatmap onto the original image."""
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8))
    jet_heatmap = jet_heatmap.resize(img.size)

    jet_heatmap = np.array(jet_heatmap) / 255.0
    superimposed_img = np.array(img) / 255.0
    superimposed_img = jet_heatmap * alpha + superimposed_img
    superimposed_img = np.clip(superimposed_img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed_img)

# Main Streamlit App
def main():
    """Main function for the Streamlit app."""
    st.sidebar.title("Upload Chest X-ray")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    st.title("Pneumonia Detection with Grad-CAM")
    
    if uploaded_file is not None:
        try:
            # Load and preprocess the image
            image = Image.open(uploaded_file).convert("RGB")
            st.write("### Uploaded Image")
            col1, col2 = st.columns([1, 1])  # Create two equal columns for alignment

            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            # Preprocess the image for Grad-CAM
            processed_image = preprocess_image(image)

            # Get prediction and confidence
            result, confidence = classify_image(model, image)
            confidence *= 100

            # Display prediction
            st.markdown(f"### Prediction: <span style='color:red;'>Positive</span>" if result else f"### Prediction: <span style='color:green;'>Negative</span>", unsafe_allow_html=True)
            st.write(f"### Probability: {confidence:.2f}%")

            # Slider for selecting convolutional layer
            layer_index = st.slider(
                "Adjust Slider To View Different Convolutional Layers",
                min_value=0,
                max_value=len(conv_layers) - 1,
                value=0,
                step=1,
                format="%d"
            )
            selected_layer = conv_layers[layer_index]

            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name=selected_layer)
            gradcam_image = save_and_display_gradcam(image, heatmap)

            # Display Grad-CAM
            with col2:
                st.image(gradcam_image, caption=f"Grad-CAM Heatmap ({selected_layer})", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing image: {e}")

# Run the app
if __name__ == "__main__":
    main()