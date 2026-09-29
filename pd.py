import gradio as gr
import json
import tensorflow as tf
from plant_disease_prediction import predict_plant_disease

# 1. Load your model and class names once at startup
model = tf.keras.models.load_model("best_plant_disease_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

# 2. Create a wrapper function for Gradio
def classify_plant(image_filepath):
    # Gradio handles the upload/camera and passes a temporary file path to this function
    result = predict_plant_disease(image_filepath, model, class_names)
    return result

# 3. Build the UI interface
iface = gr.Interface(
    fn=classify_plant,
    # sources=["upload", "webcam"] allows both file uploads and taking a photo
    # type="filepath" ensures your function gets a path string, just like "apple.jpg"
    inputs=gr.Image(sources=["upload", "webcam"], type="filepath"), 
    outputs="text",
    title="Plant Disease Predictor",
    description="Upload a picture of a plant leaf or use your camera to diagnose it."
)

# 4. Launch the web app
iface.launch(share=True)