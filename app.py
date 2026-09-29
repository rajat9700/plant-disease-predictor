import os
import gradio as gr
import json
import tensorflow as tf
from plant_disease_prediction import predict_plant_disease

print("1. Starting app...")
print("2. Loading model...")
model = tf.keras.models.load_model("best_plant_disease_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)
print("Model loaded successfully!")

# 2. Update your wrapper function to return a dictionary
def classify_plant(image_filepath):
    # Unpack the tuple your function currently returns
    disease_name, confidence = predict_plant_disease(image_filepath, model, class_names)
    
    # gr.Label expects a dictionary format: {"Class Name": probability}
    # (If your confidence is a percentage like 99.39, divide by 100 so the UI bar renders correctly)
    probability = confidence / 100.0 if confidence > 1 else confidence
    
    return {disease_name: probability}

# 3. Change the output to gr.Label()
iface = gr.Interface(
    fn=classify_plant,
    inputs=gr.Image(sources=["upload", "webcam"], type="filepath"),
    outputs=gr.Label(label="Disease Prediction"), # <-- THIS IS THE KEY CHANGE
    title="Plant Disease Predictor"
)

if __name__ == "__main__":
    print("3. Launching Gradio interface...")
    port = int(os.environ.get("PORT", 10000))
    iface.launch(server_name="0.0.0.0", server_port=port)
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
