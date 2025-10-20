from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from PIL import Image
import io
import google.generativeai as genai
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import threading
import multiprocessing as mp

# --- macOS Fix ---
# Set the start method for multiprocessing to 'spawn'.
# This is a crucial step to prevent low-level crashes on macOS when using
# both PyTorch and TensorFlow in the same application.
try:
    mp.set_start_method('spawn', force=True)
    print("INFO: Multiprocessing start method set to 'spawn'.")
except RuntimeError:
    pass # It might have been set already, which is fine.

# --- Initialize the Flask application ---
app = Flask(__name__)
CORS(app)

# --- API Key Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- Global Model Variables (Initialized as None) ---
yolo_model = None
llm_model = None
classifier_model = None
models_loaded = threading.Event() # To signal when models are loaded

def load_models():
    """
    Loads all AI models if they haven't been loaded yet.
    This function is designed to be called at the start of the first request.
    """
    global yolo_model, llm_model, classifier_model
    
    # Use a lock to prevent multiple requests from trying to load models at the same time
    with threading.Lock():
        if yolo_model is None:
            print("Loading YOLOv5 model...")
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            print("YOLOv5 model loaded.")
        
        if llm_model is None:
            print("Loading Gemini LLM model...")
            llm_model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini LLM model loaded.")
        
        if classifier_model is None:
            print("Loading AI vs Natural classifier...")
            classifier_url = "https://tfhub.dev/google/aiy/vision/classifier/real_vs_fake/1"
            classifier_model = hub.KerasLayer(classifier_url, input_shape=(224, 224, 3))
            print("Classifier model loaded.")
            
        models_loaded.set() # Signal that all models are now loaded

def classify_image(image: Image.Image):
    image = image.resize((224, 224))
    image_np = np.array(image) / 255.0
    image_batch = np.expand_dims(image_np, axis=0)
    predictions = classifier_model(image_batch)
    ai_score = tf.nn.sigmoid(predictions[0]).numpy()[0]
    return round(ai_score * 100, 1)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    # Ensure models are loaded before proceeding. 
    # This will block until the first request has finished loading the models.
    if not models_loaded.is_set():
        load_models()

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Step 1: AI vs. Natural Classification
    ai_generated_percentage = classify_image(img)
    natural_percentage = 100 - ai_generated_percentage

    # Step 2: Object Detection
    results = yolo_model(img)
    detections = results.pandas().xyxy[0]
    detected_objects = []
    for index, row in detections.iterrows():
        detected_objects.append({
            "name": row['name'].capitalize(),
            "confidence": round(row['confidence'] * 100, 1),
            "icon": '🔎'
        })
    
    # Step 3: LLM Summary Generation
    if not detected_objects:
        summary = "The model could not detect any specific objects in the image."
    else:
        object_list_str = ", ".join([f"{obj['name']}" for obj in detected_objects])
        prompt = f"""
        An image was analyzed. It is considered to be {ai_generated_percentage}% likely to be AI-generated.
        It contains the following objects: {object_list_str}.
        Based on this, provide a concise, one-paragraph summary of the image's content.
        Describe a plausible scene that would contain these items and mention the nature of the image (AI or real).
        Summary:
        """
        try:
            response = llm_model.generate_content(prompt)
            summary = response.text
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = "Could not generate summary due to an API error."

    # Combine All REAL Results
    analysis_result = {
      "aiGenerated": ai_generated_percentage,
      "natural": natural_percentage,
      "detectedObjects": detected_objects,
      "summary": summary
    }

    return jsonify(analysis_result)

if __name__ == '__main__':
    # We can now set use_reloader=True if we want, as models are loaded lazily.
    app.run(debug=True, port=5000)

