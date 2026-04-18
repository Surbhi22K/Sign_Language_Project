import base64
import numpy as np
import cv2
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_code.inference.hand_detector import HandDetector
from new_code.data.transforms import resize_and_normalize
from tensorflow.keras.models import load_model
from new_code.utils.config import get_model_path, CONFIG

app = Flask(__name__)
CORS(app)

print("Loading model for API server...")
model = load_model(get_model_path())
detector = HandDetector()

# Extract labels in order
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        img_b64 = data.get('image', '')
        if img_b64.startswith('data:image'):
            img_b64 = img_b64.split(',')[1]
            
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        # Detect, crop, resize
        hand_crop = detector.crop_hand(frame)
        found = hand_crop.shape != frame.shape
        
        processed = resize_and_normalize(hand_crop, tuple(CONFIG["image_size"]))
        processed = np.expand_dims(processed, axis=0)
        
        pred = model.predict(processed, verbose=0)[0]
        class_idx = int(np.argmax(pred))
        confidence = float(pred[class_idx])
        
        if not found and confidence < 0.95:
            return jsonify({"letter": "?", "confidence": 0, "found": False})
            
        return jsonify({
            "letter": LABELS[class_idx],
            "confidence": round(confidence * 100, 1),
            "found": True
        })
    except Exception as e:
        print("Error during prediction:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("*" * 50)
    print("  Sign Language API Server Running!")
    print("  Listening for Expo App on port 5000...")
    print("*" * 50)
    # Run on all interfaces so phone can connect
    app.run(host='0.0.0.0', port=5000, threaded=True)
