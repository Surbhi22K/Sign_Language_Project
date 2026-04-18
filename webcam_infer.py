"""
Live webcam inference on the laptop for testing real-time robustness.
Uses HandDetector to crop the hand, resizes to dataset format, and runs model prediction.
"""

import cv2
import numpy as np

from new_code.data.transforms import resize_and_normalize
from new_code.inference.hand_detector import HandDetector
from new_code.inference.sequence_decoder import decode_sequence
from new_code.utils.checkpoint import load_checkpoint
from new_code.utils.config import CONFIG, get_model_path

def live_webcam_infer(model_path=None, labels=None, confidence_threshold=0.90):
    model_path = model_path or get_model_path()
    model = load_checkpoint(model_path)

    if labels is None:
        labels = [chr(c) for c in range(ord("A"), ord("Z")) if chr(c) != "J"]
        labels = labels[: model.output_shape[-1]]

    print(f"Loaded model from {model_path}")
    print("Starting webcam... Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open laptop webcam.")
        return

    image_size = tuple(CONFIG["image_size"])
    raw_predictions = []

    with HandDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        padding=0.25,
    ) as detector:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            
            # 1: Detect and crop hand
            hand_crop = detector.crop_hand(frame)
            
            # 2: Predict if hand was found (crop is smaller than frame)
            # Actually, crop_hand returns the full frame if no hand is found.
            if hand_crop.shape != frame.shape:
                processed = resize_and_normalize(hand_crop, image_size)
                img = np.expand_dims(processed, axis=0)
                pred = model.predict(img, verbose=0)
                
                class_idx = int(np.argmax(pred, axis=1)[0])
                confidence = float(np.max(pred))
                
                # Show crop on screen for debugging
                crop_disp = cv2.resize(hand_crop, (200, 200))
                display_frame[0:200, 0:200] = crop_disp
                
                if confidence >= confidence_threshold:
                    letter = labels[class_idx] if class_idx < len(labels) else "?"
                    raw_predictions.append(letter)
                    color = (0, 255, 0)
                else:
                    letter = "-"
                    color = (0, 0, 255)
                    
                cv2.putText(display_frame, f"Conf: {confidence*100:.1f}%", (210, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                letter = "?"
                cv2.putText(display_frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Keep sequence from growing infinitely large
            if len(raw_predictions) > 100:
                raw_predictions = raw_predictions[-100:]
                
            decoded = decode_sequence(raw_predictions)
            
            # Show output
            cv2.putText(display_frame, f"Raw: {letter}", (210, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Seq: {decoded}", (10, h - 30 if (h:=frame.shape[0]) else 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            cv2.imshow("Sign Language Predictor", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    from new_code.data.dataset import SignLanguageDataset
    # Get labels
    try:
        ds = SignLanguageDataset()
        gen = ds.train_generator()
        class_labels = [k for k, _ in sorted(gen.class_indices.items(), key=lambda x: x[1])]
    except Exception as e:
        print("Warning: could not load classes from training data directory, using defaults.")
        class_labels = [chr(c) for c in range(ord("A"), ord("Z")) if chr(c) != "J"]
    
    live_webcam_infer(labels=class_labels, confidence_threshold=0.85)
