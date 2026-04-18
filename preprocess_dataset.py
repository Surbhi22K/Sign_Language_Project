import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from new_code.inference.hand_detector import HandDetector
from new_code.utils.config import CONFIG

def preprocess_dataset(input_dir: str, output_dir: str):
    """
    Reads all images in input_dir, applies HandDetector to crop the hand,
    and saves the cropped/resized images to output_dir.
    """
    print(f"Preprocessing dataset from {input_dir} to {output_dir}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # We use a lower confidence to ensure we catch hands in the varied training data
    with HandDetector(
        static_image_mode=True, 
        max_num_hands=1,
        min_detection_confidence=0.3, 
        padding=0.25
    ) as detector:
        
        detected_total = 0
        total_images = 0

        for cls in classes:
            in_cls_dir = os.path.join(input_dir, cls)
            out_cls_dir = os.path.join(output_dir, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
            
            images = [f for f in os.listdir(in_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in tqdm(images, desc=f"Processing {cls}"):
                in_path = os.path.join(in_cls_dir, img_name)
                out_path = os.path.join(out_cls_dir, img_name)
                
                total_images += 1
                frame = cv2.imread(in_path)
                if frame is None:
                    continue
                
                # HandDetector will return a cropped square if a hand is found, 
                # or original frame if not found.
                crop = detector.crop_hand(frame)
                
                # Check if detector found a hand (stats will show it)
                if crop.shape != frame.shape:
                    detected_total += 1
                
                # Ensure the crop is 64x64
                target_size = CONFIG["image_size"]
                resized_crop = cv2.resize(crop, target_size)
                
                cv2.imwrite(out_path, resized_crop)
                
        print("\n--- Preprocessing Complete ---")
        print(f"Total processed     : {total_images}")
        print(f"Hands successfully cropped: {detected_total} ({detected_total/max(1, total_images)*100:.1f}%)")
        print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    in_dir = "/mnt/data/Documents/surbhi/dataset"
    out_dir = "/mnt/data/Documents/surbhi/dataset_cropped"
    preprocess_dataset(in_dir, out_dir)
