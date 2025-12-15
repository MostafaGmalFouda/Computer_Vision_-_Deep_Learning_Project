import base64
import numpy as np
import cv2
import os
import glob
import gc
import random
import pathlib
from collections import deque, Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
import math

# --- AI Libraries ---
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import torch.nn.functional as F
import mediapipe as mp # MediaPipe

# ==========================================
# 1. Configuration / Path Settings
# ==========================================

RAW_DATASET_PATH = r"D:\ASL_Project\dataset\Train_Alphabet" 
PROCESSED_DATASET_PATH = r"D:\ASL_Project\dataset\Train_Alphabet_Processed_128"
TEST_FOLDER_PATH = r'E:\Computer_Vision_-_Deep_Learning_Project\dataset\Test_Alphabet'

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, r'E:\Computer_Vision_-_Deep_Learning_Project\Model\best_model.pth') 

# ==========================================
# 1.5 MediaPipe Setup
# ==========================================
mp_hands = mp.solutions.hands
HANDS_DETECTOR = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
HAND_CROP_PADDING = 60

# ==========================================
# 2. Model Setup
# ==========================================

IMAGE_SIZE = 128 
IMAGES_PER_PAGE = 20

# Load mean & std
mean = torch.tensor([0.4822, 0.4417, 0.3973])
std = torch.tensor([0.2228, 0.2257, 0.2268])

# Load Classes
try:
    full_dataset = datasets.ImageFolder(PROCESSED_DATASET_PATH) 
    ASL_CLASSES = full_dataset.classes
except Exception as e:
    ASL_CLASSES = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ASL_CNN Model
class ASL_CNN(nn.Module):
    def __init__(self, num_classes=len(ASL_CLASSES)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            dummy = self.features(dummy)
            self.flatten_dim = dummy.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load Model Weights
model = ASL_CNN(num_classes=len(ASL_CLASSES)).to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found!")

# Define Preprocess Transform
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()) 
])


# ==========================================
# 3. App Setup
# ==========================================
app = Flask(__name__)
CORS(app)

# ==========================================
# 4. APIs
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data_base64 = data.get('image', None)
        if not image_data_base64: return jsonify({'error': 'No image'}), 400

        # Base64 to OpenCV image
        img_bytes = base64.b64decode(image_data_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_opencv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_opencv is None: return jsonify({'error': 'Decode failed'}), 400

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        h, w, _ = img_opencv.shape
        
        # MediaPipe: Detect Hand
        results = HANDS_DETECTOR.process(img_rgb)
        
        predicted_char = "No Hand"
        confidence_score = 0
        hand_img_base64 = None
        
        if results.multi_hand_landmarks:
            # Calculate Bounding Box
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in results.multi_hand_landmarks[0].landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Apply Padding
            pad = HAND_CROP_PADDING
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

            # Crop and Preprocess
            hand_img_cropped = img_rgb[y_min:y_max, x_min:x_max]
            
            if hand_img_cropped.size > 0:
                pil_img = Image.fromarray(hand_img_cropped) 
                input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

                # Prediction
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1) 
                    conf, pred_idx = torch.max(probs, 1)

                predicted_char = ASL_CLASSES[pred_idx.item()]
                confidence_score = int(conf.item() * 100)
                
                # Convert cropped hand image back to Base64 for display
                hand_img_cropped_bgr = cv2.cvtColor(hand_img_cropped, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpeg', hand_img_cropped_bgr)
                hand_img_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                
            else:
                 predicted_char = "No ROI"
        
        # Cleanup
        del img_bytes, np_arr, img_opencv, img_rgb
        gc.collect()

        # Output
        return jsonify({
            'predicted_class': predicted_char, 
            'confidence': confidence_score,
            'hand_image': hand_img_base64,
            'hand_detected': bool(results.multi_hand_landmarks)
        })
    
    except Exception as e: 
        print(f"Error in predict route: {e}")
        return jsonify({'error': str(e)}), 500

# --- API: Test Random Image ---
def get_all_test_images(root_folder):
    image_paths = []
    if not os.path.exists(root_folder): return []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

@app.route('/test_random_image', methods=['GET'])
def test_random_image():
    try:
        all_images = get_all_test_images(TEST_FOLDER_PATH)
        if not all_images: return jsonify({'error': f'No images found'}), 404
        
        img_path = random.choice(all_images)
        actual_label = pathlib.Path(img_path).parent.name
        
        frame = cv2.imread(img_path)
        if frame is None: return jsonify({'error': 'Failed to read image'}), 500
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)
            predicted_label = ASL_CLASSES[idx.item()]
            
        status = "CORRECT" if predicted_label.lower() == actual_label.lower() else "WRONG"
        
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        del frame, rgb_frame, pil_img, input_tensor
        gc.collect()
        
        return jsonify({'image': img_base64, 'actual': actual_label, 'predicted': predicted_label, 'status': status})
    except Exception as e: return jsonify({'error': str(e)}), 500

# --- APIs for Data Visualization ---
DATASET_PATHS_CACHE = {} 
BASE64_CACHE = {} 

@app.route('/get_all_image_paths/<class_name>/<process_type>', methods=['GET'])
def get_all_image_paths(class_name, process_type):
    global DATASET_PATHS_CACHE
    if class_name not in ASL_CLASSES: return jsonify({'error': 'Invalid class'}), 400
    cache_key = f"{class_name}_{process_type}"
    if cache_key in DATASET_PATHS_CACHE: return jsonify({'image_count': len(DATASET_PATHS_CACHE[cache_key])})

    if process_type.lower() == 'before': base_path = RAW_DATASET_PATH
    else: base_path = PROCESSED_DATASET_PATH

    class_folder = os.path.join(base_path, class_name)
    
    if not os.path.isdir(class_folder) and process_type.lower() == 'before':
        potential = os.path.join(base_path, 'Before', class_name)
        if os.path.isdir(potential): class_folder = potential

    if not os.path.isdir(class_folder): return jsonify({'error': f'Folder not found: {class_folder}'}), 404

    exts = ('*.jpg', '*.jpeg', '*.png')
    paths = []
    for ext in exts: paths.extend(glob.glob(os.path.join(class_folder, ext)))
    DATASET_PATHS_CACHE[cache_key] = paths
    return jsonify({'class_name': class_name, 'image_count': len(paths)})

@app.route('/get_image_batch/<class_name>/<int:page_number>/<process_type>', methods=['GET'])
def get_image_batch(class_name, page_number, process_type):
    global DATASET_PATHS_CACHE, BASE64_CACHE
    cache_key = f"{class_name}_{process_type}"
    if cache_key not in DATASET_PATHS_CACHE: return jsonify({'error': 'Refresh first'}), 400 
    
    paths = DATASET_PATHS_CACHE[cache_key]
    start = (page_number - 1) * IMAGES_PER_PAGE
    batch_paths = paths[start : start + IMAGES_PER_PAGE]
    
    images_base64 = []
    for path in batch_paths:
        if path in BASE64_CACHE:
            images_base64.append(BASE64_CACHE[path])
            continue
        try:
            img = cv2.imread(path)
            if img is not None:
                _, buffer = cv2.imencode('.jpeg', img)
                b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                
                BASE64_CACHE[path] = b64
                images_base64.append(b64)
        except: pass
            
    return jsonify({'class_name': class_name, 'images': images_base64})


if __name__ == '__main__':
    print(f"Server Running.")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)