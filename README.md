# 🖐️ Hand Sign Recognition using Deep Learning & Computer Vision

This project presents a complete **Computer Vision + Deep Learning pipeline** for recognizing hand signs (alphabets) from images and real-time webcam input.

---

## 📌 Project Overview

The system includes **data exploration, preprocessing, augmentation, CNN training, evaluation, and real-time inference**.  
Goal: Build a robust hand sign recognition system.

---

## 🔍 Data Exploration (EDA)

- Number of classes
- Images per class
- Image size statistics
- Corrupted images detection
- Visualization of:
  - Image size distributions
  - Class balance
  - Sample images per class

---

## ⚙️ Preprocessing & Augmentation

**Hand Detection:** MediaPipe Hands (dynamic bounding box)  
**Augmentations:** Brightness, contrast, blur, noise, rotation, perspective  
**Final Image Size:** 128 × 128 RGB

---

## 🏗 Model Architecture

- 4 Convolutional Blocks (Conv + BatchNorm + ReLU + MaxPool)  
- Fully connected classifier (Flatten → Dense → Dropout)  
- Automatic flatten size calculation  

**Loss:** CrossEntropyLoss  
**Optimizer:** Adam  
**Scheduler:** ReduceLROnPlateau  

---

## 🚀 Training Pipeline

- 80% Train / 20% Validation split  
- Training with mean/std normalization  
- Best model saved automatically  

---

## 📊 Evaluation

- Loss curves  
- Validation accuracy  
- Test accuracy  
- Confusion matrix  
- Classification report  

---

## 🎥 Real-Time Hand Sign Recognition

- Webcam input
- Real-time hand detection
- Bounding box + class + confidence score
- Color-coded confidence

Press **`q`** to exit

---

## 🛠 Tech Stack

Python, PyTorch, OpenCV, MediaPipe, NumPy, Matplotlib, Seaborn, scikit-learn

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python preprocessing.py
python train.py
python test.py
python webcam_test.py
