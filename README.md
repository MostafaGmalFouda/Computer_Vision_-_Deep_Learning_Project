# 🖐️ ASL Hand Sign Recognition Project

This project provides a comprehensive system for recognizing American Sign Language (ASL) alphabets using **Deep Learning** and **Computer Vision**. It includes a complete pipeline from data analysis to a real-time interactive Web Dashboard.

---

## 📝 Problem Definition

American Sign Language (ASL) is a vital communication tool for the deaf and hard-of-hearing community. However, understanding and interpreting ASL manually can be time-consuming and prone to errors.

Challenges addressed by this project:

Variability in hand positions, angles, and orientations.

Different lighting conditions and background noise in images and videos.

The need for real-time recognition for interactive applications.

Ensuring high accuracy across all 26 alphabet letters plus a "Blank" class.

## Objective:
Develop an automated system that can detect, preprocess, and classify hand signs in both static images and live webcam video using Deep Learning and Computer Vision techniques, providing accurate and reliable ASL recognition.

---

## 👥 Our Team

| Name | Role | Responsibilities |
| :--- | :--- | :--- |
| **[Moaz Ibrahim Abdallah El-Sayed](https://github.com/MoazIbrahem)** | **Team Leader** | Developed the Web Frontend, Backend (Flask API), and System Integration. |
| **[Mostafa Osama El-Sayed](https://github.com/MustafaOsama)** | **Developer** | Handled Data Preprocessing and MediaPipe Hand Detection logic. |
| **[Mohamed Abdallah Mohamed](https://github.com/mohamed78186)** | **Developer** | Model Architecture design.|
| **[Mohamed El-Shoura](https://github.com/MohammedElshora2005)** | **Developer** | Training.|
| **[Mostafa Gamal Fouda](https://github.com/MostafaGmalFouda)** | **Developer** | Evaluation and Testing. |

---

## 📂 Dataset
The dataset contains images for all 26 alphabets plus a "Blank" class. You can download it here:
[Download ASL Hand Sign Dataset](https://drive.google.com/drive/folders/1FfNud8I6dCAJxmOqXhhuo45CkpinfuuD)

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** MediaPipe, OpenCV
* **Backend:** Flask, Flask-CORS
* **Frontend:** HTML5, CSS3, JavaScript
* **Analysis:** NumPy, Matplotlib, Seaborn, Scikit-learn

---

## ⚙️ Development Pipeline

### 🔍 1. Exploratory Data Analysis (EDA)
* Analyzing class distribution (26 letters + Blank).
* Visualizing image size statistics and detecting corrupted files.
* Sampling images to understand data diversity and class balance.

### 🛠️ 2. Preprocessing & Augmentation
* **Hand Detection:** Utilizing **MediaPipe Hands** to dynamically crop the image around the hand, reducing background noise and improving accuracy.
* **Augmentation:** Applying brightness, contrast, rotation, and noise filters to improve model generalization.
* **Normalization:** Resizing images to 128x128 pixels and applying Mean/Std normalization.

### 🚀 3. Training Pipeline

* **Data Split:** 80% Training / 20% Validation split.
* **Architecture:** A custom CNN with 4 Convolutional blocks (Conv + BatchNorm + ReLU + MaxPool) followed by a fully connected classifier with Dropout.
* **Best Model:** Automatically saving the `best_model.pth` based on the highest validation accuracy during the training epochs.

### 📊 4. Evaluation
* **Metrics:** Tracking Loss Curves and Accuracy for both training and validation sets.
* **Analysis:** Generating a **Confusion Matrix** to identify similar-looking signs and a full **Classification Report** (Precision, Recall, F1-Score).

---

## ▶️ How to Run

> 💡 **Important Note:** Ensure all required libraries are installed before starting:
> ```bash
> pip install -r requirements.txt
> ```

### 1️⃣ Step 1: Model Training
* Open the **Jupyter Notebook** (`model.ipynb`) provided in the project.
* Update the dataset paths in the first cells to match your local directory.
* Select **Run All Cells**. This will process the data, train the model, and save `best_model.pth`.

### 2️⃣ Step 2: API Configuration
* Open `main.py`.
* Update the following paths to match your local folders:
    * `RAW_DATASET_PATH`: Path to original images.
    * `PROCESSED_DATASET_PATH`: Path to cropped/processed images.
    * `TEST_FOLDER_PATH`: Path to the test dataset.
    * `MODEL_PATH`: Path to the saved `best_model.pth`.

### 3️⃣ Step 3: Start the Server
* Run the Flask API from your terminal:
    ```bash
    python main.py
    ```

### 4️⃣ Step 4: Launch the Dashboard
* Open `index.html` in your web browser.
* You can now browse the dataset, run random tests, or use your webcam for real-time recognition.
