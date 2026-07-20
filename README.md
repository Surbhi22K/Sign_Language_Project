# 🤟 SignLite: Real-Time ASL Recognition with NLP Text Formation

A lightweight deep learning-based American Sign Language (ASL) recognition system that performs real-time hand gesture recognition on mobile devices using **MobileNetV2**, **TensorFlow Lite**, and **Natural Language Processing (NLP)** for automatic word and sentence formation.

## 📖 Overview

SignLite is an end-to-end Sign Language Recognition (SLR) system designed to recognize static American Sign Language (ASL) alphabets in real time. The recognized letters are processed using an NLP pipeline to generate meaningful words and sentences. The optimized model is converted to TensorFlow Lite and deployed on an Android application for efficient on-device inference.

## 🚀 Features

- Real-time ASL alphabet recognition
- MobileNetV2-based transfer learning model
- TensorFlow Lite optimized model for mobile deployment
- NLP-based word and sentence formation
- Lightweight model suitable for edge devices
- Interactive Android application
- Free Mode for live recognition
- Practice Mode for learning ASL with real-time feedback
- Confidence score for each prediction
- Low memory and fast inference

## 🏗️ Project Architecture

```
Input Image
      │
      ▼
 Image Preprocessing
      │
      ▼
 MobileNetV2 Feature Extractor
      │
      ▼
 Classification Head
(Dense + ReLU + Dropout + Softmax)
      │
      ▼
 Predicted Letter
      │
      ▼
 NLP Text Formation
      │
      ▼
 Word & Sentence Generation
      │
      ▼
 Android Application
```

## 🛠️ Technologies Used

- Python 3.10
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- TensorFlow Lite
- Android
- React Native

## 📂 Project Structure

```
SignLite/
│
├── dataset/
├── models/
├── tflite_model/
├── preprocessing/
├── training/
├── prediction/
├── nlp/
├── mobile_app/
├── utils/
├── requirements.txt
└── README.md
```

## 📊 Model Performance

| Metric | Value |
|---------|--------|
| Validation Accuracy | 97.29% |
| Best Validation Accuracy | 98.75% |
| Precision | 97.72% |
| Recall | 97.29% |
| F1 Score | 97.24% |
| Original Model Size | 28.45 MB |
| TFLite Model Size | 3.03 MB |

## 📱 Mobile Application

The application contains two operating modes:

### Free Mode

- Live camera recognition
- Real-time ASL prediction
- NLP-based text generation
- Confidence score display

### Practice Mode

- Learn individual ASL alphabets
- Reference hand gesture images
- Live prediction
- Instant visual feedback
- Confidence score

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SignLite.git
```

Move into the project directory

```bash
cd SignLite
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the training script

```bash
python train.py
```

Run prediction

```bash
python predict.py
```

## 📈 Dataset

- American Sign Language (ASL) Alphabet Dataset
- 24 static classes (A–Y excluding J and Z)
- Total Images: 2,427
- Image Size: 64 × 64 RGB

## 🔬 Research Contributions

- MobileNetV2-based transfer learning framework
- Lightweight TensorFlow Lite deployment
- NLP-based automatic word and sentence formation
- End-to-end mobile sign language recognition system
- Dual-mode Android application for recognition and learning

## 🎯 Future Work

- Dynamic gesture recognition (J and Z)
- Continuous sign language recognition
- Transformer/LSTM integration
- Larger and more diverse datasets
- Higher image resolution
- BERT-based language model integration
- Improved edge optimization

## 👩‍💻 Author

**Surbhi Kumari**

M.Tech (CSE - AI & Data Science)

Indian Institute of Information Technology Kota

Email: 2025KPAD1004@iiitkota.ac.in

## 📜 License

This project is developed for research and educational purposes.
