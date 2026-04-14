#  Medical Image Classification using CNN

##  Project Overview

This project focuses on building an AI-powered medical imaging system to automatically detect **Pneumonia (Chest X-rays)** and **Brain Tumors (MRI scans)** using Convolutional Neural Networks (CNNs). It aims to assist healthcare professionals with faster and more accurate diagnosis.

---

##  Objectives

* Develop CNN models for medical image classification
* Detect:

  * Pneumonia from Chest X-rays
  * Brain Tumors from MRI scans
* Improve model performance using preprocessing & augmentation
* Evaluate models using accuracy and loss metrics
* Enable real-time prediction for new images

## 🧠 Technologies Used

* Python
* TensorFlow & Keras
* Google Colab (GPU)
* NumPy, Pandas
* OpenCV, Pillow
* Matplotlib, Seaborn

##  Dataset

* Chest X-ray dataset (NORMAL / PNEUMONIA)
* Brain MRI dataset (Tumor / No Tumor)
* Sources: Public datasets (Kaggle)

---

##  System Workflow

1. Data Collection
2. Preprocessing & Augmentation
3. CNN Model Design
4. Training & Validation
5. Model Evaluation
6. Prediction

---

##  Model Architecture

* Conv2D + MaxPooling layers
* Flatten layer
* Dense layers with Dropout
* Sigmoid activation (Binary Classification)

---

##  Results

| Model                   | Accuracy   |
| ----------------------- | ---------- |
| Chest X-ray (Pneumonia) | 93.67% |
| Brain MRI (Tumor)       | 88.00% |

* Strong generalization using data augmentation
* Reduced overfitting with Dropout
* Efficient training using GPU

---

##  Features

* Automated disease detection
* Real-time image prediction
* Lightweight CNN architecture
* Scalable for other diseases

---

##  Applications

* Clinical diagnosis support
* Telemedicine systems
* Public health screening
* Medical education

---

##  Future Enhancements

* Transfer Learning (VGG16, ResNet50)
* Explainable AI (Grad-CAM)
* Multi-disease detection
* Web/Mobile deployment
* Cloud integration

---

##  Conclusion

This project demonstrates how deep learning can enhance medical diagnostics by improving accuracy, reducing workload, and enabling early disease detection. It highlights the potential of AI in transforming healthcare systems.

---

##  Author

Chamarathi Harika
B.Tech – CSE (Data Science)
