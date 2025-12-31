# CIFAR-10 Image Classifier using Convolutional Neural Networks

## 📌 Project Overview
This project implements an image classification system using a Convolutional Neural Network (CNN) built with TensorFlow.
The model is trained on the CIFAR-10 dataset to classify images into ten different object categories.

This project was developed as the final project submission for **CS50x 2025** and focuses on understanding the implementation of python and end-to-end deep learning pipeline .

![Image Alt](https://github.com/Adityabaan/CIFAR-10-Image-Classifier-using-Convolutional-Neural-Networks/blob/e07ca17793952487f012fc009a37342d4d480789/CS50's%20Introduction%20to%20Computer%20Science.PNG)

---

## 🧠 Technologies Used
- Python
- TensorFlow (Keras)
- NumPy
- Matplotlib

---

## 📂 Dataset Information

### CIFAR-10 Dataset
The CIFAR-10 dataset is a widely used benchmark dataset for image classification tasks.

- **Official Dataset URL:**  
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

- **Total Images:** 60,000 color images  
- **Image Size:** 32 × 32 pixels  
- **Training Images:** 50,000  
- **Testing Images:** 10,000  
- **Number of Classes:** 10  

### Classes
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

The dataset is loaded and processed using TensorFlow utilities for training and evaluation.

---

## ⚙️ Project Workflow

### 1. Environment Setup
Required Python libraries are imported for model building, numerical computation, and visualization.

### 2. Dataset Loading
The CIFAR-10 dataset is loaded and split into training and testing sets.

### 3. Data Preprocessing
Image pixel values are normalized to the range [0, 1] to improve training efficiency and stability.

### 4. Data Augmentation
Random image transformations such as horizontal flipping and rotation are applied during training to reduce overfitting and improve generalization.

### 5. Model Architecture Design
A Convolutional Neural Network (CNN) is designed using:
- Convolution layers for feature extraction  
- Pooling layers for dimensionality reduction  
- Dense layers for classification  

### 6. Model Compilation
The model is compiled using:
- Adam optimizer  
- Categorical cross-entropy loss  
- Accuracy metric  

### 7. Model Training
The CNN is trained on the training dataset for multiple epochs with validation monitoring.

### 8. Performance Evaluation
The trained model is evaluated on the test dataset to measure classification accuracy.

### 9. Prediction and Results
The model predicts the class labels of unseen images and compares them with actual labels.

---

## 📊 Results
The model achieves reasonable classification accuracy on the CIFAR-10 test dataset, demonstrating the effectiveness of CNNs for image recognition tasks.

---

## 🎯 Learning Outcomes
- Practical understanding of Convolutional Neural Networks  
- Experience working with image datasets  
- Hands-on training with TensorFlow and Keras  
- Understanding of data preprocessing and augmentation techniques  

---

## 🧑‍💻 Author
Adityabaan Tripathy

---

## 📬 Connect & Feedback

Questions, collaboration ideas, or feedback are always welcome!  
- 📧 Email: [adityabaantripathy@gmail.com](mailto:adityabaantripathy@gmail.com)  
- 🌐 LinkedIn: [Adityabaan Tripathy](https://www.linkedin.com/in/adityabaan-tripathy-6b245323b/)  
- 🐙 GitHub: [Adityabaan Tripathy](https://github.com/Adityabaan)

---

## 📜 Acknowledgements
- CIFAR-10 Dataset (University of Toronto)
- TensorFlow Documentation
- CS50 by Harvard University

