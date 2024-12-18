# Facial Expression Recognition using CNN and ResNet-50

This repository showcases an award-winning implementation of **Facial Expression Recognition (FER)** using deep learning techniques. The project includes models built with a **Custom CNN Architecture** and **ResNet-50**, along with real-time emotion detection capabilities. The models classify facial expressions into seven categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. The dataset used is the **FER-2013 dataset**.

---

## Project Overview

Facial expressions are an essential part of understanding human emotions. This project aims to achieve accurate facial expression recognition using modern deep learning techniques, with features including:

- **Custom CNN Architecture** trained for high performance.
- **ResNet-50 Transfer Learning** fine-tuned for robust accuracy.
- **Real-time Prediction** using a live webcam feed for emotion detection.

---

## Dataset

The **FER-2013 dataset** contains grayscale images of facial expressions:

- **Training Data:** 20,099 images  
- **Validation Data:** 8,610 images  
- **Test Data:** 7,178 images  
- **Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral  
- **Image Size:** All images are resized to 48x48 pixels to standardize input for the models.

---

## Achievements

- **Winner of the Microsoft Code Crunch ML Challenge:** This repository’s `CNN.ipynb` was instrumental in achieving this recognition.
- **Best Model:** The file `CNNmodelfinal.keras` is the highest-performing model trained using the `CNN.ipynb` notebook.

---

## Models and Notebooks

### **Uploaded Models**

1. **`CNNmodelfinal.keras`:** Best-performing model created using the `CNN.ipynb` notebook.  
2. **`resnetfinetune.h5`:** ResNet-50 model fine-tuned with the `resnet.ipynb` notebook.  
3. **`resnet googleapis.h5`:** Pre-trained ResNet-50 weights used as the base for fine-tuning in `resnet.ipynb`.  

### **Notebooks**

#### 1. `CNN.ipynb`
- Implements a **Custom CNN Architecture** for facial expression recognition.
- **Key Layers:** Convolutional, Batch Normalization, Dropout, Fully Connected.
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1 Score.
- **Augmentation:** Random rotations, shifts, zooming, and flipping.
- **Output:** The model `CNNmodelfinal.keras` was created and achieved the best performance on the FER-2013 dataset.

#### 2. `livefeed.ipynb`
- Loads the `CNNmodelfinal.keras` model for real-time emotion prediction.
- Uses **OpenCV Haar Cascade** for face detection.
- Captures live video feed and overlays bounding boxes with predicted emotions.

#### 3. `resnet.ipynb`
- Fine-tunes a **ResNet-50 model** for facial expression recognition.
- **Pre-trained Weights:** `resnet googleapis.h5` used as the base.
- Adds a custom fully connected head for FER classification.
- Outputs the fine-tuned model `resnetfinetune.h5`.

---

## Key Features

- **Custom CNN Architecture:** Tailored from scratch for FER.
- **ResNet-50 Transfer Learning:** Fine-tuned for higher robustness and accuracy.
- **Real-time Inference:** Emotion detection using a live webcam feed.
- **Extensive Preprocessing:** Data augmentation for better generalization.
- **Custom Metrics:** Includes F1 Score, Precision, and Recall.

---

## Preprocessing and Augmentation

1. **Rescaling:** Pixel values normalized to [0, 1].  
2. **Augmentation:**  
   - Random rotations (±2 degrees)  
   - Width and height shifts (20%)  
   - Random zoom (20%)  
   - Horizontal flips  
3. **Test Data:** Only rescaled, no augmentation applied.

---

## Results

### **Custom CNN (`CNN.ipynb`)**
- **Parameters:** ~490,000  
- **Performance:** Highly efficient for smaller-scale deployment.

### **ResNet-50 (`resnet.ipynb`)**
- **Parameters:** ~24 million  
- **Performance:** Achieves greater accuracy and robustness for FER tasks.

### **Real-time Testing (`livefeed.ipynb`)**
- **Latency:** Predictions processed at ~10-15ms per frame.  
- **Output:** Predicted emotion is overlaid on the live video feed.

---

## Dependencies

This project requires the following libraries:

- **TensorFlow** >= 2.0  
- **Keras**  
- **OpenCV**  
- **NumPy**  
- **Matplotlib**  
- **SciPy**  
- **Pandas**  

## Usage

### **Training the Models**
- Use `CNN.ipynb` to train the custom CNN model.  
- Use `resnet.ipynb` for training and fine-tuning the ResNet-50 model.

### **Real-time Testing**
- Run `livefeed.ipynb` to load the `CNNmodelfinal.keras` model and start the live webcam feed.  
- Press `Q` to exit the live feed.

### **Using Pre-trained Models**
- Place the models (`CNNmodelfinal.keras`, `resnetfinetune.h5`, or `resnet googleapis.h5`) in the appropriate directory as required by the notebooks.

---

## Acknowledgments

- **FER-2013 Dataset:** [Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013)  
- **ResNet-50 Architecture:** He et al., 2015  
- **TensorFlow/Keras:** Deep learning framework.  

Feel free to raise issues or contribute to the project!

