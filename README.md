# Rice Leaf Disease Detection

## 📌 Project Overview
Rice is a staple food for billions, but diseases can drastically reduce crop yields. This project (PRCP-1001) focuses on building a robust Machine Learning/Deep Learning model to automatically detect and classify common diseases in rice plants using images of their leaves. 

By leveraging computer vision, this model aims to assist in early disease diagnosis, enabling timely intervention and securing crop production.

## 🗂️ Dataset Information
The dataset consists of images of rice leaves categorized into distinct health states. 
* **Target Classes:**
  1. Bacterial Leaf Blight
  2. Brown Spot
  3. Leaf Smut
* **Format:** JPEG/PNG images
* *(Optional: Add the total number of images or the dataset source here)*

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras 
* **Computer Vision:** OpenCV / PIL
* **Data Manipulation & Math:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## ⚙️ Project Workflow
1. **Data Preprocessing:** * Resizing images to a uniform dimension.
   * Normalizing pixel values for faster convergence.
   * Image augmentation (rotation, flipping, zooming) to handle dataset variance and prevent overfitting.
2. **Exploratory Data Analysis (EDA):** * Visualizing sample images from each disease class.
   * Analyzing class distribution to check for imbalances.
3. **Model Architecture:** * Developed a Convolutional Neural Network (CNN) from scratch (or utilizing Transfer Learning models like ResNet/VGG16) to extract spatial features from the leaf images.
4. **Model Evaluation:** * Generating evaluation metrics including Accuracy, Precision, Recall, and F1-Score.
   * Plotting the Confusion Matrix to identify misclassifications between visually similar diseases (e.g., Brown Spot vs. Leaf Smut).

## 🚀 How to Run the Project
1. **Clone this repository:**
   ```bash
   git clone [https://github.com/VISHNUREDDY352/RICE-LEAF.git](https://github.com/VISHNUREDDY352/RICE-LEAF.git)

