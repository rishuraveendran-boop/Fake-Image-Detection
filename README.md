# CNN-Based Fake Image Detection Using SVM and Grad-CAM

A CNN-based image classification system designed to distinguish between **real and fake images** using deep feature extraction, Support Vector Machine (SVM) classification, and Grad-CAM-based visual explainability.

## Project Overview

The project uses a **Convolutional Neural Network (CNN)** to extract meaningful visual features from images and an **SVM classifier** to classify images as either **REAL** or **FAKE**.

To make the predictions more interpretable, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize the regions of an image that contribute most to the model's prediction.

The project also includes image preprocessing, model training, evaluation, and a Streamlit-based interface for testing images.

## Key Features

- CNN-based feature extraction
- SVM-based image classification
- Binary classification: REAL vs FAKE
- Grad-CAM visual explainability
- Image preprocessing and normalization
- Confusion matrix and training performance analysis
- Streamlit web interface
- Video detection support
- Dataset creation and preprocessing scripts

## System Architecture

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
CNN Feature Extraction
     |
     v
SVM Classifier
     |
     v
REAL / FAKE Prediction
     |
     v
Grad-CAM Visualization

Fake-Image-Detection/
│
├── appog.py
├── preprocessing.py
├── create_universal_dataset.py
├── python check_dataset.py
├── test.py
├── train_model.py
├── train_universal_model.py
├── video_detector.py
│
├── class_distribution.png
├── confusion_matrix.png
├── preprocessing_fake.png
├── preprocessing_real.png
├── sample_fake_images.png
├── sample_real_images.png
└── training_history.png

Model

The classification pipeline follows a hybrid CNN + SVM approach.

The CNN is responsible for learning and extracting visual representations from the input images. These extracted features are then passed to an SVM classifier for the final REAL/FAKE classification.

This approach separates deep feature extraction from the final classification stage.

Grad-CAM Explainability

Grad-CAM is used to provide a visual explanation of the model's decision.

Instead of only returning a prediction, the system can highlight important regions of an image that influenced the CNN's prediction.

This helps make the classification process more interpretable.

Dataset and Preprocessing

The images are processed before being passed to the model.

The preprocessing pipeline includes:

Image resizing
Pixel normalization
Dataset preparation
Training, validation, and test splitting
Visualization of real and fake samples

Large processed datasets and trained model files are excluded from this repository using .gitignore.

Evaluation

The project includes evaluation visualizations such as:

Confusion Matrix
Class Distribution
Training History
Sample Real Images
Sample Fake Images
Preprocessing Results
Running the Application
1. Clone the repository
git clone https://github.com/rishuraveendran-boop/Fake-Image-Detection.git
cd Fake-Image-Detection
2. Install dependencies
pip install tensorflow scikit-learn opencv-python numpy joblib streamlit matplotlib plotly pillow
3. Run the Streamlit application
streamlit run appog.py

The application will open in your browser.

Important Note

The trained model files and large processed datasets are not included in the GitHub repository because of their large file sizes.

These include:

.keras model files
.pkl model files
.npy processed datasets

The corresponding files are excluded through .gitignore.

Future Improvements
Improve detection performance on unseen image-generation techniques
Expand the dataset with newer generative models
Improve video-level detection
Add additional explainability techniques
Optimize the model for real-time inference
Deploy the application as a hosted web application
Author

Rishu Raveendran

MSc Artificial Intelligence and Machine Learning