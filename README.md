Sign Language Recognition - Static Model

This repository contains a Jupyter Notebook for implementing a static sign language recognition system. The notebook leverages deep learning techniques for classifying sign language gestures using image data.

Project Overview -->

This project focuses on recognizing static hand signs in images and translating them into corresponding letters or numbers. It aims to assist individuals with hearing or speech impairments by providing a tool for effective communication.

Features -->

Preprocessing of hand gesture images for model training.

Implementation of a Convolutional Neural Network (CNN) for sign language recognition.

Evaluation using metrics such as confusion matrix and classification report.

Visualization of training and validation performance.

Dependencies -->

The following Python libraries are required for this project:

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
import zipfile

Getting Started -->

Prerequisites
Ensure you have Python 3.9 or later installed along with Jupyter Notebook. Install the required libraries by running:
- pip install -r requirements.txt

Usage

Clone the repository:
git clone https://github.com/yourusername/sign-language-recognition.git

Navigate to the project directory:
cd sign-language-recognition

Launch the Jupyter Notebook:
jupyter notebook
- Open SLR_STATIC.ipynb and run the cells sequentially to train and evaluate the model.

Results
- The notebook includes sections for:
- Visualizing training and validation metrics such as accuracy and loss.
- Displaying a confusion matrix and a classification report for model performance evaluation.

Example Output
- Training and validation accuracy/loss plots.
- Confusion matrix showcasing classification performance.

Directory Structure

├── SLR_STATIC.ipynb         # Main notebook file
├── data                     # Directory containing image datasets (add your dataset here)
├── models                   # Directory for saving trained models
├── results                  # Directory for evaluation results and visualizations
└── README.md                # Project documentation

Contributing -->
- Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

Acknowledgments -->
- Special thanks to open-source contributors and datasets used in this project for enabling its development.
