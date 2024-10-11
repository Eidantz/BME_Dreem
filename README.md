# Sleep Stage Classification Using EEG and IMU Data

This project implements a deep learning model to classify sleep stages using EEG (Electroencephalogram) and IMU (Inertial Measurement Unit) data. The model leverages a combination of Short-Time Fourier Transform (STFT) for feature extraction from EEG signals, GRU (Gated Recurrent Units) with attention mechanisms, and convolutional neural networks (CNNs) to handle both raw data and transformed features. The final model is trained using subject-wise K-fold cross-validation and optimized for multi-class classification of different sleep stages.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Overview
Sleep stage classification is a key aspect of understanding sleep patterns and detecting sleep-related disorders. This project uses EEG signals and IMU data to classify sleep stages into different categories such as Wake, NREM (Non-Rapid Eye Movement) stages 1-3, and REM (Rapid Eye Movement) From Dreem Dataset. 

Key features of this project:
- **EEG Signal Processing**: Short-Time Fourier Transform (STFT) is applied to EEG data to capture time-frequency domain features.
- **GRU & Attention**: The model uses GRU layers with an attention mechanism to focus on relevant time steps.
- **CNN for Raw Data**: A CNN is applied to raw EEG data for feature extraction.
- **Custom Loss Function**: A weighted Sparse Categorical Crossentropy loss is implemented to handle class imbalance.
- **K-Fold Cross-Validation**: Ensures robust model training by using subject-wise splits for cross-validation.

## Installation
To get started with this project, you need to set up a Python environment with the required dependencies.


### Prerequisites
Ensure you have Python 3.7 or later installed. Then, follow these steps to install the necessary packages.

1. **Clone the repository**:
```bash
   git clone https://github.com/Eidantz/BME_Dreem.git
   cd BME_Dreem
```
2. **Install dependencies: Install the required Python packages using pip:**:
```bash
pip install -r requirements.txt
```
## Data Preprocessing
The data consists of EEG and IMU signals stored in HDF5 (`.h5`) and CSV format. The key steps in preprocessing include:

1. **Normalization**: 
   EEG signals are clipped to a range between -150 and 150 and then normalized between -1 and 1 to prepare them for input into the model.

2. **STFT Transformation**: 
   Short-Time Fourier Transform (STFT) is applied to the EEG signals to capture both the time and frequency domain information. The STFT converts time-series EEG data into a spectrogram, allowing the model to learn from both temporal and spectral features.

3. **Positional Embeddings**: 
   Positional embeddings are generated from metadata (such as the index window) to provide the model with additional context about the position of the data in the sequence. This is done using cosine transformations to create positional encodings.

4. **Splitting**: 
   The data is split into training and validation sets using subject-wise K-fold cross-validation. This ensures that data from the same subject does not appear in both the training and validation sets, preventing data leakage and providing a more robust model evaluation.

## Model Architecture
The architecture combines multiple components to handle different aspects of the input data:

### STFT Input:
- **TimeDistributed Dense Layers**: 
  The STFT-transformed data is passed through dense layers, applied using the `TimeDistributed` wrapper, to reduce the number of filters and channels.
  
- **Bidirectional GRU**: 
  A Bidirectional GRU (Gated Recurrent Unit) layer is used to capture temporal dependencies in the STFT data. It processes the data in both forward and backward directions to improve the model's ability to capture temporal patterns.

- **Attention Layer**: 
  An attention mechanism is applied to focus the model on the most important parts of the sequence. This helps the model prioritize relevant time steps when making predictions.

### Positional Embedding Input:
- **Dense Layers**: 
  The positional embeddings are passed through several dense layers with ReLU activation to allow the model to learn from metadata and positional information.

### Raw EEG Input:
- **1D CNN**: 
  Raw EEG data is processed using a 1D Convolutional Neural Network (CNN). Multiple convolution and max-pooling layers are applied to extract features from the raw signal.

### Final Output:
- The outputs from the STFT, positional embedding, and raw EEG paths are concatenated and passed through a final softmax layer for multi-class classification. The softmax layer outputs the predicted probabilities for each sleep stage.

## Training and Evaluation
The model is trained using subject-wise K-fold cross-validation. Key aspects of the training and evaluation process include:

- **Early Stopping**: 
  The training process uses early stopping, which monitors the validation loss and halts training if the model starts to overfit. This helps to ensure better generalization and prevent overfitting.

- **Custom Loss Function**: 
  A custom weighted Sparse Categorical Crossentropy loss is used to handle class imbalance. By assigning higher weights to underrepresented classes, the model is encouraged to perform better on these classes.

- **Optimizer**: 
  The model is trained using the Adam optimizer with a learning rate of 0.001. Adam is chosen for its ability to adapt the learning rate during training, leading to better convergence.
## Results
The model is evaluated on several metrics to assess its performance:

- **Accuracy**: 
  Accuracy is the percentage of correct predictions made by the model. It provides an overall measure of how well the model performs in classifying the sleep stages.

- **Confusion Matrix**: 
  A confusion matrix is used to visualize the classification performance across different sleep stages. It shows the number of times a sleep stage was correctly predicted, as well as the number of misclassifications. This matrix helps to identify which sleep stages are harder for the model to distinguish.

- **Classification Report**: 
  A classification report provides more detailed performance metrics, including:
  - **Precision**: The number of true positive predictions divided by the total number of positive predictions.
  - **Recall**: The number of true positives divided by the actual number of positives.
  - **F1-Score**: A weighted average of precision and recall that takes both false positives and false negatives into account.
  
Training and validation curves (accuracy and loss) are plotted to help monitor the model's performance during training. These plots are useful for understanding whether the model is overfitting or underfitting the training data.
