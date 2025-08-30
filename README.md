## Facial Expression Recognition using CNN
This repository contains a Jupyter Notebook implementation of a Convolutional Neural Network (CNN) for facial expression recognition. The model is trained on a dataset of grayscale facial images to classify expressions into seven categories.

Dataset
The dataset used in this project is the "Face Expression Recognition Dataset" from Kaggle, containing 28,821 training images and 7,066 validation images across 7 emotion classes:
Predicts multiple emotions such as:

Angry 😠

Disgust 🤢

Fear 😨

Happy 😀

Neutral 😐

Sad 😢

Surprise 😲

Model Architecture
The CNN model consists of the following layers:

Convolutional Layers:

Conv2D (64 filters, 3×3 kernel) + Batch Normalization + MaxPooling + Dropout

Conv2D (128 filters, 3×3 kernel) + Batch Normalization + MaxPooling + Dropout

Conv2D (256 filters, 3×3 kernel) + Batch Normalization + MaxPooling + Dropout

Fully Connected Layers:

Flatten layer

Dense layer (512 units) + Batch Normalization + Dropout

Output layer (7 units with softmax activation)

Training
The model is compiled with:

Optimizer: Adam with learning rate 0.001

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Data augmentation is applied during training including:

Rotation (20 degrees)

Zoom (20%)

Horizontal flipping

Rescaling (1./255)

Requirements
The notebook requires the following Python libraries:

TensorFlow/Keras

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Usage
Ensure the dataset is available in the directory structure:

/kaggle/input/face-expression-recognition-dataset/images/train/

/kaggle/input/face-expression-recognition-dataset/images/validation/

Run the notebook cells sequentially to:

Import libraries and load dataset

Perform exploratory data analysis (EDA)

Build and compile the CNN model

Train the model on the facial expression dataset

Results
The notebook includes visualization of sample images from the dataset and the model architecture summary. The final model achieves classification of facial expressions into the seven emotion categories.

