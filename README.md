# Installing dlib on Windows 10

This is a step-by-step guide to install dlib on Windows 10. Make sure to follow the steps carefully and in order. (For Python 3.8 only)

## Prerequisites

- Python version: 3.8.0 (Date: 2019-10-14)

## Installation Steps

1. **Install dlib**

    Enter the following command in the terminal:

    ```bash
    pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
    ```

2. **Install CMake**

    Then, install CMake using the following command:

    ```bash
    pip install cmake
    ```

3. **Update pip**

    Next, update pip using the following command:

    ```bash
    pip install --upgrade pip
    ```

4. **Install face_recognition**

    Finally, you can install `face_recognition` using the following command:

    ```bash
    pip install face-recognition
    ```

## Results

The experiments conducted using various models yielded the following results:

### CNN Model Training Experiments

| Exp No | Configuration | Train Acc | Train Loss | Val Acc | Val Loss | Test Acc | ROC AUC | Conclusion |
|--------|----------------|-----------|------------|---------|----------|----------|---------|------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 68.8%    | 1.2        | 20.9%   | 2.9      | 25.51%   | 0.75    | Not Converged (High training loss) |
| 2      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 50 | 89.5%    | 0.5        | 30.2%   | 2.5      | 30.4%    | 0.78    | Converged (Overfitting) |
| 3      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.5, Epoch: 70 | 96.42%   | 0.18       | 28.66%  | 2.98     | 31.7%    | 0.84    | Converged (Overfitting) |
| 4      | RS: 54, Img: 224x224, Dense: 128, Dropout: 0.5, Epoch: 70 | 97.07%   | 0.13       | 30.33%  | 3.24     | 32.2%    | 0.87    | Model complexity insufficient |

### Pretrained Model (VGG16) Training Experiments

| Exp No | Configuration | Train Acc | Train Loss | Val Acc | Val Loss | Test Acc | ROC AUC | Conclusion |
|--------|----------------|-----------|------------|---------|----------|----------|---------|------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 94.52%   | 0.13       | 44.67%  | 2.24     | 43.85%   | 0.93    | Converged (Overfitting) |
| 2      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 50 | 79.51%   | 0.86       | 34.84%  | 2.5      | 31.15%   | 0.82    | Converged (Overfitting, dropout impact) |
| 3      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.15, Epoch: 70 | 98.27%   | 0.09       | 38.52%  | 2.63     | 40.16%   | 0.91    | Converged (Overfitting, increase image size) |
| 4      | RS: 54, Img: 224x224, Dense: 128, Dropout: 0.15, Epoch: 80 | 91.67%   | 0.5        | 31.56%  | 2.67     | 31.15%   | 0.90    | Model complexity insufficient |

### Optimized Pretrained Model (VGG16) with Haar Cascade Experiments

| Exp No | Configuration | Train Acc | Train Loss | Val Acc | Val Loss | Test Acc | ROC AUC | Conclusion |
|--------|----------------|-----------|------------|---------|----------|----------|---------|------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 98.34%   | 0.11       | 70.42%  | 1.44     | 70.42%   | 0.97    | Converged |

### Best Model (Using Optimized Technique)

| Exp No | Model Name | Configuration | Train Acc | Train Loss | Val Acc | Val Loss | Test Acc | ROC AUC | Conclusion |
|--------|------------|---------------|-----------|------------|---------|----------|----------|---------|------------|
| 1      | VGG16      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 98.34%   | 0.11       | 70.42%  | 1.44     | 70.42%   | 0.97    | Converged |
| 2      | ResNet50   | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 4.66%    | 3.39       | 4.92%   | 3.39     | 4.92%    | 0.59    | Not Converged (Model too complicated) |
| 3      | DenseNet121| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 93.5%    | 0.19       | 64.79%  | 2.09     | 65.10%   | 0.96    | Converged |
| 4      | InceptionV3| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 87.35%   | 0.4        | 48.83%  | 2.95     | 51.76%   | 0.94    | Converged |
| 5      | MobileNetV2| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 92.43%   | 0.23       | 63.85%  | 2.08     | 58.82%   | 0.94    | Converged |
| 6      | EfficientNetB0| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 4.74%    | 3.39       | 4.69%   | 3.39     | 3.92%    | 0.56    | Not Converged (Model too complicated) |

## Face Recognition Attendance System

The face recognition system includes features for user registration, attendance checking, and additional functionalities like identifying eyes and smiles. Below is a brief overview of the system and placeholders for demonstration images.

### Tkinter GUI

The system uses Tkinter for the user interface, which includes buttons for checking attendance, identifying faces, and registering new users. Each button is associated with a specific function.

### CV2 and Face Recognition

OpenCV (cv2) is used for real-time image processing. The `face_recognition` library, built on dlib, provides functions to find, encode, and compare faces.

#### Register New User

New users can be registered by capturing their facial data and storing the corresponding embeddings. Below is a placeholder for images demonstrating user registration.

![Register New User](placeholder-for-register-new-user-image)

![Successful Registration](placeholder-for-successful-registration-image)

![Failed Registration](placeholder-for-failed-registration-image)

#### Check Attendance

The system logs attendance by identifying users in real-time and comparing live captures with stored facial embeddings. Below is a placeholder for images demonstrating the attendance check feature.

![Attendance Check](placeholder-for-attendance-check-image)

![User Not Registered](placeholder-for-user-not-registered-image)

![Attendance Logged](placeholder-for-attendance-logged-image)

#### Identify Eyes & Smile

Additional features include identifying eyes and smiles using Haar cascades provided by OpenCV. Below is a placeholder for images demonstrating these features.

![Identify Eyes and Smile](placeholder-for-identify-eyes-smile-image)

![Identify Eyes and Smile](placeholder-for-identify-eyes-smile-image-2)

### Anti-Spoofing System

The anti-spoofing system ensures security by using the MiniFASNet architecture, which includes auxiliary supervision of the Fourier spectrum to differentiate between real and fake faces. Below is a placeholder for images demonstrating the spoofing detection system.

![Spoofing Detection](placeholder-for-spoofing-detection-image)

![Spoofing Detection](placeholder-for-spoofing-detection-image-2)

![Spoofing Detection](placeholder-for-spoofing-detection-image-3)

## Conclusion

The VGG16 model with Haar Cascade optimization achieved the best performance in this study. Configured with a random seed of 54, image size of 128x128, dense layer size of 128, dropout rate of 0.3, and trained for 30 epochs, it achieved a train accuracy of 98.34%, train loss of 0.11, validation accuracy of 70.42%, validation loss of 1.44, test accuracy of 70.42%, and a micro-average ROC curve area (AUC) of 0.97. Other models like ResNet50 and EfficientNetB0 did not converge effectively, while DenseNet121, InceptionV3, and MobileNetV2 showed decent performance but did not surpass VGG16.

These findings suggest that VGG16 optimized with Haar Cascade and appropriate configurations provides the best balance between accuracy and performance. Future work will focus on further fine-tuning hyperparameters, exploring more complex architectures, and reducing overfitting to enhance performance even further.