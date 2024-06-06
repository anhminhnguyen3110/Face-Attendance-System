# Video demonstration of the project
[Click here to watch the video](https://youtu.be/tf6SQ8tDwUs)
[![Watch the video](https://media.cnn.com/api/v1/images/stellar/prod/200209224400-20200209-facial-recognition-gfx.jpg?q=w_1600,h_900,x_0,y_0,c_fill)](https://youtu.be/tf6SQ8tDwUs)

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

# Dataset

For this project, a face recognition dataset consisting of 31 different classes, each representing a distinct individual, will be utilized. The dataset is structured with folders corresponding to each class containing a collection of images of the respective person. Notably, all individuals in the dataset are famous personalities, allowing for a diverse and challenging set of face images to evaluate the performance of the developed models. The images within each class capture various angles, expressions, and lighting conditions, providing a comprehensive representation of the individual's facial features. This dataset's format aligns well with the requirements of the face recognition task as it enables training and evaluation of models to accurately classify and recognize individuals based on their facial characteristics. The availability of multiple images per class also allows for thorough testing and validation of the developed systems, ensuring their robustness and generalization capabilities. The dataset can be accessed at the following link.

**Dataset Link:** [Dataset](https://example.com/dataset-link)

## Results

The experiments conducted using various models yielded the following results:

### CNN Model Training Experiments

| Exp No | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|----------------|---------|---------|------------------|----------------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 20.9%   | 0.75    | ![Training Loss](training_loss_exp1.png) | ![Training Accuracy](training_accuracy_exp1.png) |
| 2      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 50 | 30.2%   | 0.78    | ![Training Loss](training_loss_exp2.png) | ![Training Accuracy](training_accuracy_exp2.png) |
| 3      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.5, Epoch: 70 | 28.66%  | 0.84    | ![Training Loss](training_loss_exp3.png) | ![Training Accuracy](training_accuracy_exp3.png) |
| 4      | RS: 54, Img: 224x224, Dense: 128, Dropout: 0.5, Epoch: 70 | 30.33%  | 0.87    | ![Training Loss](training_loss_exp4.png) | ![Training Accuracy](training_accuracy_exp4.png) |

### Pretrained Model (VGG16) Training Experiments

| Exp No | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|----------------|---------|---------|------------------|----------------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 44.67%  | 0.93    | ![Training Loss](training_loss_exp1_vgg.png) | ![Training Accuracy](training_accuracy_exp1_vgg.png) |
| 2      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 50 | 34.84%  | 0.82    | ![Training Loss](training_loss_exp2_vgg.png) | ![Training Accuracy](training_accuracy_exp2_vgg.png) |
| 3      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.15, Epoch: 70 | 38.52%  | 0.91    | ![Training Loss](training_loss_exp3_vgg.png) | ![Training Accuracy](training_accuracy_exp3_vgg.png) |
| 4      | RS: 54, Img: 224x224, Dense: 128, Dropout: 0.15, Epoch: 80 | 31.56%  | 0.90    | ![Training Loss](training_loss_exp4_vgg.png) | ![Training Accuracy](training_accuracy_exp4_vgg.png) |

### Optimized Pretrained Model (VGG16) with Haar Cascade Experiments

| Exp No | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|----------------|---------|---------|------------------|----------------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 70.42%  | 0.97    | ![Training Loss](training_loss_exp1_haar.png) | ![Training Accuracy](training_accuracy_exp1_haar.png) |

### Best Model (Using Optimized Technique)

| Exp No | Model Name | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|------------|---------------|---------|---------|------------------|----------------------|
| 1      | VGG16      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 70.42%  | 0.97    | ![Training Loss](training_loss_best.png) | ![Training Accuracy](training_accuracy_best.png) |
| 2      | ResNet50   | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 4.92%   | 0.59    | ![Training Loss](training_loss_resnet.png) | ![Training Accuracy](training_accuracy_resnet.png) |
| 3      | DenseNet121| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 64.79%  | 0.96    | ![Training Loss](training_loss_densenet.png) | ![Training Accuracy](training_accuracy_densenet.png) |
| 4      | InceptionV3| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 48.83%  | 0.94    | ![Training Loss](training_loss_inception.png) | ![Training Accuracy](training_accuracy_inception.png) |
| 5      | MobileNetV2| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 63.85%  | 0.94    | ![Training Loss](training_loss_mobilenet.png) | ![Training Accuracy](training_accuracy_mobilenet.png) |
| 6      | EfficientNetB0| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 4.69%   | 0.56    | ![Training Loss](training_loss_efficient.png) | ![Training Accuracy](training_accuracy_efficient.png) |

# Face Recognition Attendance System

The face recognition system includes features for user registration, attendance checking, and additional functionalities like identifying eyes and smiles. Below is a brief overview of the system and placeholders for demonstration images.

### Tkinter GUI

The system uses Tkinter for the user interface, which includes buttons for checking attendance, identifying faces, and registering new users. Each button is associated with a specific function.

### CV2 and Face Recognition

OpenCV (cv2) is used for real-time image processing. The `face_recognition` library, built on dlib, provides functions to find, encode, and compare faces.

#### Register New User

New users can be registered by capturing their facial data and storing the corresponding embeddings. Below is a placeholder for images demonstrating user registration.

![Register New User](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/449504ee-e1b5-43e9-87f5-7b56217a7af0)

![Successful Registration](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/4766fa59-ec22-46b4-b3ac-dee4dbd7f9cf)

![Failed Registration](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/c8b05c01-4f9e-4aa4-aeff-4d74dea25041)

#### Check Attendance

The system logs attendance by identifying users in real-time and comparing live captures with stored facial embeddings. Below is a placeholder for images demonstrating the attendance check feature.

![Attendance Check](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/40d19bb2-6e65-47ba-b4ca-923283ded1fe)

![User Not Registered](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/06ef2c15-bc47-4f8a-a64d-0505cd7340ee)

#### Identify Eyes & Smile

Additional features include identifying eyes and smiles using Haar cascades provided by OpenCV. Below is a placeholder for images demonstrating these features.

![Identify Eyes and Smile Box](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/e8a761a5-6496-4830-a020-6b6e3cb64462)

![Identify Eyes and Smile](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/38260757-f697-43dc-9c72-9fdc8abcc8d8)

### Anti-Spoofing System

The anti-spoofing system ensures security by using the MiniFASNet architecture, which includes auxiliary supervision of the Fourier spectrum to differentiate between real and fake faces. Below is a placeholder for images demonstrating the spoofing detection system.
![MiniFASNet architecture (MiniVision, 2020)](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/290c1575-6a21-4509-b9a0-9fad78af0b9e)

![Spoofing Detection](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/87dafc8e-b64a-4917-b251-3adf6c83b94e)

![Spoofing Detection](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/61fb2d03-3860-459b-8ed8-7d25e45abeed)

![Spoofing Detection](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/7a4552ce-7bee-4816-ae8e-a041aa95b6b6)

## Conclusion

The VGG16 model with Haar Cascade optimization achieved the best performance in this study. Configured with a random seed of 54, image size of 128x128, dense layer size of 128, dropout rate of 0.3, and trained for 30 epochs, it achieved a train accuracy of 98.34%, train loss of 0.11, validation accuracy of 70.42%, validation loss of 1.44, test accuracy of 70.42%, and a micro-average ROC curve area (AUC) of 0.97. Other models like ResNet50 and EfficientNetB0 did not converge effectively, while DenseNet121, InceptionV3, and MobileNetV2 showed decent performance but did not surpass VGG16.

These findings suggest that VGG16 optimized with Haar Cascade and appropriate configurations provides the best balance between accuracy and performance. Future work will focus on further fine-tuning hyperparameters, exploring more complex architectures, and reducing overfitting to enhance performance even further.
