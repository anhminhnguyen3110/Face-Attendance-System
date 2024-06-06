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

For this project, a face recognition dataset consisting of 31 different classes, each representing a distinct individual, will be utilized. The dataset is structured with folders corresponding to each class containing a collection of images of the respective person. Notably, all individuals in the dataset are famous personalities, allowing for a diverse and challenging set of face images to evaluate the performance of the developed models. The images within each class capture various angles, expressions, and lighting conditions, providing a comprehensive representation of the individual's facial features. 

**Dataset Link:** [Dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset)

## Results

The experiments conducted using various models yielded the following results:

### CNN Model Training Experiments

| Exp No | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|----------------|---------|---------|------------------|----------------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 20.9%   | 0.75    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/16ad0ac9-13ad-4ba6-8ca8-6947d6c6b73c) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/826299e0-dd05-455c-8da6-ad3b0a1d18ee) |
| 2      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 50 | 30.2%   | 0.78    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/a4e923f9-495d-44c6-a056-184121a1b4c7) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/43501e81-beae-4f46-8a9b-72e4a809b1c4) |
| 3      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.5, Epoch: 70 | 28.66%  | 0.84    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/a39f2951-00f6-48f2-8add-16d7c6da9008) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/db1ff572-321e-4d4f-ad0f-755257127294) |
| 4      | RS: 54, Img: 224x224, Dense: 128, Dropout: 0.5, Epoch: 70 | 30.33%  | 0.87    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/e07f3fd0-bc37-4a39-804c-7aafe53ae109) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/5c489771-15b7-485c-9755-6121ed467e74) |

#### Best CNN Model ROC Curve
<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/5d7164b9-df75-4263-a5e4-80cbaad68931" width="800"/>

### Pretrained Model (VGG16) Training Experiments

| Exp No | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|----------------|---------|---------|------------------|----------------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 44.67%  | 0.93    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/ad3b0f43-047d-43dd-a389-3fa596edf800) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/90dd680a-98e9-4f3b-bb6f-2be6a4f8d761) |
| 2      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 50 | 34.84%  | 0.82    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/d3e5ef38-00a1-428e-95cd-ae1443b013e0) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/2cb6dbdb-343a-4715-af81-a85b0b04295e) |
| 3      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.15, Epoch: 70 | 38.52%  | 0.91    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/9e1ba7b5-ba5b-44fa-a59e-403afd1f82fd) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/d9e9e29d-2966-4ecd-affe-149e79fa5e4b) |
| 4      | RS: 54, Img: 224x224, Dense: 128, Dropout: 0.15, Epoch: 80 | 31.56%  | 0.90    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/3660a5aa-10e9-40ec-a22f-f1e7426adf85) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/ae0f3991-e3c5-4482-b08d-833f4ae8cd76) |

#### Best Pretrained Model (VGG16) ROC Curve
<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/b06c992e-4ef5-4564-b354-ed6a2b4ff18a" width="800"/>

### Optimized Pretrained Model (VGG16) with Haar Cascade Experiments

| Exp No | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|----------------|---------|---------|------------------|----------------------|
| 1      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0, Epoch: 30 | 70.42%  | 0.97    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/f9f214f7-8548-4cd4-b6f7-aeaec39fa699) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/e0be24fb-8815-4205-a8f0-c1341983350d) |

#### Best Optimized Pretrained Model (VGG16) ROC Curve
<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/bdb8d773-1e45-4dc0-a6d9-ceb8a287be4c" width="800"/>

### Experiments to find out the top performance model (Using Optimized Technique)

| Exp No | Model Name | Configuration | Val Acc | ROC AUC | Chart Train Loss | Chart Train Accuracy |
|--------|------------|---------------|---------|---------|------------------|----------------------|
| 1      | VGG16      | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 70.42%  | 0.97    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/496110fe-0ff2-4a20-8f26-15a1904bfde4) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/7e1c8a90-4fd5-46d1-928a-7527fca4fcb1) |
| 2      | ResNet50   | RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 4.92%   | 0.59    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/dcc7a640-a878-4b9b-891d-5379485c7077) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/e9251197-ed25-4ccd-83bb-ac4bbd2dc7f1) |
| 3      | DenseNet121| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 30 | 64.79%  | 0.96    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/edc975bd-43d4-4a91-b969-033dcafe7376) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/db34480e-6ed5-4633-ac4a-2e4dbac17fca) |
| 4      | InceptionV3| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 48.83%  | 0.94    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/d07bf1af-7dda-4093-84bb-9ed35d274127) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/c2c4f0a0-75ea-4178-bdba-146ce0328f43) |
| 5      | MobileNetV2| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 63.85%  | 0.94    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/ee030b17-7c6c-4ee5-b6e5-bbd51098fce6) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/6a1a4de0-a7be-42c6-b90d-b1aa03ff7940) |
| 6      | EfficientNetB0| RS: 54, Img: 128x128, Dense: 128, Dropout: 0.3, Epoch: 40 | 4.69%   | 0.56    | ![Training Loss](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/80097ad8-d254-4bfc-bc7d-033ed031027f) | ![Training Accuracy](https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/7ce4f267-5288-4162-b396-cf350b868703)
 |

#### Best Performed Model (VGG16) ROC Curve
<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/401595b8-8bcb-4178-ad31-70da2a90fad9" width="800"/>

# Face Recognition Attendance System

The face recognition system includes features for user registration, attendance checking, and additional functionalities like identifying eyes and smiles. Below is a brief overview of the system and placeholders for demonstration images.

### Tkinter GUI

The system uses Tkinter for the user interface, which includes buttons for checking attendance, identifying faces, and registering new users. Each button is associated with a specific function.

### CV2 and Face Recognition

OpenCV (cv2) is used for real-time image processing. The `face_recognition` library, built on dlib, provides functions to find, encode, and compare faces.

#### Register New User

New users can be registered by capturing their facial data and storing the corresponding embeddings. Below is a placeholder for images demonstrating user registration.

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/449504ee-e1b5-43e9-87f5-7b56217a7af0" width="500"/>

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/4766fa59-ec22-46b4-b3ac-dee4dbd7f9cf" width="500"/>

#### Check Attendance

The system logs attendance by identifying users in real-time and comparing live captures with stored facial embeddings. Below is a placeholder for images demonstrating the attendance check feature.

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/40d19bb2-6e65-47ba-b4ca-923283ded1fe" width="500"/>

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/06ef2c15-bc47-4f8a-a64d-0505cd7340ee" width="500"/>

#### Identify Eyes & Smile

Additional features include identifying eyes and smiles using Haar cascades provided by OpenCV. Below is a placeholder for images demonstrating these features.

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/e8a761a5-6496-4830-a020-6b6e3cb64462" width="500"/>

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/38260757-f697-43dc-9c72-9fdc8abcc8d8" width="500"/>

### Anti-Spoofing System

The anti-spoofing system ensures security by using the MiniFASNet architecture, which includes auxiliary supervision of the Fourier spectrum to differentiate between real and fake faces. Below is a placeholder for images demonstrating the spoofing detection system.

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/87dafc8e-b64a-4917-b251-3adf6c83b94e" width="500"/>

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/61fb2d03-3860-459b-8ed8-7d25e45abeed" width="500"/>

<img src="https://github.com/anhminhnguyen3110/Face-attendance-system/assets/57170354/7a4552ce-7bee-4816-ae8e-a041aa95b6b6" width="1000"/>

## Conclusion

The VGG16 model with Haar Cascade optimization achieved the best performance in this study. Configured with a random seed of 54, image size of 128x128, dense layer size of 128, dropout rate of 0.3, and trained for 30 epochs, it achieved a train accuracy of 98.34%, train loss of 0.11, validation accuracy of 70.42%, validation loss of 1.44, test accuracy of 70.42%, and a micro-average ROC curve area (AUC) of 0.97. Other models like ResNet50 and EfficientNetB0 did not converge effectively, while DenseNet121, InceptionV3, and MobileNetV2 showed decent performance but did not surpass VGG16.

These findings suggest that VGG16 optimized with Haar Cascade and appropriate configurations provides the best balance between accuracy and performance. Future work will focus on further fine-tuning hyperparameters, exploring more complex architectures, and reducing overfitting to enhance performance even further.
