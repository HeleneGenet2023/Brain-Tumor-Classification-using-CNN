# Brain-Tumor-Classification-using-CNN

# Introduction

This project utilizes Convolutional Neural Networks (CNN) for the classification of brain tumors from MRI images. Early detection and classification of brain tumors are vital steps in the planning of effective treatment strategies. This project aims to assist in this by automatically classifying brain tumors into one of the four categories: glioma, meningioma, no tumor, and pituitary.

# Dataset
The dataset used in this project is a combination of three datasets: figshare, SARTAJ, and Br35H. It comprises 7023 MRI images classified into the aforementioned four classes. Notably, some adjustments have been made to the original SARTAJ dataset to ensure the correct categorization of glioma class images. You can find and explore the dataset using the following link: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

# Installation
To install the necessary pacjages to run the script, use the following commands:
!pip install numpy tensorflow scikit-learn
!pip install opencv-python-headless

# Usage

1. Data Preprocessing: The preprocess_image function is used to preprocess MRI images before feeding them into the model. It performs operations such as thresholding and contour detection to crop and resize the images to a uniform size.

2. Model: The script uses a Sequential model from the Keras library, comprised of convolutional layers followed by max-pooling layers, and fully connected layers at the end. The model is compiled with the Adam optimizer and categorical crossentropy as the loss function.

3. Training and Evaluation: The script includes sections to define constants for image dimensions and batch size, data augmentation, and normalization. After defining the model, it is trained on the training dataset and validated on a validation set. The trained model is then saved as 'brain_tumor_cnn_model.h5'.

4. Testing: After training, the script evaluates the model on a test dataset, prints the accuracy, and displays the confusion matrix to visualize the performance of the model on the test data.

# Running the script
Make sure to adjust train_dir and test_dir to point to the correct directories where your datasets are stored.
Execute the script in a Python environment where all the necessary packages are installed.

# Results

The performance of the model was evaluated using a test dataset containing 1311 images, which were categorized into four classes: glioma, meningioma, no tumor, and pituitary. The evaluation metrics used were precision, recall, and F1-score, along with a confusion matrix to visualize the performance of the model. Below are the detailed results of the evaluation:

## Confusion Matrix
The confusion matrix is a tabulation that describes the performance of the model on the test dataset. It shows the number of true positive, true negative, false positive, and false negative predictions made by the model. Here is the confusion matrix obtained from the evaluation:

### Confusion Matrix

|            | Predicted: Glioma | Predicted: Meningioma | Predicted: No Tumor | Predicted: Pituitary |
|------------|-------------------|------------------------|---------------------|----------------------|
| **Glioma**       | 282               | 17                    | 0                   | 1                    |
| **Meningioma**   | 16                | 243                   | 35                  | 12                   |
| **No Tumor**     | 0                 | 6                     | 399                 | 0                    |
| **Pituitary**    | 9                 | 4                     | 0                   | 287                  |



From the matrix, it can be seen that the model has a high accuracy in classifying the images into the correct categories, with a large number of true positives and a relatively low number of false positives and false negatives.

## Classification Report
The classification report provides detailed information on the precision, recall, and F1-score for each class, along with the accuracy of the model. Here is a summary of the classification report:

|    Class    | Precision | Recall | F1-Score | Support |
|:-----------:|:---------:|:------:|:--------:|:-------:|
|   glioma    |    0.92   |  0.94  |   0.93   |   300   |
| meningioma  |    0.90   |  0.79  |   0.84   |   306   |
|   notumor   |    0.92   |  0.99  |   0.95   |   405   |
|  pituitary  |    0.96   |  0.96  |   0.96   |   300   |
|  **Accuracy**  |          |       |   0.92   |   1311  |
| **Macro Avg**  |    0.92   |  0.92  |   0.92   |   1311  |
|**Weighted Avg**|    0.92   |  0.92  |   0.92   |   1311  |




## Analysis

- Precision: The model exhibits high precision across all classes, indicating that it has a high true positive rate with a lower rate of false positives. The precision values are between 0.90 and 0.96 for the different classes.
- Recall: The recall values indicate that the model has a high true positive rate. The recall values range from 0.79 to 0.99, indicating a strong ability to correctly identify positive cases within each class.
- F1-Score: The F1-score is the harmonic mean of precision and recall, providing balance between the two metrics.  The F1-Scores are high for all the classes, indicating a good balance between precision and recall.
- Accuracy: The model correctly classifies 92% of the images in the test dataset.

# Conclusion

The model demonstrates good performance in classifying brain tumors from MRI images with an accuracy of 92%. It achieves good precision and recall values across all classes, indicating that it can correctly identify and categorize a high percentage of brain tumors with a low rate of false positives and false negatives. This suggests that the model could potentially be a valuable tool in assisting with the early detection and classification of brain tumors. Future work could focus on further optimizing the model and expanding the dataset to improve performance, particuliarly in the meningioma class where the recall is slightly lower compared to other classes.



