# Image Manipulation and Model Evaluation Project

## Overview
This project involves preprocessing, manipulating, and evaluating images to test a model's robustness under various conditions. The steps include preprocessing the dataset, training a deep learning model, and evaluating its performance on manipulated and white-balanced test sets.

---

## Phases of the Project

### 1. **Dataset Preparation**
- Loaded the image dataset for classification.
- Normalized image pixel values to the range [0, 1] for faster model convergence.
- Split the dataset into training and test sets.

---

### 2. **Model Architecture and Training**
- Defined a deep learning model with convolutional layers, pooling layers, and dense layers.
- Used the Adam optimizer and sparse categorical cross-entropy as the loss function.
- Trained the model on the normalized training dataset, achieving an accuracy of **77.03%** on the normal test set.

---

### 3. **Image Manipulation: Brightness and Contrast Adjustment**
- Adjusted the brightness and contrast of test set images using OpenCV's `convertScaleAbs` function.
  - **Contrast (alpha)**: Multiplied pixel values to adjust contrast.
  - **Brightness (beta)**: Added constant value to adjust brightness.
- Saved manipulated images as `.png` files for further evaluation.

#### **Result**:
- Model accuracy on the manipulated test set: **11.13%**.

---

### 4. **White Balance Adjustment with Gray World Algorithm**
- Implemented the Gray World algorithm to apply white balance:
  - Calculated average values of the Red, Green, and Blue channels.
  - Scaled each channel to match the overall average color.
- Applied this technique to the manipulated test set images.

#### **Result**:
- Model accuracy on the white-balanced test set: **10.41%**.

---

### 5. **Model Evaluation and Comparison**
- Evaluated the model's performance on:
  1. **Original Test Set**: Accuracy = **77.03%**
  2. **Manipulated Test Set**: Accuracy = **11.13%**
  3. **White-Balanced Test Set**: Accuracy = **10.41%**
- Observed a significant drop in accuracy on manipulated and white-balanced datasets, indicating limited model generalization.

---

## Insights and Improvements

### Observations
- The model struggles to generalize under altered image conditions (brightness, contrast, and color balance).
- Accuracy significantly drops on manipulated and white-balanced test sets.

### Potential Solutions
1. **Data Augmentation**:
   - Incorporate brightness and contrast adjustments during training.
   - Apply random color transformations for robustness.

2. **Fine-Tuning**:
   - Train the model on datasets with varied lighting and contrast.

3. **Regularization**:
   - Use dropout and L2 regularization to prevent overfitting.

4. **Pretrained Models**:
   - Leverage models trained on diverse datasets (e.g., ImageNet).

5. **Learning Rate Adjustment**:
   - Experiment with learning rate schedules for better convergence.

---

## Conclusion
The project highlights the importance of robust preprocessing and model generalization. While the model performs well on the original dataset, additional strategies such as data augmentation and fine-tuning are required to handle real-world variations effectively.

You can access the notebook on Kaggle using the link below:

[Kaggle Notebook: Image Processing with CNN](https://www.kaggle.com/code/fatmanurkantar/image-processing-with-cnn)

