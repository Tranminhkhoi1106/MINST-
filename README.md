# MNIST Handwritten Digit Classification using Random Forest

This project implements a **Random Forest Classifier** to recognize handwritten digits from the **MNIST dataset**. It demonstrates how classical machine learning can effectively handle image classification tasks without requiring deep neural networks.

---

## 1. Project Overview

The MNIST dataset is a benchmark collection of **70,000 grayscale images** (28×28 pixels) of handwritten digits (0–9).
This project aims to:

* Preprocess and flatten image data for classical ML input.
* Train a Random Forest classifier to predict digits.
* Evaluate accuracy, precision, recall, and F1-score.
* Test the trained model on an external image.

---

## 2. Data Loading & Preprocessing

* **Dataset Source:** `keras.datasets.mnist`
* **Train/Test Split:** 60,000 images for training, 10,000 for testing.

### Steps:

1. Loaded the dataset:

   ```python
   from keras.datasets import mnist
   (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
   ```
2. Displayed example images using:

   ```python
   plt.imshow(train_images[i])
   ```
3. **Flattened** images from 28×28 into 784-dimensional vectors:

   ```python
   train_images_flattened = train_images.reshape(train_images.shape[0], -1)
   test_images_flattened = test_images.reshape(test_images.shape[0], -1)
   ```
4. Created DataFrames and added labels:

   ```python
   df_train['Target'] = train_labels
   df_test['Target'] = test_labels
   ```
5. Checked data integrity:

   ```python
   df_train.isnull().sum().any()  # Result: False
   ```

=> **No missing data or corrupt images** were found.

---

## 3. Statistical Overview

* Each sample contains **784 pixel features**.
* Targets are balanced across digits 0–9.
* Since MNIST is pre-normalized (pixel range 0–255), Random Forest can handle it without scaling.

---

## 4. Model Development

A **Random Forest Classifier** from Scikit-learn was trained with 244 estimators (trees).

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=244)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

* The choice of **244 estimators** was based on balancing accuracy and computational cost.
* The model uses ensemble averaging to reduce overfitting and improve prediction stability.

---

## 5. Model Evaluation

The model achieved **97% accuracy** on the test dataset — strong performance for a non-deep learning model.

### 🔢 Classification Report

| Digit | Precision | Recall | F1-Score | Support |
| :---- | :-------: | :----: | :------: | :-----: |
| 0     |    0.97   |  0.99  |   0.98   |   980   |
| 1     |    0.99   |  0.99  |   0.99   |   1135  |
| 2     |    0.96   |  0.97  |   0.97   |   1032  |
| 3     |    0.96   |  0.96  |   0.96   |   1010  |
| 4     |    0.98   |  0.97  |   0.98   |   982   |
| 5     |    0.98   |  0.96  |   0.97   |   892   |
| 6     |    0.98   |  0.98  |   0.98   |   958   |
| 7     |    0.97   |  0.96  |   0.97   |   1028  |
| 8     |    0.97   |  0.96  |   0.96   |   974   |
| 9     |    0.96   |  0.96  |   0.96   |   1009  |

**Overall Performance:**

* **Accuracy:** 0.97
* **Macro Avg (Precision / Recall / F1):** 0.97 / 0.97 / 0.97
* **Weighted Avg:** 0.97 / 0.97 / 0.97

### 📊 Confusion Matrix

The confusion matrix was visualized using Seaborn’s heatmap:

```python
sns.heatmap(cm, annot=True, cmap='coolwarm')
```

<img width="785" height="813" alt="download" src="https://github.com/user-attachments/assets/7c2578aa-e362-42b2-85c4-f35837e0e6fc" />

This visualization shows excellent separation among all digit classes, with only minor confusion between similar-looking digits (e.g., 3 and 5, 4 and 9).

---

## 🖼️ 6. Custom Image Prediction

A new handwritten digit image was tested with the trained model:

1. Loaded the image using **PIL (Python Imaging Library): (<img width="247" height="303" alt="Annotation 2025-07-25 171204" src="https://github.com/user-attachments/assets/4f040347-a067-4d06-81fc-ab57d62cb433" />
rl).
2. Converted to grayscale (`'L'` mode).
3. Resized to **28×28 pixels**.
4. Flattened and reshaped to `(1, 784)`.
5. Predicted using:

   ```python
   predicted_class = model.predict(new_image_flattened)
   print("Predicted digit:", predicted_class[0])
   ```
6. Displayed the preprocessed image:

   ```python
   plt.imshow(new_image_reshaped, cmap='gray')
   plt.axis('off')
   plt.title("Preprocessed Image")
   plt.show()
   ```
   
<img width="389" height="411" alt="download (1)" src="https://github.com/user-attachments/assets/29933121-2954-4fe8-8e2c-eeee0e5972f9" />

7. Predicted Number: 2

---

## 7. Insights & Discussion

* **Accuracy of 97%** demonstrates Random Forest’s robustness on structured datasets like MNIST.
* It performs well without convolutional feature extraction.
* Misclassifications mainly occur between digits with overlapping handwriting shapes.

 **Advantages:**

* Simple to train and interpret.
* Resistant to overfitting with enough trees.

 **Limitations:**

* High memory usage (many trees).
* Not ideal for large or high-resolution image datasets.

---

## 8. Future Improvements

* Compare with **Convolutional Neural Networks (CNNs)** for feature learning.
* Use **Principal Component Analysis (PCA)** to reduce dimensionality before training.
* Implement **Grid Search** or **RandomizedSearchCV** for hyperparameter tuning.

---

## 9. Key Takeaways

✅ Random Forest achieves **high accuracy** on MNIST without deep learning.
✅ Minimal preprocessing (flattening only) is sufficient.
✅ Excellent baseline model for image classification benchmarks.

---
