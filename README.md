# Handwritten Digit Classification Using Artificial Neural Networks (ANN)

## **Overview**
This project demonstrates the use of an Artificial Neural Network (ANN) for classifying handwritten digits from the **MNIST dataset**. The MNIST dataset contains 28x28 grayscale images of handwritten digits (0-9), along with their corresponding labels. The ANN is implemented using TensorFlow and Keras.

---

## **Dataset**
- **Source**: MNIST Dataset
- **Training Data**: 60,000 images
- **Test Data**: 10,000 images
- **Image Dimensions**: 28x28 pixels (grayscale)

---

## **Project Workflow**
1. **Data Loading and Preprocessing**
   - The MNIST dataset is loaded using `keras.datasets.mnist`.
   - Pixel values are normalized to the range `[0,1]` by dividing by 255.

2. **Model Architecture**
   - A Sequential model is built with the following layers:
     - `Flatten`: Converts the 28x28 input images into a 1D array of size 784.
     - `Dense (128 neurons)`: Fully connected layer with ReLU activation.
     - `Dense (64 neurons)`: Fully connected layer with ReLU activation.
     - `Dense (10 neurons)`: Output layer with Softmax activation for multi-class classification.

3. **Compilation**
   - **Loss Function**: Sparse Categorical Crossentropy
   - **Optimizer**: Adam
   - **Metrics**: Accuracy

4. **Model Training**
   - The model is trained for 30 epochs with a validation split of 20%.

5. **Evaluation**
   - The model's performance is evaluated on the test set using accuracy.

6. **Visualization**
   - Training and validation loss are plotted to analyze model performance.

---

## **Results**
- Test Accuracy: **97.65%**

The model achieves high accuracy in classifying handwritten digits from the MNIST dataset.

---

## **Model Summary**

| Layer (Type)       | Output Shape | Parameters |
|--------------------|--------------|------------|
| Flatten            | (None, 784) | 0          |
| Dense (ReLU)       | (None, 128) | 100,480    |
| Dense (ReLU)       | (None, 64)  | 8,256      |
| Dense (Softmax)    | (None, 10)  | 650        |

- Total Parameters: `109,386`
- Trainable Parameters: `109,386`
- Non-Trainable Parameters: `0`

---

## **Conclusion**
This project demonstrates how an ANN can effectively classify handwritten digits from grayscale images using TensorFlow/Keras. The high test accuracy highlights the capability of neural networks in handling image classification tasks.

Feel free to experiment with different architectures or hyperparameters to further optimize performance!

---

### How to Use This Repository
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/handwritten-digit-classification-ann.git
   ```
2. Navigate to the directory:
   ```bash
   cd handwritten-digit-classification-ann
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook handwritten-digit-classification-using-ann.ipynb
   ```
4. Execute the cells in the notebook to train and evaluate the model.

---

### Dependencies
To run this project, install the following Python libraries:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install them using:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

### Repository Contents
- `handwritten-digit-classification-using-ann.ipynb`: Jupyter Notebook containing the implementation.
- `README.md`: Project documentation.

---

### License
This project is licensed under the MIT License. Feel free to use and modify it as needed!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/51708891/32ebd210-217e-4a6e-9f39-b60486be9ea4/handwritten-digit-classification-using-ann.ipynb

---
