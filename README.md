# Handwritten Digit Classification using Logistic Regression  

This repository contains a Jupyter Notebook that demonstrates how to classify handwritten digits using **Logistic Regression** with the **Scikit-learn** library.  

## Dataset  
The notebook utilizes the `digits` dataset from `sklearn.datasets`, which consists of 8x8 pixel grayscale images of handwritten digits (0-9).  

## Features  
- Load the dataset and visualize sample images  
- Preprocess the data and split it into training and testing sets  
- Train a **Logistic Regression** model  
- Evaluate the model's performance  
- Generate a **confusion matrix** to analyze classification results  

---

## Installation  

To run this project, you need Python and the required dependencies installed. Use the following command to install dependencies:  

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Usage  

Clone the repository and open the Jupyter Notebook to explore the code and experiment with different models.

```bash
git clone <your-repository-url>
cd <your-repository-folder>
jupyter notebook
```

---

## Code Overview  

### Import Necessary Libraries  

```python
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
```

### Load Dataset  

```python
digits = load_digits()
print("Dataset structure:", dir(digits))
```

### Visualizing Sample Images  

```python
plt.gray()
for i in range(5):
  plt.matshow(digits.images[i])
```

### Splitting Data into Training and Testing Sets  

```python
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)
```

### Training the Model  

```python
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
```

### Evaluating Model Performance  

```python
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
```

### Making Predictions  

```python
sample_index = 67
plt.matshow(digits.images[sample_index])
predicted_label = model.predict([digits.data[sample_index]])
print("Predicted Label:", predicted_label)
```

### Generating a Confusion Matrix  

```python
y_predicted = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", conf_matrix)
```

---

## Results  

- The model achieves a high accuracy on the test dataset.  
- The confusion matrix helps visualize misclassifications.  
- The trained model can predict handwritten digits based on pixel data.  

---

## Contributing  

Feel free to fork this repository, create a feature branch, and submit a pull request! 🚀  

---

## License  

This project is open-source and available under the **MIT License**.  
