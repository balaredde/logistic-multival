# Jupyter Notebook Overview

## Description


## Code Examples

```python
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
```

```python
digits = load_digits()
dir(digits)
```

```python
digits.data[0]
```

```python
plt.gray()
for i in range(5):
  plt.matshow(digits.images[i])
```

```python
digits.images[0]
```

```python
digits.target[0:5]
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)
```

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
```

```python
model.score(X_test, y_test)
```

```python
plt.matshow(digits.images[67])
```

```python
digits.target[67]
```

```python
model.predict([digits.data[67]])
```

```python
from sklearn.metrics import confusion_matrix
y_predicted = model.predict(X_test)
confusion_matrix(y_test, y_predicted)
```
