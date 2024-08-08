# **Title of Project**
# Iris Flower Classification

## **Import Library**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

print("Data Info:")
print(data.info())
print("\nData Description:")
print(data.describe())
print("\nFirst Few Rows:")
print(data.head())

sns.pairplot(data, hue='species', palette='viridis')
plt.show()

X = data.drop('species', axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
prediction = model.predict(new_data)
print("\nPrediction for new data [5.1, 3.5, 1.4, 0.2]:")
print(iris.target_names[prediction][0])

