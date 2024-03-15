from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_model = SVC()

# Fit the model on the training set
svm_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier with Random Forest as the base estimator
svm_rf_model = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
svm_rf_model.estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training set
svm_rf_model.fit(X_train, y_train)
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Initialize Random Forest classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

st.title('Iris Flower Classification')
st.write('This app performs classification of iris flowers using SVM and Random Forest algorithms.')

# Make predictions and display accuracy for SVM
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
st.write('SVM Accuracy:', accuracy_svm)
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create the SVM classifier
clf = SVC(kernel='linear', probability=True, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing data
predictions = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions, target_names=iris.target_names)

# Create a Streamlit app
st.title('Iris Flower Classification SVM (Indiani)')
st.write('Model training and evaluation results:')
st.write(f'Training set size: {len(X_train)}')
st.write(f'Testing set size: {len(X_test)}')
st.write(f'Model accuracy: {accuracy}')
st.write('Confusion Matrix:')
st.write(conf_matrix)
st.write('Classification Report:')
st.write(class_report)

# Add user input for feature values
sepal_length = st.slider('Sepal length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider('Sepal width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider('Petal length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider('Petal width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Make a prediction
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display the prediction
species = iris.target_names[prediction][0]
st.write(f'The predicted species is {species}') 

# Make predictions and display accuracy for Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
st.write('Random Forest Accuracy:', accuracy_rf)
# Make predictions on the testing set
y_pred_svm_rf = svm_rf_model.predict(X_test)

# Calculate accuracy
accuracy_svm_rf = accuracy_score(y_test, y_pred_svm_rf)
print("SVM with Random Forest Accuracy:", accuracy_svm_rf)

import joblib

# Simpan model ke dalam file H5
joblib.dump(clf, 'svm_model.h5')
