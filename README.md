# EXPERIMENT NO: 04
# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries for data handling, modeling, and evaluation.

2. Load the Iris dataset using load_iris() from sklearn.datasets.

3. Create a DataFrame from the dataset and add a new column for the target (species).

4. Display the first few rows to understand the structure.

5. Split the DataFrame into:

     features: all columns except the target.

     target: the species label.

6. Use train_test_split() to divide the data into 80% training and 20% testing sets.

7. Initialize an SGDClassifier with default parameters (max_iter=1000, tol=1e-3).

8. Train the classifier on the training data using .fit().

9. Make predictions on the test data using .predict().

10. Evaluate model performance using accuracy_score() and print the result.

11. Compute and display the confusion matrix using confusion_matrix().

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: HARI PRIYA M 
RegisterNumber: 212224240047
*/
```
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    iris_data = load_iris()
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    iris_df['species'] = iris_data.target
    print(iris_df.head())
    
    features = iris_df.drop('species', axis=1)
    target = iris_df['species']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = SGDClassifier(max_iter=1000, tol=1e-3)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.3f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

## Output:

Displaying rows

![Screenshot 2025-05-12 223046](https://github.com/user-attachments/assets/95d03015-77ac-4019-8f0e-8ea864cb2e6a)


Model

![Screenshot 2025-05-12 223057](https://github.com/user-attachments/assets/1894d0ea-7506-40c1-b19d-5fa56c557b58)


Evaluating Accuracy

![Screenshot 2025-05-12 223710](https://github.com/user-attachments/assets/4bed7174-3051-4ea1-968c-5d0030caa1d0)


Confusion Matrix

![Screenshot 2025-05-12 223112](https://github.com/user-attachments/assets/bcad297f-00d1-4d0a-a06c-f2585abab252)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
