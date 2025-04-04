# Bagging_Iris.py

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Print the content of the Iris dataset (Preliminary work)
print(iris)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# Define a range for the number of estimators (N)
estimator_range = range(2, 21)

# Initialize lists to store models and scores
scores = []

# Loop over the range of N (number of estimators)
for n_estimators in estimator_range:
    # Create a BaggingClassifier with DecisionTreeClassifier as estimator
    clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n_estimators, random_state=22)
    # Fit the model
    clf.fit(X_train, y_train)
    # Calculate and store the accuracy score
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

# Step 3: Plot the accuracy score vs. N
plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scores, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel("Number of Estimators (N)", fontsize=14)
plt.ylabel("Accuracy Score", fontsize=14)
plt.title("Bagging: Accuracy vs. Number of Estimators", fontsize=16)
plt.xticks(estimator_range)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig('BaggingIris_plot.pdf', dpi=300)
plt.show()
