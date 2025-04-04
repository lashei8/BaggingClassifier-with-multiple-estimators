# Draw_DT.py

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# Create a BaggingClassifier with N = 3 estimators
clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=3, oob_score=True, random_state=22)
clf.fit(X_train, y_train)

# Check if the number of out-of-bag samples is reasonable
print(f"Out-of-bag score: {clf.oob_score_:.3f}")
print(f"Is the number of out-of-bag samples reasonable? {'Yes' if clf.oob_score_ > 0 else 'No'}")

# Plot each of the decision trees
for i, estimator in enumerate(clf.estimators_):
    plt.figure(figsize=(12, 8))
    plot_tree(estimator, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.title(f"Decision Tree {i+1}")
    plt.savefig("DrawDT_plot.pdf", dpi=300)
    plt.show()


