from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
iris = load_iris()
X, y, class_names = iris.data, iris.target, [str(name) for name in iris.target_names]
DecisionTreeClassifier().fit(X, y)
plt.figure(figsize=(12, 8))
plot_tree(DecisionTreeClassifier().fit(X, y), feature_names=iris.feature_names, class_names=class_names, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
