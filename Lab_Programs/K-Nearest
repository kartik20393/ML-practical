from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 
iris = load_iris() 
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42) 
knn_classifier = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train) 
predictions = knn_classifier.predict(X_test) 
print(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}") 
print("Classification Report:\n", metrics.classification_report(y_test, predictions)) 
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, predictions))
