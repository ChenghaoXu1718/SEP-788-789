import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import timeit
# Process data
path = "/Users/xukaibin/Desktop/SEP/SEP788/database/iris.data"
tags = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
database = pd.read_csv(path, names=tags)
X = database.iloc[:, 0:4]
Y = database.iloc[:, 4]
print(database)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# implement KNN
start_knn = timeit.default_timer()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
knn.predict(X_test)
stop_knn = timeit.default_timer()

knn_accuracy = knn.score(X_test, Y_test)
knn_cm = metrics.confusion_matrix(Y_test, knn.predict(X_test))
knn_f1 = metrics.f1_score(Y_test, knn.predict(X_test), average='micro')
print("\nKnn accuracy is: " + str(knn_accuracy),
      "\nKnn f1 score is: " + str(knn_f1),
      "\nKnn confusion matrix is: " + str(knn_cm),
      "\nKnn runtime is: " + str(stop_knn - start_knn))

# implement NB
start_nb = timeit.default_timer()
GNB = GaussianNB()
GNB.fit(X_train, Y_train)
GNB.predict(X_test)
stop_nb = timeit.default_timer()

nb_accuracy = GNB.score(X_test, Y_test)
nb_cm = metrics.confusion_matrix(Y_test, GNB.predict(X_test))
nb_f1 = metrics.f1_score(Y_test, GNB.predict(X_test), average='micro')
print("\nNB accuracy is: " + str(nb_accuracy),
      "\nNB f1 score is: " + str(nb_f1),
      "\nNB confusion matrix is: " + str(nb_cm),
      "\nNB runtime is: " + str(stop_nb - start_nb))

# implement SVM
start_svm = timeit.default_timer()
SVM = svm.SVC()
SVM.fit(X_train, Y_train)
SVM.predict(X_test)
stop_svm = timeit.default_timer()

svm_accuracy = SVM.score(X_test, Y_test)
svm_cm = metrics.confusion_matrix(Y_test, SVM.predict(X_test))
svm_f1 = metrics.f1_score(Y_test, SVM.predict(X_test), average='micro')
print("\nsvm accuracy is: " + str(svm_accuracy),
      "\nsvm f1 score is: " + str(svm_f1),
      "\nsvm confusion matrix is: " + str(svm_cm),
      "\nsvm runtime is: " + str(stop_svm - start_svm))



