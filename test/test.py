import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics

# Carico il dataset
path = '../color_extr/data.csv'
data = pd.read_csv(path, usecols = [i for i in range(5)])

# Creo train e test set
X_train, X_test, Y_train, Y_test = train_test_split(data[['presenza_foglie','assenza_foglie','presenza_bachi_sfondo','assenza_bachi_sfondo']], data[['classificazione']], test_size=0.13)

# Normalizzo i dati
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Logistic Regression
# Alleno il modello e ottengo le predizioni
model_lr = LogisticRegression()
lr = model_lr.fit(X_train, Y_train)
predictions_lr = model_lr.predict(X_test)

# Controllo accuratezza risultati
train_score = model_lr.score(X_train, Y_train)
test_score = model_lr.score(X_test, Y_test)
print("### LOGISTIC:\n\tAccuracy of training: "+str(train_score)+"\n\tAccuracy of test: "+str(test_score))

# SVM
# Alleno il modello e ottengo le predizioni
clf = svm.SVC(kernel='linear') # Linear Kernel
svmm = clf.fit(X_train, Y_train)
predictions_svmm = clf.predict(X_test)

# Controllo accuratezza risultati
train_score = metrics.accuracy_score(Y_test, predictions_svmm)
test_score = svmm.score(X_train, Y_train)
print("### SVM:\n\tAccuracy of training: "+str(train_score)+"\n\tAccuracy of test: "+str(test_score))

# K-nearest neighbour
# Alleno il modello e ottengo le predizioni
model_knn = classifier = KNeighborsClassifier(n_neighbors=3)
knn = model_knn.fit(X_train, Y_train)
predictions_knn = model_knn.predict(X_test)

# Controllo accuratezza risultati
train_score = metrics.accuracy_score(Y_test, predictions_knn)
test_score = knn.score(X_train, Y_train)
print("### KNN:\n\tAccuracy of training: "+str(train_score)+"\n\tAccuracy of test: "+str(test_score))