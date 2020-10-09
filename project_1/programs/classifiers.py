import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time


df1 = pd.read_csv('new_data.csv')
df2 = pd.read_csv('test_set.csv')
dataset1 = df1.values
x_train=dataset1[:,5:21]
y_train=dataset1[:,4]
dataset2 = df2.values
x_test=dataset2[:,5:21]
y_test=dataset2[:,4]

#KNN

KNN = KNeighborsClassifier(n_neighbors = 5, p = 2)
KNN_para = KNN.get_params()
start = time.time()
KNN.fit(x_train, y_train)
stop = time.time()
time1=stop-start
y_predict_KNN = KNN.predict(x_test)
KNN_accuracy = accuracy_score(y_test, y_predict_KNN)
print("K-NN :\nparameters: ",KNN_para,"\naccuracy: ",KNN_accuracy,"\ntraining time: ",time1)

#DT

DT = DecisionTreeClassifier(max_depth=15, splitter='random',criterion= 'entropy')
DT_para = DT.get_params()
start = time.time()
DT.fit(x_train, y_train)
stop = time.time()
time2=stop-start
y_predict_DT = DT.predict(x_test)
DT_accuracy = accuracy_score(y_test, y_predict_DT)
DT_para = DT.get_params()
print("\nDecision Tree :\nparameters: ",DT_para,"\naccuracy: ",DT_accuracy,"\ntraining time: ",time2)


#SVM

SVM = SVC(kernel='linear', C=1)
start = time.time()
SVM.fit(x_train, y_train)
stop = time.time()
time3=stop-start
y_predict_SVM = SVM.predict(x_test)
SVM_accuracy = accuracy_score(y_test, y_predict_SVM)
SVM_para = SVM.get_params()
print("\nSVM :\nparameters: ",SVM_para,"\naccuracy: ",SVM_accuracy,"\ntraining time: ",time3)


#MLP

MLP = MLPClassifier(activation='identity',max_iter=3000,solver='lbfgs')
start = time.time()
MLP.fit(x_train, y_train)
stop = time.time()
time4=stop-start
y_predict_MLP = MLP.predict(x_test)
MLP_accuracy = accuracy_score(y_test, y_predict_MLP)
MLP_para = MLP.get_params()
print("\nMLP :\nparameters: ",MLP_para,"\naccuracy: ",MLP_accuracy,"\ntraining time: ",time4)

#voting ensemble methods

KNN = KNeighborsClassifier(n_neighbors = 5, p = 3)
KNN1 = KNN.fit(x_train,y_train)
KNN_proba=KNN1.predict_proba(x_test)

DT = DecisionTreeClassifier(max_depth=8, splitter='random',random_state=85)
DT1 = DT.fit(x_train,y_train)
DT_proba=DT1.predict_proba(x_test)

SVM = SVC(kernel='linear', C=2, probability=True)
SVM1 = SVM.fit(x_train,y_train)
SVM_proba=SVM1.predict_proba(x_test)

MLP = MLPClassifier(activation='identity',max_iter=5000,solver='lbfgs')
MLP1 = MLP.fit(x_train,y_train)
MLP_proba=MLP1.predict_proba(x_test)

#mean
mean_matrix = []
mean = (KNN_proba + DT_proba + SVM_proba + MLP_proba) / 4
[row,col] = mean.shape
for i in range(row):
    if mean[i][0]>mean[i][1]:
        mean_matrix.append(1)
    else:
        mean_matrix.append(2)
time5=stop-start
mean_accuracy = accuracy_score(y_test, mean_matrix)
print("\nMean :\naccuracy: ",mean_accuracy)


#max
temp = []
for j in range(row):
    x = max(KNN_proba[j][0], DT_proba[j][0],SVM_proba[j][0],MLP_proba[j][0])
    y = max(KNN_proba[j][1], DT_proba[j][1],SVM_proba[j][1],MLP_proba[j][1])
    temp.append([x,y])
max_matrix = []
for m in range(row):
    if temp[m][0]>temp[m][1]:
        max_matrix.append(1)
    else:
        max_matrix.append(2)
stop=time.time()
max_accuracy = accuracy_score(y_test, max_matrix)
print("\nMax :\naccuracy: ",max_accuracy)







