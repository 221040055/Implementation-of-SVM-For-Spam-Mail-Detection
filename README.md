# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement.
2.Read the given csv file and print the number of contents to be displayed.
3.Split the dataset using train_test_split.
4.Calculate Y_Pred and accuracy.
5.
Display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Hemavathi.N
RegisterNumber:  212221040055
*/
```import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

## Output:
![Screenshot 2023-06-02 142247](https://github.com/221040055/Implementation-of-SVM-For-Spam-Mail-Detection/assets/135315330/4c1228f8-682b-4192-acf7-a6c32e55e7a2)
![Screenshot 2023-06-02 142300](https://github.com/221040055/Implementation-of-SVM-For-Spam-Mail-Detection/assets/135315330/394ffa46-9bac-4a77-b119-f4728542407b)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
