# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or sum values using .isnull() and .sum() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Predict the values of array.

5.Calculate the accuracy by importing the required modules from sklearn.

6.Apply new unknown values.

## Program:
```python
'''
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by : K MADHAVA REDDY 
RegisterNumber : 212223240064 
'''
import pandas as pd 
data = pd.read_csv('Employee.csv')
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### Head()
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/9c92f996-2722-41f6-ad4d-1e45716054b7)

### info()
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/3cf530aa-e6b8-4009-936a-60ae7fac2f4e)

### isnull().sum()
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/e689125f-63cf-4616-ba1b-94963266e1c7)

### Left value counts
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/49ffeb1b-e3fc-476c-a1c0-8d853e527d82)

### Head()(After transform of salary)
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/21551d3c-391a-4365-a2a0-dd71686b093c)

### After removing left and departments columns
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/4d7c519f-a256-47ae-b115-8ff196f4bc37)

### accuracy
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/394b0e59-b655-4a7a-85c3-bf3b9dd6f90a)

### prediction
![image](https://github.com/Madhavareddy09/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742470/7f740299-ac84-469e-aef0-074d70dd1965)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
