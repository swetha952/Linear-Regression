"""Generate a dataset that is NOT linearly separable (e.g., two concentric circles). 
Train logistic regression and observe its failure. Then add polynomial features (x1^2, x2^2, x1*x2)
 and train again. Show that the accuracy improves."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
N=200
X=[]
y=[]

#Class 0- Inner Circle
for _ in range(N//2):
    angle=np.random.uniform(0,2*np.pi)
    radius=np.random.uniform(0,2)

    x1=radius* np.cos(angle)
    x2=radius * np.sin(angle)

    X.append([x1,x2])
    y.append(0)

#Class 1- Outer Circle
for _ in range(N//2):
    angle=np.random.uniform(0.2*np.pi)
    radius=np.random.uniform(0,2)

    x1=radius* np.cos(angle)
    x2=radius* np.sin(angle)

    X.append([x1,x2])
    y.append(1)

X=np.array(X)
y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic regression training
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Normal Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))

#Polynomial features
X_poly=np.column_stack([X[:, 0],
    X[:, 1],
    X[:, 0] ** 2,
    X[:, 1] ** 2,
    X[:, 0] * X[:, 1]])

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred_poly=model.predict(X_test)

print("Polynomial Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_poly))