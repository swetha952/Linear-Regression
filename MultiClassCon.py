"""Implement a multi-class confusion matrix for the 3-class softmax model. Compute per-class precision and recall.
Which class is hardest to classify?"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)

#generating 2 classes
X0=np.random.randn(100,2)+[2,2]
X1=np.random.randn(100,2)+[5,5]
X2=np.random.randn(100,2)+[8,2]

X=np.vstack([X0,X1,X2])
y=np.array([0]*100+[1]*100+[2]*100) #creating labels for each class, first 100->1,next 100->2 and rest 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

cm=[
    [0,0,0],
    [0,0,0],
    [0,0,0]
]

for act,pred in zip(y_test,y_pred):
    cm[act][pred]+=1

print("Confusion Matrix: ")
for row in cm:
    print(row)

for c in range(3):
    TP=cm[c][c]
    FP = sum(cm[row][c] for row in range(3)) - TP

    FN = sum(cm[c]) - TP

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    print("\nClass", c)
    print("Precision:", precision)
    print("Recall:", recall)
