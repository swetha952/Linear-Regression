from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0=np.random.randn(100,2)+[2,2]
X_1=np.random.randn(100,2)+[5,5]
X_sk=np.vstack([X_0,X_1])
y_sk=np.array([0] * 100 + [1] * 100)

X_train,X_test,y_train,y_test=train_test_split(X_sk,y_sk,test_size=0.2,random_state=42)

scalar=StandardScaler()
X_tr_sc=scalar.fit_transform(X_train)
X_te_sc=scalar.transform(X_test)

ls=LogisticRegression()
ls.fit(X_tr_sc,y_train)
y_pred=ls.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))