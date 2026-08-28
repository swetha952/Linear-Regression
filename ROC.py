"""Build an ROC curve from scratch. For 100 threshold values from 0 to 1, compute the
 true positive rate and false positive rate. Calculate the AUC (area under the curve) 
using the trapezoidal rule."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Generate data
X0 = np.random.randn(100, 2) + [2, 2]
X1 = np.random.randn(100, 2) + [5, 5]

X = np.vstack([X0, X1])
y = np.array([0] * 100 + [1] * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Probability of Class 1
probabilities = model.predict_proba(X_test)[:, 1]


thresholds = np.linspace(0, 1, 100)

TPR = []
FPR = []

for threshold in thresholds:

    y_pred = [1 if p >= threshold else 0 for p in probabilities]

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for actual, predicted in zip(y_test, y_pred):

        if actual == 1 and predicted == 1:
            TP += 1

        elif actual == 0 and predicted == 0:
            TN += 1

        elif actual == 0 and predicted == 1:
            FP += 1

        elif actual == 1 and predicted == 0:
            FN += 1

    if TP + FN == 0:
        tpr = 0
    else:
        tpr = TP / (TP + FN)

    if FP + TN == 0:
        fpr = 0
    else:
        fpr = FP / (FP + TN)

    TPR.append(tpr)
    FPR.append(fpr)


# Sort points by FPR
points = sorted(zip(FPR, TPR))

FPR_sorted = [point[0] for point in points]
TPR_sorted = [point[1] for point in points]


# Trapezoidal rule
auc = 0

for i in range(1, len(FPR_sorted)):

    width = FPR_sorted[i] - FPR_sorted[i - 1]

    height = (TPR_sorted[i] + TPR_sorted[i - 1]) / 2

    auc += width * height


print("AUC:", auc)