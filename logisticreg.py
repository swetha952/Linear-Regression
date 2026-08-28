#sigmoid fn and data generation
"""Converts any number into a value between 0 and 1.
Used to represent the probability of belonging to class 1"""
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

"""Creates 200 artificial data points.
100 points belong to Class 0, centered around (2,2).
100 points belong to Class 1, centered around (5,5).
The data is shuffled so the classes aren't ordered."""

random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

#logistic regression from scratch
"""Uses the generated data to learn how to distinguish Class 0 and Class 1.
It calculates probabilities using the sigmoid function.
Usually:
probability ≥ 0.5 → Class 1
probability < 0.5 → Class 0"""
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

#confusion matrix and metrics from scratch
"""Checks how well the model predicted.
Gives:
Accuracy → overall correct predictions
Precision → how many predicted 1s were actually 1
Recall → how many actual 1s were correctly found
F1-score → balance between precision and recall"""
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

#decision boundary analysis
"""Finds the line that separates Class 0 and Class 1.
Since you have 2 features, the boundary is basically a line on a graph."""
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

#multicalss with softmax
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

#threshold tuning
"""Softmax is used when there are more than 2 classes.
Example:
Class 0 → Cat
Class 1 → Dog
Class 2 → Bird
The current data actually has only 2 classes, so Softmax isn't necessary here."""
class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [0.0, 0.0]
        self.bias = 0.0

    def fit(self, X, y):
        n = len(X)

        for _ in range(self.epochs):
            dw = [0.0, 0.0]
            db = 0.0

            for i in range(n):
                z = self.weights[0] * X[i][0] + self.weights[1] * X[i][1] + self.bias
                prediction = sigmoid(z)

                error = prediction - y[i]

                dw[0] += error * X[i][0]
                dw[1] += error * X[i][1]
                db += error

            self.weights[0] -= self.learning_rate * dw[0] / n
            self.weights[1] -= self.learning_rate * dw[1] / n
            self.bias -= self.learning_rate * db / n

    def predict_proba(self, x):
        z = self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias
        return sigmoid(z)

    def predict(self, X):
        return [1 if self.predict_proba(x) >= 0.5 else 0 for x in X]
    
class ClassificationMetrics:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self):
        correct = sum(a == b for a, b in zip(self.y_true, self.y_pred))
        return correct / len(self.y_true)

    def precision(self):
        tp = sum(a == 1 and b == 1 for a, b in zip(self.y_true, self.y_pred))
        fp = sum(a == 0 and b == 1 for a, b in zip(self.y_true, self.y_pred))

        if tp + fp == 0:
            return 0

        return tp / (tp + fp)

    def recall(self):
        tp = sum(a == 1 and b == 1 for a, b in zip(self.y_true, self.y_pred))
        fn = sum(a == 1 and b == 0 for a, b in zip(self.y_true, self.y_pred))

        if tp + fn == 0:
            return 0

        return tp / (tp + fn)

    def f1(self):
        p = self.precision()
        r = self.recall()

        if p + r == 0:
            return 0

        return 2 * p * r / (p + r)
# Split data into training and testing
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]
model = LogisticRegression(learning_rate=0.01, epochs=1000)

model.fit(X_train, y_train)
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")