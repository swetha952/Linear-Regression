from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


np.random.seed(42)
X=np.random.uniform(1,10,(100,1))

#Doom=2x+0.5x^2+noise
y=2*X.squeeze()+0.5*(X.squeeze()**2)+np.random.normal(0,2.0,100)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Linear regression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_pr=lr.predict(X_test)
lr_r2=r2_score(y_test,lr_pr)
lr_mean=mean_squared_error(y_test,lr_pr)


print("--- LINEAR REGRESSION ---")
print("Coefficient:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R-squared:", lr_r2)
print("MSE:", lr_mean)
print("--------------------------")
print(" ")

#Polynomial Regression
poly=PolynomialFeatures(degree=2,include_bias=False)

X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test) 

poly_model=LinearRegression()

poly_model.fit(X_train_poly,y_train)
poly_pr=poly_model.predict(X_test_poly)
poly_r2=r2_score(y_test,poly_pr)
poly_mean=mean_squared_error(y_test,poly_pr)

print("--- POLYNOMIAL REGRESSION ---")
print("R-squared:", poly_r2)
print("MSE:", poly_mean)
print("--------------------------")
print(" ")

#Ridge Regression
scalar=StandardScaler()

X_train_sc=scalar.fit_transform(X_train)
X_test_sc=scalar.transform(X_test)

ridge=Ridge(alpha=1.0)

ridge.fit(X_train_sc,y_train)
ridge_pr=ridge.predict(X_test_sc)
ridge_r2=r2_score(y_test,ridge_pr)
ridge_mean=mean_squared_error(y_test,ridge_pr)

print("--- RIDGE REGRESSION ---")
print("Coefficient:", ridge.coef_[0])
print("R-squared:", ridge_r2)
print("MSE:", ridge_mean)
print("--------------------------")
print(" ")


scores = {
    "Linear Regression": lr_r2,
    "Polynomial Regression": poly_r2,
    "Ridge Regression": ridge_r2
}

best_model=max(scores,key=scores.get)

print("--- MODEL COMPARISON ---")
for model,score in scores.items():
    print(model, "R-squared:", score)

print("\nBest Model:", best_model)

tasks = float(input("\nEnter number of pending tasks: "))
user_input = np.array([[tasks]])

if best_model == "Linear Regression":

    prediction = lr.predict(user_input)[0]

elif best_model == "Polynomial Regression":

    user_poly = poly.transform(user_input)
    prediction = poly_model.predict(user_poly)[0]

else:

    user_scaled = scalar.transform(user_input)
    prediction = ridge.predict(user_scaled)[0]


print("\n===== DEADLINE DOOM PREDICTION =====")

print("Pending Tasks:", tasks)
print("Predicted Doom Score:", round(prediction, 2))

if prediction < 15:
    print("Doom Level:CHILL")

elif prediction < 30:
    print("Doom Level:SLIGHTLY COOKED")

elif prediction < 50:
    print("Doom Level:COOKED")

else:
    print("Doom Level:ABSOLUTELY DOOMED")