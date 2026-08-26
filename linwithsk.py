from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

np.random.seed(42)
X=np.random.uniform(0, 10, (100, 1))
y=3.0*X.squeeze()+7.0+np.random.normal(0,2.0,100)

#splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=SklearnLR()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

print("--- Scikit-learn Linear Regression ---")
print(f"Coefficient (w): {lr.coef_[0]:.4f}")
print(f"Intercept (b): {lr.intercept_:.4f}")

r2_score(y_test,y_pred)
print(f"R-squared (test): {r2_score(y_test, y_pred):.4f}")

mean_squared_error(y_test, y_pred)
print(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")

poly=PolynomialFeatures(degree=2,include_bias=False)
X_poly_sk = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
lr_poly = SklearnLR()
lr_poly.fit(X_poly_sk, y_train)
lr_poly.predict(X_poly_test)
r2_score(y_test, lr_poly.predict(X_poly_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"Ridge R-squared: {r2_score(y_test, ridge.predict(X_test_scaled)):.4f}")
print(f"Ridge coefficient: {ridge.coef_[0]:.4f}")