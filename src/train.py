from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

auto = pd.read_csv("data/processed/auto_features.csv")

# Separate train and test data

X = auto.drop("price", axis=1)

y = auto["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train the model!

model = LinearRegression()

model.fit(X_train, y_train)

# Display the coefficient of determination
print('Coefficient of determination (train):{:.3f}'.format(model.score(X_train, y_train)))
print('Coefficient of determination (test):{:.3f}'.format(model.score(X_test, y_test)))

# Display regression coefficients and intercept
print('\nRegression coefficients\n{}'.format(pd.Series(model.coef_, index=X.columns)))
print('Intercept: {:.3f}'.format(model.intercept_))

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")