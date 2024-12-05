import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

filename = "./data/08_pima-indians-diabetes.data.csv"

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(Y_pred)

Y_pred_binary = (Y_pred > 0.5).astype(int)
print(Y_pred_binary)

accuracy = accuracy_score(Y_test, Y_pred_binary)

print("-" * 25)
print("Actual Values: ", Y_test)
print("Predicted Values: ", Y_pred_binary)
print("-" * 25)
print("Accuracy", accuracy)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(Y_pred_binary)), Y_pred_binary, color='red', label='Predicted Values', marker='x')

plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Data Index')
plt.ylabel('Class (0 or 1')
plt.legend()

plt.savefig("./results/linear_regression.png")