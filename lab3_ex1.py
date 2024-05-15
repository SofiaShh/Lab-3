import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Real estate.csv") # читаем файл

df = df.drop_duplicates() # Предварительная обработка данных
print(df.shape)
print(df.head(5))

print("Correlation matrix:") # Анализ данных
print(df.corr())

# Датасет
X = df[
    [
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude",
    ]
]
y = df["Y house price of unit area"]

# Разделение данных
RANDOM_STATE = 762146634  # random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)  # 70% обучения и 30% тестирования

model = LinearRegression() # Модель
model.fit(X_train, y_train)

print(f"Coefficients for linear regression formula:")
print(pd.DataFrame(model.coef_, X.columns, columns=["Coefficients"]))

y_pred = model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print(f"Model evaluation: \n\t{MSE=} \n\t{R2=}")

# Визуализация
predicted_prices = model.predict(X) # Спрогнозируем цену для каждого дома 

# Визуализация реальных и прогнозируемых цен
plt.plot([min(y), max(y)], [min(y), max(y)], color="blue", linewidth=1)  # price line
plt.scatter(y, predicted_prices, color="red", s=0.5)  # predicted prices against real
plt.xlabel("Real prices")
plt.ylabel("Predicted prices")
plt.title("Real prices vs. Predicted prices")
plt.show()
