import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


df = pd.read_csv("./tweets_train.csv") # Загружаем датасет


df = df[df["text"].notna()][df["sentiment"].notna()] # Предобраотка
df = df.drop_duplicates()

print("Dataset:") # Визуализируем датасет
print(df.shape)
print(df.head(5))

le = LabelEncoder() # Кодирование категориальных данных
df["encoded-sentiment"] = le.fit_transform(df["sentiment"])

# Необходимая информация
X = df["text"]
y = df["encoded-sentiment"]

vectorizer = CountVectorizer() # Векторизация текста
X = vectorizer.fit_transform(X)
vectorizer.get_feature_names_out()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # Разделение данных

mnb = MultinomialNB() # Модель
mnb.fit(X_train, y_train)

pred = mnb.predict(X_test) # Evaluation

score = accuracy_score(y_test, pred) # Точность

print(f"Score: {score:.2f}") # Отчет по модели

# Проверим модель в действии
random_from_df = df.sample(n=10) # берем случайный сэмпл из датасета
pred = mnb.predict(vectorizer.transform(random_from_df["text"])) # прогнозируем тональность
random_from_df["predicted-sentiment"] = le.inverse_transform(pred) # обратное преобразование

# Вывод результатов
print(random_from_df[["text", "sentiment", "predicted-sentiment"]])
