import nltk

# Скачиваем библиотеку для работы с тегами (один раз)
nltk.download("averaged_perceptron_tagger")


# Функция, чтобы не дублировать код
def tag_tokens(text):
    # Выделяем токены
    tokens = nltk.word_tokenize(text)
    # Теггируем токены
    tagged = nltk.pos_tag(tokens)
    # Выводим первые 20 токенов
    for word, tag in tagged[:20]:
        print(f"({tag} {word})")


# В качестве английского текста возьмём текст песни Bo Burnham - That Funny Feeling
print("English tokens and tags:")
with open("Bo Burnham - That Funny Feeling.txt", "r", encoding="utf-8") as f:
    tag_tokens(f.read())

# В качестве русского текста возьмём текст 1 главы книги Молокина Алексея - Блюз 100 рентген
print("Russian tokens and tags:")
with open("Молокин Алексей.Блюз 100 рентген. 1 глава.txt", "r", encoding="utf-8") as f:
    tag_tokens(f.read())
