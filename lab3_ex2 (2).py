from natasha import Segmenter, Doc, DatesExtractor, MorphVocab
from natasha.emb import NewsEmbedding
from natasha.morph.tagger import NewsMorphTagger
from natasha.syntax import NewsSyntaxParser

from natasha.doc import DocSent, DocToken

# Загружаем текст книги из файла
with open("Молокин Алексей.Блюз 100 рентген. 1 глава.txt", "r", encoding="utf-8") as f:
    doc = Doc(f.read())

segmenter = Segmenter()  # Разбивает текст на предложения
emb = NewsEmbedding()  # Словарь
tagger = NewsMorphTagger(emb)  # Морфологический разбор
parser = NewsSyntaxParser(emb)  # Синтаксический разбор

doc.segment(segmenter)  # Устанавливаем сегментатор
doc.tag_morph(tagger)  # Устанавливаем морфологический разборщик
doc.parse_syntax(parser)  # Устанавливаем синтаксический разборщик

sents: list[DocSent] = doc.sents  # Предложения
print(f"{len(sents)} предложений в тексте")

print("Первые 10 токенов:")
print("               Токен |      Тип |     Ч.р. | Особенности")
print("---------------------+----------+----------+------------")
for token in doc.tokens[:10]:
    token: DocToken = token
    print(f"{token.text :>20} | {token.rel :>8} | {token.pos :>8} | {token.feats}")


# Загружаем текст с датами из файла
with open("dates.txt", "r", encoding="utf-8") as f:
    text = f.read()

dateExtractor = DatesExtractor(MorphVocab())  # Экстрактор дат
matches = dateExtractor(text)
print("Даты в тексте:")
for date in matches:
    print(f"{date.fact.day}-{date.fact.month}-{date.fact.year}")
