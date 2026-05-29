# Классификация финансовых трат на русском языке

## Финальный проект по курсу NLP, весна 2026

Многоклассовая классификация коротких русскоязычных описаний расходов по 9 категориям.

## Задача

По короткой пользовательской записи (1–10 слов) определить категорию траты:

| Категория | Примеры |
| --------- | ------- |
| еда | «продукты в пятёрочке», «молоко хлеб 300р» |
| транспорт | «такси до офиса», «заправка лукойл» |
| кафе | «кофе в старбаксе», «ужин с другом 2500р» |
| здоровье | «аптека таблетки», «анализы крови» |
| одежда | «кроссовки новые 3500₽», «джинсы зара» |
| техника | «наушники airpods», «зарядка для телефона» |
| подписки | «яндекс плюс», «netflix месяц» |
| развлечения | «кино с женой», «квест на двоих» |
| другое | «подарок маме», «цветы 800р» |

## Датасет

Синтетический датасет, сгенерированный через GigaChat-2 (temperature=0.7):

- **995 примеров**, 9 категорий
- **Train**: 796 | **Val**: 99 | **Test**: 100
- От 66 до 140 примеров на категорию

## Результаты

| Подход | Accuracy | F1 (weighted) |
| ------ | -------- | ------------- |
| Rule-based | 0.44 | 0.45 |
| TF-IDF + LR | **0.79** | **0.79** |
| GigaChat Zero-shot | 0.62 | 0.59 |
| GigaChat Few-shot | **0.79** | **0.79** |

**Основной вывод:** TF-IDF+LR на символьных n-граммах и GigaChat Few-shot показывают одинаковую точность. Few-shot примечателен тем, что не требует обучения классификатора — достаточно нескольких примеров в промпте.

## Структура проекта

```text
data/
  dataset.csv     # полный датасет (995 примеров)
  train.csv       # обучающая выборка 80% (796)
  val.csv         # валидационная выборка 10% (99)
  test.csv        # тестовая выборка 10% (100)
src/
  generate_dataset.py   # генерация датасета через GigaChat API
  baseline_rules.py     # правила (keyword matching)
  baseline_tfidf.py     # TF-IDF + логистическая регрессия
  gigachat_zeroshot.py  # GigaChat zero-shot
  gigachat_fewshot.py   # GigaChat few-shot (предложенный подход)
  run_experiments.py    # запуск всех экспериментов
  generate_report.py    # генерация report/report.pdf
results/
  results.json    # метрики всех подходов
  results.csv     # таблица результатов
report/
  report.pdf      # полный научный отчёт
requirements.txt
```

## Запуск

```bash
pip install -r requirements.txt

# 1. Сгенерировать датасет (нужен ключ GigaChat API)
python src/generate_dataset.py

# 2. Запустить все эксперименты
python src/run_experiments.py

# 3. Сгенерировать PDF отчёт
python src/generate_report.py
```

## Описание подходов

### 1. Rule-based (Baseline 1)

Словари ключевых слов — 20–30 ключевых слов на категорию (бренды, названия мест, типы товаров). Не требует обучения.

### 2. TF-IDF + Logistic Regression (Baseline 2)

Символьные n-граммы (2–4), 50 000 признаков, sublinear TF, мультиклассовая LR (C=5.0). Обучается на 796 примерах за < 1 секунды.

### 3. GigaChat Zero-shot (конкурентный подход)

Прямая классификация через LLM без примеров: промпт содержит только список категорий и текст для классификации.

### 4. GigaChat Few-shot (предложенный подход)

3 размеченных примера на каждый класс (27 пар всего) добавляются в промпт из обучающей выборки. Устраняет неоднозначность между близкими категориями без дообучения модели.

## Литература

1. Joulin et al. *Bag of Tricks for Efficient Text Classification.* EACL 2017.
2. Zhang et al. *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015.
3. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
4. Куратов, Архипов. *Адаптация глубоких двунаправленных трансформеров для русского языка.* arXiv 2019.
5. Brown et al. *Language Models are Few-Shot Learners.* NeurIPS 2020.
6. Сбербанк. *GigaChat.* [developers.sber.ru/portal/products/gigachat](https://developers.sber.ru/portal/products/gigachat), 2024.
