"""Rule-based expense classifier using keyword dictionaries."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

KEYWORDS: dict[str, list[str]] = {
    'еда': [
        'продукты', 'пятёрочка', 'пятерочка', 'перекрёсток', 'перекресток',
        'магнит', 'ашан', 'лента', 'дикси', 'окей', 'вкусвилл', 'атак',
        'хлеб', 'молоко', 'мясо', 'рыба', 'овощи', 'фрукты', 'бакалея',
        'супермаркет', 'гипермаркет', 'алкоголь', 'пиво', 'вино', 'groceries',
        'продуктовый', 'сыр', 'масло', 'яйца', 'йогурт', 'крупа',
    ],
    'транспорт': [
        'такси', 'метро', 'автобус', 'маршрутка', 'яндекс такси', 'убер', 'uber',
        'бензин', 'заправка', 'лукойл', 'газпромнефть', 'парковка', 'электричка',
        'поезд', 'ласточка', 'сапсан', 'трамвай', 'троллейбус', 'самокат',
        'проездной', 'каршеринг', 'делимобиль', 'ситимобил', 'яндекс.такси',
        'транспортная карта', 'билет на поезд',
    ],
    'кафе': [
        'кафе', 'ресторан', 'кофе', 'старбакс', 'starbucks', 'обед в', 'ужин в',
        'пицца', 'суши', 'бар', 'бургер', 'макдак', 'макдоналдс', 'kfc',
        'доставка еды', 'додо', 'яндекс еда', 'столовая', 'шаурма', 'sushi',
        'пироговая', 'блинная', 'кондитерская', 'круассан', 'капучино', 'латте',
        'эспрессо', 'завтрак в', 'ланч', 'бизнес-ланч', 'фастфуд',
    ],
    'здоровье': [
        'аптека', 'врач', 'больница', 'поликлиника', 'лекарства', 'анализы',
        'стоматолог', 'витамины', 'медицин', 'клиника', 'таблетки',
        'мазь', 'капли', 'физиотерапия', 'массаж', 'психолог', 'терапевт',
        'рецепт', 'импакт', 'асна', 'горздрав', 'ригла',
    ],
    'одежда': [
        'одежда', 'обувь', 'джинсы', 'футболка', 'куртка', 'платье', 'зара', 'zara',
        'h&m', 'спортмастер', 'брюки', 'рубашка', 'пальто', 'носки', 'бельё',
        'кроссовки', 'сапоги', 'ботинки', 'свитер', 'толстовка', 'пуховик',
        'юбка', 'шарф', 'перчатки', 'шапка', 'uniqlo', 'gloria jeans',
    ],
    'техника': [
        'телефон', 'ноутбук', 'планшет', 'наушники', 'зарядка', 'dns', 'ситилинк',
        'эльдорадо', 'мвидео', 'iphone', 'samsung', 'apple', 'xiaomi', 'кабель',
        'адаптер', 'мышка', 'клавиатура', 'монитор', 'колонка', 'powerbank',
        'чехол для', 'стекло на', 'провод', 'флешка', 'жёсткий диск',
    ],
    'подписки': [
        'netflix', 'spotify', 'яндекс плюс', 'подписка', 'vk музыка', 'premier',
        'кинопоиск', 'telegram premium', 'adobe', 'notion', 'яндекс музыка',
        'сбер прайм', 'amazon prime', 'twitch', 'youtube premium', 'apple music',
        'ivi', 'okko', 'start', 'more.tv',
    ],
    'развлечения': [
        'кино', 'театр', 'концерт', 'клуб', 'игра', 'steam', 'музей', 'выставка',
        'боулинг', 'квест', 'батут', 'каток', 'зоопарк', 'аквапарк',
        'аттракцион', 'escape room', 'игровой', 'кальян', 'nightclub',
        'билеты на', 'спектакль', 'цирк',
    ],
}


def classify_rules(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {cat: 0 for cat in KEYWORDS}
    for cat, keywords in KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1
    best_cat = max(scores, key=lambda c: scores[c])
    return best_cat if scores[best_cat] > 0 else 'другое'


def evaluate(test_path: str) -> dict:
    df = pd.read_csv(test_path)
    df['predicted'] = df['text'].apply(classify_rules)
    acc = accuracy_score(df['category'], df['predicted'])
    f1 = f1_score(df['category'], df['predicted'], average='weighted', zero_division='warn')
    print('=== Rule-based ===')
    print(classification_report(df['category'], df['predicted'], zero_division='warn'))
    return {'accuracy': float(acc), 'f1_weighted': float(f1), 'df': df}


if __name__ == '__main__':
    result = evaluate('data/test.csv')
    print(f"Accuracy: {result['accuracy']:.4f}  F1: {result['f1_weighted']:.4f}")
