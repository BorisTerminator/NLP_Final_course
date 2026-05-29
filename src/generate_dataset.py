"""
Generates synthetic Russian expense categorization dataset via GigaChat API.
Produces ~1260 labeled examples across 9 categories, saved to data/ as CSV files.
"""
import csv
import random
from pathlib import Path

from langchain_gigachat import GigaChat

GIGACHAT_CREDS = "MDE5ZGIwZDItZDkzNS03ODBiLWI3MzMtOWVhZmRlY2YxYzMxOmE2Njk2ZDE1LTg4MzYtNGQyNS1hMDQ4LTFkYjViYTliOThiNg=="

CATEGORIES = {
    'еда': 'покупки продуктов питания в супермаркетах, продуктовых магазинах, у фермеров',
    'транспорт': 'такси, метро, автобус, электричка, заправка автомобиля, парковка, проездной',
    'кафе': 'кофе, рестораны, столовые, бары, доставка еды на дом, обеды и ужины вне дома',
    'здоровье': 'аптека, врач, больница, медицинские анализы, процедуры, витамины, лекарства',
    'одежда': 'одежда, обувь, аксессуары, сумки, украшения, покупки в магазинах одежды',
    'техника': 'электроника, смартфоны, ноутбуки, компьютеры, гаджеты, аксессуары к технике',
    'подписки': 'онлайн-сервисы, стриминг видео и музыки, игровые подписки, облачные сервисы',
    'развлечения': 'кино, театр, концерты, игры, боулинг, квест-комнаты, выставки, музеи',
    'другое': 'подарки, канцелярия, книги, цветы, бытовая химия, хозяйственные товары',
}


def generate_examples(giga: GigaChat, category: str, description: str, n: int = 140) -> list[str]:
    prompt = (
        f'Сгенерируй {n} коротких описаний финансовых трат категории "{category}" '
        f'({description}).\n\n'
        'Это записи, которые пользователь вводит в приложение учёта расходов вручную.\n'
        'Примеры формата: "кофе в старбаксе", "такси до аэропорта", "продукты в пятёрочке", '
        '"ужин с другом 1500р".\n\n'
        'Требования:\n'
        '- Каждая запись на отдельной строке\n'
        '- Длина 2–8 слов\n'
        '- Разнообразие: с суммами (350р, 1200₽) и без, с названиями мест и без\n'
        '- Реалистичные русские фразы, как пишут обычные люди\n'
        '- Только список строк, без нумерации, точек и пояснений\n\n'
        f'Выдай ровно {n} строк:'
    )
    response = str(giga.invoke(prompt).content).strip()
    lines = [
        line.strip().lstrip('•·-–—0123456789.) ').strip()
        for line in response.split('\n')
        if line.strip()
    ]
    lines = [l for l in lines if l and 1 <= len(l.split()) <= 12]
    return lines[:n]


def save_csv(data: list[dict], path: Path) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'category'])
        writer.writeheader()
        writer.writerows(data)


def main():
    giga = GigaChat(
        credentials=GIGACHAT_CREDS,
        verify_ssl_certs=False,
        model='GigaChat-2',
        temperature=0.7,
        max_tokens=3000,
    )

    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    all_examples = []
    for category, description in CATEGORIES.items():
        print(f'[{category}] Generating examples...', flush=True)
        examples = generate_examples(giga, category, description, n=140)
        for text in examples:
            all_examples.append({'text': text, 'category': category})
        print(f'  Got {len(examples)} examples', flush=True)

    random.seed(42)
    random.shuffle(all_examples)

    n = len(all_examples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = all_examples[:n_train]
    val = all_examples[n_train:n_train + n_val]
    test = all_examples[n_train + n_val:]

    save_csv(all_examples, data_dir / 'dataset.csv')
    save_csv(train, data_dir / 'train.csv')
    save_csv(val, data_dir / 'val.csv')
    save_csv(test, data_dir / 'test.csv')

    print(f'\nDataset created: {n} total examples')
    print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
    print(f'Saved to: {data_dir}')


if __name__ == '__main__':
    main()
