"""GigaChat few-shot expense classifier — proposed approach with in-context examples."""
import random
import pandas as pd
from langchain_gigachat import GigaChat
from sklearn.metrics import accuracy_score, f1_score, classification_report

GIGACHAT_CREDS = "MDE5ZGIwZDItZDkzNS03ODBiLWI3MzMtOWVhZmRlY2YxYzMxOmE2Njk2ZDE1LTg4MzYtNGQyNS1hMDQ4LTFkYjViYTliOThiNg=="

CATEGORIES = ['еда', 'транспорт', 'кафе', 'здоровье', 'одежда', 'техника', 'подписки', 'развлечения', 'другое']


def _make_giga() -> GigaChat:
    return GigaChat(
        credentials=GIGACHAT_CREDS,
        verify_ssl_certs=False,
        model='GigaChat-2',
        temperature=0.1,
        max_tokens=20,
    )


def build_few_shot_block(train_path: str, n_per_class: int = 3) -> str:
    train_df = pd.read_csv(train_path)
    blocks = []
    for cat in CATEGORIES:
        cat_examples = train_df[train_df['category'] == cat]['text'].tolist()
        sampled = random.sample(cat_examples, min(n_per_class, len(cat_examples)))
        for ex in sampled:
            blocks.append(f'Трата: "{ex}"\nКатегория: {cat}')
    random.shuffle(blocks)
    return '\n\n'.join(blocks)


def classify(giga: GigaChat, text: str, few_shot_block: str) -> str:
    cats_str = ', '.join(CATEGORIES)
    prompt = (
        'Вот примеры классификации финансовых трат:\n\n'
        f'{few_shot_block}\n\n'
        '---\n'
        f'Классифицируй эту запись: "{text}"\n'
        f'Категории: {cats_str}\n'
        'Ответь ТОЛЬКО одним словом — названием категории.'
    )
    result = str(giga.invoke(prompt).content).strip().lower()
    for cat in CATEGORIES:
        if cat in result:
            return cat
    return 'другое'


def evaluate(test_path: str, train_path: str, sample_n: int = 100, n_per_class: int = 3) -> dict:
    giga = _make_giga()
    random.seed(42)
    few_shot_block = build_few_shot_block(train_path, n_per_class=n_per_class)

    df = pd.read_csv(test_path)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    print(f'GigaChat few-shot ({n_per_class} ex/class): classifying {len(df)} examples...', flush=True)
    preds = []
    for idx, text in enumerate(df['text'].tolist()):
        pred = classify(giga, str(text), few_shot_block)
        preds.append(pred)
        if (idx + 1) % 20 == 0:
            print(f'  {idx+1}/{len(df)}', flush=True)

    df['predicted'] = preds
    acc = accuracy_score(df['category'], preds)
    f1 = f1_score(df['category'], preds, average='weighted', zero_division='warn')
    print('=== GigaChat Few-shot ===')
    print(classification_report(df['category'], preds, zero_division='warn'))
    return {'accuracy': float(acc), 'f1_weighted': float(f1), 'df': df, 'predictions': preds}


if __name__ == '__main__':
    result = evaluate('data/test.csv', 'data/train.csv', sample_n=100, n_per_class=3)
    print(f"Accuracy: {result['accuracy']:.4f}  F1: {result['f1_weighted']:.4f}")
