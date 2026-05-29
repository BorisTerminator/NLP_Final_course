"""GigaChat zero-shot expense classifier — no training examples in prompt."""
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


def classify(giga: GigaChat, text: str) -> str:
    cats_str = ', '.join(CATEGORIES)
    prompt = (
        f'Категоризируй эту запись о финансовой трате: "{text}"\n\n'
        f'Категории: {cats_str}\n\n'
        'Ответь ТОЛЬКО одним словом — названием категории из списка выше.'
    )
    result = str(giga.invoke(prompt).content).strip().lower()
    for cat in CATEGORIES:
        if cat in result:
            return cat
    return 'другое'


def evaluate(test_path: str, sample_n: int = 100) -> dict:
    giga = _make_giga()
    df = pd.read_csv(test_path)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    print(f'GigaChat zero-shot: classifying {len(df)} examples...', flush=True)
    preds = []
    for idx, text in enumerate(df['text'].tolist()):
        pred = classify(giga, str(text))
        preds.append(pred)
        if (idx + 1) % 20 == 0:
            print(f'  {idx+1}/{len(df)}', flush=True)

    df['predicted'] = preds
    acc = accuracy_score(df['category'], preds)
    f1 = f1_score(df['category'], preds, average='weighted', zero_division='warn')
    print('=== GigaChat Zero-shot ===')
    print(classification_report(df['category'], preds, zero_division='warn'))
    return {'accuracy': float(acc), 'f1_weighted': float(f1), 'df': df, 'predictions': preds}


if __name__ == '__main__':
    result = evaluate('data/test.csv', sample_n=100)
    print(f"Accuracy: {result['accuracy']:.4f}  F1: {result['f1_weighted']:.4f}")
