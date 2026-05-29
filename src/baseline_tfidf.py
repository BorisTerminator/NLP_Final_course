"""TF-IDF (char n-grams) + Logistic Regression expense classifier."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline


MODEL_PATH = Path(__file__).parent.parent / 'models' / 'tfidf_lr.pkl'


def train(train_path: str, val_path: str | None = None) -> Pipeline:
    train_df = pd.read_csv(train_path)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=50_000,
            sublinear_tf=True,
        )),
        ('lr', LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver='lbfgs',
        )),
    ])

    pipeline.fit(train_df['text'], train_df['category'])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

    if val_path:
        val_df = pd.read_csv(val_path)
        val_preds = pipeline.predict(val_df['text'])
        val_acc = accuracy_score(val_df['category'], val_preds)
        print(f'Val accuracy: {val_acc:.4f}')

    return pipeline


def evaluate(test_path: str) -> dict:
    pipeline: Pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(test_path)
    preds = pipeline.predict(df['text'])
    acc = accuracy_score(df['category'], preds)
    f1 = f1_score(df['category'], preds, average='weighted', zero_division='warn')
    print('=== TF-IDF + LR ===')
    print(classification_report(df['category'], preds, zero_division='warn'))
    return {'accuracy': float(acc), 'f1_weighted': float(f1), 'df': df, 'predictions': preds.tolist()}


if __name__ == '__main__':
    train('data/train.csv', 'data/val.csv')
    result = evaluate('data/test.csv')
    print(f"Accuracy: {result['accuracy']:.4f}  F1: {result['f1_weighted']:.4f}")
