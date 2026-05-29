"""Runs all 4 classifiers and saves results to results/."""
import json
import sys
from pathlib import Path

# allow importing siblings from src/
sys.path.insert(0, str(Path(__file__).parent))

import baseline_rules
import baseline_tfidf
import gigachat_zeroshot
import gigachat_fewshot

RESULTS_DIR = Path(__file__).parent.parent / 'results'
DATA_DIR = Path(__file__).parent.parent / 'data'


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    print('\n' + '=' * 60)
    print('1/4  RULE-BASED BASELINE')
    print('=' * 60)
    r = baseline_rules.evaluate(str(DATA_DIR / 'test.csv'))
    results['Rule-based'] = {'accuracy': r['accuracy'], 'f1_weighted': r['f1_weighted']}

    print('\n' + '=' * 60)
    print('2/4  TF-IDF + LOGISTIC REGRESSION')
    print('=' * 60)
    baseline_tfidf.train(str(DATA_DIR / 'train.csv'), str(DATA_DIR / 'val.csv'))
    r = baseline_tfidf.evaluate(str(DATA_DIR / 'test.csv'))
    results['TF-IDF + LR'] = {'accuracy': r['accuracy'], 'f1_weighted': r['f1_weighted']}

    print('\n' + '=' * 60)
    print('3/4  GIGACHAT ZERO-SHOT')
    print('=' * 60)
    r = gigachat_zeroshot.evaluate(str(DATA_DIR / 'test.csv'), sample_n=100)
    results['GigaChat Zero-shot'] = {'accuracy': r['accuracy'], 'f1_weighted': r['f1_weighted']}

    print('\n' + '=' * 60)
    print('4/4  GIGACHAT FEW-SHOT  (proposed)')
    print('=' * 60)
    r = gigachat_fewshot.evaluate(
        str(DATA_DIR / 'test.csv'),
        str(DATA_DIR / 'train.csv'),
        sample_n=100,
        n_per_class=3,
    )
    results['GigaChat Few-shot'] = {'accuracy': r['accuracy'], 'f1_weighted': r['f1_weighted']}

    # Save JSON
    with open(RESULTS_DIR / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save CSV
    import pandas as pd
    rows = [
        {'Approach': k, 'Accuracy': f"{v['accuracy']:.4f}", 'F1 (weighted)': f"{v['f1_weighted']:.4f}"}
        for k, v in results.items()
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / 'results.csv', index=False)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for name, metrics in results.items():
        print(f"  {name:<25}  Acc={metrics['accuracy']:.4f}  F1={metrics['f1_weighted']:.4f}")
    print(f'\nSaved to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
