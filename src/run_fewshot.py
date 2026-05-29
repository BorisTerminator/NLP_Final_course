import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pandas as pd
import gigachat_fewshot

DATA = Path(__file__).parent.parent / 'data'
RESULTS = Path(__file__).parent.parent / 'results'

r = gigachat_fewshot.evaluate(str(DATA / 'test.csv'), str(DATA / 'train.csv'), sample_n=100, n_per_class=3)
print(f'FEW-SHOT acc: {r["accuracy"]:.4f}  f1: {r["f1_weighted"]:.4f}')

results = json.loads((RESULTS / 'results_partial.json').read_text(encoding='utf-8'))
results['GigaChat Few-shot'] = {'accuracy': r['accuracy'], 'f1_weighted': r['f1_weighted']}

(RESULTS / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

rows = [{'Approach': k, 'Accuracy': round(v['accuracy'], 4), 'F1_weighted': round(v['f1_weighted'], 4)}
        for k, v in results.items()]
pd.DataFrame(rows).to_csv(RESULTS / 'results.csv', index=False)

print('\n=== FINAL RESULTS ===')
for name, m in results.items():
    print(f'  {name}: acc={m["accuracy"]:.4f}  f1={m["f1_weighted"]:.4f}')
