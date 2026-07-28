# expense-category-classifier

Nine-class categorisation of very short free-text expense notes, with four approaches
benchmarked against each other on the same split.

The interesting result is negative: a 1990s-era linear model matches a modern LLM, so the
LLM does not earn its place in production.

## The task

Given a note a user types into an expense tracker — one to ten words, Russian, often
misspelled and mixing brand names with amounts — assign one of nine categories.

| Category | Typical input |
|---|---|
| groceries | «продукты в пятёрочке», «молоко хлеб 300р» |
| transport | «такси до офиса», «заправка лукойл» |
| eating out | «кофе в старбаксе», «ужин с другом 2500р» |
| health | «аптека таблетки», «анализы крови» |
| clothing | «кроссовки новые 3500₽», «джинсы зара» |
| electronics | «наушники airpods», «зарядка для телефона» |
| subscriptions | «яндекс плюс», «netflix месяц» |
| entertainment | «кино с женой», «квест на двоих» |
| other | «подарок маме», «цветы 800р» |

Two properties make this harder than the class count suggests: the inputs are far too
short for context to disambiguate, and several categories overlap by construction —
a pharmacy purchase is *health*, but a supermarket purchase that includes medicine is
*groceries*.

## Results

Measured on a held-out 100-example test set. Raw numbers in
[`results/results.json`](results/results.json).

| Approach | Accuracy | F1 (weighted) |
|---|---|---|
| Rule-based keywords | 0.44 | 0.449 |
| **TF-IDF + Logistic Regression** | **0.79** | **0.790** |
| LLM zero-shot | 0.62 | 0.589 |
| LLM few-shot | 0.79 | 0.787 |

**Reading the table.** The keyword baseline sets the floor. Zero-shot prompting beats it
but loses badly to a trained linear model — the LLM knows language, not this label
taxonomy. Few-shot closes the gap entirely: twenty-seven examples in the prompt buy the
same accuracy as supervised training.

TF-IDF + LR and few-shot finish in a statistical tie, and that decides the engineering
question. The linear model trains in under a second, runs locally, costs nothing per call
and has no rate limit. It is the correct choice.

Where few-shot stays valuable is the cold start: it reaches production accuracy with no
labelled training set at all, which makes it the right tool while a taxonomy is still
changing and the wrong one once it has settled.

## Data

No public corpus of Russian expense notes exists, so the dataset is synthetic — generated
with GigaChat-2 at temperature 0.7, then split.

| | Examples |
|---|---|
| Total | 995 |
| Train | 796 |
| Validation | 99 |
| Test | 100 |

Per-category counts range from 66 to 140. Generating the data with an LLM and then showing
that a linear model beats that same LLM at the task is a deliberate part of the design.

## Approaches in detail

**Rule-based** — 20–30 keywords per category covering brands, place names and product
types. No training. Serves as the floor any learned model must clear.

**TF-IDF + Logistic Regression** — character n-grams of length 2–4, 50 000 features,
sublinear term frequency, multiclass logistic regression at `C=5.0`. Character n-grams
rather than words, because the inputs are short and misspelled: `пятёрочка` and
`пятерочка` must land in the same place. Trains on 796 examples in under a second.

**LLM zero-shot** — the prompt carries the category list and the text, nothing else.

**LLM few-shot** — three labelled examples per class, twenty-seven in total, drawn from
the training split and added to the prompt. This is what resolves the near-miss pairs the
zero-shot prompt confuses.

## Repository layout

| Path | Purpose |
|---|---|
| `src/generate_dataset.py` | Synthetic dataset generation via the GigaChat API |
| `src/baseline_rules.py` | Keyword baseline |
| `src/baseline_tfidf.py` | TF-IDF + logistic regression |
| `src/gigachat_zeroshot.py` | Zero-shot prompting |
| `src/gigachat_fewshot.py` | Few-shot prompting |
| `src/run_experiments.py` | Runs every approach, writes `results/` |
| `src/run_fewshot.py` | Few-shot run on its own |
| `src/generate_report.py` | Builds `report/report.pdf` |
| `data/` | Full dataset plus train / validation / test splits |
| `results/` | `results.json` and `results.csv` |
| `report/report.pdf` | Full write-up |

## Running it

```bash
pip install -r requirements.txt

python src/generate_dataset.py   # needs a GigaChat API key
python src/run_experiments.py
python src/generate_report.py
```

## Stack

Python · scikit-learn · GigaChat API · pandas

## References

1. Joulin et al. *Bag of Tricks for Efficient Text Classification.* EACL 2017.
2. Zhang et al. *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015.
3. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
4. Kuratov, Arkhipov. *Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language.* arXiv 2019.
5. Brown et al. *Language Models are Few-Shot Learners.* NeurIPS 2020.
6. Sber. *GigaChat.* [developers.sber.ru/portal/products/gigachat](https://developers.sber.ru/portal/products/gigachat), 2024.
