# Short Query Analysis — Topic 0

## Overview
- **23 short queries** (<8 words) tested from Topic 0 (the hardest cluster)
- **Baseline R@5**: 87.0% (20/23)
- **HyDE answer embedding**: 87.0% — no improvement
- **HyDE query rewriting (3x)**: 87.0% — no improvement
- **HyDE hybrid RRF**: 87.0% — no improvement

## All 23 Short Queries

| # | Query | Expected FAQ | Course |
|---|-------|-------------|--------|
| 1 | what are embeddings | What are embeddings? | llm-zoomcamp |
| 2 | how do u turn words into numbers that a computer can understand | What are embeddings? | llm-zoomcamp |
| 3 | **how to prevent multicollinearity in linear regression** | Why do we sometimes use random_state and not at other times? | ml-zoomcamp |
| 4 | **correlating numerical and categorical data** | How do you find the correlation matrix? | ml-zoomcamp |
| 5 | what is f1 score | What is the difference between F1 score and accuracy? | ml-zoomcamp |
| 6 | what is roc auc | What is the difference between AUROC and ROC AUC? | ml-zoomcamp |
| 7 | tf gpu memory error | Out of memory errors when running tensorflow | ml-zoomcamp |
| 8 | tensorflow oom fix | Out of memory errors when running tensorflow | ml-zoomcamp |
| 9 | **docker model not updating** | Getting the same result | ml-zoomcamp |
| 10 | mlflow ui windows error | mlflow ui on Windows FileNotFoundError | mlops-zoomcamp |
| 11 | leaderboard not showing | Leaderboard: I am not on the leaderboard | ml-zoomcamp |
| 12 | homework q3 help | Homework Q4: Is r same as alpha in Scikit-Learn Ridge? | ml-zoomcamp |
| 13 | dlt exercise 3 | Homework: dlt Exercise 3 - Merge a generator concerns | de-zoomcamp |
| 14 | sum ages filter jobs | Homework: dlt Exercise 3 | de-zoomcamp |
| 15 | singular matrix error | Singular Matrix Error | ml-zoomcamp |
| 16 | xgboost value error | ValueError: not enough values to unpack when parsing XGBoost | ml-zoomcamp |
| 17 | rmse squared false | TypeError: got unexpected keyword argument 'squared' | ml-zoomcamp |
| 18 | high rmse validation | Overfitting: Absurdly high RMSE on the validation dataset | ml-zoomcamp |
| 19 | tensorflow serving mac | Docker: tensorflow serving image illegal instruction | ml-zoomcamp |
| 20 | nvidia smi loop | Running 'nvidia-smi' in a loop without using 'watch' | ml-zoomcamp |
| 21 | reproducibility projects | Reproducibility: Do we have to run everything? | de-zoomcamp |
| 22 | adding metric errors | Adding additional metric | de-zoomcamp |
| 23 | gcp cloud function csv | GCP BQ - Tip: Using Cloud Function to read csv.gz files | de-zoomcamp |

## Failures (All Methods)

These 3 queries fail baseline, answer embedding, query rewriting, AND hybrid RRF:

### 1. "how to prevent multicollinearity in linear regression"
- **Expected**: "Why do we sometimes use random_state and not at other times?" (ml-zoomcamp)
- **Issue**: Query is specific (multicollinearity), FAQ is generic (random_state). Semantic mismatch.

### 2. "correlating numerical and categorical data"
- **Expected**: "How do you find the correlation matrix?" (ml-zoomcamp)
- **Issue**: Query asks about mixed data types, FAQ is about correlation matrix in general.

### 3. "docker model not updating"
- **Expected**: "Getting the same result" (ml-zoomcamp)
- **Issue**: Query mentions Docker, FAQ doesn't. FAQ title is vague ("Getting the same result").

## Key Insight

These failures aren't vocabulary gaps — they're **semantic mismatches** where the generated query implies a different intent than the FAQ title. HyDE can't fix this because the fake answers it generates align with the query's intent, not the FAQ's actual content.
