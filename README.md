# Kaggle Playground S5E9 : BPM Prediction (Task 1)

Data prep, cleaning, EDA and feature engineering for the Beats-per-Minute regression task (metric: RMSE).

## Repo contents
| File | Purpose |
|---|---|
| `bpm_task1_eda_fe.ipynb` | Full pipeline with outputs |
| `eda_visuals/` | 4 small PNGs (distributions, BPM-vs-feature deciles, correlation heatmap, permutation importance) |
| `cleaned_train_sample.csv` | 5,000-row sample of cleaned + engineered train data (25 features + BPM) |
| `.gitignore` | Keeps `train.csv`/`test.csv` and full exports out of git |

Full cleaned files are **not** committed (they exceed GitHub limits). Run the notebook with `FULL_EXPORT = True` to regenerate them locally.

## Pipeline
1. **Load** train (524,164 × 11) and test (174,722 × 10).
2. **Quality:** 0 missing values, 0 duplicates; outliers <1.3% per column (skewed tails, not errors).
3. **Cleaning:** winsorise features at 0.1/99.9 percentiles (fit on train, applied to test; 4,171 of 4.7M values changed). Target untouched.
4. **Split:** 80/20 train/validation, `random_state=42`. `StandardScaler` fit on the train split only (used for Ridge; trees don't need it).
5. **EDA:** distributions, decile plots, correlations, permutation importance.

## Feature-engineering notes
16 new features: duration (min/log/quantile band), loudness band, interactions (Rhythm×Energy, Loudness×Energy, Mood×Energy, Live×Vocal), Vocal+Instrumental, Acoustic−Energy, Energy/Acoustic, Loudness per minute and mean/std/max/min across the seven 0–1 audio scores.

## Results (validation RMSE)
| Model | RMSE |
|---|---|
| Mean baseline | 26.4454 |
| Ridge, raw | 26.4439 |
| Ridge, engineered | 26.4439 |
| HistGB, raw | 26.4385 |
| HistGB, engineered | 26.4393 |

**Honest takeaway:** raw features have essentially no linear/monotonic relationship with BPM (all |Spearman| < 0.01), and the engineered features did not improve validation RMSE beyond noise (~0.01). `MoodScore`, `TrackDurationMs` and `RhythmScore` rank highest in permutation importance, but by tiny margins. For later tasks, expect gains to come from tuning/ensembling and testing more exotic interactions rather than simple aggregates.
