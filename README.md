# TikTok Video Engagement Prediction — Day-30 View Count Forecast

**WeCloudData DS Bootcamp — In-Class Kaggle Competition**

Predicting the cumulative view count a TikTok video will reach at **Day 30**, using only video metadata, creator statistics, and daily engagement metrics observed during the first five days after posting (Days 0–5).

---

## 🎯 Problem Statement

- **Target:** `target_day30_views` — cumulative views at day 30 (raw scale)
- **Evaluation metric:** Root Mean Squared Error (RMSE) on the **raw** view-count scale
- **Constraint (no leakage):** predictions must rely solely on historical engagement from Day 0 to Day 5
- **Challenge:** the target follows a heavy **power-law distribution** — a small number of viral videos dominate the squared error, so wild over-predictions are heavily penalized

**Dataset source:** [huggingface.co/datasets/lingbow/tiktok-video-engagement-200k](https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k) (sampled subset used for this competition, text features removed)

---

## 🔍 Approach Summary

### 1. Data Exploration
- Verified engagement-table coverage: **Day 0 is missing for ~62% of videos**, while Days 1–5 are available for ~98%. Decision: build engagement features only from **Days 1–5**, using each video's **last available day**.
- Confirmed the target's power-law shape (raw histogram dominated by near-zero values, `log1p` transform normalizes it). The **top 10 videos alone account for a large share of total squared views** — directly motivating repeated CV and stable ensembling over a single aggressive model.
- Confirmed the target is **cumulative**: day-30 views are (almost) always ≥ last observed views (99.99% of training rows), with a typical growth multiple of ~1.16–1.6x from day 5 to day 30. This motivated two techniques used later: **prediction clipping** and a **growth-target** formulation.
- Checked creator overlap between train/test: **97.5%** of test videos belong to creators already seen in training → random K-Fold chosen as the primary validation scheme, with GroupKFold (by creator) as a stricter robustness check.

### 2. Data Cleaning
- No duplicate `video_id`s.
- Missing categorical values (`ratio`, `desc_language`, `music_selected_from`, `topic`) filled with `'unknown'`.
- Missing numeric values handled **inside** each model's pipeline (LightGBM/CatBoost read NaN natively; Random Forest/Ridge impute within the pipeline) to avoid leaking information from validation folds.

### 3. Feature Engineering
Built exclusively from Days 1–5 (no Day-0 dependency):
- **Last-available engagement:** views, likes, comments, shares, collects, downloads (and WhatsApp shares)
- **Snapshots & trends:** Day-1/Day-3/last-day values, daily increments, growth ratios (Day1→5, Day4→5, Day1→3), acceleration ratios
- **Interaction ratios:** like/comment/share/collect/download-to-view ratios
- **Creator features:** followers, following, total favorites, video count — merged with `merge_asof` on the video's creation date to avoid future leakage
- **Video metadata:** duration, aspect ratio, language, AI-generated flag, ad flag, hashtags, emojis, sentiment scores, post hour/day-of-week/month

### 4. Target Transformation & Validation
Two target formulations were compared:
1. `log1p(day30_views)` — standard log target
2. `log1p(day30_views) - log1p(last_known_views)` — **growth target**, letting the model focus on "how much more will this grow?"

Evaluated with **repeated 5-fold CV** (2 seeds), reporting three metrics per model:
- **Raw RMSE** (competition metric, primary selection criterion)
- **Log RMSE** (diagnostic, less outlier-driven)
- **Raw RMSE excluding top-10 viral training videos** (typical-video performance)

A GroupKFold-by-creator split was run as a secondary robustness check.

### 5. Models Compared
| Model | Notes |
|---|---|
| LightGBM | Gradient boosting, tuned learning rate + early stopping |
| Random Forest | Warm-start tree growth curve to pick tree count |
| Ridge Regression | Linear baseline, log-transformed numeric features, one-hot categoricals |
| CatBoost | Native categorical handling, early stopping |

### 6. Ensembling
Equal-weight blends of out-of-fold predictions (deliberately **not** learned weights, to avoid overfitting the blend to a handful of viral outliers).

### 7. Post-processing
- **Clipping:** every prediction is floored at the video's last observed view count (cumulative views cannot decrease). Verified experimentally: clipping cut LightGBM's raw RMSE from **151,342 → 82,473** (≈45% improvement).
- **Final sanity checks:** no negative predictions, no prediction below last known views, submission format matches `sample_submission.csv` exactly (row order and `video_id` values).

---

## 📊 Results (Repeated 5-Fold CV, Raw RMSE)

| Model | Raw RMSE | Raw RMSE (SD) | Log RMSE |
|---|---:|---:|---:|
| **Best Blend (LGB + RF + Ridge + CatBoost)** | **58,481** | 1,662 | 0.2885 |
| Blend (LGB + RF + Ridge) | 61,569 | 2,333 | 0.2885 |
| Random Forest (log target) | 74,140 | 563 | 0.2971 |
| Ridge (log target) | 77,629 | 274 | 0.3032 |
| CatBoost (growth target) | 80,686 | 1,998 | 0.2995 |
| CatBoost (log target) | 80,885 | 3,804 | 0.2986 |
| LightGBM (log target) | 82,473 | 1,517 | 0.2947 |
| Random Forest (growth target) | 97,739 | 4,079 | 0.3122 |
| LightGBM (growth target) | 130,557 | 11,763 | 0.3046 |

**The final ensemble is selected automatically from this table by lowest raw RMEnter** — no manual model choice.

### GroupKFold Robustness Check
| Split | Log RMSE | Raw RMSE |
|---|---:|---:|
| Random 5-fold (main) | 0.2947 | 82,473 |
| GroupKFold by creator | 0.3085 | 81,141 |

Performance is close between the two, confirming the model isn't overly dependent on creator overlap between train and test.

---

## 🔎 Error Analysis Highlights

- The best blend is **well-calibrated on typical videos**: mean absolute % error stays around **10–14%** across every content topic.
- It **systematically under-predicts the largest viral outliers** — 9 of the top 10 largest absolute errors are under-predictions, including the single largest training video (16.3M actual vs. ~1.5M predicted early in development, before final blending). This is an expected consequence of log-target models being pulled toward the bulk of the data — exactly where raw RMSE is most sensitive.
- Feature importance confirms early engagement level (`play_count_last`, `play_count_d3`) dominates predictive signal, as expected for this problem.

---

## ✅ Competition Compliance

- **Tabular modeling only** — no scraping, LLMs, or multimodal embeddings used.
- **No data leakage** — all engagement features derive strictly from Days 0–5; creator features are merged as of the video's creation date.
- **Reproducible** — notebook runs end-to-end from raw CSVs to `submission.csv` with fixed random seeds (`SEED = 42`).

---
## Author

Prepared as part of the WeCloudData DS Bootcamp in-class competition (TikTok Video Engagement Prediction). 

## Analyzed by Abeer Alshahrani.
