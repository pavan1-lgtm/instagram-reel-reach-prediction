# Instagram Reel Reach Prediction Before Posting Using Machine Learning

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pavan1-lgtm/instagram-reel-reach-prediction)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Data-Synthetic%20Benchmark-amber.svg)]()

> **Project Mandate:** Estimate an Instagram Reel's expected reach and empirical confidence range using **only features known prior to publication**.

---

## ⚡ Run Instantly in GitHub (1-Click in Browser)

You do **not** need to install Python or Git on your computer to run this project. You can run the complete web app directly in GitHub using **GitHub Codespaces**:

1. Click here: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pavan1-lgtm/instagram-reel-reach-prediction)
2. Wait ~30 seconds while GitHub creates your cloud development environment and automatically installs all dependencies.
3. In the terminal at the bottom of Codespaces, run:
   ```bash
   uvicorn app.main:app --port 8000
   ```
4. A popup will appear in the bottom right: **"Open in Browser"**. Click it to view and interact with the live glassmorphism application!

### 🌐 Localhost Quick Access
When running the application locally, click any link below to open:
- 🚀 **Live Web App:** [http://localhost:8000](http://localhost:8000) *(or [http://127.0.0.1:8000](http://127.0.0.1:8000))*
- 📖 **Interactive API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- 🔍 **Model Metadata Endpoint:** [http://localhost:8000/api/meta](http://localhost:8000/api/meta)

---

## 1. Problem Statement & Motivation

Predicting social media reach is typically done retrospectively using engagement signals (likes, shares, watch time). However, creators and digital marketing teams require decision support **before posting**:
- *Should this Reel be 22 seconds or 55 seconds?*
- *Is it worth scheduling for Sunday evening versus Tuesday morning?*
- *Does pairing with trending audio provide a measurable reach advantage?*

This project implements an end-to-end Machine Learning pipeline that predicts expected reach ($\hat{y}$) and calibrates an **empirical 80% confidence range** for a planned Reel based on account baseline volume and planned video metadata.

---

## 2. Non-Negotiable Methodological Principles

### A. Zero Data Leakage (Pre-Posting Features Only)
Under no circumstances does this model consume post-launch signals (likes, comments, saves, shares, retention rate, average percentage viewed). Those metrics are consequences of reach, not causes. Using post-launch telemetry to predict reach is a severe form of **target leakage**. Every feature fed to our model is fully known when the Reel sits in the creator's draft folder.

### B. Statistical Expectation vs. Meta Ranking Algorithm
This project is a statistical regression model estimating expected distribution from historical patterns. It **does NOT replicate Meta's proprietary recommendation algorithm**, graph neural networks, or millisecond auction infrastructure. All app text and documentation strictly use the phrasing **"associated with"**, never **"causes"**.

### C. Honest Synthetic Data Disclosure
Meta does not publish raw tabular creator reach logs. This demonstration uses a calibrated synthetic dataset of **4,015 Reels across 400 accounts** generated with realistic industry effect sizes and deliberate mess (missing values, impossible durations, duplicate entries, extreme tail leverage). Results demonstrate **engineering and statistical methodology**, not production Instagram behavior.

---

## 3. Pipeline Architecture

```text
[ Raw Pre-Posting Inputs ]
         │
         ├── Account Baselines: followers, avg_previous_reach, engagement_rate, account_age, posts_30d
         └── Planned Reel Specs: reel_duration, content_category, posting_day, posting_hour,
                                 caption_length, hashtag_count, has_trending_audio, is_original_content
         │
         ▼
[ Custom Feature Engineering ] (ReelFeatureEngineer)
         │
         ├── reach_to_follower_ratio = avg_previous_reach / max(followers, 1)
         ├── caption_density = caption_length / (hashtag_count + 1)   <- (+1 Laplace Smoothing)
         └── posting_time_category = Morning (5-11), Afternoon (12-16), Evening (17-21), Night (22-4)
         │
         ▼
[ Scikit-Learn ColumnTransformer ] (Fit Strictly on Training Split)
         │
         ├── Numeric Pipeline:     SimpleImputer(strategy='median') -> StandardScaler()
         └── Categorical Pipeline: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(drop='first')
         │
         ▼
[ Estimator / Regressor ]
         │
         ├── Primary Model: Ordinary Least Squares Linear Regression (Interpretable baseline)
         └── Comparison Models: Ridge Regression, Random Forest, Gradient Boosting
         │
         ▼
[ Non-Negative Empirical Range Engine ] (src/predict.py)
         │
         ├── point_estimate = max(0, round(raw_prediction))
         ├── range_low = max(0, round(point_estimate * 0.6489))      <- 10th percentile error ratio
         └── range_high = round(point_estimate * 1.3912)             <- 90th percentile error ratio
```

---

## 4. Empirical Evaluation & Benchmark Results

> **Terminology Rule:** Regression models do not have an "accuracy" percentage. We report Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and Coefficient of Determination (**$R^2$**).

### A. Model Comparison (Test Set & 5-Fold Cross-Validation)

| Model Architecture | Test MAE (People) | Test RMSE (People) | Test $R^2$ | 5-Fold CV MAE (Mean ± Std) | 5-Fold CV $R^2$ (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Historical Avg Reach)** | 4,402.7 | 9,999.4 | 0.7813 | *N/A* | *N/A* |
| **Linear Regression (Baseline OLS)** | 3,235.0 | 6,942.9 | **0.8946** | 3,409.7 ± 309.4 | 0.8619 ± 0.0442 |
| **Ridge Regression ($\alpha=10.0$)** | 3,225.8 | 6,954.6 | 0.8942 | 3,401.8 ± 311.1 | 0.8620 ± 0.0443 |
| **Random Forest (100 Trees)** | 3,243.9 | 7,637.0 | 0.8724 | 3,361.5 ± 218.4 | 0.8453 ± 0.0370 |
| **Gradient Boosting (120 Trees)** | **3,037.4** | 7,257.0 | 0.8848 | **3,287.0 ± 284.1** | 0.8481 ± 0.0392 |

### B. Cold-Start Account Robustness Audit (`GroupShuffleSplit`)
To verify whether the model generalizes to completely new creators without account identity leakage, we tested a split grouped by `account_id` (320 accounts in train, 80 in test, zero creator overlap):
- **Random 80/20 Row Split:** Test MAE = **3,235.0** | Test $R^2$ = **0.8946**
- **GroupShuffleSplit (Unseen Accounts):** Test MAE = **3,241.8** | Test $R^2$ = **0.9087**
- *Conclusion:* The model does not memorize creators; it accurately estimates reach for first-time cold-start accounts based on account baselines.

---

## 5. Frontend Design (Glassmorphism & Accessibility)

The user interface is built with **vanilla HTML5, CSS3, and modern JavaScript** (no external framework, no node build step, and **zero CDN dependencies**, guaranteeing offline availability during oral defense).

### What is Glassmorphism?
Glassmorphism is a modern UI design aesthetic simulating translucent frosted glass sitting above a vibrant, colorful surface. It creates physical depth, layering, and high aesthetic polish.

### Core CSS Properties That Create the Effect
1. **Multi-Layered Optical Blur:**
   ```css
   background: rgba(255, 255, 255, 0.08);
   backdrop-filter: blur(20px) saturate(160%);
   -webkit-backdrop-filter: blur(20px) saturate(160%);
   border: 1px solid rgba(255, 255, 255, 0.18);
   border-radius: 24px;
   box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25), 0 8px 32px rgba(0, 0, 0, 0.35);
   ```
2. **Chromatic Layering:** Glass only looks like glass when color moves behind it. Four GPU-accelerated blurred gradient orbs (violet `#7c3aed`, pink `#ec4899`, orange `#f97316`, cyan `#22d3ee`) drift continuously using transform-only animations (`translate3d`).
3. **No Nested Blur:** Inner form inputs use plain translucent backgrounds (`rgba(255, 255, 255, 0.05)`). `backdrop-filter` is applied exclusively to the top-level cards, preserving 60 FPS performance.

### Accessibility & Contrast Compliance (WCAG AA)
- All primary text (`#f8fafc`) and secondary text (`#cbd5e1`) exceed the WCAG AA minimum contrast ratio of 4.5:1 against the dark indigo background (`#0b0820`).
- **Fallbacks:**
  - `@supports not (backdrop-filter: blur(1px))` provides opaque tinted surfaces for older browsers.
  - `@media (prefers-reduced-transparency: reduce)` replaces glass with solid dark containers.
  - `@media (prefers-reduced-motion: reduce)` disables orb drift and number count-up animations.
- Fully operable via keyboard with high-contrast `:focus-visible` rings and `aria-live="polite"` result announcements.

### UI States Overview
- **Empty State:** Explanatory prompt inviting the creator to input characteristics or click *Load Example*.
- **Loading State:** Shimmer skeleton placeholders matching the exact card dimensions during API calculation.
- **Result State:** Large point estimate with smooth count-up animation, formatted text range (`expected range 8.6K – 18.4K`), percentage comparison vs historical average, empirical horizontal range visualizer bar, and actionable suggestion chips with distinct icons and tags.

---

## 6. Project Limitations

1. **Synthetic Data Constraint:** Effect sizes reflect documented social media benchmarks, not internal proprietary Meta weights.
2. **Unobservable Qualitative Signals:** Tabular metadata cannot score creative delivery, on-camera charisma, pacing in the first 2 seconds, or visual aesthetic quality.
3. **Heteroscedasticity at High Reach:** Residual variance expands on large viral accounts; estimates must always be interpreted as probabilistic ranges.
4. **Algorithmic Platform Drift:** Platform distribution shifts seasonally and algorithm updates occur without notice, requiring scheduled model recalibration.
5. **Correlation vs. Causation:** Identified feature uplifts represent empirical historical associations, not guaranteed algorithmic causes.

---

## 7. How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset (or inspect template)
```bash
python src/generate_data.py
```
*(To use real data, populate `data/real_reels_template.csv` and point the loader to it).*

### 3. Run Pipeline Training, Comparison, & Diagnostics
```bash
python -m src.train
```

### 4. Run Automated Unit & Integration Tests
```bash
pytest -v
```

### 5. Launch the Web Application
```bash
uvicorn app.main:app --reload
```
Open in your browser: **[http://localhost:8000](http://localhost:8000)** *(or [http://127.0.0.1:8000](http://127.0.0.1:8000))*

---

## 8. Manual Visual QA Checklist

- [x] **Glassmorphism Visual Check:** Blurred gradient orbs are visible and drifting slowly behind the translucent cards.
- [x] **WCAG AA Contrast Check:** All text is crisp and readable against both the dark background and the blurred glass cards.
- [x] **Responsive Layout (375px):** Resize browser to 375px width; verify that the two columns stack into a clean single column without horizontal overflow.
- [x] **Keyboard Navigation:** Tab through all form inputs; verify that bright cyan focus rings appear on every element.
- [x] **Dynamic Synthetic Badge:** Verify that the header shows *"Demo model trained on synthetic data"* loaded from `/api/meta`.
- [x] **Reduced-Motion Verification:** Enable reduced motion in OS settings; verify that orbs remain stationary and count-up animation snaps immediately to the final number.
