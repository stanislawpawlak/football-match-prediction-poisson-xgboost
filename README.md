# Football Match Outcome Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/XGBoost%20%2F%20cuML%20%2F%20statsmodels-GPU-green.svg)](#)
[![Optuna](https://img.shields.io/badge/Hyperparameter%20Tuning-Optuna-orange.svg)](https://optuna.org/)

An end-to-end machine learning and statistical modeling pipeline designed to forecast 3-way \\(1X2\\) football match outcomes (*Home Win*, *Draw*, *Away Win*) across **16 European leagues (2000–2025)** using over **134,000 matches**.

> **Academic Context:** Developed as part of the B.Sc. thesis *"Modelowanie wyników meczów piłkarskich z wykorzystaniem metod statystycznych i uczenia maszynowego"* at Warsaw University of Technology (Faculty of Mathematics and Information Science / MiNI PW)

---

## Key Highlights

- **Massive Scale & Strict Methodology:** Processed 230,000+ raw matches down to a clean, non-leaking dataset of **134,310 matches**. Uses a strict **chronological train-test split** (`shuffle=False`) to prevent data leakage.
- **Advanced Feature Engineering:** Dynamic team **Elo ratings**, relative Elo advantage differentials, short-term form momentum vectors (2 &middot; Form<sub>3</sub> - Form<sub>5</sub>), and `HashingEncoder` (32 components) for high-cardinality team identity compression.
- **Statistical vs. Machine Learning Comparison:** Benchmarks classical stochastics (**Bivariate Poisson Regression with Dixon-Coles adjustment**, **Multinomial Logistic Regression / MNLogit**) against ML algorithms (**XGBoost**, **cuML Random Forest**, **Decision Trees**).
- **Automated Bayesian Optimization:** Utilizes **Optuna** (50 trials per model) optimizing for **LogLoss minimization** on GPU infrastructure.
- **Comprehensive Probabilistic Evaluation:** Assessed across **Accuracy**, **LogLoss (Cross-Entropy)**, **Brier Score**, and **Permutation Feature Importance**.

---

##  Dataset & Preprocessing Pipeline

- **Raw Dataset:** 230,557 historical matches spanning 2000–2025 across 37 global divisions (sourced from the Club-Football-Match-Data repository).
- **Data Filtering & Leakage Elimination:**
  - Removed all post-match statistics (shots, fouls, corners, cards, half-time results, etc.) to prevent data leakage.
  - Excluded `MatchTime` (>50% missing values) and 21 niche leagues with >50% feature missingness (retaining 16 core European leagues including Premier League, La Liga, Serie A, Bundesliga and Ekstraklasa to maintain consistency with the topic).
- Handled Elo missingness via a 3-tier strategy: `forward fill` → expanding league historical mean → fallback base rating (1500).
- **Final Evaluation Split:**
  - **Train + Validation Set (90%):** ~120,879 matches (2000–2023) split 80:20 chronologically for Optuna tuning, followed by retraining the optimized models on this full 90%         combined set.
  - **Holdout Test Set (10%):** **13,431 matches** spanning **2023–2025** used for final model benchmarking.

---

## Feature Engineering & Pipeline Architecture

Built using a modular `scikit-learn` `ColumnTransformer` pipeline:

| Feature Group | Raw Variables | Derived / Engineered Features | Processing & Encoding |
| :--- | :--- | :--- | :--- |
| **Elo Metrics** | `HomeElo`, `AwayElo` | `EloDifference` (Home - Away), `EloAdvantage` (EloDiff / EloTotal), `EloTotal` | `SimpleImputer(median)` + `StandardScaler` |
| **Form Dynamics** | `Form3Home/Away`, `Form5Home/Away` | `Form3Difference`, `Form5Difference`, `FormMomentumDiff` ((2 &middot; Form<sub>3</sub> - Form<sub>5</sub>)<sub>Home</sub> - (2 &middot; Form<sub>3</sub> - Form<sub>5</sub>)<sub>Away</sub>) | `SimpleImputer(median)` + `StandardScaler` |
| **Team Identity** | `HomeTeam`, `AwayTeam` | High-cardinality categorical compression | `ce.HashingEncoder(n_components=32)` |
| **League Division** | `Division` | 16 European Leagues | `OneHotEncoder(handle_unknown='ignore')` |
| **Calendar / Date** | `MatchDate` | `DayOfWeek`, `Month`, `IsWeekend` | `StandardScaler` |

---

## Mathematical & Algorithmic Foundations

### 1. Bivariate Poisson Regression with Dixon-Coles Correction
Model goal scoring intensity λ (home) and μ (away) via log-link function:

```math
\ln(\lambda_t) = \beta_0 + \beta_1(1) + \beta_2 \text{EloDiff} + \beta_3 \text{EloTotal} + \beta_4 \text{Form5Diff} + \gamma_{\text{league}}
```

```math
\ln(\mu_t) = \beta_0 + \beta_1(0) - \beta_2 \text{EloDiff} + \beta_3 \text{EloTotal} - \beta_4 \text{Form5Diff} + \gamma_{\text{league}}
```

Adjusted joint probability density with Dixon-Coles low-score multiplier τ(x, y):

```math
P(X=x, Y=y) = \tau(x, y) \cdot \frac{\lambda^x e^{-\lambda}}{x!} \cdot \frac{\mu^y e^{-\mu}}{y!}
```

Estimated in two stages: MLE for parameters β, followed by bounded scalar optimization for correlation ρ = -0.06.

### 2. Extreme Gradient Boosting (XGBoost GPU)
2nd-order Taylor expansion of multi-class LogLoss objective function. Optimal leaf weight for leaf j and class c ∈ {0, 1, 2}:

```math
w^*_{jmc} = -\frac{\sum_{i \in I_{jmc}} g_{imc}}{\sum_{i \in I_{jmc}} h_{imc} + \lambda}
```

Where g<sub>imc</sub> = p<sub>ic</sub> - y<sub>ic</sub> (prediction error) and h<sub>imc</sub> = 2 p<sub>ic</sub>(1 - p<sub>ic</sub>) (diagonal upper bound of Hessian).

##  Benchmark Results & Model Comparison

Evaluated on **13,431 holdout test matches (2023–2025)**:

| Model | Accuracy (%) | LogLoss (Lower is Better) | Brier Score (Lower is Better) | Key Tuning Parameters |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost (GPU CUDA)** | 50.44% | **1.0003** | **0.5979** | `max_depth=4`, `lr=0.113`, `subsample=0.71`, `colsample=0.86`, `gamma=2.57` |
| **Random Forest (cuML)** | 50.45% | **1.0006** | **0.5981** | `n_estimators=152`, `max_depth=6`, `max_features=0.55`, `max_samples=0.76` |
| **Poisson (Dixon-Coles)** | 50.45% | **1.0016** | **0.5987** | $\hat{\rho} = -0.06$, 2-Stage MLE |
| **Decision Tree** | 50.35% | **1.0016** | **0.5987** | `max_depth=4`, `min_samples_split=158`, `min_samples_leaf=51` |
| **Logistic Reg (cuML)** | **50.58%** | **1.0015** | **0.5987** | `penalty=elasticnet`, $C=0.011$, `l1_ratio=0.60` |
| **MNLogit (statsmodels)** | **50.55%** | **1.0021** | **0.5990** | Unregularized Multiclass Logistic Regression ($p$-values computed)|
| **Pure Elo Baseline** | 50.48% | 1.0021 | 0.5990 | Single Home/Away Elo Logistic Regression |
| **Prior Baseline (Prior)** | 44.17% | 1.0730 | 0.6489 | Empirical class distribution predictor |
| **Uniform Random (33/33/33)** | 33.61% | 1.0986 | 0.6667 | $\frac{1}{3}$ constant probability |
| **Always Home Win** | 44.17% | 20.1218 | 1.1165 | Naive hard classification |

---

## Key Findings & Insights

1. **Elo Rating Dominance:** Permutation Feature Importance proves that `EloDifference` and `EloAdvantage` account for most of total pedictive power across all ML algorithms.
2. **Probability Calibration:** Non-linear ML models (XGBoost & Random Forest) achieve the lowest LogLoss and Brier Score, offering superior probability calibration despite similar accuracy to linear models for this dataset.
3. **The Draw Prediction Bottleneck:** Hard classification ($\text{argmax}$) struggles to predict draws ($X$), as draw probabilities rarely exceed the home win threshold—underlining the necessity of soft probabilistic predictions for sports betting value identification.
4. **League Predictability Disparity:**
   - **Highly Predictable Leagues:** Dutch Eredivisie (`N1`) & Portuguese Primeira Liga (`P1`) achieve **LogLoss ~0.918** and **Accuracy ~57-58%**, likely driven by clear team quality stratification.
   - **Volatile / Stochastic Leagues:** Italian Serie B (`I2`) & Spanish Segunda División (`SP2`) exhibit **LogLoss ~1.065** and **Accuracy ~42%**, potentially due to tight parity and draw frequency.
  
## Data Source & Acknowledgements

The historical match data utilized in this project was sourced from the [Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data) repository. Per the author's request, the original dataset is cited as follows:

> **Gábor, A. (2026).** *Club Football Match Data*. Retrieved from https://github.com/xgabora/Club-Football-Match-Data/
> 
> **Author:** Adam Gábor | **ORCID:** [0009-0001-9252-5976](https://orcid.org/0009-0001-9252-5976) | **Affiliation:** Slovak University of Technology in Bratislava
