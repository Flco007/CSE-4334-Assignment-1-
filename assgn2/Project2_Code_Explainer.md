# Project 2 Code Explainer and Demo Q&A

**Course:** CSE 4334/5334 Data Mining, Fall 2026, Project 2: Classification Models & Imbalanced Evaluation
**Team:** Joseph Fitchet, Rodrigo Pleitez
**Covers:** the submitted `final_submission.ipynb` (every section, both people's code) and `assignment2_report.docx` (Challenges A, B, C and the AI audit)

This replaces the earlier Person A explainer. Use it to prepare for the live demo:

* **Sections 1–3** explain what the notebook does and why.
* **Section 4** summarizes the report answers.
* **Section 5** is the question bank. Each question has follow-ups, because graders usually ask "why?" at least twice.
* **Section 6** lists the weak spots a grader might poke, with honest answers.

All numbers below are copied from the outputs saved in the submitted notebook unless marked otherwise.

---

## 1. The 60-second version

> We predict whether someone earns more than $50K from the 1994 census (UCI Adult, 32,561 people, 24.1% positive). Every model is a scikit-learn **Pipeline**: the preprocessing is fitted **inside each cross-validation fold**, so the test fold never influences the imputers, scaler, encoder or SMOTE. That's what "zero data leakage" means here.
>
> All models are scored by one function, `evaluate()`, which runs the same 5-fold stratified CV and reports accuracy, precision, recall, F1 and ROC-AUC. We tuned a decision tree and k-NN with `GridSearchCV`, picking the winner by **F1** rather than accuracy, because a model that predicts "≤50K" for everyone already gets 75.9% accuracy. We also built a RIPPER rule learner and wrapped it so it works like any scikit-learn model.
>
> For the imbalance, we put SMOTE inside an `imblearn` pipeline. Then we swept the decision threshold from 0 to 1 under the costs $100 per missed high earner and $10 per wasted contact. The cheapest threshold was **0.12**, close to the theoretical **0.091**.

---

## 2. Results at a glance

### 2.1 Model comparison (5-fold stratified CV, threshold 0.5, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (always ≤50K) | 0.759 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| Decision tree, default | 0.816 ± 0.005 | 0.618 ± 0.012 | 0.625 ± 0.018 | 0.621 ± 0.010 | 0.751 ± 0.007 |
| **Decision tree, tuned** | **0.855** ± 0.007 | **0.721** ± 0.019 | **0.652** ± 0.022 | **0.685** ± 0.015 | **0.895** ± 0.006 |
| k-NN, tuned | 0.843 ± 0.006 | 0.701 ± 0.015 | 0.605 ± 0.013 | 0.650 ± 0.013 | 0.895 ± 0.007 |
| RIPPER | 0.822 ± 0.002 | 0.718 ± 0.025 | 0.433 ± 0.026 | 0.539 ± 0.015 | 0.693 ± 0.010 |

### 2.2 Tuning results

| Search | Best settings | Best mean CV F1 |
|---|---|---|
| Decision tree (180 candidates × 5 folds = 900 fits) | entropy, `max_depth=14`, `min_samples_split=50` | 0.6847 |
| k-NN (20 candidates × 5 folds) | euclidean, `n_neighbors=31`, `weights="distance"` | 0.6497 |
| Decision tree + SMOTE | picked as the better SMOTE model | 0.6761 |
| AI's pipeline, debugged (80/20 split) | entropy, depth 10, leaf 4, split 10 | 0.668 CV; F1 0.67 and recall 0.85 on its test split |

### 2.3 Cost analysis (decision tree + SMOTE, out-of-fold probabilities)

| Threshold | FN | FP | Total cost |
|---|---|---|---|
| Predict everyone ≤50K | 7,841 | 0 | $784,100 |
| Predict everyone >50K | 0 | 24,720 | $247,200 |
| **Best found: t = 0.12** | **424** | **9,410** | **$136,500** (about $4.19 per person) |
| Theory says | — | — | t = 10 / (10 + 100) = **0.091** |

At t = 0.12 the model finds 94.6% of high earners (recall 0.946) at a precision of 0.44.

---

## 3. The notebook, section by section

### 3.1 Setup cells (cells 0–2)

| Cell | What it does | Why |
|---|---|---|
| `%pip install wittgenstein` | Installs the RIPPER library into the running kernel | scikit-learn has no rule learner |
| Imports | Loads pandas, matplotlib, scikit-learn and the tree and dummy models | — |
| Constants | `RANDOM_STATE = 21`, `CV = StratifiedKFold(5, shuffle=True, random_state=21)`, `SCORING` (5 metrics), `COST_FN, COST_FP = 100, 10`, column lists | One place defines the folds, metrics, costs and seed. Every model uses the same ones, so results are comparable. |

**The `CV` object matters most.** It's created once and passed everywhere: to `evaluate()`, to every `GridSearchCV`, and to `cross_val_predict`. Every model therefore sees the exact same 5 train/test splits.

### 3.2 Task 1: `load_data()` and `make_preprocessor()` (cell 4)

**`load_data()` does *stateless* cleaning.** Stateless means it learns nothing from the data, so it's safe to run once on all rows:

| Argument / line | Problem it solves |
|---|---|
| `names=COLUMNS, header=None` | `adult.data` has no header row. Without this, the first person becomes the column names. |
| `skipinitialspace=True` | Every value after a comma starts with a space (`" Private"`, `" ?"`) |
| `na_values="?"` | Turns `?` into a real NaN so the imputer can see it. Without it, `?` becomes its own category. |
| `(df["income"].str.rstrip(".") == ">50K").astype(int)` | Target: 1 for >50K, 0 for ≤50K. `rstrip(".")` also handles `adult.test`, whose labels end in a period. |
| `X = df.drop(columns=["income"] + DROP_COLS)` | Features only. `DROP_COLS` is empty, so all 14 attributes are kept, matching the spec. |

**`make_preprocessor()` returns an *unfitted* recipe.** Its parts are *stateful*: they learn medians, means, standard deviations and category lists. So they must be fitted inside each fold, never on all the data:

```
ColumnTransformer
├── "num": median imputer → StandardScaler      (6 numeric columns)
└── "cat": most-frequent imputer → OneHotEncoder(handle_unknown="ignore")   (8 categorical → 99 columns)
```

* **Output shape:** (32,561, 105), which is 6 numeric plus 99 one-hot columns.
* **Category counts:** workclass 8, education 16, marital-status 7, occupation 14, relationship 6, race 5, sex 2, native-country 41.
* **Missing values found:** workclass 1,836, occupation 1,843, native-country 583.
* **Structure** follows scikit-learn's "Column Transformer with Mixed Types" example.
* **The shape check** (`fit_transform(X).shape`) is inspection only. It never feeds a model.

### 3.3 Task 2: the `evaluate()` harness (cell 6)

This is the one function every reported number comes from:

```
for each of the 5 folds in CV.split(X, y):
    model = clone(pipeline)                 # fresh, unfitted copy
    model.fit(train rows)                   # fits imputers, scaler, encoder, SMOTE, model: training rows only
    proba = model.predict_proba(test rows)  # P(>50K) for rows it never saw
    pred  = proba > 0.5
    record accuracy, precision, recall, F1 (from pred) and ROC-AUC (from proba)
    save proba into oof_proba[test rows]
return mean ± std per metric, and oof_proba
```

* **`clone()`** makes sure fold 2 doesn't start from fold 1's fitted model.
* **`oof_proba`** is the "out-of-fold" probability. After the 5 folds, every one of the 32,561 rows has exactly one probability, produced by a model that never trained on it. It's what you'd use for ROC/PR curves and the cost sweep.
* **`zero_division=0`** stops the dummy model from crashing precision. It never predicts positive, so precision would be 0/0.
* **`np.std`** gives the spread across the 5 folds, which is what the ± values show.

### 3.4 Baselines (cells 17–19)

* **`results = {}`** collects every model's output. `show()` prints mean ± std.
* **Dummy (majority class):** 75.9% accuracy, with 0 precision, recall and F1. This is the **accuracy paradox** from Challenge A, in our own numbers.
* **Default decision tree:** no depth limit, so it grows until every leaf is pure. F1 0.621, ROC-AUC 0.751. That low AUC comes from pure leaves giving only probabilities of 0 or 1, so the ranking is crude.

### 3.5 Task 3a: decision tree tuning (cells 21–25)

* **Grid:**
  * `max_depth` 3–20, as the spec requires
  * `min_samples_split` ∈ {2, 10, 50, 100, 200}. The spec gives no values; these run from the default to about 1% of a training fold.
  * `criterion` gini or entropy
* **`GridSearchCV` settings:**
  * `scoring=SCORING` records all 5 metrics for every candidate.
  * `refit="f1"` picks the winner by F1.
  * `cv=CV` uses the same folds as `evaluate()`.
  * `return_train_score=True` records training scores, which the validation curve needs.
  * `n_jobs=-1` uses all CPU cores.
* **The `clf__` prefix** is how a `Pipeline` exposes the parameters of its step named `clf`.
* **Top-10 table:** the top 8 candidates are within 0.004 F1 of each other. That's less than one standard deviation (about 0.015), so the exact winner is partly luck of the folds.
* **Validation curve** (F1 vs. depth, at `min_samples_split=50`):
  * Solid lines are test folds, with ±1 std shading. Dashed lines are training folds.
  * Depth 3–8: both scores are low and close together, which is underfitting.
  * Depth 13–14: test F1 peaks at about 0.68.
  * Beyond that, training F1 keeps climbing to about 0.75 while test F1 drifts down. That widening gap is overfitting.
  * Gini and entropy overlap almost everywhere, so depth matters and criterion barely does.
* **Tuned tree through `evaluate()`:** F1 **0.685**, ROC-AUC **0.895**, up from 0.621 and 0.751.

### 3.6 Task 3b: k-NN tuning (cells 27–33)

* **Pipeline:** the same `make_preprocessor()`, then `KNeighborsClassifier`. Its step is named `"model"`, so the grid keys are `model__...`.
* **Grid:** `n_neighbors` ∈ {1, 3, 7, 15, 31}, `weights` uniform or distance, `metric` euclidean or manhattan. That's 20 candidates, exactly the spec's grid.
* **Best:** euclidean, k = 31, distance weighting, with CV F1 0.650.
* **The k-NN validation curve** has the same layout as the tree's, with x = number of neighbors.
* **Why k-NN needs scaling:** distance is computed across all 105 columns. Without `StandardScaler`, capital-gain (std about 7,385, max 99,999) would swamp age (std about 13.6). That's Challenge B.

### 3.7 Rule-based model: RIPPER (cells 11, 34–36)

**`RipperClassifier` (cell 11)** wraps `wittgenstein.RIPPER` so it behaves like a scikit-learn model. It follows scikit-learn's estimator template:

| Piece | Rule it follows |
|---|---|
| `class RipperClassifier(ClassifierMixin, BaseEstimator)` | Mixin on the left. Reversed, `is_classifier()` silently fails. |
| `__init__` only stores the hyperparameters | `clone()` and `GridSearchCV` rebuild the object from them |
| `fit()` builds a new `lw.RIPPER(...)`, fits with `pos_class=1`, returns `self` | wittgenstein's own `fit` returns `None`, which breaks pipelines |
| `classes_ = [0, 1]`; `predict_proba` returns `[P(≤50K), P(>50K)]` | Column order must match `classes_` for ROC-AUC |
| `_to_frame()` | wittgenstein needs a DataFrame, and only bins columns that are numeric |
| `@property ruleset_` | Read-only view of the learned rules |

**`make_rule_preprocessor()` imputes only.** It uses the same median and most-frequent imputers as Task 1, but no scaling and no one-hot encoding:
* RIPPER cuts numeric columns into 10 bins itself, so scaling changes nothing.
* It tests categories directly (`relationship=Husband`), so one-hot would only add 99 binary columns.
* On fold 1, imputation-only beat the full preprocessor: F1 0.556 vs. 0.523, AUC 0.704 vs. 0.683.

**Results:** precision 0.718, which matches the tuned tree, but recall only 0.433, for F1 0.539. Anyone no rule covers is predicted ≤50K.

**Rules learned** (from our earlier run; the rules cell has no saved output in the submitted notebook): 53 rules. Nearly all start with `marital-status=Married-civ-spouse`, then add education level, an executive/managerial occupation, hours worked, or a capital-gain band.

### 3.8 Task 4: SMOTE (cells 41–44)

* **Two `imblearn` pipelines:** preprocessor → `SMOTE(random_state=21)` → model.
  * One for a decision tree, with `max_depth` 3–20, `min_samples_split` {2, 5, 10, 20}, and both criteria.
  * One for k-NN, with the same k-NN grid.
* **Why the `imblearn` Pipeline:** its samplers only run during `fit`, so SMOTE resamples each training fold and is skipped when predicting on the test fold. A regular scikit-learn `Pipeline` can't hold a sampler.
* **What SMOTE does in one fold:** a training fold has about 6,273 positives and 19,776 negatives. For each positive row, SMOTE picks one of its 5 nearest positive neighbors and creates a new row at a random point on the line between them. That adds about 13,500 synthetic positives, until the classes are 50/50.
* **Result:** the decision tree with SMOTE won, CV F1 0.676, versus 0.685 without SMOTE. SMOTE didn't raise F1 at the 0.5 threshold. What it does is shift the trade-off: more recall, less precision.

### 3.9 Task 4: the cost sweep (cells 9, 45–47)

* **`best_search`** is whichever SMOTE grid search had the higher F1, which was the decision tree.
* **`cross_val_predict(best_model, X, y, cv=CV, method="predict_proba")`** gives out-of-fold probabilities. It refits the whole SMOTE pipeline in each fold, so every probability comes from a model that didn't train on that row.
* **`cost_sweep(y, oof_probabilities)`** tries t = 0.00, 0.01, …, 1.00. For each threshold it predicts positive when P ≥ t, builds the confusion matrix, and computes cost = 100·FN + 10·FP. It also records precision, recall and F1.
* **`plot_metrics(results)`** (actually a cost plot, see Section 6) prints the cheapest row and plots total cost against threshold. The minimum is at **t = 0.12, $136,500**.

### 3.10 AI audit: the debugged pipeline (cell 48)

* **The cell:** it's the AI's code exactly, except the loading lines are replaced with `X, y = load_data()`.
* **Audit verdict:** SMOTE was placed **correctly**, inside `imblearn.pipeline.Pipeline` between the `ColumnTransformer` and the tree. It is not applied globally before CV, so there's no leakage.
* **Bug 1, loading:** a plain `read_csv("data.csv")` can't load Adult. There's no header row, and `?` isn't recognized as missing.
* **Bug 2, scoring:** `scoring="f1"` crashes on the text labels '>50K' / '≤50K' with `ValueError: pos_label=1 is not a valid label`.
* **Debugged result:** CV F1 0.668. On the AI's 80/20 test split, F1 is 0.67 and recall 0.85.

---

## 4. Report answers, summarized

### Challenge A: accuracy paradox and the 0.091 threshold

* **Paradox:** predicting ≤50K for everyone gets 75.9% accuracy but recall 0. It never finds a single high earner, which is the whole point of targeted marketing.
* **Derivation:**
  * Predicting + costs (1 − p)·C(−,+) in expectation, because you're wrong with probability 1 − p.
  * Predicting − costs p·C(+,−).
  * Predict + when (1 − p)·10 < p·100. That gives p > 10/110 = **0.091**.
* **Why it moves from 0.5:** 0.5 is only optimal when both errors cost the same. Here a miss costs 10 times more than a wasted contact, so you should contact anyone with even a 9.1% chance.

### Challenge B: scale and sparsity in k-NN

* **Scale:** Euclidean distance adds squared differences. A $10,000 capital-gain gap contributes 10⁸, while a 40-year age gap contributes 1,600. So capital-gain decides who the neighbors are.
* **Voronoi cells:** for 1-NN, each training point owns the region of space closer to it than to any other point. With unscaled features, those cell boundaries line up almost entirely with the capital-gain axis.
* **Sparsity:** one-hot encoding turns native-country into 41 columns that are mostly 0. In high dimensions, distances between random points become nearly equal (distance concentration), so "nearest" stops meaning much.

### Challenge C: tree splitting vs. FOIL gain, and rarest-first

* **Trees:** a tree picks the split that makes its child groups purest. Gini and entropy are concave, so almost any split counts as an improvement. They're symmetric, so a tree spends most of its effort on the 76% majority.
* **RIPPER:** RIPPER grows one rule at a time with FOIL gain, which only rewards making the rule more precise about >50K while still covering many positives. Uncovered rows are left for the next rule.
* **Rarest-first:** RIPPER writes rules only for the rare class, tests each rule on held-out data, and defaults to ≤50K.
* **The evidence:**
  * The unpruned tree reached depth 51 with 4,682 leaves.
  * 2,034 of its 2,247 >50K leaves (91%) held five or fewer people.
  * RIPPER's precision was 0.718, against the unpruned tree's 0.618.
  * The cost is recall: 0.433 against 0.625.

---

## 5. Demo question bank

Each question has follow-ups (↳). Practice saying the answers out loud.

### 5.1 Pipeline and leakage

**Q1. What is data leakage, and how did you prevent it?**
Leakage is when information from the test data influences training, which makes scores look better than they'd be on new data. Every preprocessing step that learns from data sits inside a `Pipeline`. The pipeline is fitted fresh on each training fold, so the test fold never affects the medians, scaling, categories or SMOTE.

* ↳ **Which steps learn from the data?** The median imputer learns medians, the most-frequent imputer learns modes, `StandardScaler` learns means and standard deviations, `OneHotEncoder` learns the category list, and SMOTE learns minority neighbors. Turning `?` into NaN or naming columns learns nothing, so that happens once in `load_data()`.
  * ↳↳ **What would actually go wrong if you scaled before CV?** Each test fold would be scaled using a mean and std that include its own rows. The model would get a small peek at the test data. For scaling the effect is small. For SMOTE it's large (Q22).
* ↳ **Why `clone()` inside `evaluate()`?** So each fold starts from a fresh, unfitted copy. Without it, fold 2 would start from a model already fitted on fold 1's data.
  * ↳↳ **Doesn't `GridSearchCV` already do that?** Yes. `GridSearchCV` and `cross_val_score` clone internally. `evaluate()` is our own loop, so it has to clone explicitly.

**Q2. Why a `ColumnTransformer` and not just one `Pipeline`?**
A `Pipeline` runs steps in sequence on all columns. A `ColumnTransformer` runs different pipelines side by side on different columns, numeric and categorical, then concatenates the results.

* ↳ **What happens to a column that isn't listed?** By default (`remainder="drop"`), it's dropped. All 14 of ours are listed.
  * ↳↳ **How would you drop `fnlwgt`?** Add it to `DROP_COLS`. `load_data()` drops it, and `make_preprocessor()` filters it out of the column lists automatically.

**Q3. Why are the parameters named `clf__max_depth`?**
A Pipeline exposes each step's parameters as `<step name>__<parameter>`. Our tree step is named `clf`, so `GridSearchCV` uses `clf__max_depth`.

* ↳ **Why a double underscore?** Parameter names already contain single underscores (`max_depth`), so a double one makes the split unambiguous.
  * ↳↳ **How would you tune the imputer inside the preprocessor?** `prep__num__imputer__strategy`, which reads as step `prep` → transformer `num` → step `imputer` → parameter `strategy`.

### 5.2 Task 1 preprocessing

**Q4. Why median imputation for numbers?**
The spec requires it, and it's robust to outliers. Capital-gain is 0 for 91.7% of people but goes up to 99,999, so a mean would be pulled toward the extremes.

* ↳ **Do any numeric columns actually have missing values?** No. All the missing values are in three categorical columns: workclass, occupation and native-country. The numeric imputer is a safeguard.
  * ↳↳ **Then why keep it?** It's part of the spec's master `ColumnTransformer`. It also protects the pipeline if new data arrives with missing numbers.

**Q5. Why most-frequent imputation for categories?**
It's the spec's choice and the simplest valid fill: the most common value for that column, learned from the training fold.

* ↳ **Is that a good assumption for workclass and occupation?** Not perfect. Those two are often missing together, and that pattern itself may carry signal. `SimpleImputer(add_indicator=True)` would add a "was missing" column to capture it.
  * ↳↳ **Why didn't you?** The spec defines the preprocessing exactly, and changing it would break comparability with the required design. It's a reasonable improvement to mention.

**Q6. What does `handle_unknown="ignore"` do, and why does it matter?**
If a test fold has a category the encoder never saw in training, it gets all zeros instead of crashing.

* ↳ **Can that happen here?** Yes. `Holand-Netherlands` appears in exactly **one** row. In whichever fold that row lands in the test set, the training data has never seen it.
  * ↳↳ **Isn't all-zeros wrong information?** It's treated as "none of the known countries," which is the most neutral option. With one row out of 32,561, the effect on results is negligible.

**Q7. Why 105 columns?**
Six numeric columns plus 99 one-hot columns, from 8 + 16 + 7 + 14 + 6 + 5 + 2 + 41 categories.

* ↳ **Why one-hot instead of numbering the categories 0, 1, 2…?** Numbering implies an order and distances that don't exist, for example that "Sales" is between "Tech" and "Admin". One-hot gives each category its own independent column.

### 5.3 Cross-validation and metrics

**Q8. Why `StratifiedKFold`?**
Each fold keeps the 24/76 class ratio. With plain KFold, a fold could end up with noticeably fewer positives, which makes precision and recall noisy.

* ↳ **Why `shuffle=True`?** The file might be ordered somehow. Shuffling, with a fixed seed, mixes rows randomly but reproducibly.
  * ↳↳ **Why seed 21?** Any fixed number works. What matters is that one seed is used everywhere, so every model sees the same folds and results are reproducible.

**Q9. Why five metrics, and which matters most?**
Accuracy is required but misleading here. Precision asks how often a ">50K" prediction is right. Recall asks how many real high earners were found. F1 balances the two. ROC-AUC measures ranking quality across all thresholds. For this problem, **F1 and ROC-AUC** matter most, and for the business question, **total cost**.

* ↳ **Why did the tuned tree and k-NN get the same AUC (0.895) but different F1?** AUC measures how well a model ranks people, regardless of threshold. F1 depends on the 0.5 cut-off. The two models rank people about equally well, but the tree's probabilities happen to sit better around 0.5.
  * ↳↳ **Why is the default tree's AUC so low (0.751)?** An unlimited tree has pure leaves, so almost every probability is exactly 0 or 1. With only two distinct values there's barely any ranking, and AUC suffers.

**Q10. What are "out-of-fold" probabilities?**
After 5-fold CV, every row has been in the test fold exactly once. Its out-of-fold probability is the one predicted by the model that didn't train on it. That gives one honest probability per row for the whole dataset.

* ↳ **What are they used for?** ROC/PR curves and the cost sweep, which need a probability for every row.

### 5.4 Decision tree

**Q11. Why tune on F1 instead of accuracy?**
The dummy model already gets 75.9% accuracy. Tuning on accuracy rewards models that lean toward predicting ≤50K, which is the accuracy paradox.

* ↳ **Why not ROC-AUC?** That was the alternative. AUC would have picked gini, depth 12, split 200 (AUC 0.907, F1 0.667). F1 was chosen because the model is used at a fixed threshold.
  * ↳↳ **Does the choice matter much?** Not much. The top candidates under either metric are close, and both beat the default tree by a wide margin.

**Q12. What do `max_depth` and `min_samples_split` control?**
`max_depth` caps how many questions the tree can ask in a row. `min_samples_split` says a node must hold at least that many rows before it can split. Both stop the tree from memorizing small groups, which is pre-pruning.

* ↳ **Why did depth 14 win and not 20?** Past about depth 14, the tree starts fitting noise. Training F1 keeps rising, but test F1 drifts down, as the validation curve shows.
  * ↳↳ **What's `min_samples_leaf`, and how is it different?** It's the minimum number of rows each *leaf* must keep after a split, rather than the minimum needed to *attempt* one. It's often a smoother control. The spec only asked for `min_samples_split`.

**Q13. Why did entropy win over gini?**
It edged ahead by a hair. The top 8 candidates are within 0.004 F1, and gini first appears at rank 9. The bands overlap almost everywhere on the plot, so the honest answer is that the criterion barely matters here.

* ↳ **How do gini and entropy differ mathematically?** Both are 0 for a pure node and largest at a 50/50 mix. Entropy uses −Σ p log₂ p and Gini uses 1 − Σ p². Entropy penalizes mixed nodes slightly more. In practice they pick the same split most of the time.

**Q14. Is your tuned score (0.685) a fair estimate of performance on new data?**
It's slightly optimistic. The same folds chose the parameters and scored the result, which is model-selection bias (Cawley & Talbot, 2010).

* ↳ **How big is the bias?** Small here. The top candidates are within one standard deviation of each other, so we didn't pick a lucky outlier.
  * ↳↳ **How would you remove it?** Nested CV: an inner loop for tuning and an outer loop for scoring. Or score the final model once on the separate `adult.test` file.

### 5.5 k-NN

**Q15. Why must k-NN data be scaled?**
k-NN chooses neighbors by distance. Without scaling, the feature with the biggest numbers dominates: capital-gain has std 7,385 against age's 13.6. `StandardScaler` puts every numeric feature on mean 0 and std 1.

* ↳ **Does scaling fix everything?** No. Capital-gain is still extremely skewed: 91.7% zeros plus a few huge values. Scaling compresses the normal values, and outliers still sit far out.
  * ↳↳ **What else could help?** A log transform (`log1p`) on capital-gain and capital-loss, or `RobustScaler`. Neither was required by the spec.

**Q16. Why did k = 31 win, the biggest value in the grid?**
A larger k averages over more neighbors, which smooths out noise. That's especially useful in 105 dimensions where single neighbors are unreliable.

* ↳ **Does winning at the edge of the grid mean anything?** Yes. The best k might be even larger. The spec fixed the grid at {1, 3, 7, 15, 31}, so we stayed with it, but extending it would be the next experiment.

**Q17. Why is k-NN's training F1 exactly 1.0 with `weights="distance"`?**
When you score training data, each point finds *itself* as a neighbor at distance 0. Distance weighting gives it overwhelming weight, so it always predicts its own label. It isn't a sign of a good model, just of how distance weighting works on training data.

* ↳ **So is it overfitting?** Training F1 is meaningless for distance-weighted k-NN. Judge it on test F1 (0.650), which is close to the uniform-weight version (0.649).

**Q18. Why euclidean over manhattan?**
Euclidean was slightly better, 0.650 vs. 0.644, within one standard deviation. With one-hot columns, the two treat a category mismatch differently: euclidean contributes √2 per mismatch and manhattan contributes 2. Otherwise they behave similarly.

### 5.6 RIPPER

**Q19. How does RIPPER work in one breath?**
It learns rules for >50K one at a time.
* Each rule is grown condition by condition, picking the condition with the best FOIL gain on two-thirds of the data.
* It's then pruned on the other third.
* Rules stop being added when they no longer explain the data better than they cost to describe (the MDL principle).
* An optimization pass revisits every rule.
* Anything no rule covers is predicted ≤50K.

* ↳ **What is FOIL gain measuring?** How much more precise the rule becomes on >50K after adding a condition, multiplied by how many positives it still covers. So it rewards conditions that are both precise and broad.
  * ↳↳ **Example from your data?** Adding `marital-status = Married-civ-spouse` raises the rule's share of high earners from 24% to 45% while keeping 6,692 positives. Adding `education-num > 13` then raises it to about 80%.

**Q20. Why wrap it, instead of calling wittgenstein directly?**
Our harness and grid searches expect a scikit-learn estimator: `fit` returns `self`, `predict_proba` columns line up with `classes_`, and `clone()` must work. wittgenstein's `fit` returns `None` and has no `classes_`, so it wouldn't work inside a Pipeline.

* ↳ **Why must `__init__` only store parameters?** `clone()` and `GridSearchCV` rebuild the model from those parameters. `set_params()` also changes them directly *without* running `__init__`, so any logic in `__init__` would be skipped.
  * ↳↳ **Would your wrapper pass scikit-learn's `check_estimator` tests?** Not all of them. It has no `validate_data` or `__sklearn_tags__`. It's built for our pipeline's inputs, and adding those two would make it fully compliant.

**Q21. Why is RIPPER's recall so low (0.433)?**
Anyone who doesn't match one of its rules defaults to ≤50K. So recall can never exceed how much of the >50K class the rules cover. Its rules are precise (0.718) but don't cover everyone.

* ↳ **Is it caused by the 0.5 threshold?** We checked. On fold 1, every row a rule covered had probability above 0.5, so RIPPER's own rule decision and the 0.5 cut-off give identical predictions.
  * ↳↳ **How would you raise its recall?** Lower the threshold (Task 4), oversample the minority class, or allow more rules. Raising `prune_size` or `max_rules` and comparing would be the experiment.

### 5.7 SMOTE

**Q22. Why must SMOTE be inside the pipeline?**
SMOTE builds synthetic positives from neighboring real positives. Applied before CV, synthetic rows in the training folds are built from test-fold rows, so the model has effectively seen the test set. The test folds also become 50% positive instead of 24%, which isn't realistic.

* ↳ **Did you measure the effect?** Yes, in our test run (not in the submitted notebook). The same tree got CV F1 **0.671** with SMOTE inside the pipeline and **0.859** with SMOTE applied globally first. That 0.19 gap is pure leakage.
  * ↳↳ **Why does `imblearn`'s Pipeline allow this and scikit-learn's doesn't?** `imblearn` knows samplers change the number of rows. It runs them only during `fit` and skips them during `predict`, so test data is never resampled.

**Q23. Is there a problem with SMOTE after one-hot encoding?**
Yes. SMOTE interpolates between rows, so a synthetic person can get `sex_Male = 0.37`. In our test, 41% of synthetic rows had fractional one-hot values. It isn't leakage, but those rows aren't realistic people.

* ↳ **What's the fix?** `SMOTENC`, which handles categorical columns by picking an existing category instead of averaging. It has to run before encoding.

**Q24. SMOTE didn't improve F1 (0.676 vs. 0.685). Was it pointless?**
No. SMOTE trades precision for recall. In the debugged AI pipeline, recall reached 0.85 against the tuned tree's 0.65. F1 at 0.5 barely moves, but when misses cost 10 times more than false alarms, higher recall is what you want.

* ↳ **Could you get the same effect without SMOTE?** Largely, yes. `class_weight="balanced"` or simply lowering the threshold also push the model toward recall. SMOTE is one of several remedies.

### 5.8 Cost analysis

**Q25. Why is the theoretical threshold 0.091?**
Predicting positive costs $10 if wrong, and predicting negative costs $100 if wrong. Predict positive when the expected cost of positive is lower: (1 − p)·10 < p·100, which gives p > 10/110 ≈ 0.091.

* ↳ **Why did you find 0.12 instead?** The 0.091 result assumes the probabilities are *calibrated*: a "0.10" really means a 10% chance. Tree leaf frequencies aren't calibrated, and training on SMOTE-balanced data pushes probabilities upward, so the cheapest cut-off moves.
  * ↳↳ **How would you make them match?** Calibrate the probabilities with `CalibratedClassifierCV`, or skip SMOTE and use the raw tree's probabilities. Both bring the empirical optimum closer to 0.091.

**Q26. How good is $136,500?**
Predicting everyone ≤50K costs $784,100 (every high earner missed). Contacting everyone costs $247,200. The model at t = 0.12 costs **$136,500**, 83% below doing nothing and 45% below contacting everyone. It finds 94.6% of high earners.

* ↳ **Is choosing the threshold on the same out-of-fold predictions you report a problem?** Slightly. The threshold was tuned and scored on the same data, so $136,500 is a little optimistic. The clean version picks the threshold on one part of the data and reports cost on another, such as `adult.test`.

### 5.9 AI audit

**Q27. Did the AI place SMOTE correctly?**
Yes. It put SMOTE inside `imblearn.pipeline.Pipeline`, between the `ColumnTransformer` and the classifier, so it only resamples training folds.

* ↳ **Then what did you fix?** Two bugs that stopped it running on Adult. The loading (`data.csv`, no header, `?` not treated as missing) became `load_data()`. And `scoring="f1"` crashed on the text labels, because F1 needs a positive class of 1 and `load_data()` encodes >50K as 1.
  * ↳↳ **Anything else you'd change?** Its grid doesn't match the spec (depth [None, 5, 10, 20]), it uses one 80/20 split with F1 only, and its SMOTE has the fractional one-hot problem (Q23). The debugged version deliberately fixes only the two bugs, so the difference from the original is easy to see.

---

## 6. Weak spots a grader may notice, with honest answers

These are visible in the submitted notebook. Knowing them in advance is better than being surprised.

| What they might see | What it is | What to say |
|---|---|---|
| Cell 38 is under "task 2 benchmark bar plot", but `plot_metrics()` plots cost vs. threshold | The Task 2 grouped bar chart with error bars wasn't finished. The function was reused for the cost plot. | "The per-model mean ± std is printed by `show()` for every model, and that's the data the bar chart would use. The chart itself is the missing piece." |
| No ROC or PR curves | Not implemented. The `oof_proba` needed for them is saved for every model in `results`. | "Every model's out-of-fold probabilities are stored, so the curves come straight from `roc_curve(y, results[name]["oof_proba"])`." |
| Cell 46: `results = cost_sweep(...)` | This overwrites the dict of model results with the cost table | It's a naming clash. It happens after every model was scored, so no printed metric is affected. A separate name like `cost_df` is the fix. |
| Summary table (cell 39) and rules cell (cell 36) have no output | They weren't run in the final session | The numbers are printed by `show()` in the cells above. The rules (53, mostly starting with Married-civ-spouse) come from the earlier run. |
| k-NN plot title says `min_samples_split=distance` | Copy-paste from the tree plot. The variable holds `weights`. | Cosmetic. The value shown is the `weights` setting. |
| `cost_sweep` uses `>=` while its docstring says `>`, and hard-codes 100 and 10 | Minor inconsistency | They give the same answer in practice. Using `COST_FN`/`COST_FP` would be cleaner. |
| Some cells have execution count `None` or out-of-order counts | Cells were re-run in pieces | The code is designed to run top to bottom. A full run takes about 20+ minutes (grid searches and RIPPER), so don't do it live. |
| The SMOTE models aren't in the comparison table | They were tuned with `GridSearchCV`, but not passed through `evaluate()` | Their CV F1 is reported by `best_score_` (0.676 for the tree). Passing them through `evaluate()` would add full rows. |
| The tuned tree is scored on the same folds it was tuned on | Model-selection bias (Q14) | Small here, and nested CV or `adult.test` would remove it |
| The report's Part 2 contains the chat logs and the notebook has the debugged code, but the SMOTE verdict sentence isn't written in the report | The verdict is in `Ai_integration.ipynb`, which isn't in the zip | Be ready to say it out loud (Q27) |

---

## 7. Live demo tips

* **Don't re-run the slow cells live.** The decision tree grid takes about 5 minutes, the SMOTE grids longer, and RIPPER about 7 minutes. Walk through the saved outputs instead.
* **Cells worth running live**, since they're fast and show understanding:
  * `load_data()` and the missing-value counts
  * `make_preprocessor().fit_transform(X).shape`
  * the dummy baseline, about 1 second
  * `dt_search.best_params_`
* **Order for the walkthrough:**
  1. The problem and the accuracy paradox (dummy row)
  2. Pipeline and leakage (Q1)
  3. Preprocessing
  4. `evaluate()`
  5. Tree tuning and the validation curve
  6. k-NN
  7. RIPPER
  8. SMOTE
  9. Cost sweep (0.12 vs. 0.091)
  10. AI audit
* **Split the talking**, since the demo grades both members' participation:
  * Joseph: Task 1, the decision tree, RIPPER, Challenge C, the AI audit
  * Rodrigo: k-NN, SMOTE, the cost sweep, Challenges A and B
* **If you don't know an answer,** say what you'd check and how. That counts more than guessing.

---

## 8. Sources

* scikit-learn: [Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html); [Multi-metric evaluation with GridSearchCV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html); [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html); [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html); [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
* imbalanced-learn: [SMOTE and SMOTENC](https://imbalanced-learn.org/stable/over_sampling.html); [imblearn Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
* wittgenstein RIPPER: [GitHub](https://github.com/imoscovitz/wittgenstein) (v0.3.5)
* Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
* Cohen, W. W. (1995). Fast Effective Rule Induction. *Proc. ICML*, 115–123.
* Quinlan, J. R. (1990). Learning Logical Definitions from Relations. *Machine Learning*, 5, 239–266.
* Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *Proc. IJCAI*, 973–978. (The threshold derivation in Challenge A.)
* Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When Is "Nearest Neighbor" Meaningful? *Proc. ICDT*, 217–235. (Distance concentration in Challenge B.)
* Cawley, G. C., & Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *JMLR*, 11, 2079–2107.
