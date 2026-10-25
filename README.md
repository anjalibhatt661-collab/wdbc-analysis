# Nuclear features and breast mass diagnosis: a cautious public-dataset analysis

## Biological question

Do nuclear size, shape and texture measurements taken from fine-needle-aspirate (FNA) images of breast masses differ between **malignant** and **benign** diagnoses, and which of the 30 measured features still show a reliable difference after controlling the false discovery rate across all 30 tests?

This is an association question. It is not a diagnostic-accuracy study, and nothing here shows that any feature causes malignancy.

## Data

**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository (Wolberg, Street and Mangasarian, 1995; CC BY 4.0). It is a real public clinical dataset, here as the copy bundled with scikit-learn.

- 569 samples: 212 malignant, 357 benign.
- 30 numeric features computed from digitised images of cell nuclei: 10 measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each as a **mean**, a **standard error** (`_se`) and a **worst** (largest-values) summary.
- `data/wdbc.csv` is the table used. `prep_data.py` shows how it was exported. It is not needed to run the analysis.

## Methods

1. **Test per feature:** two-sided Mann-Whitney U test, malignant vs benign. A rank test was used because the features are skewed and have outliers.
2. **Effect size:** AUC = the probability that a random malignant sample has a larger value than a random benign one (0.5 = no difference), with a 95% bootstrap confidence interval (2,000 resamples within each group). P-values alone are not informative here because with 569 samples they become extremely small.
3. **Multiple-testing control:** Benjamini-Hochberg false discovery rate (FDR) at 0.05 across all 30 tests. Bonferroni is reported alongside as a stricter comparison.
4. **Null check:** the diagnosis labels were shuffled 200 times and the whole procedure was repeated, to confirm the correction behaves as intended when there is nothing to find.
5. **Replication check:** the data were split into two stratified halves. Features were chosen in the discovery half (BH, 0.05) and then re-tested only in the held-out half.
6. **Dependence check:** a Spearman correlation matrix of the features, to see how far the 30 tests are really independent.

All random steps use seed 42. Parameters are in one block at the top of `analysis.py`.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Takes about 10 seconds. Results are written to `results/`. Running it twice gives byte-identical tables. The script works on any CSV with a group column and numeric feature columns: change `DATA_CSV`, `GROUP_COL`, `CASE_LABEL` and `CONTROL_LABEL` at the top.

## Results

- **27 of 30 features** differ significantly at BH FDR 0.05. 26 survive Bonferroni.
- **Not significant:** `texture_se` (AUC 0.51), `mean_fractal_dimension` (0.48), `smoothness_se` (0.47).
- **Strongest effects:** `worst_perimeter` (AUC 0.975, 95% CI 0.964 to 0.986), `worst_radius` (0.970), `worst_area` (0.970), `worst_concave_points` (0.967), `mean_concave_points` (0.964). A random malignant sample has a larger `worst_perimeter` than a random benign one about 97.5% of the time.
- **Weak but significant:** `fractal_dimension_se` has AUC 0.62, a small effect.
- **Null check:** with shuffled labels, 1.41 of 30 features per run reached p < 0.05 uncorrected (close to the 1.5 expected by chance). After BH the mean was 0.06 discoveries per run, and only 4 of 200 shuffles (2%) produced any discovery.
- **Replication:** 27 features were selected in the discovery half (n = 284) and 26 replicated in the held-out half (n = 285), in the same direction.
- **Correlation:** 26 of 435 feature pairs have |Spearman rho| > 0.9, and 7 principal components explain 90% of the variance, so the 30 tests carry far less than 30 independent pieces of information.

Full per-feature table: `results/feature_tests.csv`. Figures are in `results/figures/`.

## Interpretation

Malignant FNA samples in this dataset have larger, more irregular nuclei than benign ones. The clearest signals are size measures (radius, perimeter, area) and contour irregularity (concavity, concave points), especially the "worst" summaries, which capture the most extreme nuclei in a sample. Texture, smoothness and symmetry differ less, and fractal dimension barely at all.

Three cautions apply to reading this result:

- **The strongest features are largely one signal.** Radius, perimeter and area are mathematically linked, and the mean and worst versions of a measurement are strongly correlated. "27 significant features" should not be read as 27 independent findings.
- **`symmetry_se` is a weak finding.** It passes BH (q = 0.031) but not Bonferroni (adjusted p = 0.84), it does not replicate in the held-out half (AUC 0.49, q = 0.86), and its direction is lower in malignant samples. It should be treated as not established.
- **Significance is not importance.** Almost every p-value is tiny because n is large. The effect sizes (AUC) and their intervals are the informative part.

## Limitations and confounders

- **No covariates, so no adjustment.** The data have no age, menopausal status, mass size, aspiration technique, imaging batch or operator. Any of these could differ between benign and malignant groups and change the measurements. This analysis cannot separate them from the diagnosis.
- **Possible circularity.** The diagnosis labels come from pathology. Pathologists classify these cells partly by nuclear appearance, which is also what the features measure, so some of this association may be built in rather than discovered.
- **Selection and prevalence.** The samples come from one clinical series of masses that were sampled, not from a screening population. The 37% malignant share is far above what a screening setting would see, so any predictive use would not transfer.
- **Independence is assumed, not verified.** The copy used has no patient identifiers. If several samples came from one patient, the tests treat them as more independent than they are, and p-values would be too small.
- **Correlated tests.** BH is valid under positive dependence, which is the likely situation here, but the effective number of independent tests is much smaller than 30 (see Results).
- **Univariate only.** Each feature is tested alone. No multivariable model was fitted, and no diagnostic performance is claimed. The split-half check is internal; it is not an independent external validation.
- **Image-derived measures.** The features depend on how the nuclei were segmented and measured, which is a method-specific step, not a direct biological readout.
- **One dataset, one time point.** Nothing here has been tested in another cohort.

## Decisions

- Mann-Whitney rather than a t-test, because of skewed distributions.
- BH as the primary correction because the tests are correlated and the goal is exploratory; Bonferroni is shown to illustrate how much stricter control changes the picture (one feature).
- One split-half replication check with a fixed seed, kept simple on purpose.
- AUC as the main effect size because it is easy to interpret and does not depend on the units of each feature.

## Files

| Path | Purpose |
|---|---|
| `analysis.py` | The whole analysis (run this) |
| `data/wdbc.csv` | Input data |
| `prep_data.py` | How the CSV was exported (optional) |
| `requirements.txt` | Python packages |
| `results/feature_tests.csv` | Per-feature statistics, p-values and adjusted p-values |
| `results/null_check.csv` | Label-permutation results |
| `results/replication.csv` | Discovery vs validation results |
| `results/summary.txt` | Key numbers in one place |
| `results/figures/` | Effect-size plot, boxplots of the top features, correlation heatmap |

## Reference

Street WN, Wolberg WH, Mangasarian OL. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE Electronic Imaging, 1993. Data: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic).
