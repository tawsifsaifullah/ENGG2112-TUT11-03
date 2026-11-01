# ENGG2112-TUT11-02: Investigating if machine learning models can predict EV battery state of health and observing the adaptability of the models in different temperature extremes

Models that estimate battery capacity from voltage readings are often trained at a single temperature, but measuring capacity at a new temperature requires time-consuming discharge testing. We train eight regression models to estimate an unseen battery cell's next-discharge capacity from the voltage relaxation after charging, and investigate how much labelled data from a new temperature (35 °C and 45 °C) a model trained at 25 °C needs.

## Team
- Eamon (550667766): Executive Summary, Related Work and Conclusion
- Saifullah (550687672): Background Motivation, Objective and Problem Statement, Potential for Wider Adoption; project notebook and repository
- Tristan (540820456): Methodology; full study training scripts and experiment runs
- Nathan (530820172): Simulation Results and Analysis, results figures

## Repository contents
- `ENGG2112_Battery_SoH.ipynb`: the full pipeline, step by step (data loading, cleaning and preprocessing, feature engineering, model training, evaluation, temperature transfer and feature importance).
- `data/features.csv`: the extracted feature table (22,239 cycles from 66 cells, one row per cycle).
- `data/feature_names.json`: the 24 model inputs.
- `Dataset_1_NCA_battery/CY25-1_1-#3.csv`: one raw cell file, used in the notebook to show the raw data and test the feature extraction.
- `full_study_code/`: the scripts that produced the results in our report.
- `requirements.txt`: Python packages.

### Why a notebook and separate scripts
The results in the report come from the full study scripts, which run 32 training jobs (eight model families under two input variants, plus four extra seeds for each MLP) with nested leave-one-cell-out validation, seven donor-data budgets and SHAP. They take several hours of CPU time, so they are kept as `.py` files. The notebook follows the same data, features and cell-level tests in a simpler form (fixed model settings, scikit-learn models, one seed, permutation importance instead of SHAP), so its numbers are close to, but not exactly the same as, the report.

## Dataset
Zhu et al. (2022), "Data-Driven Capacity Estimation of Commercial Lithium-Ion Batteries from Voltage Relaxation", Nature Communications 13, 2261. Dataset 1 (NCA) from Zenodo record 6405084: https://zenodo.org/records/6405084

The full raw dataset (`Dataset_1_NCA_battery.zip`, about 580 MB) is too large for GitHub. The notebook runs from `data/features.csv`. To rebuild the features from the raw data, download and unzip the dataset into `Dataset_1_NCA_battery/` and set `REBUILD_FEATURES = True` in the notebook.

## How to run
1. Install: `pip install -r requirements.txt`
2. Open `ENGG2112_Battery_SoH.ipynb` and run all cells from the top.

## Results
Mean absolute error (MAE) on 35 unseen 25 °C cells (report Table 4):

| Model | MAE (Ah) | SoH error (% points) | Improvement over baseline (%) |
|---|---|---|---|
| Small MLP | 0.045 | 1.29 | 72 |
| Support vector regression (RBF) | 0.046 | 1.31 | 71 |
| Large MLP | 0.047 | 1.34 | 71 |
| Histogram boosting | 0.051 | 1.46 | 68 |
| Random forest | 0.055 | 1.58 | 65 |
| Extra trees | 0.055 | 1.57 | 66 |
| Ridge regression | 0.060 | 1.72 | 62 |
| Mean baseline | 0.160 | 4.56 | - |

- The small MLP, large MLP and SVR met the 0.05 Ah target. They lie within 0.0015 Ah of each other, so no model is clearly the best.
- Using the 25 °C data alone, no model met the 0.05 Ah target at 35 °C or 45 °C; histogram boosting came closest at 0.053 and 0.066 Ah.
- Removing the two constant inputs (ambient temperature and discharge rate) cut the small MLP's 45 °C error from 0.357 to 0.088 Ah.
- Cycles from other cells at the target temperature were required to reach the target. At 45 °C the small MLP met the target with a 10 % donor-data budget (about 1,500 labelled cycles).
- Cells at 25 °C were worn to 80 % SoH roughly three times sooner than those at 35 °C and 45 °C.

In the notebook: Section 2 (Table 1 and Figure 2), Section 5 (25 °C results), Section 6 (transfer) and Section 7 (feature importance).

## Use of AI
Generative AI (ChatGPT and Claude) assisted with code preparation, results analysis and report drafting, as declared in the report appendix. All code was reviewed and run by the group.
