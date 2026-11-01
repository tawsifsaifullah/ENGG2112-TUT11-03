# Full experimental model training code

These scripts produced the results in our report (32 training jobs, run on CPU on 23 October 2026).

- `prepare_data.py`: extracts the relaxation features from the raw Dataset 1 files and writes `data/features.csv`
- `models.py`: the eight model families and their settings (report Table 2)
- `run_model.py`: one experiment per model: nested leave-one-cell-out at 25 °C, learning curves, and transfer to 35 °C and 45 °C with seven donor-data budgets
- `run_all.py`: runs all 32 jobs in order (eight families under two input variants at seed 42, plus seeds 43–46 for both MLPs)
- `summarise.py`: combines the results into comparison tables and figures
- `explain.py`: SHAP analysis of the source-only models
- `preflight.py`, `test_integrity.py`: checks on the data and code before running
- `requirements.txt`: packages used for the full study; the MLPs also need PyTorch (version 2.9.1 was used)

The full run takes several hours of CPU time. For a shorter, step-by-step version of the same pipeline, see `Main_project_code_battery_SoH.ipynb` in the main folder.
