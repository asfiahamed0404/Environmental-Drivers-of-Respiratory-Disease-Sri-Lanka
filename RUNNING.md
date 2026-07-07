# Running the Full Pipeline

This project uses a two-stage pipeline:
1. Final EDA and feature engineering from `EDA-and-Analysis-3/research_analysis_final.py`
2. Final model execution from `Model/model_pipeline.py`

The pipeline is already configured to prefer a local Python virtual environment at `.venv/Scripts/python.exe` on Windows.

## Recommended Setup

Use Python 3.11 in a virtual environment inside the project root.

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not available, create the environment with your installed Python 3.11 executable instead.

## Run the Full Pipeline

From the project root, run:

```bash
python pipeline.py
```

If you want to force the local environment explicitly on Windows:

```bash
.\.venv\Scripts\python.exe pipeline.py
```

## Running on Google Colab

1. **Runtime → Change runtime type** → Hardware accelerator: **CPU**, Runtime version: **2026.04** → Save.
2. Clone the repo and enter it:
   ```python
   !git clone https://github.com/asfiahamed0404/Environmental-Drivers-of-Respiratory-Disease-Sri-Lanka.git
   %cd Environmental-Drivers-of-Respiratory-Disease-Sri-Lanka
   ```
3. Verify the pinned versions are already present in this runtime:
   ```python
   import numpy, sklearn, xgboost, shap
   print(numpy.__version__, sklearn.__version__, xgboost.__version__, shap.__version__)
   # Expected: 2.0.2 1.6.1 3.2.0 0.51.0
   ```
   If any version doesn't match, run `!pip install -r requirements.txt` before proceeding.
4. Run the pipeline:
   ```python
   !python pipeline.py
   ```

## What the Pipeline Does

1. Downloads the raw CSV from the Hugging Face dataset `shazan18/environmental-drivers-respiratory-disease-sri-lanka` as part of `EDA-and-Analysis-3/research_analysis_final.py`.
2. Runs the final EDA script and writes the engineered dataset to `outputs_final_v2/research_analysis/dataset_feature_engineered.csv`.
3. Copies the engineered dataset into `Model/dataset_feature_engineered.csv`.
4. Runs `Model/model_pipeline.py` from the `Model/` working directory so its relative dataset path resolves correctly.
5. Produces the model outputs, SHAP summaries, district plots, and the final FAH ranking tables.

## Required Packages

The model pipeline expects these pinned packages:

- pandas==2.2.2
- numpy==2.0.2
- scipy==1.16.3
- scikit-learn==1.6.1
- xgboost==3.2.0
- shap==0.51.0
- matplotlib==3.10.0
- seaborn==0.13.2

## Notes

- The engineered dataset must exist in `Model/dataset_feature_engineered.csv` before the model step starts.
- The model script reads `dataset_feature_engineered.csv` using a relative path, so it must be launched from the `Model/` folder. The pipeline handles this automatically.
- The raw EDA data is fetched from Hugging Face first and falls back to the local CSV only if the download is unavailable.
- If you rerun the pipeline, it will regenerate the engineered dataset and then rerun the model end-to-end.
