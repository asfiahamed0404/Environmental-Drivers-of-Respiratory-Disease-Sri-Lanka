# Environmental Drivers of Respiratory Disease

This project analyzes environmental drivers of respiratory disease across 25 districts in Sri Lanka from 2014 to 2024. It combines vegetation, fire, pollutant, carbon flux, and health metrics to build the Forest-Air-Health (FAH) Risk Index and train the final model pipeline with SHAP-based interpretation.

## Main Files

- [pipeline.py](pipeline.py): runs the full workflow end to end.
- [EDA-and-Analysis-3/research_analysis_final.py](EDA-and-Analysis-3/research_analysis_final.py): generates the engineered dataset.
- [Model/model_pipeline.py](Model/model_pipeline.py): trains the model and produces the final outputs.
- [requirements.txt](requirements.txt): pinned Python dependencies.
- [RUNNING.md](RUNNING.md): full run instructions for both local and Google Colab environments.

## Quick Start (Local)

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py
```

## Quick Start (Google Colab)

Set Runtime version to **2026.04** (Runtime → Change runtime type), then:

```python
!git clone https://github.com/asfiahamed0404/Environmental-Drivers-of-Respiratory-Disease-Sri-Lanka.git
%cd Environmental-Drivers-of-Respiratory-Disease-Sri-Lanka
!python pipeline.py
```

See [RUNNING.md](RUNNING.md) for full details, troubleshooting, and package version checks.

## What It Produces

The pipeline generates the engineered dataset, runs model training, computes SHAP explanations, and writes the final FAH ranking tables and plots.

## Notes

- The model step must run from the `Model/` folder so its relative dataset path resolves correctly.
- No manual dataset download is needed — `pipeline.py` automatically fetches the raw dataset from Hugging Face on first run.
- The EDA stage downloads the raw CSV from Hugging Face and falls back to `Data/final_cleaned_dataset.csv` if needed.
- The engineered dataset must exist in `Model/dataset_feature_engineered.csv` before model training starts.

## Reproducibility

To reproduce the paper's exact reported numbers, run the pipeline on Google Colab (runtime snapshot **2026.04**) with the pinned package versions in [requirements.txt](requirements.txt) rather than a local environment.
