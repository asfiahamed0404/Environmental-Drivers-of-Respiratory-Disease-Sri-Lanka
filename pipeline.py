import os
import subprocess
import sys
import shutil
from pathlib import Path


def get_python_executable():
    venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

def run_pipeline():
    print("======================================================")
    print("      ENVIRONMENTAL-RESPIRATORY MODELING PIPELINE      ")
    print("======================================================")
    
    # ---------------------------------------------------------
    # Paths Definition
    # ---------------------------------------------------------
    engineered_data = "Model/dataset_feature_engineered.csv"
    final_eda_script = "EDA-and-Analysis-3/research_analysis_final.py"
    final_eda_output = "outputs_final_v2/research_analysis/dataset_feature_engineered.csv"
    model_script = "Model/model_pipeline.py"
    
    # ---------------------------------------------------------
    # Step 1: Comprehensive Exploratory Data Analysis & Feature Engineering
    # ---------------------------------------------------------
    print(f"\n>>> STEP 1: Running Final EDA & Feature Engineering ({final_eda_script}) <<<")
    if os.path.exists(final_eda_script):
        # Run the final EDA script
        python_executable = get_python_executable()
        eda_cmd = [python_executable, final_eda_script]
        print(f"Executing: {' '.join(eda_cmd)}")
        
        # Ensure utf-8 encoding for stdout on Windows to prevent UnicodeEncodeError
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run(eda_cmd, env=env)
        
        if result.returncode != 0:
            print("Error running final EDA pipeline. Exiting.")
            sys.exit(1)
        print("Final EDA Completed Successfully. Outputs saved to 'outputs_final_v2/'.")
        
        # Copy the newly generated dataset to the Model folder
        if os.path.exists(final_eda_output):
            print(f"Copying newly generated dataset from {final_eda_output} to {engineered_data}")
            shutil.copy2(final_eda_output, engineered_data)
        else:
            print(f"Warning: The script did not produce {final_eda_output}!")
    else:
        print(f"Error: {final_eda_script} not found! Cannot proceed with feature engineering.")
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 2: Dataset Verification
    # ---------------------------------------------------------
    print("\n>>> STEP 2: Verifying Feature Engineered Dataset <<<")
    if not os.path.exists(engineered_data):
        print(f"Error: Engineered dataset {engineered_data} not found!")
        print("The model requires this dataset to proceed.")
        sys.exit(1)
    else:
        print(f"Dataset found and verified: {engineered_data}")

    # ---------------------------------------------------------
    # Step 3: Model Building and Execution
    # ---------------------------------------------------------
    print("\n>>> STEP 3: Running Model Building Pipeline <<<")
    if os.path.exists(model_script):
        model_dir = os.path.dirname(model_script)
        model_file = os.path.basename(model_script)

        if model_file.endswith(".py"):
            script_path = Path(model_script).resolve()
            script_cmd = [get_python_executable(), str(script_path)]
            model_workdir = str((Path(__file__).resolve().parent / "Model").resolve())
            print(f"Executing Script in {model_workdir}/: {' '.join(script_cmd)}")
            result = subprocess.run(script_cmd, cwd=model_workdir)
            if result.returncode != 0:
                print("Error executing the model pipeline script. Exiting.")
                sys.exit(1)
            print("Model Execution Completed Successfully.")
        else:
            # We run the notebook from its own directory so it can access its local CSVs
            nbconvert_cmd = [
                get_python_executable(), "-m", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                model_file
            ]

            print(f"Executing Notebook in {model_dir}/: {' '.join(nbconvert_cmd)}")

            # Execute nbconvert in the Model directory
            result = subprocess.run(nbconvert_cmd, cwd=model_dir)

            if result.returncode != 0:
                print("Error executing the Model notebook. Exiting.")
                sys.exit(1)
            print("Model Execution Completed Successfully.")
    else:
        print(f"Error: {model_script} not found!")
        sys.exit(1)
        
    print("\n======================================================")
    print("               PIPELINE RUN COMPLETE                   ")
    print("======================================================")

if __name__ == "__main__":
    run_pipeline()
