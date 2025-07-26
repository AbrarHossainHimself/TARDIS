#!/bin/bash

# This script reorganizes the HPC GNN Simulator project into a clean structure.
# Run this script from the root directory of your project.
#
# It's recommended to run this on a copy of your project or after committing
# your current work to Git, just in case.

set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting project reorganization..."

# === STEP 1: Create the new directory structure ===
echo "--> Creating new directories: data, notebooks, src, scripts..."
mkdir -p data
mkdir -p notebooks/01_EDA_Frontier
mkdir -p notebooks/02_EDA_PM100
mkdir -p notebooks/03_Preprocessing
mkdir -p notebooks/04_GNN_Experiments
mkdir -p notebooks/05_Scheduler_Analysis
mkdir -p src
mkdir -p scripts

# === STEP 2: Move Notebooks ===
echo "--> Moving notebooks..."
# Move Frontier, PM100, and GNN notebooks
if [ -f "Frontier/Frontier_power data.ipynb" ]; then mv "Frontier/Frontier_power data.ipynb" notebooks/01_EDA_Frontier/; fi
if [ -d "PM100" ]; then mv PM100/*.ipynb notebooks/02_EDA_PM100/; fi
if [ -d "GNN" ]; then mv GNN/*.ipynb notebooks/04_GNN_Experiments/; fi

# Move preprocessing and root-level notebooks
if [ -f "preprocessing/data_split.ipynb" ]; then mv preprocessing/data_split.ipynb notebooks/03_Preprocessing/; fi
if [ -f "FCFS.ipynb" ]; then mv FCFS.ipynb notebooks/05_Scheduler_Analysis/; fi

# Move results notebooks
if [ -f "hpc_simulator/results/multizone.ipynb" ]; then mv hpc_simulator/results/multizone.ipynb notebooks/05_Scheduler_Analysis/; fi
if [ -f "hpc_simulator/results/total_cost.ipynb" ]; then mv hpc_simulator/results/total_cost.ipynb notebooks/05_Scheduler_Analysis/; fi

# === STEP 3: Move Source Code to src/ ===
echo "--> Moving source code to src/..."
# Move all python files from hpc_simulator root to src/
if ls hpc_simulator/*.py 1> /dev/null 2>&1; then mv hpc_simulator/*.py src/; fi
# Move the multizone_scheduler package into src/
if [ -d "hpc_simulator/multizone_scheduler" ]; then mv hpc_simulator/multizone_scheduler/ src/; fi

# === STEP 4: Move runnable scripts to scripts/ ===
echo "--> Moving runnable scripts to scripts/..."
if [ -f "src/run_gnn.py" ]; then mv src/run_gnn.py scripts/; fi
if [ -f "src/multizone_scheduler/run_peak_analysis.py" ]; then mv src/multizone_scheduler/run_peak_analysis.py scripts/; fi
if [ -f "src/multizone_scheduler/run_resource_analysis.py" ]; then mv src/multizone_scheduler/run_resource_analysis.py scripts/; fi
if [ -f "src/multizone_scheduler/cost_benefit_analysis_run.py" ]; then mv src/multizone_scheduler/cost_benefit_analysis_run.py scripts/; fi
if [ -f "src/multizone_scheduler/scheduler_comparison.py" ]; then mv src/multizone_scheduler/scheduler_comparison.py scripts/; fi
if [ -f "src/multizone_scheduler/scheduler_debug.py" ]; then mv src/multizone_scheduler/scheduler_debug.py scripts/; fi


# === STEP 5: Move data and config files ===
echo "--> Moving data and config files..."
if [ -f "hpc_simulator/available_columns.txt" ]; then mv hpc_simulator/available_columns.txt data/; fi
if [ -f "hpc_simulator/sample_data.txt" ]; then mv hpc_simulator/sample_data.txt data/; fi
# Move the hyperparameters file with the predictor code
if [ -f "best_hyperparameters.json" ]; then mv best_hyperparameters.json src/; fi


# === STEP 6: Create essential repo files ===
echo "--> Creating __init__.py files..."
touch src/__init__.py
if [ -d "src/multizone_scheduler" ]; then touch src/multizone_scheduler/__init__.py; fi

echo "--> Creating a sample requirements.txt file..."
cat << EOF > requirements.txt
# This is a sample list. Please verify against your environment.
# Run 'pip freeze > requirements.txt' to get your exact versions.
pandas
numpy
matplotlib
openpyxl
# Add torch, tensorflow, or other GNN-related libraries here
# e.g., torch
# torch_geometric
EOF

echo "--> Creating a comprehensive .gitignore file..."
cat << EOF > .gitignore
# Python artifacts
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/
*.egg

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Data Files & Output
# It's best practice not to commit data files to Git.
# Use Git LFS for large files if necessary.
data/
results/
*.csv
*.xlsx
*.txt
!data/available_columns.txt # You can explicitly include small text files

# Environment
.env
venv/
env/
myenv/

# IDE specific
.idea/
.vscode/
EOF

# === STEP 7: Clean up old and generated files ===
echo "--> Cleaning up old directories and generated files..."
# Remove old, now-empty directories
rm -rf Frontier/ PM100/ GNN/ preprocessing/ hpc_simulator/

# Remove generated output files
find . -name "*_results.txt" -type f -delete
find . -name "cost_job_peak_analysis.txt" -type f -delete
find . -name "repomix-output.txt" -type f -delete

echo ""
echo "✅ Project reorganization complete!"
echo "Your files are now structured in the data/, notebooks/, src/, and scripts/ directories."
echo ""
echo "Next steps:"
echo "1. Review the new structure."
echo "2. Update your 'requirements.txt' by running 'pip freeze > requirements.txt' from your project's virtual environment."
echo "3. Initialize a Git repository ('git init'), add the files ('git add .'), and commit ('git commit -m \"Initial commit\"')."