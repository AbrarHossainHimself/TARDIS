# TARDIS: Power-Aware Scheduling for Multi-Center HPC

This project is a simulation environment for High-Performance Computing (HPC) schedulers. It includes several scheduling algorithms (FCFS, SJF, Multizone) and features an enhanced scheduler that uses a Graph Neural Network (GNN) to predict job power consumption, aiming to optimize for electricity cost and resource utilization across multiple data centers.

## Publication

The research and methods in this repository were presented at the **30th International Job Scheduling Strategies for Parallel Processing (JSSPP) Workshop in 2025**.

**Paper Title:** *Power-Aware Scheduling for Multi-Center HPC Electricity Cost Optimization*

## Project Structure

- **/data/**: Holds sample data files. (Raw data is ignored by Git).
- **/notebooks/**: Contains all Jupyter notebooks for data exploration, model development, and results analysis, organized by topic.
- **/src/**: All Python source code for the simulator, schedulers, and analysis tools.
- **/scripts/**: Standalone scripts to run simulations and analyses from the command line.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/AbrarHossainHimself/TARDIS.git
    cd TARDIS
    ```
2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run a simulation, use one of the scripts in the `/scripts` directory. For example:
```bash
python scripts/run_peak_analysis.py 
