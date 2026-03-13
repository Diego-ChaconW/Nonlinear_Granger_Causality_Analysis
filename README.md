# Nonlinear Granger Causality Analysis Using Neural Network Architectures for Sequential Data

This repository implements **nonlinear Granger causality (GC) analysis** using neural networks for four well-known chaotic maps. It accompanies the scientific article describing the methodology and results.

## Overview

We test whether one variable in a coupled chaotic system Granger-causes the other by comparing two neural network models:

- **NAR** (Restricted model): predicts the target using only its own past values.
- **NARX** (Full model): predicts the target using both its own and the potential cause's past values.

A statistically significant improvement in the full model (assessed via the **Wilcoxon signed-rank test**) indicates Granger causality.

### Supported Chaotic Maps

| Map | Dynamics | Reference |
|-----|----------|-----------|
| **Hénon** | Quadratic, 2D discrete | Hénon (1976) |
| **Ikeda** | Optical cavity, 2D discrete | Ikeda (1979) |
| **Tinkerbell** | Quadratic, 2D discrete | Nusse & Yorke (1994) |
| **Rulkov** | Neuronal bursting, 2D discrete | Rulkov (2001) |

### Neural Network Architectures

- **MLP** (Multi-Layer Perceptron)
- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)

## Repository Structure

```
nonlinear-granger-causality/
├── README.md               # This file
├── LICENSE                  # MIT license
├── CITATION.cff             # How to cite this repository
├── .gitignore
├── requirements.txt         # Python dependencies
│
├── src/                     # Core Python modules
│   ├── __init__.py          # Keras 3 compatibility setup
│   ├── maps.py              # Chaotic map generators
│   ├── config.py            # Shared constants and validators
│   ├── data.py              # Data generation, normalization, splitting
│   ├── causality.py         # Nonlinear causality test wrapper
│   └── analysis.py          # Statistics and plotting functions
│
├── notebooks/               # Interactive Jupyter notebooks
│   ├── 01_hyperparameter_search.ipynb
│   ├── 02_run_experiment.ipynb
│   └── 03_results_analysis.ipynb
│
├── scripts/                 # CLI scripts for automation
│   ├── run_hyperparameter_search.py
│   ├── run_experiment.py
│   └── run_analysis.py
│
├── data/                    # Data directory (synthetic)
│   └── README.md
│
└── results/                 # Output directory
    ├── README.md
    ├── hyperparameters/     # .pkl from step 1
    ├── experiments/         # .pkl from step 2
    └── figures/             # .png from step 3
```

## Pipeline

The analysis follows a three-step pipeline:

```
Step 1                    Step 2                    Step 3
Hyperparameter Search ──► Experiment (N runs) ──► Results Analysis
        │                         │                       │
  hyperparameters.pkl       results_final.pkl        figures + stats
```

### Step 1: Hyperparameter Search

Evaluates all combinations of neurons, lags, batch sizes, and architectures with a single training run each. Selects the best configuration based on total error (only considering configurations with p-value < 0.05).

### Step 2: Experiment Execution

Runs the selected hyperparameters with **100 random initializations** to address the neural network initialization problem. Saves intermediate checkpoints and a final `.pkl` with all metrics.

### Step 3: Results Analysis

Loads the experiment results, computes **mean ± standard deviation** for all metrics, and generates histograms comparing NAR vs NARX prediction errors.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- See `requirements.txt` for complete list

## Installation

```bash
git clone https://github.com/Diego-ChaconW/nonlinear-granger-causality.git
cd nonlinear-granger-causality
pip install -r requirements.txt
```

## Usage

### Option A: Jupyter Notebooks (Interactive)

```bash
jupyter notebook notebooks/01_hyperparameter_search.ipynb
```

Follow the notebooks in order: `01 → 02 → 03`.

### Option B: Command-Line Scripts (Automation)

```bash
# Step 1: Hyperparameter search
python scripts/run_hyperparameter_search.py \
    --map henon --direction Y_to_X --arch GRU \
    --output results/hyperparameters/

# Step 2: Run experiment (100 initializations)
python scripts/run_experiment.py \
    --map henon --direction Y_to_X --arch GRU \
    --neurons 100 --lag 5 --batch-size 16 --runs 100 \
    --output results/experiments/

# Step 3: Generate statistics and figures
python scripts/run_analysis.py \
    --pkl results/experiments/results_henon_GRU_Y_to_X_final.pkl \
    --output results/figures/
```

### Running All Maps and Directions

```bash
for MAP in henon ikeda tinkerbell rulkov; do
    for DIR in Y_to_X X_to_Y; do
        python scripts/run_experiment.py \
            --map $MAP --direction $DIR --arch GRU \
            --neurons 100 --lag 5 --batch-size 16 --runs 100
        python scripts/run_analysis.py \
            --pkl results/experiments/results_${MAP}_GRU_${DIR}_final.pkl
    done
done
```

## Causality Directions

The `nonlincausality` library expects data as `[target, cause]`:

| Direction Setting | Tests | Data Column 0 | Data Column 1 |
|-------------------|-------|---------------|---------------|
| `Y_to_X` | Does Y Granger-cause X? | X (target) | Y (cause) |
| `X_to_Y` | Does X Granger-cause Y? | Y (target) | X (cause) |

## Data Normalization

All time series are normalized to the interval **[-1, 1]** before analysis:

```
x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
```

## Statistical Significance

- Significance level: **α = 0.05**
- In hyperparameter search, only configurations with **p-value < 0.05** are considered as valid candidates.
- In experiment results, the percentage of runs achieving significance is reported.

## Citation

If you use this code, please cite:

```
@software{nonlinear_granger_causality,
  title = {Nonlinear Granger Causality Analysis for Chaotic Maps},
  url = {https://github.com/Diego-ChaconW/nonlinear-granger-causality}
}
```

See `CITATION.cff` for the full citation metadata.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Acknowledgments

This work uses the [`nonlincausality`](https://github.com/mrostecki/nonlincausality) library for nonlinear Granger causality testing with neural networks.
