# Modular versus Hierarchical: A Structural Analysis of Mathematical Collaboration

**Brian Hepler**

[bhepler.com](https://bhepler.com) | [Math Research Compass](https://mathresearchcompass.com)

This repository contains the complete, automated analysis pipeline for the paper "Modular versus Hierarchical: A Structural Signature of Topic Popularity in Mathematical Research." The workflow processes raw metadata from the arXiv preprint server, identifies research topics, builds and analyzes co-authorship networks, and performs a series of statistical and sensitivity analyses to characterize the structural differences between popular and niche fields in mathematics.

The entire pipeline is orchestrated by a `Makefile` and managed by a `config.yaml` file, ensuring full reproducibility with minimal manual intervention.

## Project Structure

```
.
├── data/
│   ├── raw/
│   └── cleaned/
├── results/
│   ├── collaboration_analysis/
│   ├── disambiguation/
│   └── ...
├── src/                          # All Python analysis scripts
├── figures/                      # Final figures for the manuscript
├── config.yaml                   # Manages data flow between scripts
├── Makefile                      # Automates the entire pipeline
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```

## Setup and Prerequisites

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/brian-hepler-phd/MRC-Network-Analysis.git
    cd MRC-Network-Analysis
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:** All required packages and their specific versions are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Obtain Raw Data:**
    *   Download the Cornell arXiv dataset from Kaggle: [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).
    *   Place the `arxiv-metadata-oai-snapshot.json` file into the `data/raw/` directory.
    *   **Note:** The pipeline assumes an initial, one-time manual filtering of this raw JSON to create `data/cleaned/math_arxiv_snapshot.csv`, containing only mathematics papers with the necessary columns (`id`, `authors`, `title`, `categories`, `abstract`, `update_date`, and `authors_parsed`).

---

## Reproducing the Analysis

The entire pipeline can be executed using simple `make` commands from the project's root directory. The `Makefile` automatically handles the dependencies between scripts.

### Core Pipeline

To run the full analysis from start to finish and generate all results and figures for the paper, use the default target:

```bash
make
```
or
```bash
make all
```

This will execute the following steps in sequence, using the `config.yaml` file to pass data between them:

1.  **`make topics`**: Identifies research topics using BERTopic.
2.  **`make network_data`**: Prepares the author-topic network dataset.
3.  **`make disambiguate`**: Performs author name disambiguation.
4.  **`make metrics`**: Builds networks and calculates all structural metrics.
5.  **`make compare`**: Runs the baseline popular vs. niche statistical comparison.
6.  **`make regression`**: Runs the fixed and enhanced regression analyses to control for network size.
7.  **`make visualize`**: Generates the final figures for the manuscript.

`make` is intelligent: if you modify a script, it will only re-run that step and all subsequent steps that depend on it, saving significant computation time.

### Other Useful Commands

*   **Clean the Workspace:** To delete all generated results and start fresh:
    ```bash
    make clean
    ```
*   **Run Sensitivity Analyses:** To run the validation scripts (cutoff sensitivity, topic model stability, COVID-19 temporal analysis):
    ```bash
    make sensitivity
    ```
*   **Run Only a Specific Step:** To run part of the pipeline (e.g., up to the metrics calculation):
    ```bash
    make metrics
    ```

---

## The Analysis Scripts (`src/`)

The `src/` directory contains the modular scripts that perform each stage of the analysis.

-   **`config_manager.py`**: A helper module for reading from and writing to `config.yaml`.
-   **`1_build_topics.py`**: Step 1 - Topic Modeling.
-   **`2_prepare_network_data.py`**: Step 2 - Data Preparation.
-   **`3_disambiguate_authors.py`**: Step 3 - Author Name Disambiguation.
-   **`4_calculated_network_metrics.py`**: Step 4 - Network Metrics Calculation.
-   **`5a_compare_groups.py`**: Step 5a - Baseline Group Comparison.
-   **`5b_bootstrap_effects.py`**: Step 5b - Bootstrap Analysis for CIs.
-   **`5c_regression_size_controls.py`**: Step 5c - Main Regression Analysis.
-   **`5d_regression_binary_vs_continuous.py`**: Step 5d - Continuous Popularity Regression.
-   **`6_generate_figures`**: Step 6 - Final Visualizations.
-   **`validation_cutoff_thresholds.py`**, **`validation_topic_modeling.py`**, **`validation_temporal.py`**: Step 7 - Robustness and Sensitivity Checks.
```