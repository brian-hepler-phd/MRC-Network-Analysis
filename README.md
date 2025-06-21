# Modular versus Hierarchical: A Structural Analysis of Mathematical Collaboration

**Brian Hepler**

[bhepler.com](https://bhepler.com) | [Math Research Compass](https://mathresearchcompass.com)

This repository contains the full analysis pipeline for the paper "Modular versus Hierarchical: A Structural Signature of Topic Popularity in Mathematical Research." The workflow processes raw metadata from the arXiv preprint server, identifies research topics, builds and analyzes co-authorship networks, and performs a series of statistical and sensitivity analyses to characterize the structural differences between popular and niche fields in mathematics.

## Project Structure

The project is organized into the following key directories:

```
.
├── data/
│   ├── raw/                      # Raw dataset downloaded from source
│   └── cleaned/                  # Intermediate, processed data files
├── results/
│   ├── collaboration_analysis/   # Outputs from network analysis & comparison
│   ├── disambiguation/           # Logs and outputs from author disambiguation
│   ├── regression_analysis_*/    # Outputs from regression models
│   └── validation/               # Outputs from sensitivity analyses
├── src/                          # All Python analysis scripts
├── figures/                      # Final figures for the manuscript
└── README.md                     # This file
```

## Prerequisites

- Python 3.9+
- Required packages can be installed via pip:
  ```bash
  pip install pandas numpy scikit-learn statsmodels networkx tqdm bertopic sentence-transformers umap-learn hdbscan matplotlib seaborn
  ```

---

## Analysis Pipeline

The analysis is structured as a multi-step pipeline. The scripts are designed to be run in the order presented below, as the output of one step is often the input for the next.

### Step 0: Data Acquisition

1.  **Obtain Raw Data**: Download the Cornell arXiv dataset from Kaggle: [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).
2.  **Placement**: Save the `arxiv-metadata-oai-snapshot.json` file into the `data/raw/` directory.
3.  **Initial Cleaning**: Manually or with a simple script, filter the raw JSON to a more manageable CSV format containing only mathematics papers. The resulting file should be saved as `data/cleaned/math_arxiv_snapshot.csv`. This file should contain the columns: `id`, `authors`, `title`, `categories`, `abstract`, `update_date`, and `authors_parsed`.

### Step 1: Topic Modeling

This step uses BERTopic to identify distinct research areas from the paper abstracts and titles.

-   **Script**: `BERTopic_analyzer.py`
-   **Purpose**: Processes the text of all mathematics papers to cluster them into semantically coherent topics.
-   **Input**: `data/cleaned/math_arxiv_snapshot.csv`
-   **Output**: A set of files in `results/topics/` identified by a timestamp, including:
    -   `document_topics_[timestamp].csv`: A mapping of each paper `id` to a `topic` number.
    -   `topic_info_[timestamp].csv`: A list of all identified topics and their representative keywords.
    -   `metadata_[timestamp].json`: A file containing metadata about the topic modeling run.
-   **Usage**:
    ```bash
    python BERTopic_analyzer.py --custom-csv data/cleaned/math_arxiv_snapshot.csv
    ```

### Step 2: Create Author-Topic Network Data

This is a manual data preparation step that combines the output from Step 1 with the author data.

-   **Action**: Create a file named `data/cleaned/author_topic_networks.csv`. This file should have three columns: `topic`, `id`, and `authors_parsed`, by joining the `document_topics_*.csv` (from Step 1) with `math_arxiv_snapshot.csv` on the paper `id`. **Crucially, papers assigned to the outlier topic (-1) should be excluded.**

### Step 3: Author Name Disambiguation

This step resolves author name variations (e.g., "J. Doe" vs. "John Doe") to ensure network accuracy.

-   **Script**: `author_disambiguation_v4.py`
-   **Purpose**: Implements a multi-stage, graph-based pipeline to create a canonical identity for each researcher, preventing both over-merging and under-merging of names.
-   **Input**: `data/cleaned/author_topic_networks.csv` (from Step 2)
-   **Output**:
    -   `data/cleaned/author_topic_networks_disambiguated_v4.csv`: The main output file with disambiguated author names.
    -   `results/disambiguation/`: Directory containing detailed logs, statistics, and a validation script for the disambiguation run.
-   **Usage**:
    ```bash
    python author_disambiguation_v4.py --data-path data/cleaned/author_topic_networks.csv
    ```

### Step 4: Network Metrics Calculation

For each topic, this step constructs a co-authorship network and computes ten structural metrics.

-   **Script**: `collaboration_network_analysis_v5.py`
-   **Purpose**: Builds a network for each of the 1,938 topics and calculates metrics for collaboration dynamics, topology, resilience, and researcher positioning. This script is foundational for all subsequent statistical analysis.
-   **Input**: `data/cleaned/author_topic_networks_disambiguated_v4.csv` (from Step 3)
-   **Output**:
    -   `results/collaboration_analysis/topic_analysis_10metrics_fixed_[timestamp].json`: The primary output, containing the ten computed metrics for every topic. This file is the main input for all subsequent analysis scripts.
    -   `results/collaboration_analysis/topic_summaries_10metrics_fixed_[timestamp].csv`: A CSV version of the above for easier inspection.
-   **Usage**:
    ```bash
    python collaboration_network_analysis_v5.py --data-path data/cleaned/author_topic_networks_disambiguated_v4.csv
    ```

### Step 5: Statistical Analysis and Comparison

This stage involves several scripts that analyze the metrics generated in Step 4.

#### 5a. Baseline Group Comparison (Popular vs. Niche)

-   **Script**: `analyze_popular_vs_niche.py`
-   **Purpose**: Performs the initial statistical comparison between popular (top 20%) and niche (bottom 20%) topics using non-parametric tests (Mann-Whitney U) and effect sizes (Cliff's Delta).
-   **Input**: `results/collaboration_analysis/topic_analysis_10metrics_fixed_[timestamp].json` (from Step 4)
-   **Output**:
    -   `results/collaboration_analysis/popular_vs_niche_analysis_[timestamp].json`: A detailed report of the statistical tests.
    -   `results/collaboration_analysis/topic_classifications_[timestamp].csv`: **A crucial file** that contains the metrics for each topic along with its 'popular' or 'niche' group assignment. This becomes an input for later regression and visualization scripts.
-   **Usage** (replace with your actual timestamped file):
    ```bash
    python analyze_popular_vs_niche.py --results-file results/collaboration_analysis/topic_analysis_10metrics_fixed_20250611_194954.json
    ```

#### 5b. Bootstrap Analysis for Confidence Intervals

-   **Script**: `bootstrap_CI_analysis.py`
-   **Purpose**: Provides more robust estimates of effect sizes and their uncertainty by constructing 95% confidence intervals for Cliff's Delta using 10,000 bootstrap resamples. This is a more rigorous alternative to the baseline comparison.
-   **Input**: `results/collaboration_analysis/topic_analysis_10metrics_fixed_[timestamp].json` (from Step 4)
-   **Output**: `results/collaboration_analysis/publication_ready_table_[timestamp].csv` and other detailed JSON/CSV reports.
-   **Usage**:
    ```bash
    python bootstrap_CI_analysis.py results/collaboration_analysis/topic_analysis_10metrics_fixed_20250611_194954.json
    ```

#### 5c. Regression Analysis (Controlling for Size)

-   **Script**: `fixed_regression_analysis.py`
-   **Purpose**: Disentangles the effects of topic popularity from network size by running OLS regression models. It uses standardized coefficients to provide interpretable effect sizes.
-   **Input**: `results/collaboration_analysis/topic_analysis_10metrics_fixed_[timestamp].json` (from Step 4)
-   **Output**: `results/regression_analysis_fixed/` containing a detailed JSON report and a publication-ready CSV table.
-   **Usage**:
    ```bash
    python fixed_regression_analysis.py
    ```
    *(Note: This script uses a hardcoded path in its `main` function; update if necessary.)*

#### 5d. Enhanced Regression (Testing Continuous Popularity)

-   **Script**: `enhanced_regression.py`
-   **Purpose**: A sensitivity check to ensure findings are not an artifact of the binary popular/niche classification. It compares the binary model against several continuous popularity measures (log-transformed, percentile, etc.).
-   **Input**: `results/collaboration_analysis/topic_classifications_[timestamp].csv` (from Step 5a)
-   **Output**: `results/regression_analysis_enhanced/` containing a detailed JSON report, a comparison table, and a visualization of model fits.
-   **Usage**:
    ```bash
    python enhanced_regression.py
    ```
    *(Note: This script also uses a hardcoded path in its `main` function; update if necessary.)*

### Step 6: Robustness and Sensitivity Analyses

These scripts validate the core assumptions of the pipeline.

-   **`sensitivity_analysis.py`**: Tests the sensitivity of the results to the 20% popular/niche cutoff by repeating the group comparison at 15%, 25%, and 30%.
-   **`bertopic_sensitivity_analysis.py`**: Tests the stability of the topic modeling step by varying key BERTopic hyperparameters and ensuring the topic structure remains consistent.
-   **`covid_temporal_sensitivity.py`**: An orchestrator script that splits the dataset into "peak pandemic" (2020-2021) and "post-peak" (2022-2025) periods and re-runs the entire pipeline on both to check for temporal stability of the findings.

### Step 7: Final Visualizations

This step generates the key figures for the manuscript.

-   **Script**: `enhanced_network_viz.py`
-   **Purpose**: Creates systematic, size-matched network visualizations comparing popular and niche topics to illustrate the core findings of the paper.
-   **Inputs**:
    1.  `results/collaboration_analysis/topic_classifications_[timestamp].csv` (from Step 5a)
    2.  `data/cleaned/author_topic_networks_disambiguated_v4.csv` (from Step 3)
-   **Output**: `figure_systematic_networks.png` and `figure_distribution_analysis.png` in the root directory.
-   **Usage**:
    ```bash
    python enhanced_network_viz.py
    ```
    *(Note: Update the hardcoded file paths inside the script's `main` function to match your timestamped files.)*
```
