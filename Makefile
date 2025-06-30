# ====================================================================
# Makefile for the "Modular versus Hierarchical" Analysis Pipeline
# Author: Brian Hepler
# ====================================================================
# This file automates the entire research workflow. It uses stamp files
# for robust, incremental builds, only re-running steps whose
# dependencies have changed.

# --- Usage ---
# make all         : Run the core data processing and analysis pipeline.
# make figures     : Generate the final manuscript figures.
# make validation  : Run all validation and sensitivity analyses.
# make clean       : Remove all generated files and stamp files.

# --- Configuration ---
PYTHON = python3
CONFIG_FILE = config.yaml
STATIC_INPUT = data/cleaned/math_arxiv_snapshot.csv

# Define stamp files that represent the completion of each major step.
STAMP_TOPICS         = .stamp_1_topics_built
STAMP_PREPARE_DATA   = .stamp_2_data_prepared
STAMP_DISAMBIGUATE   = .stamp_3_authors_disambiguated
STAMP_METRICS        = .stamp_4_metrics_calculated
STAMP_COMPARISON     = .stamp_5a_groups_compared
STAMP_BOOTSTRAP      = .stamp_5b_effects_bootstrapped
STAMP_REGRESSION     = .stamp_5c_regression_run
STAMP_ENH_REGRESSION = .stamp_5d_enhanced_regression_run
STAMP_VISUALIZE      = .stamp_6_figures_generated

# --- High-Level Targets ---
.PHONY: all clean figures validation bootstrap

all: $(STAMP_REGRESSION) $(STAMP_ENH_REGRESSION)
	@echo "✅ Core pipeline is up-to-date."

figures: $(STAMP_VISUALIZE)
	@echo "✅ Manuscript figures are up-to-date."

validation:
	@echo "\n--- Running All Validation and Sensitivity Analyses ---"
	$(PYTHON) src/validation_cutoff_thresholds.py
	$(PYTHON) src/validation_topic_modeling.py
	$(PYTHON) src/validation_temporal.py

bootstrap: $(STAMP_BOOTSTRAP)

# --- Core Pipeline Step Definitions ---

# Step 1: Topic Modeling
$(STAMP_TOPICS): $(STATIC_INPUT) src/1_build_topics.py $(CONFIG_FILE)
	@echo "\n--- Running Step 1: Topic Modeling with BERTopic ---"
	$(PYTHON) src/1_build_topics.py
	@touch $@

# Step 2: Preparing Author-Topic Data
$(STAMP_PREPARE_DATA): $(STAMP_TOPICS) src/2_prepare_author_data.py $(CONFIG_FILE)
	@echo "\n--- Running Step 2: Preparing Author-Topic Network Data ---"
	$(PYTHON) src/2_prepare_author_data.py
	@touch $@

# Step 3: Author Name Disambiguation
$(STAMP_DISAMBIGUATE): $(STAMP_PREPARE_DATA) src/3_disambiguate_authors.py $(CONFIG_FILE)
	@echo "\n--- Running Step 3: Author Name Disambiguation ---"
	$(PYTHON) src/3_disambiguate_authors.py
	@touch $@

# Step 4: Network Metrics Calculation
$(STAMP_METRICS): $(STAMP_DISAMBIGUATE) src/4_calculate_network_metrics.py $(CONFIG_FILE)
	@echo "\n--- Running Step 4: Calculating Network Metrics ---"
	$(PYTHON) src/4_calculate_network_metrics.py
	@touch $@

# Step 5a: Group Comparison (Popular vs. Niche)
$(STAMP_COMPARISON): $(STAMP_METRICS) src/5a_compare_groups.py $(CONFIG_FILE)
	@echo "\n--- Running Step 5a: Comparing Popular vs. Niche Topics ---"
	$(PYTHON) src/5a_compare_groups.py
	@touch $@

# Step 5b: Bootstrap Analysis for Confidence Intervals
$(STAMP_BOOTSTRAP): $(STAMP_METRICS) src/5b_bootstrap_effects.py $(CONFIG_FILE)
	@echo "\n--- Running Step 5b: Bootstrap CI Analysis ---"
	$(PYTHON) src/5b_bootstrap_effects.py
	@touch $@

# Step 5c: Main Regression Analysis (Size Control)
$(STAMP_REGRESSION): $(STAMP_COMPARISON) src/5c_regression_size_control.py $(CONFIG_FILE)
	@echo "\n--- Running Step 5c: Main Regression Analysis ---"
	$(PYTHON) src/5c_regression_size_control.py
	@touch $@

# Step 5d: Enhanced Regression (Binary vs. Continuous)
$(STAMP_ENH_REGRESSION): $(STAMP_COMPARISON) src/5d_regression_binary_vs_continuous.py $(CONFIG_FILE)
	@echo "\n--- Running Step 5d: Enhanced Regression Analysis ---"
	$(PYTHON) src/5d_regression_binary_vs_continuous.py
	@touch $@

# Step 6: Final Visualizations
$(STAMP_VISUALIZE): $(STAMP_COMPARISON) $(STAMP_DISAMBIGUATE) src/6_generate_figures.py $(CONFIG_FILE)
	@echo "\n--- Running Step 6: Generating Manuscript Figures ---"
	$(PYTHON) src/6_generate_figures.py
	@touch $@

# --- Housekeeping ---
.PHONY: clean

clean:
	@echo "🔥 Cleaning up generated files and stamp files..."
	rm -rf results/*
	rm -f data/cleaned/author_topic_networks.csv
	rm -f data/cleaned/author_topic_networks_disambiguated_v4.csv
	rm -f figures/figure_*.png figures/figure_*.pdf
	rm -f .stamp_*
	@echo "🧹 Workspace is clean."