# Makefile for the "Modular versus Hierarchical" analysis pipeline
# ----------------------------------------------------------------
# This file automates the entire research workflow using a true dependency
# graph with stamp files for robust, incremental builds.
#
# Author: Brian Hepler
#
# --- Usage ---
# make all         : Run the core pipeline, only rebuilding what's necessary.
# make clean       : Remove all generated files and stamps.
# make figures     : Ensure the final manuscript figures are up-to-date.
# make sensitivity : Run all validation and sensitivity analyses.
# make bootstrap   : Run the bootstrap CI analysis.

# --- Configuration ---
PYTHON = python3
CONFIG_FILE = config.yaml
STATIC_INPUT = data/cleaned/math_arxiv_snapshot.csv
SRC_FILES = $(wildcard src/*.py)

# Define the stamp files that represent the completion of each step.
STAMP_TOPICS = .stamp_topics
STAMP_NETWORK_DATA = .stamp_network_data
STAMP_DISAMBIGUATE = .stamp_disambiguate
STAMP_METRICS = .stamp_metrics
STAMP_COMPARE = .stamp_compare
STAMP_REGRESSION = .stamp_regression
STAMP_VISUALIZE = .stamp_visualize

# --- High-Level Targets ---
.PHONY: all clean figures sensitivity bootstrap

all: $(STAMP_REGRESSION)
	@echo "✅ Core pipeline is up-to-date."

figures: $(STAMP_VISUALIZE)
	@echo "✅ Manuscript figures are up-to-date."


# --- Pipeline Step Definitions (The Dependency Chain) ---

# Step 1: Topic Modeling
$(STAMP_TOPICS): $(STATIC_INPUT) src/BERTopic_analyzer.py src/config_manager.py $(CONFIG_FILE)
	@echo "\n--- Running Step 1: Topic Modeling with BERTopic ---"
	$(PYTHON) src/BERTopic_analyzer.py
	@touch $(STAMP_TOPICS)

# Step 2: Preparing Author-Topic Data
$(STAMP_NETWORK_DATA): $(STAMP_TOPICS) src/prepare_network_data.py src/config_manager.py $(CONFIG_FILE)
	@echo "\n--- Running Step 2: Preparing Author-Topic Network Data ---"
	$(PYTHON) src/prepare_network_data.py
	@touch $(STAMP_NETWORK_DATA)

# Step 3: Author Name Disambiguation
$(STAMP_DISAMBIGUATE): $(STAMP_NETWORK_DATA) src/author_disambiguation_v4.py src/config_manager.py $(CONFIG_FILE)
	@echo "\n--- Running Step 3: Author Name Disambiguation ---"
	$(PYTHON) src/author_disambiguation_v4.py
	@touch $(STAMP_DISAMBIGUATE)

# Step 4: Network Metrics Calculation
$(STAMP_METRICS): $(STAMP_DISAMBIGUATE) src/collaboration_network_analysis_v5.py src/config_manager.py $(CONFIG_FILE)
	@echo "\n--- Running Step 4: Calculating Network Metrics ---"
	$(PYTHON) src/collaboration_network_analysis_v5.py
	@touch $(STAMP_METRICS)

# Step 5a: Group Comparison (Popular vs. Niche)
$(STAMP_COMPARE): $(STAMP_METRICS) src/analyze_popular_vs_niche.py src/config_manager.py $(CONFIG_FILE)
	@echo "\n--- Running Step 5a: Comparing Popular vs. Niche Topics ---"
	$(PYTHON) src/analyze_popular_vs_niche.py
	@touch $(STAMP_COMPARE)

# Step 5c & 5d: Regression Analyses
$(STAMP_REGRESSION): $(STAMP_COMPARE) src/fixed_regression_analysis.py src/enhanced_regression.py src/config_manager.py $(CONFIG_FILE)
	@echo "\n--- Running Step 5c/5d: Regression Analyses ---"
	$(PYTHON) src/fixed_regression_analysis.py
	$(PYTHON) src/enhanced_regression.py
	@touch $(STAMP_REGRESSION)

# Step 7: Final Visualizations
# Note: Renamed from Step 6 to Step 7 to match original README.
$(STAMP_VISUALIZE): $(STAMP_COMPARE) $(STAMP_DISAMBIGUATE) src/enhanced_network_viz.py src/config_manager.py $(CONFIG_FILE)
	@echo "\n--- Running Step 7: Generating Manuscript Figures ---"
	$(PYTHON) src/enhanced_network_viz.py
	@touch $(STAMP_VISUALIZE)

# ==================================
# --- VALIDATION & EXTRA ANALYSES ---
# ==================================

.PHONY: sensitivity bootstrap

# A target to run all sensitivity analyses.
# It depends on the main data file and the final comparison step from the core pipeline.
sensitivity: $(STAMP_COMPARE) $(STATIC_INPUT)
	@echo "\n--- Running All Sensitivity and Validation Analyses ---"
	$(PYTHON) src/sensitivity_analysis.py
	$(PYTHON) src/bertopic_sensitivity_analysis.py
	$(PYTHON) src/covid_temporal_sensitivity.py --arxiv-data $(STATIC_INPUT)

# Step 5b: Run bootstrap analysis for more robust CIs.
# This depends on the metrics being calculated.
bootstrap: $(STAMP_METRICS)
	@echo "\n--- Running Step 5b: Bootstrap CI Analysis ---"
	$(PYTHON) src/bootstrap_CI_analysis.py

# ==================
# --- Housekeeping ---
# ==================

.PHONY: clean

clean:
	@echo "🔥 Cleaning up generated files and stamps..."
	rm -rf results/*
	rm -f data/cleaned/author_topic_networks.csv
	rm -f data/cleaned/author_topic_networks_disambiguated_v4.csv
	rm -f figure_*.png figure_*.pdf
	rm -f .stamp_*
	@echo "🧹 Workspace is clean."