# Makefile for the "Modular versus Hierarchical" analysis pipeline
# ----------------------------------------------------------------
# This file automates the entire research workflow, from topic modeling
# to final regression analysis and visualization.
#
# Author: Brian Hepler
#
# --- Usage ---
# make all         : Run the core pipeline from start to finish.
# make clean       : Remove all generated files from the results/ directory.
# make topics      : Run only the topic modeling step.
# make visualize   : Generate the final manuscript figures.
# make sensitivity : Run all validation and sensitivity analyses.

# Define the Python interpreter to use
PYTHON = python3

# Define phony targets (targets that aren't files)
.PHONY: all clean topics network_data disambiguate metrics compare regression bootstrap visualize sensitivity

# ============================
# --- CORE PIPELINE TARGETS ---
# ============================

# The default target: runs the main analysis for the paper.
all: regression visualize
	@echo "✅ Core pipeline completed successfully."

# Step 1: Run BERTopic to identify topics.
topics:
	@echo "\n--- Running Step 1: Topic Modeling with BERTopic ---"
	$(PYTHON) src/BERTopic_analyzer.py

# Step 2: Prepare the author-topic network file.
network_data: topics
	@echo "\n--- Running Step 2: Preparing Author-Topic Network Data ---"
	$(PYTHON) src/prepare_network_data.py

# Step 3: Run author name disambiguation.
disambiguate: network_data
	@echo "\n--- Running Step 3: Author Name Disambiguation ---"
	$(PYTHON) src/author_disambiguation_v4.py

# Step 4: Calculate the 10 network metrics for each topic.
metrics: disambiguate
	@echo "\n--- Running Step 4: Calculating Network Metrics ---"
	$(PYTHON) src/collaboration_network_analysis_v5.py

# Step 5a: Run baseline popular vs. niche group comparison.
# This creates the topic_classifications.csv file needed for regression.
compare: metrics
	@echo "\n--- Running Step 5a: Comparing Popular vs. Niche Topics ---"
	$(PYTHON) src/analyze_popular_vs_niche.py

# Step 5c & 5d: Run the main regression analyses.
# This depends on the 'compare' step which generates its input file.
regression: compare
	@echo "\n--- Running Step 5c/5d: Regression Analyses (Fixed and Enhanced) ---"
	$(PYTHON) src/fixed_regression_analysis.py
	$(PYTHON) src/enhanced_regression.py

# Step 7: Generate the final manuscript figures.
# Depends on 'disambiguate' for network data and 'compare' for classifications.
visualize: compare disambiguate
	@echo "\n--- Running Step 7: Generating Manuscript Figures ---"
	$(PYTHON) src/enhanced_network_viz.py


# ==================================
# --- VALIDATION & EXTRA ANALYSES ---
# ==================================

# A target to run all sensitivity analyses.
sensitivity:
	@echo "\n--- Running All Sensitivity and Validation Analyses ---"
	$(PYTHON) src/sensitivity_analysis.py
	$(PYTHON) src/bertopic_sensitivity_analysis.py
	$(PYTHON) src/covid_temporal_sensitivity.py

# Step 5b: Run bootstrap analysis for more robust CIs.
bootstrap: metrics
	@echo "\n--- Running Step 5b: Bootstrap CI Analysis ---"
	$(PYTHON) src/bootstrap_CI_analysis.py


# ==================
# --- HOUSEKEEPING ---
# ==================

# Clean up all generated files to start fresh.
clean:
	@echo "🔥 Cleaning up generated files..."
	rm -rf results/*
	# This find command is safer than 'rm data/cleaned/*'
	find data/cleaned -type f ! -name 'math_arxiv_snapshot.csv' -delete
	rm -f figure_*.png figure_*.pdf
	@echo "🧹 Workspace is clean."