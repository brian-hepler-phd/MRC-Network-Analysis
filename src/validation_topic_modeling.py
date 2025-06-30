#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

"""
BERTopic Parameter Sensitivity Analysis
--------------------------------------
This script runs comprehensive parameter sensitivity analysis for BERTopic
to address referee concerns about parameter selection robustness.

Usage:
    python bertopic_sensitivity_analysis.py --custom-csv data/cleaned/math_arxiv_snapshot.csv
"""

import os
import gc
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from itertools import product
from src.config_manager import ConfigManager

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sensitivity_analysis")

# Directories
RESULTS_DIR = Path("results")
VALIDATION_DIR = RESULTS_DIR / "validation"
VALIDATION_DIR.mkdir(exist_ok=True, parents=True)

# Stopwords (same as main script)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", 
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", 
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", 
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", 
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", 
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", 
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", 
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", 
    "you've", "your", "yours", "yourself", "yourselves",
    "prove", "paper", "show", "result", "consider", "using", "use", "given", "thus",
    "therefore", "hence", "obtain", "we", "our", "propose", "method", "approach",
    "introduce", "study", "analyze", "present", "develop", "data", "set", "model",
    "algorithm", "equation", "function", "theorem", "lemma", "define", "definition",
    "example", "problem", "solution", "property", "application"
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity analysis for BERTopic"
    )
    parser.add_argument(
        "--custom-csv",
        type=str,
        default="data/cleaned/math_arxiv_snapshot.csv",
        help="Path to the CSV file with pre-cleaned arXiv math data"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Sample size for sensitivity analysis (to speed up computation)"
    )
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Run extensive parameter grid (slower but more thorough)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def load_and_sample_data(csv_path, sample_size=None):
    """Load data and optionally sample for faster analysis."""
    logger.info(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} papers")
    
    # Create text field if needed
    if "text_for_nlp" not in df.columns and "title" in df.columns and "abstract" in df.columns:
        df["text_for_nlp"] = df["title"] + " " + df["abstract"]
    
    # Sample data if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {len(df)} papers for sensitivity analysis")
    
    return df


def build_topic_model(min_topic_size=15, umap_n_neighbors=15, umap_n_components=5, 
                     hdbscan_min_samples=None, umap_min_dist=0.1):
    """Build BERTopic model with specified parameters."""
    
    # Set default min_samples if not provided
    if hdbscan_min_samples is None:
        hdbscan_min_samples = max(2, min_topic_size // 3)
    
    vectorizer_model = CountVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.8
    )
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric="cosine",
        low_memory=False,
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        algorithm="best",
        core_dist_n_jobs=-1
    )
    
    representation_model = KeyBERTInspired()
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=False  # Reduce verbosity for batch runs
    )
    
    return topic_model


def get_parameter_grid(extensive=False):
    """Get parameter combinations to test."""
    
    if extensive:
        # Extensive grid - more thorough but slower
        param_grid = {
            "min_topic_size": [10, 15, 20, 25, 30],
            "umap_n_neighbors": [10, 15, 20, 30],
            "umap_n_components": [3, 5, 10, 15],
            "umap_min_dist": [0.05, 0.1, 0.2],
            "hdbscan_min_samples": [None, 5, 10]  # None means auto-calculated
        }
    else:
        # Focused grid - faster but still comprehensive
        param_grid = {
            "min_topic_size": [10, 15, 20, 25],
            "umap_n_neighbors": [10, 15, 20],
            "umap_n_components": [3, 5, 10],
            "umap_min_dist": [0.1],  # Keep default
            "hdbscan_min_samples": [None, 5]
        }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    logger.info(f"Generated {len(combinations)} parameter combinations")
    return combinations


def run_single_analysis(docs, params, run_id):
    """Run BERTopic analysis with single parameter combination."""
    
    try:
        logger.info(f"Run {run_id}: Testing params {params}")
        
        # Build and fit model
        model = build_topic_model(**params)
        topics, _ = model.fit_transform(docs)
        
        # Get basic statistics
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]
        
        results = {
            "run_id": run_id,
            "params": params,
            "success": True,
            "num_topics": len(valid_topics),
            "outlier_count": sum(1 for t in topics if t == -1),
            "outlier_ratio": sum(1 for t in topics if t == -1) / len(topics),
            "topics": topics,
            "topic_sizes": valid_topics["Count"].tolist(),
            "topic_size_stats": {
                "mean": float(valid_topics["Count"].mean()),
                "median": float(valid_topics["Count"].median()),
                "std": float(valid_topics["Count"].std()),
                "min": int(valid_topics["Count"].min()),
                "max": int(valid_topics["Count"].max())
            }
        }
        
        # Clean up model to save memory
        del model
        gc.collect()
        
        return results
        
    except Exception as e:
        logger.warning(f"Run {run_id} failed with params {params}: {str(e)}")
        return {
            "run_id": run_id,
            "params": params,
            "success": False,
            "error": str(e)
        }


def calculate_stability_metrics(results_list):
    """Calculate stability metrics across parameter combinations."""
    
    logger.info("Calculating stability metrics...")
    
    # Get successful results
    successful_results = [r for r in results_list if r["success"]]
    
    if len(successful_results) < 2:
        return {"error": "Not enough successful runs for stability analysis"}
    
    # Calculate topic assignment stability
    topic_assignments = [r["topics"] for r in successful_results]
    baseline_topics = topic_assignments[0]
    
    # ARI scores vs baseline
    ari_scores = []
    nmi_scores = []
    
    for i, topics in enumerate(topic_assignments[1:], 1):
        ari = adjusted_rand_score(baseline_topics, topics)
        nmi = normalized_mutual_info_score(baseline_topics, topics)
        ari_scores.append(ari)
        nmi_scores.append(nmi)
    
    # Topic count stability
    topic_counts = [r["num_topics"] for r in successful_results]
    outlier_ratios = [r["outlier_ratio"] for r in successful_results]
    
    stability_metrics = {
        "topic_assignment_stability": {
            "mean_ari": float(np.mean(ari_scores)),
            "std_ari": float(np.std(ari_scores)),
            "min_ari": float(np.min(ari_scores)),
            "max_ari": float(np.max(ari_scores)),
            "mean_nmi": float(np.mean(nmi_scores)),
            "std_nmi": float(np.std(nmi_scores))
        },
        "topic_count_stability": {
            "mean_count": float(np.mean(topic_counts)),
            "std_count": float(np.std(topic_counts)),
            "min_count": int(np.min(topic_counts)),
            "max_count": int(np.max(topic_counts)),
            "cv_count": float(np.std(topic_counts) / np.mean(topic_counts))  # Coefficient of variation
        },
        "outlier_ratio_stability": {
            "mean_ratio": float(np.mean(outlier_ratios)),
            "std_ratio": float(np.std(outlier_ratios)),
            "min_ratio": float(np.min(outlier_ratios)),
            "max_ratio": float(np.max(outlier_ratios))
        },
        "overall_stability_score": float(np.mean(ari_scores))  # Simple overall metric
    }
    
    return stability_metrics


def analyze_parameter_effects(results_list):
    """Analyze how different parameters affect outcomes."""
    
    logger.info("Analyzing parameter effects...")
    
    successful_results = [r for r in results_list if r["success"]]
    
    if len(successful_results) < 5:
        return {"error": "Not enough successful runs for parameter analysis"}
    
    # Convert to DataFrame for easier analysis
    data = []
    for result in successful_results:
        row = result["params"].copy()
        row.update({
            "num_topics": result["num_topics"],
            "outlier_ratio": result["outlier_ratio"],
            "mean_topic_size": result["topic_size_stats"]["mean"],
            "topic_size_std": result["topic_size_stats"]["std"]
        })
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate correlations
    numeric_cols = ["min_topic_size", "umap_n_neighbors", "umap_n_components", 
                   "num_topics", "outlier_ratio", "mean_topic_size", "topic_size_std"]
    
    # Handle None values in hdbscan_min_samples
    df["hdbscan_min_samples_numeric"] = df["hdbscan_min_samples"].fillna(0)
    numeric_cols.append("hdbscan_min_samples_numeric")
    
    correlation_matrix = df[numeric_cols].corr()
    
    # Parameter effects analysis
    param_effects = {}
    
    for param in ["min_topic_size", "umap_n_neighbors", "umap_n_components"]:
        param_effects[param] = {
            "correlation_with_num_topics": float(correlation_matrix.loc[param, "num_topics"]),
            "correlation_with_outlier_ratio": float(correlation_matrix.loc[param, "outlier_ratio"]),
            "correlation_with_mean_topic_size": float(correlation_matrix.loc[param, "mean_topic_size"])
        }
    
    # Find optimal parameter ranges
    # Define "good" results as those with reasonable topic counts and low outlier ratios
    median_topics = df["num_topics"].median()
    q75_outlier = df["outlier_ratio"].quantile(0.75)
    
    good_results = df[
        (df["num_topics"] >= median_topics * 0.8) & 
        (df["num_topics"] <= median_topics * 1.2) &
        (df["outlier_ratio"] <= q75_outlier)
    ]
    
    optimal_ranges = {}
    for param in ["min_topic_size", "umap_n_neighbors", "umap_n_components"]:
        if len(good_results) > 0:
            optimal_ranges[param] = {
                "min": int(good_results[param].min()),
                "max": int(good_results[param].max()),
                "mean": float(good_results[param].mean()),
                "recommended": int(good_results[param].median())
            }
    
    return {
        "parameter_correlations": correlation_matrix.to_dict(),
        "parameter_effects": param_effects,
        "optimal_parameter_ranges": optimal_ranges,
        "total_successful_runs": len(successful_results),
        "good_results_count": len(good_results)
    }


def run_sensitivity_analysis(docs, extensive=False):
    """Run complete sensitivity analysis."""
    
    logger.info("Starting parameter sensitivity analysis...")
    
    # Get parameter combinations
    param_combinations = get_parameter_grid(extensive=extensive)
    
    # Run analysis for each combination
    results = []
    total_runs = len(param_combinations)
    
    for i, params in enumerate(param_combinations):
        logger.info(f"Progress: {i+1}/{total_runs}")
        result = run_single_analysis(docs, params, i+1)
        results.append(result)
        
        # Clean up memory periodically
        if i % 10 == 0:
            gc.collect()
    
    # Calculate stability metrics
    stability_metrics = calculate_stability_metrics(results)
    
    # Analyze parameter effects
    parameter_analysis = analyze_parameter_effects(results)
    
    # Summary statistics
    successful_runs = [r for r in results if r["success"]]
    failed_runs = [r for r in results if not r["success"]]
    
    summary = {
        "total_runs": total_runs,
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "success_rate": len(successful_runs) / total_runs,
        "parameter_space_coverage": {
            "min_topic_size_range": list(set(r["params"]["min_topic_size"] for r in successful_runs)),
            "umap_n_neighbors_range": list(set(r["params"]["umap_n_neighbors"] for r in successful_runs)),
            "umap_n_components_range": list(set(r["params"]["umap_n_components"] for r in successful_runs))
        }
    }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "extensive" if extensive else "focused",
        "summary": summary,
        "stability_metrics": stability_metrics,
        "parameter_analysis": parameter_analysis,
        "detailed_results": results,
        "interpretation": generate_interpretation(stability_metrics, parameter_analysis, summary)
    }


def generate_interpretation(stability_metrics, parameter_analysis, summary):
    """Generate human-readable interpretation of results."""
    
    interpretation = {
        "overall_assessment": "",
        "key_findings": [],
        "recommendations": [],
        "concerns": []
    }
    
    # Overall stability assessment
    if "error" not in stability_metrics:
        stability_score = stability_metrics["overall_stability_score"]
        if stability_score > 0.8:
            interpretation["overall_assessment"] = "High stability - results are robust to parameter changes"
        elif stability_score > 0.6:
            interpretation["overall_assessment"] = "Moderate stability - some sensitivity to parameter choices"
        else:
            interpretation["overall_assessment"] = "Low stability - results are sensitive to parameter choices"
    
    # Key findings
    if "error" not in stability_metrics:
        ari_mean = stability_metrics["topic_assignment_stability"]["mean_ari"]
        topic_cv = stability_metrics["topic_count_stability"]["cv_count"]
        
        interpretation["key_findings"].append(f"Topic assignment stability (ARI): {ari_mean:.3f}")
        interpretation["key_findings"].append(f"Topic count variability (CV): {topic_cv:.3f}")
    
    if "error" not in parameter_analysis:
        # Find most influential parameters
        effects = parameter_analysis["parameter_effects"]
        most_influential = max(effects.keys(), 
                             key=lambda k: abs(effects[k]["correlation_with_num_topics"]))
        interpretation["key_findings"].append(f"Most influential parameter: {most_influential}")
    
    # Recommendations
    if "error" not in parameter_analysis and parameter_analysis["optimal_parameter_ranges"]:
        for param, ranges in parameter_analysis["optimal_parameter_ranges"].items():
            interpretation["recommendations"].append(
                f"{param}: recommended value {ranges['recommended']} (range: {ranges['min']}-{ranges['max']})"
            )
    
    # Concerns
    if summary["success_rate"] < 0.9:
        interpretation["concerns"].append(f"Low success rate: {summary['success_rate']:.2%} of parameter combinations failed")
    
    if "error" not in stability_metrics:
        if stability_metrics["topic_count_stability"]["cv_count"] > 0.3:
            interpretation["concerns"].append("High variability in topic counts across parameter settings")
        
        if stability_metrics["outlier_ratio_stability"]["max_ratio"] > 0.3:
            interpretation["concerns"].append("Some parameter combinations produce high outlier ratios")
    
    return interpretation


def main():
    """Main execution function."""
    args = parse_arguments()

    config = ConfigManager()
    args.custom_csv = config.get_path('cleaned_snapshot', section='static_inputs')
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load and prepare data
    df = load_and_sample_data(args.custom_csv, args.sample_size)
    docs = df["text_for_nlp"].tolist()
    
    # Run sensitivity analysis
    results = run_sensitivity_analysis(docs, extensive=args.extensive)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = VALIDATION_DIR / f"parameter_sensitivity_analysis_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Sensitivity analysis complete. Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    
    summary = results["summary"]
    print(f"Total parameter combinations tested: {summary['total_runs']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    
    if "error" not in results["stability_metrics"]:
        stability = results["stability_metrics"]
        print(f"\nStability Metrics:")
        print(f"  Topic assignment stability (ARI): {stability['topic_assignment_stability']['mean_ari']:.3f} ± {stability['topic_assignment_stability']['std_ari']:.3f}")
        print(f"  Topic count stability (CV): {stability['topic_count_stability']['cv_count']:.3f}")
        print(f"  Overall stability score: {stability['overall_stability_score']:.3f}")
    
    print(f"\nInterpretation:")
    interpretation = results["interpretation"]
    print(f"  Overall assessment: {interpretation['overall_assessment']}")
    
    if interpretation["key_findings"]:
        print(f"  Key findings:")
        for finding in interpretation["key_findings"]:
            print(f"    - {finding}")
    
    if interpretation["recommendations"]:
        print(f"  Recommendations:")
        for rec in interpretation["recommendations"]:
            print(f"    - {rec}")
    
    if interpretation["concerns"]:
        print(f"  Concerns:")
        for concern in interpretation["concerns"]:
            print(f"    - {concern}")
    
    print("="*60)


if __name__ == "__main__":
    main()