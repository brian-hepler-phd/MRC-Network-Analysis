#!/usr/bin/env python3
"""
BERTopic Analysis for Math Research Compass
------------------------------------------
This script performs topic modeling analysis on arXiv papers:
1. Loads and processes pre-cleaned arXiv data from CSV
2. Analyzes topics using BERTopic
3. Saves topic modeling results for downstream analysis

Usage:
    python BERTopic_analyzer.py --custom-csv data/cleaned/math_arxiv_snapshot.csv
"""

import os
import gc
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Import BERTopic and dependencies
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
logger = logging.getLogger("topic_analyzer")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set number of threads for different libraries
os.environ["OMP_NUM_THREADS"] = "8"          # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "8"     # OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "8"          # MKL threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"   # Accelerate threads
os.environ["NUMEXPR_NUM_THREADS"] = "8"      # NumExpr threads

# Constants
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
TOPIC_DIR = RESULTS_DIR / "topics"
for dir_path in [DATA_DIR, RESULTS_DIR, TOPIC_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Common English stopwords plus math/research specific ones
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
    # Research/math specific stopwords
    "prove", "paper", "show", "result", "consider", "using", "use", "given", "thus",
    "therefore", "hence", "obtain", "we", "our", "propose", "method", "approach",
    "introduce", "study", "analyze", "present", "develop", "data", "set", "model",
    "algorithm", "equation", "function", "theorem", "lemma", "define", "definition",
    "example", "problem", "solution", "property", "application"
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze topics in arXiv math papers using BERTopic"
    )
    parser.add_argument(
        "--custom-csv",
        type=str,
        default="data/cleaned/math_arxiv_snapshot.csv",
        help="Path to the CSV file with pre-cleaned arXiv math data"
    )
    parser.add_argument(
        "--categories", 
        nargs="+", 
        default=None,
        help="Optional: Specific math categories to filter from the CSV (e.g., math.AG math.AT). If not specified, all math categories will be analyzed."
    )
    parser.add_argument(
        "--years", 
        type=int, 
        default=5,
        help="Number of years of data to analyze"
    )
    parser.add_argument(
        "--min-topic-size", 
        type=int, 
        default=15,
        help="Minimum cluster size for topics"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def load_custom_csv(csv_path):
    """
    Load a pre-cleaned CSV file with arXiv papers.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with paper data
    """
    logger.info(f"Loading custom CSV file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} papers from {csv_path}")
        
        # Convert date columns to datetime
        if "update_date" in df.columns:
            df["updated_date"] = pd.to_datetime(df["update_date"])
            
        # We need a 'published_date' for filtering
        if "published_date" not in df.columns and "updated_date" in df.columns:
            df["published_date"] = df["updated_date"]
        
        # Create year field for filtering
        if "published_date" in df.columns:
            df["year"] = pd.to_datetime(df["published_date"]).dt.year
        
        # Create text field for NLP
        if "text_for_nlp" not in df.columns and "title" in df.columns and "abstract" in df.columns:
            df["text_for_nlp"] = df["title"] + " " + df["abstract"]
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()


def filter_recent_years(df, years):
    """Filter dataframe to only include papers from the last N years."""
    if df.empty or "year" not in df.columns:
        return df
        
    current_year = datetime.now().year
    start_year = current_year - years
    
    filtered_df = df[df["year"] >= start_year].copy()
    logger.info(f"Filtered to {len(filtered_df)} papers from {start_year}-{current_year}")
    
    return filtered_df


def build_advanced_topic_model(
    min_topic_size=15,
    n_gram_range=(1, 3)
):
    """
    Build an enhanced BERTopic model with improved parameters for large datasets.
    """
    # Set up the vectorizer with n-grams
    vectorizer_model = CountVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=n_gram_range,
        min_df=5,
        max_df=0.8
    )
    
    # Use a better-tuned embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # UMAP - Optimized for large datasets
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,        # Reduced from 10 for faster processing
        min_dist=0.1,          # Slightly increased for faster convergence
        metric="cosine",
        low_memory=False,      # Faster but uses more memory
        random_state=42
    )
    
    # HDBSCAN - Optimized for large datasets
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        algorithm="best",      # Let HDBSCAN choose the fastest algorithm
        core_dist_n_jobs=-1    # Use all CPU cores
    )
    
    # Create custom representation model
    representation_model = KeyBERTInspired()
    
    # Create and return the model with optimized settings
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=False,  # Set to False for better performance
        verbose=True
    )
    
    return topic_model


def run_topic_modeling(
    df, 
    min_topic_size=15
):
    """
    Run topic modeling on the dataset.
    
    Args:
        df: DataFrame with preprocessed paper data
        min_topic_size: Minimum size of topics
    
    Returns:
        Tuple of (fitted topic model, document topics, document topic probabilities)
    """
    logger.info("Building topic model...")
    model = build_advanced_topic_model(min_topic_size=min_topic_size)
    
    # Prepare documents
    docs = df["text_for_nlp"].tolist()
    logger.info(f"Running topic modeling on {len(docs)} documents")
    
    # Fit the model
    topics, probs = model.fit_transform(docs)
    
    # Log topic statistics
    topic_info = model.get_topic_info()
    num_topics = len(topic_info[topic_info["Topic"] != -1])
    logger.info(f"Found {num_topics} topics (excluding outliers)")
    
    return model, topics, probs


def save_topic_data(model, df, topics):
    """
    Save topic modeling results for later use in the dashboard.
    
    Args:
        model: Fitted BERTopic model
        df: DataFrame with paper data
        topics: Document topic assignments
    """
    logger.info("Saving topic modeling results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save topic info
    topic_info = model.get_topic_info()
    topic_info.to_csv(TOPIC_DIR / f"topic_info_{timestamp}.csv", index=False)
    
    # 2. Save topic keywords (top terms for each topic)
    keywords = {}
    for topic_id in topic_info["Topic"]:
        if topic_id != -1:  # Skip outlier topic
            keywords[topic_id] = model.get_topic(topic_id)
    
    with open(TOPIC_DIR / f"topic_keywords_{timestamp}.json", "w") as f:
        # Convert NumPy values to Python native types for JSON serialization
        serializable_keywords = {}
        for topic_id, terms in keywords.items():
            serializable_keywords[int(topic_id)] = [
                [term, float(weight)] for term, weight in terms
            ]
        json.dump(serializable_keywords, f)
    
    # 3. Save document-topic mappings with flexible column handling
    doc_topics_dict = {
        "id": df["id"].tolist() if "id" in df.columns else range(len(df)),
        "title": df["title"].tolist(),
        "topic": topics
    }
    
    # Add published_date if available
    if "published_date" in df.columns:
        doc_topics_dict["published_date"] = df["published_date"].dt.strftime("%Y-%m-%d").tolist()
    else:
        doc_topics_dict["published_date"] = ["2023-01-01"] * len(df)  # Default date
    
    # Add primary_category if available
    if "primary_category" in df.columns:
        doc_topics_dict["primary_category"] = df["primary_category"].tolist()
    elif "categories" in df.columns:
        # Use the first category from categories if it exists
        doc_topics_dict["primary_category"] = df["categories"].apply(
            lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "unknown"
        ).tolist()
    else:
        doc_topics_dict["primary_category"] = ["math.XX"] * len(df)  # Default category
    
    # Create and save the document-topic dataframe
    doc_topics = pd.DataFrame(doc_topics_dict)
    doc_topics.to_csv(TOPIC_DIR / f"document_topics_{timestamp}.csv", index=False)
    
    # 4. Create a metadata file with information about this run
    subjects = df["primary_category"].unique().tolist() if "primary_category" in df.columns else ["math.XX"]
    year_range = [int(df["year"].min()), int(df["year"].max())] if "year" in df.columns else [2020, 2023]
    
    metadata = {
        "timestamp": timestamp,
        "num_documents": len(df),
        "num_topics": len(topic_info[topic_info["Topic"] != -1]),
        "subjects": subjects,
        "year_range": year_range,
        "file_references": {
            "topic_info": f"topic_info_{timestamp}.csv",
            "topic_keywords": f"topic_keywords_{timestamp}.json",
            "document_topics": f"document_topics_{timestamp}.csv",
        }
    }
    
    with open(TOPIC_DIR / f"metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f)
    
    logger.info(f"All topic data saved with timestamp: {timestamp}")


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load data from the custom CSV
    df = load_custom_csv(args.custom_csv)
    
    if df.empty:
        logger.error("No data available for analysis. Exiting.")
        return
    
    # Filter by specific categories if requested
    if args.categories:
        logger.info(f"Filtering for specific categories: {', '.join(args.categories)}")
        # Create a regex pattern to match any of the specified categories
        pattern = '|'.join(args.categories)
        df = df[df['categories'].str.contains(pattern, na=False)]
        logger.info(f"After filtering: {len(df)} papers")
    else:
        logger.info(f"Analyzing all math categories in the dataset: {len(df)} papers")
    
    # Filter to recent years
    df_recent = filter_recent_years(df, args.years)
    
    if df_recent.empty:
        logger.error(f"No data found for the last {args.years} years. Exiting.")
        return
    
    # Clear memory before heavy computation
    del df
    gc.collect()
    
    # Run topic modeling
    model, topics, probs = run_topic_modeling(df_recent, min_topic_size=args.min_topic_size)
    
    # Save topic data for dashboard
    save_topic_data(model, df_recent, topics)
    
    logger.info("Topic analysis complete!")


if __name__ == "__main__":
    main()