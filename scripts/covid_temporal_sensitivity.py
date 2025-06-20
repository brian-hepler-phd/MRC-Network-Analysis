#!/usr/bin/env python3
"""
COVID-19 Temporal Sensitivity Analysis Pipeline
==============================================

This script orchestrates the complete temporal sensitivity analysis to test
whether the core structural differences (modular vs. hierarchical) found in
the study are robust to the COVID-19 pandemic period.

Workflow:
1. Split math_arxiv_snapshot.csv into pandemic (2020-2021) and post-pandemic (2022-2025) periods
2. Run BERTopic analysis on each temporal dataset
3. Create author_topic_networks.csv for each period
4. Run author disambiguation on each period
5. Run collaboration network analysis on each disambiguated dataset
6. Run popular vs niche comparison for each period
7. Compare results across temporal periods

Usage:
    python covid_temporal_sensitivity_pipeline.py --arxiv-data data/cleaned/math_arxiv_snapshot.csv
"""

import pandas as pd
import numpy as np
import json
import subprocess
import shutil
import os
from pathlib import Path
from datetime import datetime
import logging
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class COVIDTemporalSensitivityPipeline:
    """Orchestrates the complete COVID-19 temporal sensitivity analysis."""
    
    def __init__(self, arxiv_data_path):
        self.arxiv_data_path = Path(arxiv_data_path)
        self.base_dir = Path("results/covid_sensitivity")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories for each period
        self.pandemic_dir = self.base_dir / "pandemic_2020_2021"
        self.post_pandemic_dir = self.base_dir / "post_pandemic_2022_2025"
        
        for dir_path in [self.pandemic_dir, self.post_pandemic_dir]:
            dir_path.mkdir(exist_ok=True)
            (dir_path / "data" / "cleaned").mkdir(parents=True, exist_ok=True)
            (dir_path / "results").mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def split_temporal_datasets(self):
        """Split the arXiv dataset into pandemic and post-pandemic periods."""
        logger.info("=" * 80)
        logger.info("STEP 1: Splitting dataset into temporal periods")
        logger.info("=" * 80)
        
        # Load the full dataset
        logger.info(f"Loading arXiv data from {self.arxiv_data_path}")
        df = pd.read_csv(self.arxiv_data_path)
        logger.info(f"Loaded {len(df)} papers")
        
        # Convert date columns
        date_col = 'update_date' if 'update_date' in df.columns else 'published_date'
        df['date'] = pd.to_datetime(df[date_col])
        
        # Split by date
        pandemic_mask = (df['date'] >= '2020-01-01') & (df['date'] <= '2021-12-31')
        post_pandemic_mask = (df['date'] >= '2022-01-01') & (df['date'] <= '2025-06-18')
        
        pandemic_df = df[pandemic_mask].copy()
        post_pandemic_df = df[post_pandemic_mask].copy()
        
        logger.info(f"Peak Pandemic (2020-2021): {len(pandemic_df)} papers")
        logger.info(f"Post-Peak (2022-2025): {len(post_pandemic_df)} papers")
        
        # Save the split datasets
        pandemic_path = self.pandemic_dir / "data" / "cleaned" / "math_arxiv_snapshot.csv"
        post_pandemic_path = self.post_pandemic_dir / "data" / "cleaned" / "math_arxiv_snapshot.csv"
        
        pandemic_df.to_csv(pandemic_path, index=False)
        post_pandemic_df.to_csv(post_pandemic_path, index=False)
        
        logger.info(f"Saved pandemic dataset to {pandemic_path}")
        logger.info(f"Saved post-pandemic dataset to {post_pandemic_path}")
        
        return pandemic_path, post_pandemic_path
    
    def run_bertopic_analysis(self, data_path, output_dir):
        """Run BERTopic analysis on a dataset."""
        logger.info(f"\nRunning BERTopic analysis on {data_path.name}")
        
        # Get absolute path to the script
        script_path = Path.cwd() / "BERTopic_analyzer.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--custom-csv", str(data_path),
            "--years", "5",  # Will analyze all data in the file
            "--min-topic-size", "15"
        ]
        
        # Set environment to save results in the output directory
        env = os.environ.copy()
        env['RESULTS_DIR'] = str(output_dir / "results")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logger.error(f"BERTopic analysis failed: {result.stderr}")
            raise RuntimeError("BERTopic analysis failed")
        
        logger.info("BERTopic analysis completed successfully")
        
        # BERTopic saves files with pattern: metadata_{timestamp}.json
        # Look for the most recent metadata file
        metadata_files = list(Path("results/topics").glob("metadata_*.json"))
        if metadata_files:
            # Get the most recent one
            latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
            
            # Copy all related files to our output directory
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            
            timestamp = metadata['timestamp']
            dest_dir = output_dir / "results" / "topics"
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all files with this timestamp
            for file_type, filename in metadata['file_references'].items():
                src_file = Path("results/topics") / filename
                if src_file.exists():
                    shutil.copy2(src_file, dest_dir / filename)
            
            # Also copy the metadata file
            shutil.copy2(latest_metadata, dest_dir / latest_metadata.name)
            
            return dest_dir / latest_metadata.name
        else:
            raise FileNotFoundError("No BERTopic results found")
    
    def create_author_topic_networks(self, arxiv_data_path, metadata_path, output_dir):
        """Create author_topic_networks.csv by combining arXiv data with topic assignments."""
        logger.info(f"\nCreating author_topic_networks.csv")
        
        # Load arXiv data
        arxiv_df = pd.read_csv(arxiv_data_path)
        
        # Load metadata to get file references
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load the document topics file
        doc_topics_file = metadata_path.parent / metadata['file_references']['document_topics']
        doc_topics_df = pd.read_csv(doc_topics_file)
        
        # CRITICAL: Exclude outlier topic (-1)
        doc_topics_df = doc_topics_df[doc_topics_df['topic'] != -1]
        logger.info(f"Excluded outlier topic (-1), keeping {len(doc_topics_df)} papers in valid topics")
        
        # Create mapping from paper ID to topic
        paper_topics = {}
        for idx, row in doc_topics_df.iterrows():
            paper_id = row['id']
            topic_id = row['topic']
            paper_topics[paper_id] = topic_id
        
        # Create author_topic_networks DataFrame
        author_topic_data = []
        for idx, row in arxiv_df.iterrows():
            if row['id'] in paper_topics:
                author_topic_data.append({
                    'topic': paper_topics[row['id']],
                    'id': row['id'],
                    'authors_parsed': row['authors_parsed']
                })
        
        author_topic_df = pd.DataFrame(author_topic_data)
        output_path = output_dir / "data" / "cleaned" / "author_topic_networks.csv"
        author_topic_df.to_csv(output_path, index=False)
        
        # Log topic distribution
        topic_counts = author_topic_df['topic'].value_counts()
        logger.info(f"Created author_topic_networks.csv with {len(author_topic_df)} entries")
        logger.info(f"Number of topics: {len(topic_counts)}")
        logger.info(f"Papers per topic - Mean: {topic_counts.mean():.1f}, Median: {topic_counts.median():.1f}")
        
        return output_path
    
    def run_author_disambiguation(self, data_path, output_dir):
        """Run author disambiguation on the dataset."""
        logger.info(f"\nRunning author disambiguation")
        
        # Get absolute path to the script
        script_path = Path.cwd() / "author_disambiguation_v4.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data-path", str(data_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Author disambiguation failed: {result.stderr}")
            raise RuntimeError("Author disambiguation failed")
        
        logger.info("Author disambiguation completed successfully")
        
        # The output file should be in the same directory with _disambiguated_v4 suffix
        disambiguated_path = data_path.parent / f"{data_path.stem}_disambiguated_v4.csv"
        if not disambiguated_path.exists():
            raise FileNotFoundError(f"Disambiguated file not found: {disambiguated_path}")
        
        return disambiguated_path
    
    def run_collaboration_network_analysis(self, data_path, output_dir):
        """Run collaboration network analysis."""
        logger.info(f"\nRunning collaboration network analysis")
        
        # Get absolute path to the script
        script_path = Path.cwd() / "collaboration_network_analysis_v5.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data-path", str(data_path),
            "--compare-popular-niche"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Collaboration network analysis failed: {result.stderr}")
            raise RuntimeError("Collaboration network analysis failed")
        
        logger.info("Collaboration network analysis completed successfully")
        
        # Find the results file - it's usually saved in the main results directory
        results_files = list(Path("results/collaboration_analysis").glob("topic_analysis_10metrics_*.json"))
        if results_files:
            # Get the most recent one
            latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
            # Copy to our output directory
            dest_path = output_dir / "results" / "collaboration_analysis" / latest_results.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_results, dest_path)
            return dest_path
        else:
            raise FileNotFoundError("No collaboration analysis results found")
    
    def run_popular_vs_niche_analysis(self, results_file, output_dir):
        """Run popular vs niche comparison analysis."""
        logger.info(f"\nRunning popular vs niche analysis")
        
        # Get absolute path to the script
        script_path = Path.cwd() / "analyze_popular_vs_niche.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--results-file", str(results_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Popular vs niche analysis failed: {result.stderr}")
            raise RuntimeError("Popular vs niche analysis failed")
        
        logger.info("Popular vs niche analysis completed successfully")
        
        # Find the results file - it's usually saved in the main results directory
        comparison_files = list(Path("results/collaboration_analysis").glob("popular_vs_niche_analysis_*.json"))
        if comparison_files:
            # Get the most recent one
            latest_comparison = max(comparison_files, key=lambda p: p.stat().st_mtime)
            # Copy to our output directory
            dest_path = output_dir / "results" / "collaboration_analysis" / latest_comparison.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_comparison, dest_path)
            return dest_path
        else:
            raise FileNotFoundError("No comparison results found")
    
    def compare_temporal_results(self, pandemic_results_path, post_pandemic_results_path):
        """Compare results across temporal periods."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPARING RESULTS ACROSS TEMPORAL PERIODS")
        logger.info("=" * 80)
        
        # Load both result sets
        with open(pandemic_results_path, 'r') as f:
            pandemic_results = json.load(f)
        
        with open(post_pandemic_results_path, 'r') as f:
            post_pandemic_results = json.load(f)
        
        # Extract test results for core metrics
        pandemic_tests = pandemic_results['detailed_results']['statistical_tests']
        post_pandemic_tests = post_pandemic_results['detailed_results']['statistical_tests']
        
        # Core metrics to compare
        core_metrics = [
            'modularity',
            'coreness_ratio',
            'avg_constraint',
            'avg_effective_size',
            'small_world_coefficient',
            'degree_centralization',
            'degree_assortativity',
            'robustness_ratio'
        ]
        
        # Create comparison table
        comparison_data = []
        consistent_count = 0
        
        print("\n" + "="*100)
        print("COVID-19 TEMPORAL SENSITIVITY ANALYSIS RESULTS")
        print("="*100)
        print("\n{:<30} {:>20} {:>20} {:>15}".format(
            "Metric", "2020-2021", "2022-2025", "Consistent?"
        ))
        print("-"*100)
        
        for metric in core_metrics:
            if metric in pandemic_tests and metric in post_pandemic_tests:
                p1 = pandemic_tests[metric]
                p2 = post_pandemic_tests[metric]
                
                # Extract Cliff's Delta and significance
                delta1 = p1['effect_sizes']['cliffs_delta']
                sig1 = p1['significant_005']
                dir1 = p1['direction']
                
                delta2 = p2['effect_sizes']['cliffs_delta']
                sig2 = p2['significant_005']
                dir2 = p2['direction']
                
                # Check consistency
                consistent = (dir1 == dir2) and sig1 and sig2
                if consistent:
                    consistent_count += 1
                
                # Format results
                result1 = f"δ={delta1:+.3f} ({dir1.split('_')[0]})"
                result2 = f"δ={delta2:+.3f} ({dir2.split('_')[0]})"
                consistency = "**YES**" if consistent else "No"
                
                print("{:<30} {:>20} {:>20} {:>15}".format(
                    metric.replace('_', ' ').title(),
                    result1,
                    result2,
                    consistency
                ))
                
                comparison_data.append({
                    'metric': metric,
                    'pandemic_delta': delta1,
                    'pandemic_direction': dir1,
                    'pandemic_significant': sig1,
                    'post_pandemic_delta': delta2,
                    'post_pandemic_direction': dir2,
                    'post_pandemic_significant': sig2,
                    'consistent': consistent
                })
        
        print("-"*100)
        print(f"\nOVERALL CONSISTENCY: {consistent_count}/{len(core_metrics)} metrics show consistent patterns")
        
        # Save detailed comparison
        comparison_output = {
            'timestamp': self.timestamp,
            'pandemic_period': '2020-2021',
            'post_pandemic_period': '2022-2025',
            'metrics_compared': core_metrics,
            'detailed_comparison': comparison_data,
            'summary': {
                'total_metrics': len(core_metrics),
                'consistent_metrics': consistent_count,
                'consistency_rate': consistent_count / len(core_metrics),
                'conclusion': 'ROBUST' if consistent_count >= len(core_metrics) * 0.7 else 'SENSITIVE'
            }
        }
        
        output_path = self.base_dir / f"temporal_comparison_{self.timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(comparison_output, f, indent=2)
        
        logger.info(f"\nDetailed comparison saved to: {output_path}")
        
        # Generate manuscript-ready table
        self.generate_manuscript_table(comparison_data)
        
        return comparison_output
    
    def generate_manuscript_table(self, comparison_data):
        """Generate LaTeX table for manuscript with enhanced reporting."""
        logger.info("\nGenerating manuscript-ready table...")
        
        latex_table = """
\\begin{table}[ht]
\\centering
\\caption{Robustness of Baseline Findings to COVID-19 Temporal Effects}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Metric} & \\multicolumn{2}{c}{\\textbf{2020-2021}} & \\multicolumn{2}{c}{\\textbf{2022-2025}} & \\textbf{Consistent?}$^a$ \\\\
& Cliff's $\\delta$ & 95\\% CI & Cliff's $\\delta$ & 95\\% CI & \\\\
\\hline
"""
        
        for data in comparison_data:
            metric_name = data['metric'].replace('_', ' ').title()
            delta1 = f"{data['pandemic_delta']:.3f}"
            delta2 = f"{data['post_pandemic_delta']:.3f}"
            
            # Format consistency
            if data['consistent']:
                consistent = "\\textbf{Yes}"
            else:
                consistent = "No"
            
            # Note: CIs would need to be calculated from the original data
            latex_table += f"{metric_name} & {delta1} & [--] & {delta2} & [--] & {consistent} \\\\\n"
        
        latex_table += """\\hline
\\multicolumn{6}{l}{\\footnotesize $^a$Consistency requires statistical significance (p < 0.05) and same direction in both periods.}
\\end{tabular}
\\label{tab:covid_sensitivity}
\\end{table}
"""
        
        output_path = self.base_dir / f"manuscript_table_{self.timestamp}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"LaTeX table saved to: {output_path}")
        
        # Also create a visual summary
        self.create_visual_summary(comparison_data)
    
    def create_visual_summary(self, comparison_data):
        """Create a visual comparison of effect sizes across periods."""
        logger.info("\nCreating visual summary...")
        
        # Prepare data for plotting
        metrics = []
        pandemic_effects = []
        post_pandemic_effects = []
        
        for data in comparison_data:
            # Only include metrics that are significant in both periods
            if data.get('pandemic_significant', True) and data.get('post_pandemic_significant', True):
                metrics.append(data['metric'].replace('_', ' ').title())
                pandemic_effects.append(data['pandemic_delta'])
                post_pandemic_effects.append(data['post_pandemic_delta'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, pandemic_effects, width, label='2020-2021', alpha=0.8)
        bars2 = ax.bar(x + width/2, post_pandemic_effects, width, label='2022-2025', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Network Metrics', fontsize=12)
        ax.set_ylabel("Cliff's Delta (Effect Size)", fontsize=12)
        ax.set_title('COVID-19 Temporal Sensitivity: Effect Sizes Across Periods', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal lines for effect size thresholds
        ax.axhline(y=0.147, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axhline(y=-0.147, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.text(len(metrics)-0.5, 0.15, 'Small', fontsize=8, color='gray')
        
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axhline(y=-0.33, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.text(len(metrics)-0.5, 0.34, 'Medium', fontsize=8, color='gray')
        
        ax.axhline(y=0.474, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axhline(y=-0.474, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.text(len(metrics)-0.5, 0.48, 'Large', fontsize=8, color='gray')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.base_dir / f"temporal_sensitivity_plot_{self.timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visual summary saved to: {fig_path}")
        
        # Also create a visual summary
        self.create_visual_summary(comparison_data)
    
    def run_complete_pipeline(self):
        """Run the complete COVID-19 temporal sensitivity analysis pipeline."""
        logger.info("="*80)
        logger.info("COVID-19 TEMPORAL SENSITIVITY ANALYSIS PIPELINE")
        logger.info("="*80)
        logger.info(f"Started at: {datetime.now()}")
        
        try:
            # Step 1: Split temporal datasets
            pandemic_data, post_pandemic_data = self.split_temporal_datasets()
            
            # Process each period
            periods = [
                ("Pandemic (2020-2021)", pandemic_data, self.pandemic_dir),
                ("Post-Pandemic (2022-2025)", post_pandemic_data, self.post_pandemic_dir)
            ]
            
            results = {}
            
            for period_name, data_path, output_dir in periods:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {period_name}")
                logger.info(f"{'='*60}")
                
                # Step 2: Run BERTopic
                metadata_path = self.run_bertopic_analysis(data_path, output_dir)
                
                # Step 3: Create author_topic_networks.csv
                author_topic_path = self.create_author_topic_networks(
                    data_path, metadata_path, output_dir
                )
                
                # Step 4: Run author disambiguation
                disambiguated_path = self.run_author_disambiguation(
                    author_topic_path, output_dir
                )
                
                # Step 5: Run collaboration network analysis
                network_results = self.run_collaboration_network_analysis(
                    disambiguated_path, output_dir
                )
                
                # Step 6: Run popular vs niche analysis
                comparison_results = self.run_popular_vs_niche_analysis(
                    network_results, output_dir
                )
                
                results[period_name] = comparison_results
            
            # Step 7: Compare results across periods
            comparison = self.compare_temporal_results(
                results["Pandemic (2020-2021)"],
                results["Post-Pandemic (2022-2025)"]
            )
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Completed at: {datetime.now()}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="COVID-19 Temporal Sensitivity Analysis Pipeline"
    )
    parser.add_argument(
        "--arxiv-data",
        default="data/cleaned/math_arxiv_snapshot.csv",
        help="Path to the full arXiv dataset"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    pipeline = COVIDTemporalSensitivityPipeline(args.arxiv_data)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
