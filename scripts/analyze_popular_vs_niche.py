#!/usr/bin/env python3
"""
Popular vs Niche Topic Comparison Analysis
==========================================

Analyzes pre-computed collaboration network metrics to compare popular vs niche topics.
Takes results from collaboration_network_analysis_v3.py and performs statistical comparison.

Usage:
    python analyze_popular_vs_niche.py --results-file topic_analysis_10metrics_20250611_194954.json
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PopularNicheAnalyzer:
    """
    Analyzes differences between popular and niche topics using pre-computed metrics.
    """
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results_dir = Path("results/collaboration_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # The 10 core metrics we're comparing
        self.core_metrics = [
            'collaboration_rate',
            'repeated_collaboration_rate', 
            'degree_centralization',
            'degree_assortativity',
            'modularity',
            'small_world_coefficient',
            'coreness_ratio',
            'robustness_ratio',
            'avg_constraint',
            'avg_effective_size'
        ]
        
        logger.info(f"Initialized PopularNicheAnalyzer with results: {self.results_file}")
    
    def load_results(self) -> pd.DataFrame:
        """Load the pre-computed topic analysis results."""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        logger.info("Loading pre-computed topic analysis results...")
        
        with open(self.results_file, 'r') as f:
            topic_data = json.load(f)
        
        # Convert to DataFrame
        topic_list = []
        for topic_id, data in topic_data.items():
            topic_record = {
                'topic_id': int(topic_id),
                'total_papers': data['total_papers'],
                'collaboration_papers': data['collaboration_papers']
            }
            
            # Add the 10 core metrics
            for metric in self.core_metrics:
                topic_record[metric] = data.get(metric, 0.0)
            
            topic_list.append(topic_record)
        
        df = pd.DataFrame(topic_list)
        logger.info(f"Loaded {len(df)} topics with {len(self.core_metrics)} metrics each")
        
        return df
    
    def identify_popular_niche_topics(self, df: pd.DataFrame, percentile_cutoff: float = 0.2) -> tuple:
        """
        Identify popular and niche topics based on total paper count.
        
        Args:
            df: DataFrame with topic data
            percentile_cutoff: Percentage for top/bottom groups (0.2 = 20%)
            
        Returns:
            Tuple of (popular_df, niche_df)
        """
        logger.info(f"Identifying popular and niche topics using {percentile_cutoff*100}% cutoff...")
        
        # Sort by total papers
        df_sorted = df.sort_values('total_papers', ascending=False)
        
        # Calculate cutoff indices
        n_topics = len(df_sorted)
        cutoff_size = int(n_topics * percentile_cutoff)
        
        # Extract popular and niche groups
        popular_df = df_sorted.head(cutoff_size).copy()
        niche_df = df_sorted.tail(cutoff_size).copy()
        
        logger.info(f"Popular topics: {len(popular_df)} topics (papers: {popular_df['total_papers'].min()}-{popular_df['total_papers'].max()})")
        logger.info(f"Niche topics: {len(niche_df)} topics (papers: {niche_df['total_papers'].min()}-{niche_df['total_papers'].max()})")
        
        return popular_df, niche_df
    
    def calculate_descriptive_statistics(self, popular_df: pd.DataFrame, niche_df: pd.DataFrame) -> dict:
        """Calculate descriptive statistics for both groups."""
        logger.info("Calculating descriptive statistics...")
        
        stats_results = {}
        
        for metric in self.core_metrics:
            pop_values = popular_df[metric].dropna()
            niche_values = niche_df[metric].dropna()
            
            stats_results[metric] = {
                'popular': {
                    'count': len(pop_values),
                    'mean': float(pop_values.mean()),
                    'std': float(pop_values.std()),
                    'median': float(pop_values.median()),
                    'min': float(pop_values.min()),
                    'max': float(pop_values.max()),
                    'q25': float(pop_values.quantile(0.25)),
                    'q75': float(pop_values.quantile(0.75))
                },
                'niche': {
                    'count': len(niche_values),
                    'mean': float(niche_values.mean()),
                    'std': float(niche_values.std()),
                    'median': float(niche_values.median()),
                    'min': float(niche_values.min()),
                    'max': float(niche_values.max()),
                    'q25': float(niche_values.quantile(0.25)),
                    'q75': float(niche_values.quantile(0.75))
                }
            }
        
        return stats_results
    
    def perform_statistical_tests(self, popular_df: pd.DataFrame, niche_df: pd.DataFrame) -> dict:
        """Perform statistical tests comparing popular vs niche topics."""
        logger.info("Performing statistical tests...")
        
        test_results = {}
        
        for metric in self.core_metrics:
            pop_values = popular_df[metric].dropna().values
            niche_values = niche_df[metric].dropna().values
            
            if len(pop_values) < 3 or len(niche_values) < 3:
                test_results[metric] = {
                    'test_used': 'insufficient_data',
                    'error': f'Insufficient data: popular n={len(pop_values)}, niche n={len(niche_values)}'
                }
                continue
            
            try:
                # Test for normality (only for small samples, assume non-normal for large)
                if len(pop_values) <= 5000 and len(niche_values) <= 5000:
                    pop_shapiro = stats.shapiro(pop_values)
                    niche_shapiro = stats.shapiro(niche_values)
                    normal_pop = pop_shapiro.pvalue > 0.05
                    normal_niche = niche_shapiro.pvalue > 0.05
                else:
                    normal_pop = False
                    normal_niche = False
                    pop_shapiro = (None, 0.01)
                    niche_shapiro = (None, 0.01)
                
                # Choose appropriate test
                if normal_pop and normal_niche:
                    # Both normal - use t-test
                    statistic, p_value = stats.ttest_ind(pop_values, niche_values)
                    test_used = 'independent_t_test'
                else:
                    # Non-normal - use Mann-Whitney U
                    statistic, p_value = stats.mannwhitneyu(pop_values, niche_values, alternative='two-sided')
                    test_used = 'mann_whitney_u'
                
                # Calculate effect sizes
                mean_diff = np.mean(pop_values) - np.mean(niche_values)
                
                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(pop_values) - 1) * np.var(pop_values, ddof=1) + 
                     (len(niche_values) - 1) * np.var(niche_values, ddof=1)) / 
                    (len(pop_values) + len(niche_values) - 2)
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                # Cliff's delta (non-parametric effect size)
                cliffs_delta = self._calculate_cliffs_delta(pop_values, niche_values)
                
                # Interpret effect sizes
                cohens_d_interpretation = self._interpret_cohens_d(cohens_d)
                cliffs_delta_interpretation = self._interpret_cliffs_delta(cliffs_delta)
                
                test_results[metric] = {
                    'test_used': test_used,
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant_005': p_value < 0.05,
                    'significant_001': p_value < 0.01,
                    'significant_bonferroni': p_value < (0.05 / len(self.core_metrics)),
                    'normality_tests': {
                        'popular_shapiro_p': float(pop_shapiro[1]) if pop_shapiro[1] is not None else None,
                        'niche_shapiro_p': float(niche_shapiro[1]) if niche_shapiro[1] is not None else None,
                        'both_normal': normal_pop and normal_niche
                    },
                    'effect_sizes': {
                        'cohens_d': float(cohens_d),
                        'cohens_d_interpretation': cohens_d_interpretation,
                        'cliffs_delta': float(cliffs_delta),
                        'cliffs_delta_interpretation': cliffs_delta_interpretation,
                        'mean_difference': float(mean_diff),
                        'percent_difference': float((mean_diff / np.mean(niche_values)) * 100) if np.mean(niche_values) != 0 else 0
                    },
                    'sample_sizes': {
                        'popular': len(pop_values),
                        'niche': len(niche_values)
                    },
                    'direction': 'popular_higher' if mean_diff > 0 else 'niche_higher'
                }
                
            except Exception as e:
                test_results[metric] = {
                    'test_used': 'failed',
                    'error': str(e)
                }
        
        return test_results
    
    def _calculate_cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0
        
        # Count pairs where group1 > group2 vs group1 < group2
        greater = sum(np.sum(group1[i] > group2) for i in range(n1))
        less = sum(np.sum(group1[i] < group2) for i in range(n1))
        
        return (greater - less) / (n1 * n2)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta effect size."""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def create_summary_report(self, popular_df: pd.DataFrame, niche_df: pd.DataFrame, 
                            descriptive_stats: dict, test_results: dict) -> dict:
        """Create comprehensive summary report."""
        logger.info("Creating summary report...")
        
        # Count significant results
        significant_005 = sum(1 for r in test_results.values() if r.get('significant_005', False))
        significant_001 = sum(1 for r in test_results.values() if r.get('significant_001', False))
        significant_bonferroni = sum(1 for r in test_results.values() if r.get('significant_bonferroni', False))
        
        # Identify most significant differences
        valid_results = {k: v for k, v in test_results.items() if 'p_value' in v}
        sorted_by_significance = sorted(valid_results.items(), key=lambda x: x[1]['p_value'])
        
        # Count effect size categories
        effect_sizes = {}
        for metric, result in test_results.items():
            if 'effect_sizes' in result:
                cohens_d = result['effect_sizes']['cohens_d']
                interpretation = result['effect_sizes']['cohens_d_interpretation']
                effect_sizes[metric] = {
                    'cohens_d': cohens_d,
                    'interpretation': interpretation,
                    'direction': result['direction']
                }
        
        # Pattern analysis
        popular_higher = sum(1 for r in test_results.values() if r.get('direction') == 'popular_higher' and r.get('significant_005', False))
        niche_higher = sum(1 for r in test_results.values() if r.get('direction') == 'niche_higher' and r.get('significant_005', False))
        
        summary = {
            'analysis_overview': {
                'total_topics_analyzed': len(popular_df) + len(niche_df),
                'popular_topics_count': len(popular_df),
                'niche_topics_count': len(niche_df),
                'popular_papers_range': f"{popular_df['total_papers'].min()}-{popular_df['total_papers'].max()}",
                'niche_papers_range': f"{niche_df['total_papers'].min()}-{niche_df['total_papers'].max()}",
                'metrics_compared': len(self.core_metrics)
            },
            'significance_summary': {
                'significant_at_005': significant_005,
                'significant_at_001': significant_001,
                'significant_bonferroni': significant_bonferroni,
                'total_tests': len([r for r in test_results.values() if 'p_value' in r])
            },
            'pattern_summary': {
                'popular_higher_count': popular_higher,
                'niche_higher_count': niche_higher,
                'dominant_pattern': 'popular_higher' if popular_higher > niche_higher else 'niche_higher' if niche_higher > popular_higher else 'mixed'
            },
            'top_differences': sorted_by_significance[:5],
            'effect_sizes': effect_sizes,
            'detailed_results': {
                'descriptive_statistics': descriptive_stats,
                'statistical_tests': test_results
            }
        }
        
        return summary
    
    def save_results(self, summary_report: dict, popular_df: pd.DataFrame, niche_df: pd.DataFrame):
        """Save analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON report
        report_file = self.results_dir / f"popular_vs_niche_analysis_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        # Save topic classifications
        popular_df['group'] = 'popular'
        niche_df['group'] = 'niche'
        combined_df = pd.concat([popular_df, niche_df], ignore_index=True)
        
        classifications_file = self.results_dir / f"topic_classifications_{timestamp}.csv"
        combined_df.to_csv(classifications_file, index=False)
        
        logger.info(f"Results saved with timestamp: {timestamp}")
        return timestamp
    
    def print_summary(self, summary_report: dict):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("🔬 POPULAR vs NICHE MATHEMATICAL TOPICS COMPARISON")
        print("="*80)
        
        overview = summary_report['analysis_overview']
        significance = summary_report['significance_summary']
        patterns = summary_report['pattern_summary']
        
        print(f"\n📊 ANALYSIS OVERVIEW:")
        print(f"   Total topics analyzed: {overview['total_topics_analyzed']}")
        print(f"   Popular topics (top 20%): {overview['popular_topics_count']} topics")
        print(f"   │  Paper range: {overview['popular_papers_range']} papers")
        print(f"   Niche topics (bottom 20%): {overview['niche_topics_count']} topics")
        print(f"   │  Paper range: {overview['niche_papers_range']} papers")
        print(f"   Metrics compared: {overview['metrics_compared']}")
        
        print(f"\n📈 STATISTICAL SIGNIFICANCE:")
        print(f"   Significant differences (p < 0.05): {significance['significant_at_005']}/{significance['total_tests']} metrics")
        print(f"   Highly significant (p < 0.01): {significance['significant_at_001']}/{significance['total_tests']} metrics")
        print(f"   Bonferroni corrected (p < 0.005): {significance['significant_bonferroni']}/{significance['total_tests']} metrics")
        
        print(f"\n🎯 PATTERN ANALYSIS:")
        if patterns['dominant_pattern'] == 'popular_higher':
            print(f"   🔺 Popular topics show higher values: {patterns['popular_higher_count']} metrics")
            print(f"   🔻 Niche topics show higher values: {patterns['niche_higher_count']} metrics")
            print(f"   → POPULAR TOPICS are more hierarchical/centralized")
        elif patterns['dominant_pattern'] == 'niche_higher':
            print(f"   🔺 Popular topics show higher values: {patterns['popular_higher_count']} metrics")
            print(f"   🔻 Niche topics show higher values: {patterns['niche_higher_count']} metrics")
            print(f"   → NICHE TOPICS are more hierarchical/centralized")
        else:
            print(f"   🔄 Mixed pattern - no clear dominance")
            print(f"   🔺 Popular higher: {patterns['popular_higher_count']} | 🔻 Niche higher: {patterns['niche_higher_count']}")
        
        # Show top significant differences
        print(f"\n🏆 TOP SIGNIFICANT DIFFERENCES:")
        for i, (metric, result) in enumerate(summary_report['top_differences'], 1):
            if 'p_value' in result:
                direction = "↑" if result['direction'] == 'popular_higher' else "↓"
                effect_size = result['effect_sizes']['cohens_d']
                effect_interp = result['effect_sizes']['cohens_d_interpretation']
                percent_diff = result['effect_sizes']['percent_difference']
                
                print(f"   {i}. {metric}")
                print(f"      Popular {direction} by {abs(percent_diff):.1f}% (p={result['p_value']:.4f})")
                print(f"      Effect size: {effect_size:.3f} ({effect_interp})")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if patterns['dominant_pattern'] == 'popular_higher':
            print(f"   Popular mathematical topics tend to have more:")
            print(f"   • Hierarchical collaboration structures")
            print(f"   • Centralized author networks") 
            print(f"   • Constrained researcher positions")
            print(f"   → This suggests 'hot' research areas create competitive, elite-dominated networks")
        elif patterns['dominant_pattern'] == 'niche_higher':
            print(f"   Niche mathematical topics tend to have more:")
            print(f"   • Egalitarian collaboration structures")
            print(f"   • Distributed author networks")
            print(f"   • Bridging researcher positions")
            print(f"   → This suggests specialized areas require more collaborative, bridge-building approaches")
        else:
            print(f"   Mixed patterns suggest:")
            print(f"   • Both popular and niche topics have distinct advantages in different dimensions")
            print(f"   • Mathematical collaboration may depend more on topic content than popularity")
        
        print(f"\n📁 Detailed results saved in results/collaboration_analysis/")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Popular vs Niche Topic Comparison")
    parser.add_argument("--results-file", 
                       default="results/collaboration_analysis/topic_analysis_10metrics_20250611_194954.json",
                       help="Path to pre-computed topic analysis results JSON file")
    parser.add_argument("--percentile-cutoff", type=float, default=0.2,
                       help="Percentile cutoff for popular/niche classification (default: 0.2 = 20%)")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = PopularNicheAnalyzer(args.results_file)
        
        # Load pre-computed results
        df = analyzer.load_results()
        
        # Identify popular and niche topics
        popular_df, niche_df = analyzer.identify_popular_niche_topics(df, args.percentile_cutoff)
        
        # Calculate descriptive statistics
        descriptive_stats = analyzer.calculate_descriptive_statistics(popular_df, niche_df)
        
        # Perform statistical tests
        test_results = analyzer.perform_statistical_tests(popular_df, niche_df)
        
        # Create summary report
        summary_report = analyzer.create_summary_report(popular_df, niche_df, descriptive_stats, test_results)
        
        # Save results
        timestamp = analyzer.save_results(summary_report, popular_df, niche_df)
        
        # Print summary
        analyzer.print_summary(summary_report)
        
        print(f"\n✅ Analysis complete! Results saved with timestamp: {timestamp}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()