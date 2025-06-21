#!/usr/bin/env python3
"""
Popular vs Niche Topic Comparison Analysis with Sensitivity Testing
==================================================================

Analyzes pre-computed collaboration network metrics to compare popular vs niche topics
across multiple cutoff thresholds (15%, 20%, 25%, 30%) for sensitivity analysis.

Usage:
    python popular_niche_comparison_sensitivity.py --results-file topic_analysis_10metrics_20250604_115101.json
    python popular_niche_comparison_sensitivity.py --results-file topic_analysis_10metrics_20250604_115101.json --cutoffs 0.15 0.20 0.25 0.30
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import argparse
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """
    Analyzes differences between popular and niche topics using pre-computed metrics
    across multiple cutoff thresholds for sensitivity analysis.
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
        
        logger.info(f"Initialized SensitivityAnalyzer with results: {self.results_file}")
    
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
    
    def identify_popular_niche_topics(self, df: pd.DataFrame, percentile_cutoff: float) -> tuple:
        """
        Identify popular and niche topics based on total paper count.
        
        Args:
            df: DataFrame with topic data
            percentile_cutoff: Percentage for top/bottom groups
            
        Returns:
            Tuple of (popular_df, niche_df)
        """
        # Sort by total papers
        df_sorted = df.sort_values('total_papers', ascending=False)
        
        # Calculate cutoff indices
        n_topics = len(df_sorted)
        cutoff_size = int(n_topics * percentile_cutoff)
        
        # Extract popular and niche groups
        popular_df = df_sorted.head(cutoff_size).copy()
        niche_df = df_sorted.tail(cutoff_size).copy()
        
        return popular_df, niche_df
    
    def perform_statistical_tests(self, popular_df: pd.DataFrame, niche_df: pd.DataFrame) -> dict:
        """Perform statistical tests comparing popular vs niche topics."""
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
                # Test for normality
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
                    statistic, p_value = stats.ttest_ind(pop_values, niche_values)
                    test_used = 'independent_t_test'
                else:
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
                
                # Cliff's delta
                cliffs_delta = self._calculate_cliffs_delta(pop_values, niche_values)
                
                test_results[metric] = {
                    'test_used': test_used,
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant_005': p_value < 0.05,
                    'significant_001': p_value < 0.01,
                    'significant_bonferroni': p_value < (0.05 / len(self.core_metrics)),
                    'effect_sizes': {
                        'cohens_d': float(cohens_d),
                        'cohens_d_interpretation': self._interpret_cohens_d(cohens_d),
                        'cliffs_delta': float(cliffs_delta),
                        'cliffs_delta_interpretation': self._interpret_cliffs_delta(cliffs_delta),
                        'mean_difference': float(mean_diff),
                        'percent_difference': float((mean_diff / np.mean(niche_values)) * 100) if np.mean(niche_values) != 0 else 0
                    },
                    'sample_sizes': {
                        'popular': len(pop_values),
                        'niche': len(niche_values)
                    },
                    'direction': 'popular_higher' if mean_diff > 0 else 'niche_higher',
                    'popular_mean': float(np.mean(pop_values)),
                    'niche_mean': float(np.mean(niche_values))
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
    
    def run_sensitivity_analysis(self, df: pd.DataFrame, cutoffs: list) -> dict:
        """Run analysis across multiple cutoff thresholds."""
        logger.info(f"Running sensitivity analysis across {len(cutoffs)} cutoffs: {cutoffs}")
        
        sensitivity_results = {}
        
        for cutoff in cutoffs:
            logger.info(f"Analyzing cutoff: {cutoff*100}%")
            
            # Identify popular and niche topics for this cutoff
            popular_df, niche_df = self.identify_popular_niche_topics(df, cutoff)
            
            # Perform statistical tests
            test_results = self.perform_statistical_tests(popular_df, niche_df)
            
            # Calculate summary statistics
            cutoff_summary = {
                'cutoff': cutoff,
                'cutoff_percent': cutoff * 100,
                'popular_count': len(popular_df),
                'niche_count': len(niche_df),
                'popular_papers_range': f"{popular_df['total_papers'].min()}-{popular_df['total_papers'].max()}",
                'niche_papers_range': f"{niche_df['total_papers'].min()}-{niche_df['total_papers'].max()}",
                'popular_papers_mean': float(popular_df['total_papers'].mean()),
                'niche_papers_mean': float(niche_df['total_papers'].mean()),
                'paper_ratio': float(popular_df['total_papers'].mean() / niche_df['total_papers'].mean()) if niche_df['total_papers'].mean() > 0 else 0,
                'test_results': test_results
            }
            
            # Count significant results
            significant_005 = sum(1 for r in test_results.values() if r.get('significant_005', False))
            significant_001 = sum(1 for r in test_results.values() if r.get('significant_001', False))
            significant_bonferroni = sum(1 for r in test_results.values() if r.get('significant_bonferroni', False))
            
            cutoff_summary['significance_counts'] = {
                'significant_005': significant_005,
                'significant_001': significant_001,
                'significant_bonferroni': significant_bonferroni,
                'total_tests': len([r for r in test_results.values() if 'p_value' in r])
            }
            
            # Pattern analysis
            popular_higher = sum(1 for r in test_results.values() if r.get('direction') == 'popular_higher' and r.get('significant_005', False))
            niche_higher = sum(1 for r in test_results.values() if r.get('direction') == 'niche_higher' and r.get('significant_005', False))
            
            cutoff_summary['pattern_analysis'] = {
                'popular_higher_count': popular_higher,
                'niche_higher_count': niche_higher,
                'dominant_pattern': 'popular_higher' if popular_higher > niche_higher else 'niche_higher' if niche_higher > popular_higher else 'mixed'
            }
            
            sensitivity_results[cutoff] = cutoff_summary
        
        return sensitivity_results
    
    def create_sensitivity_summary(self, sensitivity_results: dict) -> dict:
        """Create a cross-cutoff sensitivity summary."""
        logger.info("Creating sensitivity summary...")
        
        # Extract key metrics across cutoffs
        cutoffs = list(sensitivity_results.keys())
        
        # Track effect sizes across cutoffs
        effect_size_tracking = {}
        significance_tracking = {}
        direction_consistency = {}
        
        for metric in self.core_metrics:
            effect_size_tracking[metric] = []
            significance_tracking[metric] = []
            direction_consistency[metric] = []
            
            for cutoff in cutoffs:
                test_result = sensitivity_results[cutoff]['test_results'].get(metric, {})
                
                if 'effect_sizes' in test_result:
                    effect_size_tracking[metric].append({
                        'cutoff': cutoff,
                        'cohens_d': test_result['effect_sizes']['cohens_d'],
                        'cliffs_delta': test_result['effect_sizes']['cliffs_delta'],
                        'interpretation': test_result['effect_sizes']['cohens_d_interpretation']
                    })
                    
                    significance_tracking[metric].append({
                        'cutoff': cutoff,
                        'p_value': test_result['p_value'],
                        'significant_005': test_result['significant_005'],
                        'significant_bonferroni': test_result['significant_bonferroni']
                    })
                    
                    direction_consistency[metric].append({
                        'cutoff': cutoff,
                        'direction': test_result['direction']
                    })
        
        # Calculate stability metrics
        stability_analysis = {}
        for metric in self.core_metrics:
            if effect_size_tracking[metric]:
                # Effect size stability (coefficient of variation)
                effect_sizes = [item['cohens_d'] for item in effect_size_tracking[metric]]
                if len(effect_sizes) > 1 and np.std(effect_sizes) > 0:
                    cv_effect_size = np.std(effect_sizes) / abs(np.mean(effect_sizes)) if np.mean(effect_sizes) != 0 else 0
                else:
                    cv_effect_size = 0
                
                # Significance consistency
                sig_count = sum(1 for item in significance_tracking[metric] if item['significant_005'])
                sig_consistency = sig_count / len(significance_tracking[metric]) if significance_tracking[metric] else 0
                
                # Direction consistency
                directions = [item['direction'] for item in direction_consistency[metric]]
                if directions:
                    direction_mode = max(set(directions), key=directions.count)
                    direction_consistency_pct = directions.count(direction_mode) / len(directions)
                else:
                    direction_mode = 'unknown'
                    direction_consistency_pct = 0
                
                stability_analysis[metric] = {
                    'effect_size_cv': float(cv_effect_size),
                    'effect_size_mean': float(np.mean(effect_sizes)),
                    'effect_size_range': [float(min(effect_sizes)), float(max(effect_sizes))],
                    'significance_consistency': float(sig_consistency),
                    'direction_consistency': float(direction_consistency_pct),
                    'dominant_direction': direction_mode,
                    'stability_score': float((1 - cv_effect_size) * sig_consistency * direction_consistency_pct) if cv_effect_size < 1 else 0
                }
        
        # Overall sensitivity summary
        overall_summary = {
            'cutoffs_tested': cutoffs,
            'cutoff_percentages': [c * 100 for c in cutoffs],
            'sample_size_ranges': {
                cutoff: {
                    'popular': sensitivity_results[cutoff]['popular_count'],
                    'niche': sensitivity_results[cutoff]['niche_count']
                } for cutoff in cutoffs
            },
            'significance_patterns': {
                cutoff: sensitivity_results[cutoff]['significance_counts'] for cutoff in cutoffs
            },
            'stability_analysis': stability_analysis,
            'most_stable_metrics': sorted(
                [(metric, data['stability_score']) for metric, data in stability_analysis.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            'least_stable_metrics': sorted(
                [(metric, data['stability_score']) for metric, data in stability_analysis.items()],
                key=lambda x: x[1]
            )[:5]
        }
        
        return {
            'sensitivity_results': sensitivity_results,
            'effect_size_tracking': effect_size_tracking,
            'significance_tracking': significance_tracking,
            'direction_consistency': direction_consistency,
            'overall_summary': overall_summary
        }
    
    def create_visualizations(self, sensitivity_summary: dict):
        """Create visualization plots for sensitivity analysis."""
        logger.info("Creating sensitivity analysis visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Effect Size Stability Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sensitivity Analysis: Effect Sizes Across Cutoffs', fontsize=16, fontweight='bold')
        
        # Get stability analysis from the correct location
        stability_analysis = sensitivity_summary['overall_summary']['stability_analysis']
        
        # Plot 1: Cohen's d across cutoffs for top 4 most stable metrics
        stable_metrics = [metric for metric, _ in sensitivity_summary['overall_summary']['most_stable_metrics'][:4]]
        
        ax1 = axes[0, 0]
        for metric in stable_metrics:
            tracking_data = sensitivity_summary['effect_size_tracking'][metric]
            cutoffs = [item['cutoff'] * 100 for item in tracking_data]
            cohens_d = [item['cohens_d'] for item in tracking_data]
            ax1.plot(cutoffs, cohens_d, marker='o', linewidth=2, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Cutoff Percentage (%)')
        ax1.set_ylabel("Cohen's d")
        ax1.set_title('Effect Size Stability (Most Stable Metrics)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Significance consistency
        ax2 = axes[0, 1]
        metrics_for_sig = list(stability_analysis.keys())[:8]
        sig_consistency = [stability_analysis[m]['significance_consistency'] for m in metrics_for_sig]
        
        bars = ax2.bar(range(len(metrics_for_sig)), sig_consistency, alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Significance Consistency')
        ax2.set_title('Significance Consistency Across Cutoffs')
        ax2.set_xticks(range(len(metrics_for_sig)))
        ax2.set_xticklabels([m.replace('_', '\n') for m in metrics_for_sig], rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, sig_consistency):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 3: Direction consistency
        ax3 = axes[1, 0]
        direction_consistency = [stability_analysis[m]['direction_consistency'] for m in metrics_for_sig]
        
        bars = ax3.bar(range(len(metrics_for_sig)), direction_consistency, alpha=0.7, color='orange')
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Direction Consistency')
        ax3.set_title('Direction Consistency Across Cutoffs')
        ax3.set_xticks(range(len(metrics_for_sig)))
        ax3.set_xticklabels([m.replace('_', '\n') for m in metrics_for_sig], rotation=45, ha='right')
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, direction_consistency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 4: Overall stability score
        ax4 = axes[1, 1]
        stability_scores = [stability_analysis[m]['stability_score'] for m in metrics_for_sig]
        
        bars = ax4.bar(range(len(metrics_for_sig)), stability_scores, alpha=0.7, color='green')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Stability Score')
        ax4.set_title('Overall Stability Score')
        ax4.set_xticks(range(len(metrics_for_sig)))
        ax4.set_xticklabels([m.replace('_', '\n') for m in metrics_for_sig], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, stability_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.results_dir / f"sensitivity_analysis_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_file}")
        return plot_file
    
    def save_results(self, sensitivity_summary: dict):
        """Save sensitivity analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON report
        report_file = self.results_dir / f"sensitivity_analysis_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(sensitivity_summary, f, indent=2, default=str)
        
        # Save summary table as CSV
        stability_data = []
        for metric, data in sensitivity_summary['overall_summary']['stability_analysis'].items():
            stability_data.append({
                'metric': metric,
                'effect_size_mean': data['effect_size_mean'],
                'effect_size_cv': data['effect_size_cv'],
                'significance_consistency': data['significance_consistency'],
                'direction_consistency': data['direction_consistency'],
                'dominant_direction': data['dominant_direction'],
                'stability_score': data['stability_score']
            })
        
        stability_df = pd.DataFrame(stability_data)
        stability_df = stability_df.sort_values('stability_score', ascending=False)
        
        csv_file = self.results_dir / f"stability_summary_{timestamp}.csv"
        stability_df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved with timestamp: {timestamp}")
        return timestamp
    
    def print_sensitivity_summary(self, sensitivity_summary: dict):
        """Print comprehensive sensitivity analysis summary."""
        print("\n" + "="*80)
        print("🔬 SENSITIVITY ANALYSIS: Popular vs Niche Topics Across Multiple Cutoffs")
        print("="*80)
        
        overall = sensitivity_summary['overall_summary']
        
        print(f"\n📊 CUTOFFS TESTED:")
        for cutoff in overall['cutoffs_tested']:
            sample_info = overall['sample_size_ranges'][cutoff]
            print(f"   {cutoff*100:4.0f}%: {sample_info['popular']:3d} popular topics, {sample_info['niche']:3d} niche topics")
        
        print(f"\n🎯 STABILITY ANALYSIS:")
        print(f"   Most Stable Metrics (high consistency across cutoffs):")
        for i, (metric, score) in enumerate(overall['most_stable_metrics'], 1):
            direction = overall['stability_analysis'][metric]['dominant_direction']
            sig_consistency = overall['stability_analysis'][metric]['significance_consistency']
            print(f"   {i}. {metric.replace('_', ' ').title()}: {score:.3f} ({direction}, {sig_consistency:.0%} significant)")
        
        print(f"\n   Least Stable Metrics (variable across cutoffs):")
        for i, (metric, score) in enumerate(overall['least_stable_metrics'], 1):
            direction = overall['stability_analysis'][metric]['dominant_direction']
            sig_consistency = overall['stability_analysis'][metric]['significance_consistency']
            print(f"   {i}. {metric.replace('_', ' ').title()}: {score:.3f} ({direction}, {sig_consistency:.0%} significant)")
        
        print(f"\n📈 SIGNIFICANCE PATTERNS:")
        print(f"   Cutoff  | p<0.05 | p<0.01 | Bonferroni")
        print(f"   --------|--------|--------|----------")
        for cutoff in overall['cutoffs_tested']:
            sig_data = overall['significance_patterns'][cutoff]
            print(f"   {cutoff*100:5.0f}%  |  {sig_data['significant_005']:2d}/10  |  {sig_data['significant_001']:2d}/10  |   {sig_data['significant_bonferroni']:2d}/10")
        
        print(f"\n🔍 KEY FINDINGS:")
        
        # Find most consistent effects
        highly_stable = [metric for metric, score in overall['most_stable_metrics'] if score > 0.7]
        if highly_stable:
            print(f"   ✅ ROBUST EFFECTS (stable across all cutoffs):")
            for metric in highly_stable[:3]:
                data = overall['stability_analysis'][metric]
                print(f"      • {metric.replace('_', ' ').title()}: {data['dominant_direction']} (effect size CV: {data['effect_size_cv']:.3f})")
        
        # Find inconsistent effects
        unstable = [metric for metric, score in overall['least_stable_metrics'] if score < 0.3]
        if unstable:
            print(f"   ⚠️  SENSITIVE EFFECTS (vary with cutoff choice):")
            for metric in unstable:
                data = overall['stability_analysis'][metric]
                print(f"      • {metric.replace('_', ' ').title()}: inconsistent (effect size CV: {data['effect_size_cv']:.3f})")
        
        # Overall conclusion
        stable_count = sum(1 for _, score in overall['most_stable_metrics'] if score > 0.5)
        total_metrics = len(overall['stability_analysis'])
        
        print(f"\n🎯 OVERALL ROBUSTNESS:")
        print(f"   {stable_count}/{total_metrics} metrics show stable patterns across cutoffs")
        if stable_count >= total_metrics * 0.7:
            print(f"   → Results are ROBUST to cutoff choice")
        elif stable_count >= total_metrics * 0.5:
            print(f"   → Results are MODERATELY ROBUST to cutoff choice")
        else:
            print(f"   → Results are SENSITIVE to cutoff choice - interpret with caution")
        
        print(f"\n📁 Detailed results and visualizations saved in results/collaboration_analysis/")


def main():
    """Main sensitivity analysis function."""
    parser = argparse.ArgumentParser(description="Sensitivity Analysis: Popular vs Niche Topics")
    parser.add_argument("--results-file", 
                       default="results/collaboration_analysis/topic_analysis_10metrics_20250604_115101.json",
                       help="Path to pre-computed topic analysis results JSON file")
    parser.add_argument("--cutoffs", nargs="+", type=float, 
                       default=[0.15, 0.20, 0.25, 0.30],
                       help="List of percentile cutoffs to test (default: 0.15 0.20 0.25 0.30)")
    parser.add_argument("--create-plots", action="store_true", default=True,
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = SensitivityAnalyzer(args.results_file)
        
        # Load pre-computed results
        df = analyzer.load_results()
        
        # Run sensitivity analysis across multiple cutoffs
        sensitivity_results = analyzer.run_sensitivity_analysis(df, args.cutoffs)
        
        # Create comprehensive sensitivity summary
        sensitivity_summary = analyzer.create_sensitivity_summary(sensitivity_results)
        
        # Create visualizations
        if args.create_plots:
            plot_file = analyzer.create_visualizations(sensitivity_summary)
        
        # Save results
        timestamp = analyzer.save_results(sensitivity_summary)
        
        # Print summary
        analyzer.print_sensitivity_summary(sensitivity_summary)
        
        print(f"\n✅ Sensitivity analysis complete! Results saved with timestamp: {timestamp}")
        if args.create_plots:
            print(f"📊 Visualizations saved to results/collaboration_analysis/")
        
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()