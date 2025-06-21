#!/usr/bin/env python3
"""
Bootstrap Analysis with Real Topic Data - Enhanced with Cliff's Delta
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

def load_real_topic_data(json_file):
    """Load the real topic metrics from your JSON file."""
    print(f"Loading real topic data from {json_file}...")
    
    with open(json_file, 'r') as f:
        topic_data = json.load(f)
    
    # Convert to DataFrame
    topic_list = []
    for topic_id, data in topic_data.items():
        topic_record = {
            'topic_id': int(topic_id),
            'total_papers': data['total_papers'],
            'collaboration_rate': data['collaboration_rate'],
            'repeated_collaboration_rate': data['repeated_collaboration_rate'],
            'degree_centralization': data['degree_centralization'],
            'degree_assortativity': data['degree_assortativity'],
            'modularity': data['modularity'],
            'small_world_coefficient': data['small_world_coefficient'],
            'coreness_ratio': data['coreness_ratio'],
            'robustness_ratio': data['robustness_ratio'],
            'avg_constraint': data['avg_constraint'],
            'avg_effective_size': data['avg_effective_size']
        }
        topic_list.append(topic_record)
    
    df = pd.DataFrame(topic_list)
    print(f"Loaded {len(df)} topics with real network metrics")
    
    return df

def classify_popular_niche(df, percentile_cutoff=0.2):
    """Classify topics as popular or niche based on paper count."""
    df_sorted = df.sort_values('total_papers', ascending=False)
    
    n_topics = len(df_sorted)
    cutoff_size = int(n_topics * percentile_cutoff)
    
    popular_df = df_sorted.head(cutoff_size).copy()
    niche_df = df_sorted.tail(cutoff_size).copy()
    
    print(f"Popular topics: {len(popular_df)} topics (papers: {popular_df['total_papers'].min()}-{popular_df['total_papers'].max()})")
    print(f"Niche topics: {len(niche_df)} topics (papers: {niche_df['total_papers'].min()}-{niche_df['total_papers'].max()})")
    
    return popular_df, niche_df

def calculate_cliffs_delta(group1, group2):
    """
    Calculate Cliff's delta (non-parametric effect size).
    
    Cliff's delta = (number of pairs where group1 > group2 - number of pairs where group1 < group2) 
                   / (total number of pairs)
    
    Range: [-1, 1]
    - Positive values indicate group1 tends to be larger
    - Negative values indicate group2 tends to be larger
    - 0 indicates no difference
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0
    
    # Count pairs where group1 > group2 vs group1 < group2
    greater = sum(np.sum(group1[i] > group2) for i in range(n1))
    less = sum(np.sum(group1[i] < group2) for i in range(n1))
    
    return (greater - less) / (n1 * n2)

def interpret_cliffs_delta(delta):
    """Interpret Cliff's delta effect size magnitude."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"

def bootstrap_difference_ci(group1, group2, n_bootstrap=10000, confidence=0.95):
    """Bootstrap CI for difference in means between two groups."""
    differences = []
    
    for _ in range(n_bootstrap):
        boot1 = np.random.choice(group1, size=len(group1), replace=True)
        boot2 = np.random.choice(group2, size=len(group2), replace=True)
        diff = np.mean(boot1) - np.mean(boot2)
        differences.append(diff)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(differences, (alpha/2) * 100)
    ci_upper = np.percentile(differences, (1 - alpha/2) * 100)
    observed_diff = np.mean(group1) - np.mean(group2)
    
    return {
        'observed_difference': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': not (ci_lower <= 0 <= ci_upper),
        'percent_difference': (observed_diff / abs(np.mean(group2))) * 100 if np.mean(group2) != 0 else 0
    }

def bootstrap_cliffs_delta_ci(group1, group2, n_bootstrap=10000, confidence=0.95):
    """
    Bootstrap confidence interval for Cliff's delta (non-parametric effect size).
    
    This is the non-parametric equivalent to Cohen's d and is more appropriate
    when using non-parametric tests like Mann-Whitney U.
    """
    bootstrap_deltas = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample both groups
        boot1 = np.random.choice(group1, size=len(group1), replace=True)
        boot2 = np.random.choice(group2, size=len(group2), replace=True)
        
        # Calculate Cliff's delta for this bootstrap sample
        delta = calculate_cliffs_delta(boot1, boot2)
        bootstrap_deltas.append(delta)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_deltas, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_deltas, (1 - alpha/2) * 100)
    observed_delta = calculate_cliffs_delta(group1, group2)
    
    return {
        'observed_cliffs_delta': observed_delta,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'interpretation': interpret_cliffs_delta(observed_delta),
        'significant_effect': not (ci_lower <= 0 <= ci_upper),
        'direction': 'group1_higher' if observed_delta > 0 else 'group2_higher' if observed_delta < 0 else 'no_difference'
    }

def bootstrap_cohens_d_ci(group1, group2, n_bootstrap=10000, confidence=0.95):
    """Bootstrap CI for Cohen's d (kept for comparison, but Cliff's delta is preferred for non-parametric data)."""
    def cohens_d(x, y):
        n1, n2 = len(x), len(y)
        pooled_std = np.sqrt(((n1-1)*np.var(x, ddof=1) + (n2-1)*np.var(y, ddof=1)) / (n1+n2-2))
        return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
    
    bootstrap_ds = []
    for _ in range(n_bootstrap):
        boot1 = np.random.choice(group1, size=len(group1), replace=True)
        boot2 = np.random.choice(group2, size=len(group2), replace=True)
        bootstrap_ds.append(cohens_d(boot1, boot2))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_ds, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_ds, (1 - alpha/2) * 100)
    observed_d = cohens_d(group1, group2)
    
    return {
        'observed_cohens_d': observed_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant_effect': not (ci_lower <= 0 <= ci_upper)
    }

def analyze_metric(popular_data, niche_data, metric_name):
    """Analyze a single metric with bootstrap CIs."""
    print(f"\n--- {metric_name.upper()} ---")
    
    # Basic stats
    pop_mean = np.mean(popular_data)
    niche_mean = np.mean(niche_data)
    pop_median = np.median(popular_data)
    niche_median = np.median(niche_data)
    
    print(f"Popular mean: {pop_mean:.4f}, median: {pop_median:.4f}")
    print(f"Niche mean: {niche_mean:.4f}, median: {niche_median:.4f}")
    
    # Bootstrap difference CI
    diff_result = bootstrap_difference_ci(popular_data, niche_data)
    print(f"Mean difference: {diff_result['observed_difference']:.4f} "
          f"[{diff_result['ci_lower']:.4f}, {diff_result['ci_upper']:.4f}]")
    print(f"Significant difference: {diff_result['significant']}")
    print(f"Percent difference: {diff_result['percent_difference']:.1f}%")
    
    # Bootstrap Cliff's delta CI (NON-PARAMETRIC - recommended)
    cliffs_result = bootstrap_cliffs_delta_ci(popular_data, niche_data)
    print(f"Cliff's δ: {cliffs_result['observed_cliffs_delta']:.4f} "
          f"[{cliffs_result['ci_lower']:.4f}, {cliffs_result['ci_upper']:.4f}] "
          f"({cliffs_result['interpretation']} effect)")
    print(f"Direction: {cliffs_result['direction']}")
    print(f"Significant effect: {cliffs_result['significant_effect']}")
    
    # Bootstrap Cohen's d CI (PARAMETRIC - for comparison only)
    cohens_result = bootstrap_cohens_d_ci(popular_data, niche_data)
    print(f"Cohen's d: {cohens_result['observed_cohens_d']:.4f} "
          f"[{cohens_result['ci_lower']:.4f}, {cohens_result['ci_upper']:.4f}] "
          f"(for comparison only)")
    
    return {
        'metric': metric_name,
        'popular_mean': pop_mean,
        'popular_median': pop_median,
        'niche_mean': niche_mean,
        'niche_median': niche_median,
        'difference': diff_result,
        'cliffs_delta': cliffs_result,
        'cohens_d': cohens_result  # Keep for comparison
    }

def perform_mann_whitney_test(popular_data, niche_data):
    """Perform Mann-Whitney U test for comparison with bootstrap results."""
    try:
        statistic, p_value = stats.mannwhitneyu(popular_data, niche_data, alternative='two-sided')
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01,
            'significant_bonferroni': p_value < (0.05 / 10)  # Bonferroni correction for 10 tests
        }
    except Exception as e:
        return {
            'error': str(e),
            'statistic': None,
            'p_value': None,
            'significant_005': False,
            'significant_001': False,
            'significant_bonferroni': False
        }

def save_bootstrap_results(results, mann_whitney_results, popular_df, niche_df, output_dir="results/collaboration_analysis"):
    """Save comprehensive bootstrap analysis results."""
    from datetime import datetime
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save detailed results as JSON
    detailed_results = {
        'analysis_metadata': {
            'timestamp': timestamp,
            'analysis_type': 'bootstrap_cliffs_delta_analysis',
            'n_bootstrap_samples': 10000,
            'confidence_level': 0.95,
            'popular_topics_count': len(popular_df),
            'niche_topics_count': len(niche_df),
            'popular_papers_range': f"{popular_df['total_papers'].min()}-{popular_df['total_papers'].max()}",
            'niche_papers_range': f"{niche_df['total_papers'].min()}-{niche_df['total_papers'].max()}"
        },
        'bootstrap_results': results,
        'mann_whitney_results': mann_whitney_results
    }
    
    json_file = output_path / f"bootstrap_cliffs_delta_analysis_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    # 2. Save summary table as CSV
    summary_data = []
    for metric in results.keys():
        if metric in mann_whitney_results:
            row = {
                'metric': metric,
                'popular_mean': results[metric]['popular_mean'],
                'popular_median': results[metric]['popular_median'],
                'niche_mean': results[metric]['niche_mean'],
                'niche_median': results[metric]['niche_median'],
                'mean_difference': results[metric]['difference']['observed_difference'],
                'mean_diff_ci_lower': results[metric]['difference']['ci_lower'],
                'mean_diff_ci_upper': results[metric]['difference']['ci_upper'],
                'percent_difference': results[metric]['difference']['percent_difference'],
                'cliffs_delta': results[metric]['cliffs_delta']['observed_cliffs_delta'],
                'cliffs_delta_ci_lower': results[metric]['cliffs_delta']['ci_lower'],
                'cliffs_delta_ci_upper': results[metric]['cliffs_delta']['ci_upper'],
                'cliffs_delta_interpretation': results[metric]['cliffs_delta']['interpretation'],
                'cliffs_delta_direction': results[metric]['cliffs_delta']['direction'],
                'bootstrap_significant': results[metric]['cliffs_delta']['significant_effect'],
                'mann_whitney_u_statistic': mann_whitney_results[metric]['statistic'],
                'mann_whitney_p_value': mann_whitney_results[metric]['p_value'],
                'mann_whitney_significant_005': mann_whitney_results[metric]['significant_005'],
                'mann_whitney_significant_001': mann_whitney_results[metric]['significant_001'],
                'mann_whitney_significant_bonferroni': mann_whitney_results[metric]['significant_bonferroni'],
                'cohens_d': results[metric]['cohens_d']['observed_cohens_d'],
                'cohens_d_ci_lower': results[metric]['cohens_d']['ci_lower'],
                'cohens_d_ci_upper': results[metric]['cohens_d']['ci_upper']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = output_path / f"bootstrap_summary_table_{timestamp}.csv"
    summary_df.to_csv(csv_file, index=False)
    
    # 3. Save methodology-aligned results (Bootstrap + Cliff's Delta focus)
    methodology_results = {
        'analysis_approach': {
            'statistical_test': 'Mann-Whitney U (non-parametric)',
            'effect_size': 'Cliffs Delta (non-parametric)',
            'confidence_intervals': 'Bootstrap (distribution-free)',
            'rationale': 'Methodologically consistent non-parametric approach for non-normal network data'
        },
        'significant_effects': {
            'bootstrap_cliffs_delta': [m for m, r in results.items() if r['cliffs_delta']['significant_effect']],
            'mann_whitney_p005': [m for m, r in mann_whitney_results.items() if r['significant_005']],
            'mann_whitney_bonferroni': [m for m, r in mann_whitney_results.items() if r['significant_bonferroni']]
        },
        'effect_sizes_by_magnitude': {
            'large_effects': [m for m, r in results.items() if r['cliffs_delta']['interpretation'] == 'large'],
            'medium_effects': [m for m, r in results.items() if r['cliffs_delta']['interpretation'] == 'medium'],
            'small_effects': [m for m, r in results.items() if r['cliffs_delta']['interpretation'] == 'small'],
            'negligible_effects': [m for m, r in results.items() if r['cliffs_delta']['interpretation'] == 'negligible']
        },
        'top_differences': sorted(
            [(m, r['cliffs_delta']['observed_cliffs_delta']) for m, r in results.items()],
            key=lambda x: abs(x[1]), reverse=True
        )
    }
    
    methodology_file = output_path / f"methodology_aligned_results_{timestamp}.json"
    with open(methodology_file, 'w') as f:
        json.dump(methodology_results, f, indent=2)
    
    # 4. Create publication-ready table
    pub_table_data = []
    for metric in results.keys():
        if metric in mann_whitney_results:
            cliffs = results[metric]['cliffs_delta']
            mw = mann_whitney_results[metric]
            
            # Format for publication
            cliffs_formatted = f"{cliffs['observed_cliffs_delta']:.3f} [{cliffs['ci_lower']:.3f}, {cliffs['ci_upper']:.3f}]"
            p_formatted = f"{mw['p_value']:.2e}" if mw['p_value'] < 0.001 else f"{mw['p_value']:.3f}"
            
            pub_row = {
                'Metric': metric.replace('_', ' ').title(),
                'Popular Mean (SD)': f"{results[metric]['popular_mean']:.3f}",
                'Niche Mean (SD)': f"{results[metric]['niche_mean']:.3f}",
                'Cliff\'s δ [95% CI]': cliffs_formatted,
                'Effect Size': cliffs['interpretation'].title(),
                'Mann-Whitney U': f"{mw['statistic']:.0f}" if mw['statistic'] else 'N/A',
                'p-value': p_formatted,
                'Bonferroni Sig.': '**' if mw['significant_bonferroni'] else '*' if mw['significant_005'] else ''
            }
            pub_table_data.append(pub_row)
    
    pub_table_df = pd.DataFrame(pub_table_data)
    pub_table_file = output_path / f"publication_ready_table_{timestamp}.csv"
    pub_table_df.to_csv(pub_table_file, index=False)
    
    print(f"\n📁 Results saved to {output_path}/:")
    print(f"   • bootstrap_cliffs_delta_analysis_{timestamp}.json (detailed results)")
    print(f"   • bootstrap_summary_table_{timestamp}.csv (all metrics)")
    print(f"   • methodology_aligned_results_{timestamp}.json (non-parametric focus)")
    print(f"   • publication_ready_table_{timestamp}.csv (formatted for papers)")
    
    return timestamp

def main():
    """Run enhanced bootstrap analysis on real data."""
    import sys
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'results/collaboration_analysis/topic_analysis_10metrics_20250607_124314.json'
    
    df = load_real_topic_data(json_file)

    # Load real data
    # df = load_real_topic_data('results/collaboration_analysis/topic_analysis_10metrics_20250607_124314.json')
    
    # Classify popular vs niche
    popular_df, niche_df = classify_popular_niche(df)
    
    # Metrics to analyze
    metrics = [
        'collaboration_rate', 'repeated_collaboration_rate', 'degree_centralization',
        'degree_assortativity', 'modularity', 'small_world_coefficient',
        'coreness_ratio', 'robustness_ratio', 'avg_constraint', 'avg_effective_size'
    ]
    
    print(f"\n{'='*80}")
    print("ENHANCED BOOTSTRAP ANALYSIS WITH CLIFF'S DELTA")
    print("Non-parametric effect sizes for non-parametric tests")
    print(f"{'='*80}")
    
    results = {}
    mann_whitney_results = {}
    
    for metric in tqdm(metrics, desc="Processing metrics"):
        try:
            popular_values = popular_df[metric].dropna().values
            niche_values = niche_df[metric].dropna().values
            
            # Bootstrap analysis
            results[metric] = analyze_metric(popular_values, niche_values, metric)
            
            # Mann-Whitney U test for comparison
            mann_whitney_results[metric] = perform_mann_whitney_test(popular_values, niche_values)
            
        except Exception as e:
            print(f"Error processing {metric}: {e}")
    
    # Save comprehensive results
    timestamp = save_bootstrap_results(results, mann_whitney_results, popular_df, niche_df)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: BOOTSTRAP vs MANN-WHITNEY COMPARISON")
    print(f"{'='*80}")
    
    # Compare bootstrap and Mann-Whitney significance
    bootstrap_significant = [m for m, r in results.items() if r['cliffs_delta']['significant_effect']]
    mann_whitney_significant = [m for m, r in mann_whitney_results.items() if r['significant_005']]
    mann_whitney_bonferroni = [m for m, r in mann_whitney_results.items() if r['significant_bonferroni']]
    
    print(f"Bootstrap Cliff's δ significant: {len(bootstrap_significant)}/{len(results)} metrics")
    print(f"Mann-Whitney U significant (p<0.05): {len(mann_whitney_significant)}/{len(results)} metrics")
    print(f"Mann-Whitney U significant (Bonferroni): {len(mann_whitney_bonferroni)}/{len(results)} metrics")
    
    print(f"\nBootstrap significant: {', '.join(bootstrap_significant)}")
    print(f"Mann-Whitney significant: {', '.join(mann_whitney_significant)}")
    
    # Effect size summary
    print(f"\n{'='*80}")
    print("EFFECT SIZE SUMMARY (CLIFF'S DELTA)")
    print(f"{'='*80}")
    
    for metric in metrics:
        if metric in results:
            cliffs_data = results[metric]['cliffs_delta']
            print(f"{metric:25s}: δ={cliffs_data['observed_cliffs_delta']:6.3f} "
                  f"[{cliffs_data['ci_lower']:6.3f}, {cliffs_data['ci_upper']:6.3f}] "
                  f"({cliffs_data['interpretation']:>8s}) "
                  f"{'*' if cliffs_data['significant_effect'] else ' '}")
    
    print(f"\n* = Significant effect (CI excludes 0)")
    print(f"\nInterpretation: |δ| < 0.147 = negligible, 0.147-0.33 = small, 0.33-0.474 = medium, ≥0.474 = large")
    print(f"Direction: Positive δ = popular topics higher, Negative δ = niche topics higher")
    
    print(f"\n🎉 Analysis complete! Results saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()