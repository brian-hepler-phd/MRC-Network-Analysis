#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

"""
Enhanced Regression Analysis with Continuous Popularity Variables
===============================================================

Extends the fixed regression analysis to include continuous measures of popularity:
1. Log-transformed total papers (continuous popularity measure)
2. Popularity rank (1 to N ranking)
3. Popularity percentile (0-100 continuous scale)
4. Standardized popularity z-score

This provides more nuanced analysis beyond the binary popular/niche classification.
"""

import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian, Gamma, Binomial
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.config_manager import ConfigManager

class EnhancedNetworkRegressionAnalyzer:
    """
    Enhanced regression analysis with both binary and continuous popularity measures.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results_dir = Path("results/regression_analysis_enhanced")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Network metrics to analyze
        self.network_metrics = [
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
        
        print(f"Initialized Enhanced NetworkRegressionAnalyzer with data: {len(self.df)}")
    

    def prepare_enhanced_data(self) -> pd.DataFrame:
        """Prepare data with multiple popularity measures."""
        print("Preparing topic data with enhanced popularity measures...")
        df = self.df.copy()
        
        # Create 'is_popular_binary' column from 'group' for binary models
        if 'group' in df.columns:
            df['is_popular_binary'] = 0
            df.loc[df['group'] == 'popular', 'is_popular_binary'] = 1
            df.loc[df['group'] == 'niche', 'is_popular_binary'] = -1
        else:
            raise ValueError("'group' column not found in DataFrame. Ensure the 'compare' step ran successfully.")

        # Create multiple popularity measures
        df = self._create_popularity_measures(df)
        
        # Create size variables
        df['num_authors'] = df['collaboration_papers'] * 2
        df['log_num_authors'] = np.log1p(df['num_authors'])
        
        # Clean data
        df = self._clean_network_data(df)
        
        print(f"Prepared {len(df)} topics for enhanced regression analysis")
        self._print_popularity_distribution(df)
        
        return df
    
    def _create_popularity_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multiple continuous and binary popularity measures."""
        df = df.copy()
        
        # 1. Original binary classification (top/bottom 20%)
        df_sorted = df.sort_values('total_papers', ascending=False)
        n_topics = len(df_sorted)
        cutoff_size = int(n_topics * 0.2)
        
        df['is_popular_binary'] = 0
        df.loc[df_sorted.head(cutoff_size).index, 'is_popular_binary'] = 1
        df.loc[df_sorted.tail(cutoff_size).index, 'is_popular_binary'] = -1  # Niche topics
        
        # 2. Log-transformed popularity (continuous)
        df['log_total_papers'] = np.log1p(df['total_papers'])
        
        # 3. Popularity rank (1 = most popular, N = least popular)
        df['popularity_rank'] = df['total_papers'].rank(method='min', ascending=False)
        
        # 4. Popularity percentile (0-100, where 100 = most popular)
        df['popularity_percentile'] = df['total_papers'].rank(pct=True) * 100
        
        # 5. Standardized popularity z-score
        scaler = StandardScaler()
        df['popularity_zscore'] = scaler.fit_transform(df[['total_papers']]).flatten()
        
        # 6. Robust popularity measures (less sensitive to outliers)
        robust_scaler = RobustScaler()
        df['popularity_robust'] = robust_scaler.fit_transform(df[['total_papers']]).flatten()
        
        # 7. Inverted rank for easier interpretation (1 = least popular, N = most popular)
        df['popularity_rank_inv'] = n_topics - df['popularity_rank'] + 1
        
        # 8. Normalized rank (0-1 scale)
        df['popularity_rank_norm'] = (df['popularity_rank_inv'] - 1) / (n_topics - 1)
        
        return df
    
    def _clean_network_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean network metrics data."""
        df = df.copy()
        
        for metric in self.network_metrics:
            if metric not in df.columns:
                continue
                
            # Handle infinite values
            df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)
            
            # For proportion metrics, ensure they're in [0, 1]
            if metric in ['collaboration_rate', 'repeated_collaboration_rate', 'degree_centralization', 
                         'modularity', 'coreness_ratio', 'avg_constraint']:
                df[metric] = df[metric].clip(0, 1)
            
            # Remove extreme outliers (beyond 3 standard deviations)
            if df[metric].notna().sum() > 10:
                z_scores = np.abs(stats.zscore(df[metric], nan_policy='omit'))
                df.loc[z_scores > 3, metric] = np.nan
        
        return df
    
    def _print_popularity_distribution(self, df: pd.DataFrame):
        """Print summary of popularity measures."""
        print("\nPopularity Measures Summary:")
        print(f"  Total papers range: {df['total_papers'].min():.0f} - {df['total_papers'].max():.0f}")
        print(f"  Mean papers: {df['total_papers'].mean():.1f}")
        print(f"  Median papers: {df['total_papers'].median():.1f}")
        print(f"  Popular topics (binary): {(df['is_popular_binary'] == 1).sum()}")
        print(f"  Niche topics (binary): {(df['is_popular_binary'] == -1).sum()}")
        print(f"  Log popularity range: {df['log_total_papers'].min():.2f} - {df['log_total_papers'].max():.2f}")
        print(f"  Popularity percentile range: {df['popularity_percentile'].min():.1f} - {df['popularity_percentile'].max():.1f}")
    
    def standardize_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize variables for regression analysis."""
        df_std = df.copy()
        
        # Variables to standardize
        continuous_vars = (['log_num_authors', 'log_total_papers', 'popularity_zscore', 
                           'popularity_percentile', 'popularity_rank_norm', 'popularity_robust'] + 
                          self.network_metrics)
        
        scaler = StandardScaler()
        
        for var in continuous_vars:
            if var in df_std.columns and df_std[var].notna().sum() > 10:
                non_missing_mask = df_std[var].notna()
                df_std.loc[non_missing_mask, f'{var}_std'] = scaler.fit_transform(
                    df_std.loc[non_missing_mask, var].values.reshape(-1, 1)
                ).flatten()
        
        return df_std
    
    def run_enhanced_regression_analysis(self, df: pd.DataFrame) -> dict:
        """
        Run enhanced regression analysis with multiple popularity measures.
        """
        print("Running enhanced regression analysis with continuous popularity measures...")
        
        # Standardize variables
        df_std = self.standardize_variables(df)
        
        results = {}
        
        for metric in self.network_metrics:
            print(f"  Analyzing {metric}...")
            
            try:
                metric_std = f'{metric}_std'
                
                # Prepare data for this metric
                metric_df = df_std.dropna(subset=[metric_std])
                
                if len(metric_df) < 50:
                    print(f"    Insufficient data for {metric} (n={len(metric_df)})")
                    continue
                
                # Run multiple model specifications
                models = self._fit_enhanced_models(metric_df, metric, metric_std)
                
                # Extract results
                metric_results = self._extract_enhanced_results(models, metric)
                
                results[metric] = metric_results
                
            except Exception as e:
                print(f"    Error analyzing {metric}: {e}")
                results[metric] = {'error': str(e)}
        
        return results
    
    def _fit_enhanced_models(self, df: pd.DataFrame, metric: str, metric_std: str) -> dict:
        """Fit multiple regression models with different popularity measures."""
        models = {}
        
        # Binary models (from original analysis)
        binary_df = df[df['is_popular_binary'] != 0].copy()
        binary_df['is_popular_binary_adj'] = (binary_df['is_popular_binary'] == 1).astype(int)
        
        try:
            # Binary simple model
            models['binary_simple'] = smf.ols(
                f"{metric_std} ~ is_popular_binary_adj", 
                data=binary_df
            ).fit()
            
            # Binary with size control
            models['binary_size'] = smf.ols(
                f"{metric_std} ~ is_popular_binary_adj + log_num_authors_std", 
                data=binary_df
            ).fit()
        except Exception as e:
            print(f"    Binary models failed: {e}")
        
        # Continuous models
        
        # 1. Log popularity (natural log of papers)
        try:
            models['log_simple'] = smf.ols(
                f"{metric_std} ~ log_total_papers_std", 
                data=df
            ).fit()
            
            models['log_size'] = smf.ols(
                f"{metric_std} ~ log_total_papers_std + log_num_authors_std", 
                data=df
            ).fit()
        except Exception as e:
            print(f"    Log models failed: {e}")
        
        # 2. Popularity percentile
        try:
            models['percentile_simple'] = smf.ols(
                f"{metric_std} ~ popularity_percentile_std", 
                data=df
            ).fit()
            
            models['percentile_size'] = smf.ols(
                f"{metric_std} ~ popularity_percentile_std + log_num_authors_std", 
                data=df
            ).fit()
        except Exception as e:
            print(f"    Percentile models failed: {e}")
        
        # 3. Popularity z-score
        try:
            models['zscore_simple'] = smf.ols(
                f"{metric_std} ~ popularity_zscore_std", 
                data=df
            ).fit()
            
            models['zscore_size'] = smf.ols(
                f"{metric_std} ~ popularity_zscore_std + log_num_authors_std", 
                data=df
            ).fit()
        except Exception as e:
            print(f"    Z-score models failed: {e}")
        
        # 4. Normalized rank (easier interpretation)
        try:
            models['rank_simple'] = smf.ols(
                f"{metric_std} ~ popularity_rank_norm_std", 
                data=df
            ).fit()
            
            models['rank_size'] = smf.ols(
                f"{metric_std} ~ popularity_rank_norm_std + log_num_authors_std", 
                data=df
            ).fit()
        except Exception as e:
            print(f"    Rank models failed: {e}")
        
        # 5. Robust popularity (less sensitive to outliers)
        try:
            models['robust_simple'] = smf.ols(
                f"{metric_std} ~ popularity_robust_std", 
                data=df
            ).fit()
            
            models['robust_size'] = smf.ols(
                f"{metric_std} ~ popularity_robust_std + log_num_authors_std", 
                data=df
            ).fit()
        except Exception as e:
            print(f"    Robust models failed: {e}")
        
        # 6. Polynomial models (quadratic relationships)
        try:
            # Add squared term for log popularity
            df['log_total_papers_std_sq'] = df['log_total_papers_std'] ** 2
            
            models['log_quadratic'] = smf.ols(
                f"{metric_std} ~ log_total_papers_std + log_total_papers_std_sq + log_num_authors_std", 
                data=df
            ).fit()
        except Exception as e:
            print(f"    Quadratic model failed: {e}")
        
        return models
    
    def _extract_enhanced_results(self, models: dict, metric: str) -> dict:
        """Extract results from enhanced regression models."""
        results = {
            'metric': metric,
            'models': {},
            'popularity_effects': {},
            'size_effects': {},
            'model_comparison': {},
            'best_model': None
        }
        
        popularity_vars = {
            'binary_simple': 'is_popular_binary_adj',
            'binary_size': 'is_popular_binary_adj',
            'log_simple': 'log_total_papers_std',
            'log_size': 'log_total_papers_std',
            'percentile_simple': 'popularity_percentile_std',
            'percentile_size': 'popularity_percentile_std',
            'zscore_simple': 'popularity_zscore_std',
            'zscore_size': 'popularity_zscore_std',
            'rank_simple': 'popularity_rank_norm_std',
            'rank_size': 'popularity_rank_norm_std',
            'robust_simple': 'popularity_robust_std',
            'robust_size': 'popularity_robust_std',
            'log_quadratic': 'log_total_papers_std'
        }
        
        aic_values = {}
        
        for model_name, model in models.items():
            if model is None:
                continue
            
            try:
                # Extract popularity coefficient
                pop_var = popularity_vars.get(model_name)
                if pop_var and pop_var in model.params.index:
                    pop_coef = model.params[pop_var]
                    pop_pvalue = model.pvalues[pop_var]
                    pop_ci = model.conf_int().loc[pop_var]
                    
                    results['popularity_effects'][model_name] = {
                        'coefficient': float(pop_coef),
                        'p_value': float(pop_pvalue),
                        'significant': pop_pvalue < 0.05,
                        'ci_lower': float(pop_ci[0]),
                        'ci_upper': float(pop_ci[1]),
                        'effect_size_interpretation': self._interpret_effect_size(abs(pop_coef)),
                        'popularity_variable': pop_var
                    }
                
                # Extract size coefficient
                if 'log_num_authors_std' in model.params.index:
                    size_coef = model.params['log_num_authors_std']
                    size_pvalue = model.pvalues['log_num_authors_std']
                    
                    results['size_effects'][model_name] = {
                        'coefficient': float(size_coef),
                        'p_value': float(size_pvalue),
                        'significant': size_pvalue < 0.05
                    }
                
                # Model fit statistics
                results['models'][model_name] = {
                    'r_squared': float(model.rsquared),
                    'r_squared_adj': float(model.rsquared_adj),
                    'aic': float(model.aic),
                    'bic': float(model.bic),
                    'f_statistic': float(model.fvalue),
                    'f_pvalue': float(model.f_pvalue),
                    'n_obs': int(model.nobs)
                }
                
                aic_values[model_name] = model.aic
                
            except Exception as e:
                print(f"    Error extracting results for {model_name}: {e}")
        
        # Determine best model by AIC
        if aic_values:
            best_model_name = min(aic_values.keys(), key=lambda x: aic_values[x])
            results['best_model'] = {
                'model_name': best_model_name,
                'aic': aic_values[best_model_name],
                'aic_comparison': {name: aic - aic_values[best_model_name] 
                                 for name, aic in aic_values.items()}
            }
        
        # Model comparison insights
        self._add_model_comparisons(results)
        
        return results
    
    def _add_model_comparisons(self, results: dict):
        """Add model comparison insights."""
        models = results['models']
        pop_effects = results['popularity_effects']
        
        # Compare binary vs continuous approaches
        binary_models = [name for name in models.keys() if 'binary' in name]
        continuous_models = [name for name in models.keys() if 'binary' not in name]
        
        if binary_models and continuous_models:
            # Best of each type
            best_binary = max(binary_models, key=lambda x: models[x]['r_squared'])
            best_continuous = max(continuous_models, key=lambda x: models[x]['r_squared'])
            
            results['model_comparison']['binary_vs_continuous'] = {
                'best_binary': {
                    'model': best_binary,
                    'r_squared': models[best_binary]['r_squared']
                },
                'best_continuous': {
                    'model': best_continuous,
                    'r_squared': models[best_continuous]['r_squared']
                },
                'continuous_better': models[best_continuous]['r_squared'] > models[best_binary]['r_squared']
            }
        
        # Effect consistency across models
        significant_effects = [name for name, effect in pop_effects.items() 
                             if effect['significant']]
        
        results['model_comparison']['effect_consistency'] = {
            'total_models': len(pop_effects),
            'significant_models': len(significant_effects),
            'consistency_rate': len(significant_effects) / len(pop_effects) if pop_effects else 0,
            'consistent_direction': self._check_direction_consistency(pop_effects)
        }
    
    def _check_direction_consistency(self, effects: dict) -> bool:
        """Check if effects are in consistent direction across models."""
        if not effects:
            return False
        
        significant_effects = [(name, data['coefficient']) for name, data in effects.items() 
                             if data['significant']]
        
        if len(significant_effects) < 2:
            return True
        
        # Check if all significant effects have same sign
        signs = [np.sign(coef) for _, coef in significant_effects]
        return len(set(signs)) <= 1
    
    def _interpret_effect_size(self, abs_coef: float) -> str:
        """Interpret standardized coefficient as effect size."""
        if abs_coef < 0.2:
            return "Small effect"
        elif abs_coef < 0.5:
            return "Medium effect"
        elif abs_coef < 0.8:
            return "Large effect"
        else:
            return "Very large effect"
    
    def create_enhanced_publication_table(self, regression_results: dict) -> pd.DataFrame:
        """Create publication table comparing binary and continuous approaches."""
        
        pub_rows = []
        
        for metric, results in regression_results.items():
            if 'error' in results:
                continue
            
            row = {
                'Metric': metric.replace('_', ' ').title(),
                'Binary Model β': '',
                'Binary p-value': '',
                'Best Continuous Model': '',
                'Continuous β': '',
                'Continuous p-value': '',
                'R² Binary': '',
                'R² Continuous': '',
                'Continuous Better': '',
                'Effect Consistency': ''
            }
            
            # Binary model results
            if 'binary_size' in results['popularity_effects']:
                binary = results['popularity_effects']['binary_size']
                row['Binary Model β'] = f"{binary['coefficient']:.3f}"
                row['Binary p-value'] = f"{binary['p_value']:.3f}" if binary['p_value'] >= 0.001 else "<0.001"
                row['R² Binary'] = f"{results['models']['binary_size']['r_squared']:.3f}"
            
            # Best continuous model results
            if results['best_model']:
                best_name = results['best_model']['model_name']
                if best_name in results['popularity_effects'] and 'binary' not in best_name:
                    best_continuous = results['popularity_effects'][best_name]
                    row['Best Continuous Model'] = best_name.replace('_', ' ').title()
                    row['Continuous β'] = f"{best_continuous['coefficient']:.3f}"
                    row['Continuous p-value'] = f"{best_continuous['p_value']:.3f}" if best_continuous['p_value'] >= 0.001 else "<0.001"
                    row['R² Continuous'] = f"{results['models'][best_name]['r_squared']:.3f}"
            
            # Model comparison
            if 'binary_vs_continuous' in results['model_comparison']:
                comparison = results['model_comparison']['binary_vs_continuous']
                row['Continuous Better'] = "Yes" if comparison['continuous_better'] else "No"
            
            # Effect consistency
            if 'effect_consistency' in results['model_comparison']:
                consistency = results['model_comparison']['effect_consistency']
                rate = consistency['consistency_rate']
                direction = consistency['consistent_direction']
                
                if rate >= 0.8 and direction:
                    row['Effect Consistency'] = "High"
                elif rate >= 0.5 and direction:
                    row['Effect Consistency'] = "Medium"
                else:
                    row['Effect Consistency'] = "Low"
            
            pub_rows.append(row)
        
        return pd.DataFrame(pub_rows)
    
    def print_enhanced_summary(self, regression_results: dict):
        """Print comprehensive summary of enhanced regression analysis."""
        
        print("\n" + "="*90)
        print("ENHANCED REGRESSION ANALYSIS: BINARY vs CONTINUOUS POPULARITY")
        print("="*90)
        
        # Model types tested
        print("\n📊 POPULARITY MEASURES TESTED:")
        print("• Binary: Popular vs Niche (top/bottom 20%)")
        print("• Log Papers: Natural log transformation")
        print("• Percentile: 0-100 popularity percentile")
        print("• Z-score: Standardized popularity score")
        print("• Rank: Normalized popularity ranking")
        print("• Robust: Outlier-resistant popularity measure")
        
        # Results summary
        metrics_with_results = [m for m, r in regression_results.items() if 'error' not in r]
        
        print(f"\n🎯 ANALYSIS SUMMARY:")
        print(f"• {len(metrics_with_results)} metrics successfully analyzed")
        print(f"• Multiple popularity measures tested per metric")
        
        # Binary vs continuous comparison
        binary_better = 0
        continuous_better = 0
        
        for metric, results in regression_results.items():
            if 'error' in results or 'binary_vs_continuous' not in results.get('model_comparison', {}):
                continue
            
            if results['model_comparison']['binary_vs_continuous']['continuous_better']:
                continuous_better += 1
            else:
                binary_better += 1
        
        print(f"\n📈 BINARY vs CONTINUOUS COMPARISON:")
        print(f"• Continuous models better: {continuous_better} metrics")
        print(f"• Binary models better: {binary_better} metrics")
        
        if continuous_better > binary_better:
            print(f"• 🏆 FINDING: Continuous popularity measures generally perform better")
        elif binary_better > continuous_better:
            print(f"• 🏆 FINDING: Binary popularity classification performs better")
        else:
            print(f"• 🏆 FINDING: Mixed results - both approaches have merit")
        
        # Effect consistency analysis
        high_consistency = []
        low_consistency = []
        
        for metric, results in regression_results.items():
            if 'error' in results or 'effect_consistency' not in results.get('model_comparison', {}):
                continue
            
            consistency = results['model_comparison']['effect_consistency']['consistency_rate']
            if consistency >= 0.8:
                high_consistency.append(metric)
            elif consistency < 0.5:
                low_consistency.append(metric)
        
        print(f"\n🔍 EFFECT CONSISTENCY ACROSS MODELS:")
        print(f"• High consistency (≥80% models agree): {len(high_consistency)} metrics")
        for metric in high_consistency:
            print(f"    ✅ {metric}")
        
        print(f"• Low consistency (<50% models agree): {len(low_consistency)} metrics")
        for metric in low_consistency:
            print(f"    ⚠️  {metric}")
        
        # Best models summary
        print(f"\n🏆 BEST PERFORMING MODELS:")
        best_models_count = {}
        
        for metric, results in regression_results.items():
            if 'error' in results or not results.get('best_model'):
                continue
            
            best_name = results['best_model']['model_name']
            model_type = best_name.split('_')[0]  # e.g., 'log', 'percentile', 'binary'
            best_models_count[model_type] = best_models_count.get(model_type, 0) + 1
        
        for model_type, count in sorted(best_models_count.items(), key=lambda x: x[1], reverse=True):
            print(f"• {model_type.title()} models: {count} metrics")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Determine dominant finding
        total_continuous = sum(v for k, v in best_models_count.items() if k != 'binary')
        total_binary = best_models_count.get('binary', 0)
        
        if total_continuous > total_binary * 1.5:
            print(f"• 🎯 MAIN FINDING: Continuous popularity measures provide superior model fit")
            print(f"• This suggests popularity effects are gradual rather than threshold-based")
        elif total_binary > total_continuous * 1.5:
            print(f"• 🎯 MAIN FINDING: Binary popularity classification is sufficient")
            print(f"• This suggests clear threshold effects between popular and niche topics")
        else:
            print(f"• 🎯 MAIN FINDING: Both binary and continuous approaches capture important aspects")
            print(f"• Different metrics may respond differently to popularity measurement approaches")
        
        if high_consistency:
            print(f"• Effect robustness: {len(high_consistency)} metrics show consistent effects across multiple popularity measures")
        
        print(f"• Methodological insight: Testing multiple popularity measures strengthens causal inference")


def main():
    """Run the enhanced regression analysis with continuous popularity measures."""
    
    # Read in input path for loading data   
    config = ConfigManager()
    input_path = config.get_path('topic_classifications_path')
    df = pd.read_csv(input_path)

    # Initialize enhanced analyzer
    analyzer = EnhancedNetworkRegressionAnalyzer(df)
    
    # Load and prepare data with enhanced popularity measures
    prepared_df = analyzer.prepare_enhanced_data()
    
    # Run enhanced regression analysis
    regression_results = analyzer.run_enhanced_regression_analysis(prepared_df)
    
    # Create enhanced publication table
    publication_table = analyzer.create_enhanced_publication_table(regression_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = analyzer.results_dir / f"enhanced_regression_analysis_{timestamp}.json"
    
    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    clean_results = clean_for_json(regression_results)
    
    with open(results_file, 'w') as f:
        json.dump({
            'regression_results': clean_results,
            'analysis_metadata': {
                'timestamp': timestamp,
                'n_metrics_analyzed': len(regression_results),
                'popularity_measures': [
                    'binary_classification',
                    'log_total_papers', 
                    'popularity_percentile',
                    'popularity_zscore',
                    'popularity_rank_normalized',
                    'popularity_robust'
                ],
                'models_per_metric': [
                    'binary_simple', 'binary_size',
                    'log_simple', 'log_size',
                    'percentile_simple', 'percentile_size',
                    'zscore_simple', 'zscore_size',
                    'rank_simple', 'rank_size',
                    'robust_simple', 'robust_size',
                    'log_quadratic'
                ],
                'note': 'Enhanced analysis comparing binary vs continuous popularity measures'
            }
        }, f, indent=2)
    
    # Save publication table
    pub_file = analyzer.results_dir / f"enhanced_publication_table_{timestamp}.csv"
    publication_table.to_csv(pub_file, index=False)
    
    # Print enhanced summary
    analyzer.print_enhanced_summary(regression_results)
    
    print(f"\n📁 Enhanced results saved with timestamp: {timestamp}")
    print(f"   • enhanced_regression_analysis_{timestamp}.json")
    print(f"   • enhanced_publication_table_{timestamp}.csv")
    
    print(f"\n🎯 ENHANCED ANALYSIS COMPLETE!")
    print(f"   This analysis tests both binary and continuous popularity measures,")
    print(f"   providing stronger evidence for your findings beyond arbitrary cutoffs.")
    
    # Additional analysis: Create visualization comparing approaches
    create_comparison_visualization(regression_results, analyzer.results_dir, timestamp)


def create_comparison_visualization(results: dict, results_dir: Path, timestamp: str):
    """Create visualizations comparing binary vs continuous approaches."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Extract R-squared values for comparison
        binary_r2 = []
        continuous_r2 = []
        metric_names = []
        
        for metric, data in results.items():
            if 'error' in data:
                continue
                
            # Get binary R-squared (use size-controlled model)
            if 'binary_size' in data.get('models', {}):
                binary_r2.append(data['models']['binary_size']['r_squared'])
            else:
                binary_r2.append(0)
            
            # Get best continuous R-squared
            continuous_models = {name: model for name, model in data.get('models', {}).items() 
                               if 'binary' not in name}
            if continuous_models:
                best_continuous_r2 = max(model['r_squared'] for model in continuous_models.values())
                continuous_r2.append(best_continuous_r2)
            else:
                continuous_r2.append(0)
            
            metric_names.append(metric.replace('_', ' ').title())
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: R-squared comparison
        x_pos = np.arange(len(metric_names))
        width = 0.35
        
        ax1.bar(x_pos - width/2, binary_r2, width, label='Binary (Popular vs Niche)', alpha=0.8)
        ax1.bar(x_pos + width/2, continuous_r2, width, label='Best Continuous Model', alpha=0.8)
        
        ax1.set_xlabel('Network Metrics')
        ax1.set_ylabel('R-squared')
        ax1.set_title('Model Fit Comparison: Binary vs Continuous Popularity')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement in R-squared
        improvement = np.array(continuous_r2) - np.array(binary_r2)
        colors = ['green' if imp > 0 else 'red' for imp in improvement]
        
        ax2.bar(x_pos, improvement, color=colors, alpha=0.7)
        ax2.set_xlabel('Network Metrics')
        ax2.set_ylabel('R-squared Improvement\n(Continuous - Binary)')
        ax2.set_title('Improvement from Continuous Popularity Measures')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metric_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = results_dir / f"binary_vs_continuous_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   • binary_vs_continuous_comparison_{timestamp}.png")
        
        # Summary statistics
        n_improved = sum(1 for imp in improvement if imp > 0)
        avg_improvement = np.mean(improvement)
        
        print(f"\n📊 VISUALIZATION SUMMARY:")
        print(f"   • {n_improved}/{len(improvement)} metrics improved with continuous measures")
        print(f"   • Average R-squared improvement: {avg_improvement:.3f}")
        
    except ImportError:
        print("   Note: Matplotlib not available for visualization")
    except Exception as e:
        print(f"   Warning: Visualization creation failed: {e}")


if __name__ == "__main__":
    main()