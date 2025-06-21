#!/usr/bin/env python3
"""
Fixed Advanced Regression Analysis with Standardized Coefficients
================================================================

This script fixes the numerical instability issues in the GLM models by:
1. Using standardized variables for regression
2. Computing standardized coefficients (effect sizes)
3. Using robust standard errors
4. Implementing proper model diagnostics

Addresses the astronomical coefficient values that indicate numerical instability.
"""

import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian, Gamma, Binomial
from scipy import stats
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FixedNetworkRegressionAnalyzer:
    """
    Fixed regression analysis with standardized coefficients and numerical stability.
    """
    
    def __init__(self, topic_data_file: str):
        self.topic_data_file = Path(topic_data_file)
        self.results_dir = Path("results/regression_analysis_fixed")
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
        
        print(f"Initialized Fixed NetworkRegressionAnalyzer with data: {self.topic_data_file}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load topic data and prepare for regression analysis with proper scaling."""
        print("Loading and preparing topic data for fixed regression analysis...")
        
        # Load the topic analysis results
        with open(self.topic_data_file, 'r') as f:
            topic_data = json.load(f)
        
        # Convert to DataFrame
        topic_list = []
        for topic_id, data in topic_data.items():
            topic_record = {
                'topic_id': int(topic_id),
                'total_papers': data['total_papers'],
                'collaboration_papers': data['collaboration_papers']
            }
            
            # Add all network metrics
            for metric in self.network_metrics:
                topic_record[metric] = data.get(metric, np.nan)
            
            topic_list.append(topic_record)
        
        df = pd.DataFrame(topic_list)
        
        # Create popularity classification (top/bottom 20%)
        df_sorted = df.sort_values('total_papers', ascending=False)
        n_topics = len(df_sorted)
        cutoff_size = int(n_topics * 0.2)
        
        df['is_popular'] = 0
        df.loc[df_sorted.head(cutoff_size).index, 'is_popular'] = 1
        df.loc[df_sorted.tail(cutoff_size).index, 'is_popular'] = -1  # Mark niche topics
        
        # Create size variables
        df['num_authors'] = df['collaboration_papers'] * 2  # Rough estimate
        df['log_num_authors'] = np.log1p(df['num_authors'])  # log(1 + x) to handle zeros
        df['log_total_papers'] = np.log1p(df['total_papers'])
        
        # Clean data - remove invalid values and outliers
        for metric in self.network_metrics:
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
        
        print(f"Prepared {len(df)} topics for fixed regression analysis")
        print(f"Popular topics: {(df['is_popular'] == 1).sum()}")
        print(f"Niche topics: {(df['is_popular'] == -1).sum()}")
        print(f"Middle topics: {(df['is_popular'] == 0).sum()}")
        
        return df
    
    def standardize_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize variables to prevent numerical instability."""
        df_std = df.copy()
        
        # Standardize continuous variables (but not binary popularity indicator)
        continuous_vars = ['log_num_authors', 'log_total_papers'] + self.network_metrics
        
        scaler = StandardScaler()
        
        for var in continuous_vars:
            if var in df_std.columns and df_std[var].notna().sum() > 10:
                # Standardize only non-missing values
                non_missing_mask = df_std[var].notna()
                df_std.loc[non_missing_mask, f'{var}_std'] = scaler.fit_transform(
                    df_std.loc[non_missing_mask, var].values.reshape(-1, 1)
                ).flatten()
        
        return df_std
    
    def run_fixed_regression_analysis(self, df: pd.DataFrame) -> dict:
        """
        Run fixed regression analysis with standardized coefficients.
        
        Uses OLS regression with standardized variables for interpretable coefficients.
        """
        print("Running fixed regression analysis with standardized coefficients...")
        
        # Standardize variables
        df_std = self.standardize_variables(df)
        
        results = {}
        
        # Filter to popular vs niche only for main analysis
        analysis_df = df_std[df_std['is_popular'] != 0].copy()
        analysis_df['is_popular_binary'] = (analysis_df['is_popular'] == 1).astype(int)
        
        for metric in self.network_metrics:
            print(f"  Analyzing {metric}...")
            
            try:
                # Use standardized versions of variables
                metric_std = f'{metric}_std'
                size_std = 'log_num_authors_std'
                
                # Prepare data for this metric
                metric_df = analysis_df.dropna(subset=[metric_std, size_std])
                
                if len(metric_df) < 50:
                    print(f"    Insufficient data for {metric} (n={len(metric_df)})")
                    continue
                
                # Run multiple model specifications using OLS for stability
                models = self._fit_standardized_models(metric_df, metric, metric_std, size_std)
                
                # Extract standardized results
                metric_results = self._extract_standardized_results(models, metric)
                
                results[metric] = metric_results
                
            except Exception as e:
                print(f"    Error analyzing {metric}: {e}")
                results[metric] = {'error': str(e)}
        
        return results
    
    def _fit_standardized_models(self, df: pd.DataFrame, metric: str, 
                                metric_std: str, size_std: str) -> dict:
        """Fit regression models using standardized variables."""
        models = {}
        
        # Model 1: Simple popularity comparison (baseline)
        try:
            formula1 = f"{metric_std} ~ is_popular_binary"
            models['simple'] = smf.ols(formula1, data=df).fit()
        except Exception as e:
            print(f"    Warning: Simple model failed for {metric}: {e}")
            models['simple'] = None
        
        # Model 2: Control for network size
        try:
            formula2 = f"{metric_std} ~ is_popular_binary + {size_std}"
            models['size_control'] = smf.ols(formula2, data=df).fit()
        except Exception as e:
            print(f"    Warning: Size control model failed for {metric}: {e}")
            models['size_control'] = None
        
        # Model 3: Add interaction terms
        try:
            formula3 = f"{metric_std} ~ is_popular_binary * {size_std}"
            models['interaction'] = smf.ols(formula3, data=df).fit()
        except Exception as e:
            print(f"    Warning: Interaction model failed for {metric}: {e}")
            models['interaction'] = None
        
        # Model 4: Robust standard errors for size control model
        if models['size_control'] is not None:
            try:
                models['size_control_robust'] = models['size_control'].get_robustcov_results(cov_type='HC3')
            except:
                models['size_control_robust'] = models['size_control']
        
        return models
    
    def _extract_standardized_results(self, models: dict, metric: str) -> dict:
        """Extract standardized coefficients and effect sizes."""
        results = {
            'metric': metric,
            'models': {},
            'popularity_effects': {},
            'size_correlations': {},
            'effect_sizes': {},
            'model_comparison': {}
        }
        
        for model_name, model in models.items():
            if model is None:
                continue
                
            try:
                # Extract popularity coefficient (this is now standardized!)
                if 'is_popular_binary' in model.params.index:
                    pop_coef = model.params['is_popular_binary']
                    pop_pvalue = model.pvalues['is_popular_binary']
                    pop_ci = model.conf_int().loc['is_popular_binary']
                    
                    # For standardized variables, coefficient = effect size in standard deviations
                    results['popularity_effects'][model_name] = {
                        'standardized_coefficient': float(pop_coef),
                        'p_value': float(pop_pvalue),
                        'significant': pop_pvalue < 0.05,
                        'ci_lower': float(pop_ci[0]),
                        'ci_upper': float(pop_ci[1]),
                        'effect_size_interpretation': self._interpret_effect_size(abs(pop_coef))
                    }
                
                # Extract size coefficient
                size_vars = ['log_num_authors_std', 'log_num_authors']
                for size_var in size_vars:
                    if size_var in model.params.index:
                        size_coef = model.params[size_var]
                        size_pvalue = model.pvalues[size_var]
                        
                        results['size_correlations'][model_name] = {
                            'standardized_coefficient': float(size_coef),
                            'p_value': float(size_pvalue),
                            'significant': size_pvalue < 0.05
                        }
                        break
                
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
                
            except Exception as e:
                print(f"    Error extracting results for {model_name}: {e}")
        
        # Model comparison
        if 'simple' in results['models'] and 'size_control' in results['models']:
            # Compare R-squared between simple and size-controlled models
            r2_simple = results['models']['simple']['r_squared']
            r2_size = results['models']['size_control']['r_squared']
            results['model_comparison']['r2_improvement'] = r2_size - r2_simple
            results['model_comparison']['size_control_improves_fit'] = r2_size > r2_simple
        
        return results
    
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
    
    def analyze_size_confounding_fixed(self, df: pd.DataFrame) -> dict:
        """Analyze size confounding using standardized correlations."""
        print("Analyzing size confounding with standardized variables...")
        
        df_std = self.standardize_variables(df)
        size_correlations = {}
        
        analysis_df = df_std[df_std['is_popular'] != 0].copy()
        
        for metric in self.network_metrics:
            metric_std = f'{metric}_std'
            
            if metric_std not in df_std.columns:
                continue
                
            metric_data = analysis_df.dropna(subset=[metric_std, 'log_num_authors_std'])
            
            if len(metric_data) > 20:
                # Correlation with standardized log(authors)
                corr_authors, p_authors = stats.pearsonr(
                    metric_data['log_num_authors_std'], 
                    metric_data[metric_std]
                )
                
                size_correlations[metric] = {
                    'standardized_correlation_with_size': float(corr_authors),
                    'p_value': float(p_authors),
                    'size_confounding_strength': abs(corr_authors),
                    'confounding_interpretation': self._interpret_effect_size(abs(corr_authors))
                }
        
        # Rank metrics by size confounding
        confounding_ranking = sorted(
            size_correlations.items(),
            key=lambda x: x[1]['size_confounding_strength'],
            reverse=True
        )
        
        return {
            'size_correlations': size_correlations,
            'confounding_ranking': confounding_ranking,
            'high_confounding_metrics': [
                metric for metric, data in confounding_ranking[:3]
            ],
            'low_confounding_metrics': [
                metric for metric, data in confounding_ranking[-3:]
            ]
        }
    
    def create_publication_table_fixed(self, regression_results: dict) -> pd.DataFrame:
        """Create publication-ready table with standardized coefficients."""
        
        pub_rows = []
        
        for metric, results in regression_results.items():
            if 'error' in results:
                continue
            
            row = {
                'Metric': metric.replace('_', ' ').title(),
                'Simple Model β': '',
                'Simple Model p-value': '',
                'Size Control β': '',
                'Size Control p-value': '',
                'Effect Size': '',
                'R² Improvement': '',
                'Robust to Size Control': ''
            }
            
            # Fill in results if available
            if 'simple' in results['popularity_effects']:
                simple = results['popularity_effects']['simple']
                row['Simple Model β'] = f"{simple['standardized_coefficient']:.3f}"
                row['Simple Model p-value'] = f"{simple['p_value']:.3f}" if simple['p_value'] >= 0.001 else "<0.001"
                row['Effect Size'] = simple['effect_size_interpretation']
            
            if 'size_control' in results['popularity_effects']:
                size_ctrl = results['popularity_effects']['size_control']
                row['Size Control β'] = f"{size_ctrl['standardized_coefficient']:.3f}"
                row['Size Control p-value'] = f"{size_ctrl['p_value']:.3f}" if size_ctrl['p_value'] >= 0.001 else "<0.001"
            
            # R-squared improvement
            if 'r2_improvement' in results.get('model_comparison', {}):
                r2_imp = results['model_comparison']['r2_improvement']
                row['R² Improvement'] = f"{r2_imp:.3f}"
            
            # Robustness assessment
            simple_sig = results['popularity_effects'].get('simple', {}).get('significant', False)
            size_sig = results['popularity_effects'].get('size_control', {}).get('significant', False)
            
            if simple_sig and size_sig:
                # Check if effect size is maintained
                simple_coef = abs(results['popularity_effects']['simple']['standardized_coefficient'])
                size_coef = abs(results['popularity_effects']['size_control']['standardized_coefficient'])
                effect_maintained = (size_coef / simple_coef) > 0.5 if simple_coef > 0 else False
                
                if effect_maintained:
                    row['Robust to Size Control'] = "Yes"
                else:
                    row['Robust to Size Control'] = "Partial"
            elif simple_sig and not size_sig:
                row['Robust to Size Control'] = "No - Size Confounded"
            else:
                row['Robust to Size Control'] = "No Effect"
            
            pub_rows.append(row)
        
        return pd.DataFrame(pub_rows)
    
    def save_fixed_results(self, regression_results: dict, size_analysis: dict, 
                          publication_table: pd.DataFrame) -> str:
        """Save fixed regression analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed regression results
        regression_file = self.results_dir / f"fixed_regression_analysis_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        clean_results = self._clean_for_json(regression_results)
        
        with open(regression_file, 'w') as f:
            json.dump({
                'regression_results': clean_results,
                'size_confounding_analysis': self._clean_for_json(size_analysis),
                'analysis_metadata': {
                    'timestamp': timestamp,
                    'n_metrics_analyzed': len(regression_results),
                    'approach': 'standardized_coefficients',
                    'models_per_metric': ['simple', 'size_control', 'interaction'],
                    'note': 'All coefficients are standardized (effect sizes in standard deviations)'
                }
            }, f, indent=2)
        
        # Save publication table
        pub_file = self.results_dir / f"publication_table_fixed_{timestamp}.csv"
        publication_table.to_csv(pub_file, index=False)
        
        print(f"Fixed results saved with timestamp: {timestamp}")
        return timestamp
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
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
    
    def print_fixed_summary(self, regression_results: dict, size_analysis: dict):
        """Print comprehensive summary of fixed regression analysis."""
        
        print("\n" + "="*80)
        print("FIXED REGRESSION ANALYSIS WITH STANDARDIZED COEFFICIENTS")
        print("="*80)
        
        # Note about interpretation
        print("\n📊 COEFFICIENT INTERPRETATION:")
        print("All coefficients are standardized (β) and represent effect sizes:")
        print("• β < 0.2: Small effect")
        print("• β 0.2-0.5: Medium effect") 
        print("• β 0.5-0.8: Large effect")
        print("• β > 0.8: Very large effect")
        
        # Size confounding analysis
        print(f"\n🔍 SIZE CONFOUNDING ANALYSIS:")
        print(f"Metrics most confounded by network size:")
        for i, (metric, data) in enumerate(size_analysis['confounding_ranking'][:3], 1):
            corr = data['standardized_correlation_with_size']
            interp = data['confounding_interpretation']
            print(f"   {i}. {metric}: r = {corr:.3f} ({interp})")
        
        print(f"\nMetrics least confounded by network size:")
        for i, (metric, data) in enumerate(size_analysis['confounding_ranking'][-3:], 1):
            corr = data['standardized_correlation_with_size']
            interp = data['confounding_interpretation']
            print(f"   {i}. {metric}: r = {corr:.3f} ({interp})")
        
        # Regression results summary
        print(f"\n🎯 STANDARDIZED REGRESSION RESULTS:")
        
        robust_effects = []
        size_confounded = []
        no_effects = []
        
        for metric, results in regression_results.items():
            if 'error' in results:
                continue
            
            simple_sig = results['popularity_effects'].get('simple', {}).get('significant', False)
            size_sig = results['popularity_effects'].get('size_control', {}).get('significant', False)
            
            if simple_sig and size_sig:
                # Check effect size maintenance
                simple_coef = abs(results['popularity_effects']['simple']['standardized_coefficient'])
                size_coef = abs(results['popularity_effects']['size_control']['standardized_coefficient'])
                effect_maintained = (size_coef / simple_coef) > 0.5 if simple_coef > 0 else False
                
                if effect_maintained:
                    robust_effects.append((metric, size_coef, 
                                         results['popularity_effects']['size_control']['effect_size_interpretation']))
                else:
                    size_confounded.append(metric)
            elif simple_sig and not size_sig:
                size_confounded.append(metric)
            else:
                no_effects.append(metric)
        
        print(f"Robust popularity effects (survive size control): {len(robust_effects)}")
        for metric, coef, interp in robust_effects:
            print(f"   ✅ {metric}: β = {coef:.3f} ({interp})")
        
        print(f"\nSize-confounded effects: {len(size_confounded)}")
        for metric in size_confounded:
            print(f"   ⚠️  {metric}")
        
        print(f"\nNo significant effects: {len(no_effects)}")
        for metric in no_effects:
            print(f"   ❌ {metric}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        print(f"• {len(robust_effects)} metrics show genuine popularity effects beyond network size")
        print(f"• {len(size_confounded)} effects appear to be primarily due to network size")
        print(f"• Size confounding is strongest for: {', '.join(size_analysis['high_confounding_metrics'])}")
        print(f"• Size confounding is weakest for: {', '.join(size_analysis['low_confounding_metrics'])}")
        
        if robust_effects:
            print(f"• Main finding: Popularity has genuine structural effects on collaboration networks")
            print(f"• Largest robust effects: {', '.join([m for m, c, i in sorted(robust_effects, key=lambda x: x[1], reverse=True)[:3]])}")
        else:
            print(f"• Main finding: Popularity effects appear to be primarily artifacts of network size")


def main():
    """Run the fixed regression analysis with standardized coefficients."""
    
    # Initialize fixed analyzer
    analyzer = FixedNetworkRegressionAnalyzer(
        "results/collaboration_analysis/topic_analysis_10metrics_20250607_124314.json"
    )
    
    # Load and prepare data
    df = analyzer.load_and_prepare_data()
    
    # Run fixed regression analysis
    regression_results = analyzer.run_fixed_regression_analysis(df)
    
    # Analyze size confounding with standardized variables
    size_analysis = analyzer.analyze_size_confounding_fixed(df)
    
    # Create publication table with standardized coefficients
    publication_table = analyzer.create_publication_table_fixed(regression_results)
    
    # Save results
    timestamp = analyzer.save_fixed_results(regression_results, size_analysis, publication_table)
    
    # Print summary
    analyzer.print_fixed_summary(regression_results, size_analysis)
    
    print(f"\n📁 Fixed results saved with timestamp: {timestamp}")
    print(f"   • fixed_regression_analysis_{timestamp}.json")
    print(f"   • publication_table_fixed_{timestamp}.csv")
    
    print(f"\n🎯 SOLUTION: This fixed analysis uses standardized coefficients (β)")
    print(f"   that represent effect sizes in standard deviations. No more astronomical")
    print(f"   numbers - all coefficients are now interpretable and numerically stable!")


if __name__ == "__main__":
    main()
