#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

"""
Collaboration Network Analysis for Math Research Compass - Version 5
==============================================================================
For each of the 1,938 research topics, we constructed an undirected co-authorship network
where nodes represent authors and weighted edges represent collaborations. 
On each of these networks, we systematically computed ten established network metrics to quantify their structural properties. 
These metrics measured collaboration dynamics, network topology, structural resilience, and researcher positioning. 
The resulting dataset of 1,938 topics, each described by ten structural metrics, formed the basis for our statistical analysis.

Analyzes collaboration patterns with the 10 core network measures:
1. Collaboration Rate
2. Repeated Collaboration Rate  
3. Degree Centralization
4. Degree Assortativity
5. Modularity
6. Small World Coefficient (FIXED - now uses LCC for both actual and expected values)
7. Coreness (K-core ratio)
8. Robustness Ratio
9. Structural Holes Constraint
10. Effective Network Size (Brokerage)
"""

import pandas as pd
import networkx as nx
import numpy as np
import ast
import json
import random
from datetime import datetime
from collections import defaultdict, Counter
import logging
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# CONFIG 
from src.config_manager import ConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollaborationNetworkAnalyzer:
    """
    Collaboration network analyzer with 10 core topological measures.
    Version 5 with fixed Small World Coefficient calculation.
    """
    
    def __init__(self, data_path: str = "data/cleaned/author_topic_networks_disambiguated_v4.csv"):
        self.data_path = Path(data_path)
        self.results_dir = Path("results/collaboration_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.topic_results = {}
        self.cross_topic_results = {}
        
        # Storage for small-world diagnostics
        self.small_world_diagnostics = {}
        
        logger.info(f"Initialized CollaborationNetworkAnalyzer V5 with data: {self.data_path}")
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate the dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        logger.info("Loading collaboration dataset...")
        df = pd.read_csv(self.data_path)
        
        logger.info(f"Loaded {len(df)} papers")
        logger.info(f"Dataset columns: {list(df.columns)}")
        
        # Validate required columns
        required_cols = ['topic', 'authors_parsed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse authors and filter for collaboration papers
        df['authors_list'] = df['authors_parsed'].apply(self._parse_authors)
        df['num_authors'] = df['authors_list'].apply(len)
        
        # Keep all papers for team size analysis
        collaboration_df = df[df['num_authors'] > 0].copy()
        
        logger.info(f"Found {len(collaboration_df)} papers with valid author data")
        logger.info(f"Collaboration papers (>1 author): {len(collaboration_df[collaboration_df['num_authors'] > 1])}")
        
        return collaboration_df
    
    def _parse_authors(self, authors_str):
        """Parse authors from string format."""
        if pd.isna(authors_str) or not authors_str:
            return []
        
        try:
            authors_list = ast.literal_eval(authors_str)
            names = []
            for author_parts in authors_list:
                if isinstance(author_parts, list) and len(author_parts) >= 2:
                    last_name = str(author_parts[0]).strip()
                    first_name = str(author_parts[1]).strip()
                    if first_name and last_name:
                        names.append(f"{first_name} {last_name}")
                    elif last_name:
                        names.append(last_name)
            return names
        except (ValueError, SyntaxError, TypeError):
            return []
    
    def analyze_topic_networks(self, df: pd.DataFrame, sample_topics: list = None, 
                             enable_diagnostics: bool = True) -> dict:
        """
        Analyze collaboration networks for each topic with the 10 core metrics.
        
        Args:
            df: DataFrame with paper data
            sample_topics: List of specific topics to analyze (None = all topics)
            enable_diagnostics: Whether to collect detailed small-world diagnostics
        """
        logger.info("Starting 10-measure topic-wise network analysis...")
        
        # Store diagnostic mode
        self.enable_diagnostics = enable_diagnostics
        
        topics = sample_topics if sample_topics else sorted(df['topic'].unique())
        logger.info(f"Analyzing {len(topics)} topics")
        
        topic_results = {}
        
        for i, topic_id in enumerate(topics):
            if i % 100 == 0:
                logger.info(f"Processing topic {topic_id} ({i+1}/{len(topics)})")
            
            topic_df = df[df['topic'] == topic_id]
            if len(topic_df) == 0:
                continue
            
            # Analyze this topic with the 10 core metrics
            topic_analysis = self._analyze_single_topic(topic_id, topic_df)
            topic_results[topic_id] = topic_analysis
        
        logger.info(f"Completed analysis for {len(topic_results)} topics")
        return topic_results
    
    def _analyze_single_topic(self, topic_id: int, topic_df: pd.DataFrame) -> dict:
        """Analyze collaboration patterns for a single topic with 10 core metrics."""
        
        # Filter for collaboration papers (>1 author)
        collab_df = topic_df[topic_df['num_authors'] > 1]
        
        if len(collab_df) == 0:
            return self._get_empty_metrics(topic_id, len(topic_df))
        
        # Build collaboration network
        G = self._build_collaboration_network(collab_df)
        
        # Calculate all 10 core metrics
        metrics = self._calculate_all_core_metrics(G, topic_df, collab_df, topic_id)
        metrics['topic_id'] = topic_id
        metrics['total_papers'] = len(topic_df)
        metrics['collaboration_papers'] = len(collab_df)
        
        return metrics
    
    def _get_empty_metrics(self, topic_id: int, total_papers: int) -> dict:
        """Return empty metrics for topics with no collaboration."""
        return {
            'topic_id': topic_id,
            'total_papers': total_papers,
            'collaboration_papers': 0,
            'collaboration_rate': 0.0,
            'repeated_collaboration_rate': 0.0,
            'degree_centralization': 0.0,
            'degree_assortativity': 0.0,
            'modularity': 0.0,
            'small_world_coefficient': 0.0,
            'coreness_ratio': 0.0,
            'robustness_ratio': 1.0,
            'avg_constraint': 0.0,
            'avg_effective_size': 0.0
        }
    
    def _build_collaboration_network(self, papers_df: pd.DataFrame) -> nx.Graph:
        """Build collaboration network from papers."""
        G = nx.Graph()
        
        for _, paper in papers_df.iterrows():
            # Ensure unique authors per paper, preventing self-loops
            authors = list(set(paper['authors_list']))
            
            # Add nodes with paper counts
            for author in authors:
                if author not in G.nodes():
                    G.add_node(author, papers=0)
                G.nodes[author]['papers'] += 1
            
            # Add edges (collaborations)
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    if G.has_edge(author1, author2):
                        G[author1][author2]['weight'] += 1
                    else:
                        G.add_edge(author1, author2, weight=1)
        
        return G
    
    def _calculate_all_core_metrics(self, G: nx.Graph, topic_df: pd.DataFrame, 
                                  collab_df: pd.DataFrame, topic_id: int) -> dict:
        """Calculate all 10 core network metrics."""
        
        metrics = {}
        
        # 1. Collaboration Rate
        metrics['collaboration_rate'] = len(collab_df) / len(topic_df) if len(topic_df) > 0 else 0
        
        # 2. Repeated Collaboration Rate
        metrics['repeated_collaboration_rate'] = self._calculate_repeated_collaboration_rate(collab_df)
        
        # 3. Degree Centralization
        metrics['degree_centralization'] = self._calculate_degree_centralization(G)
        
        # 4. Degree Assortativity (NetworkX built-in)
        try:
            metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            metrics['degree_assortativity'] = 0.0
        
        # 5. Modularity (NetworkX with community detection)
        metrics['modularity'] = self._calculate_modularity(G)
        
        # 6. Small World Coefficient (FIXED VERSION)
        metrics['small_world_coefficient'] = self._calculate_small_world_coefficient_fixed(G, topic_id)
        
        # 7. Coreness (K-core ratio)
        metrics['coreness_ratio'] = self._calculate_coreness_ratio(G)
        
        # 8. Robustness Ratio
        metrics['robustness_ratio'] = self._calculate_robustness_ratio(G)
        
        # 9. Structural Holes Constraint (NetworkX built-in)
        metrics['avg_constraint'] = self._calculate_avg_constraint(G)
        
        # 10. Effective Network Size (NetworkX built-in)
        metrics['avg_effective_size'] = self._calculate_avg_effective_size(G)
        
        return metrics
    
    def _calculate_repeated_collaboration_rate(self, collab_df: pd.DataFrame) -> float:
        """Calculate repeated collaboration rate."""
        collaboration_pairs = defaultdict(int)
        
        for _, paper in collab_df.iterrows():
            authors = paper['authors_list']
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    pair = tuple(sorted([author1, author2]))
                    collaboration_pairs[pair] += 1
        
        if len(collaboration_pairs) == 0:
            return 0.0
        
        repeat_collaborations = sum(1 for count in collaboration_pairs.values() if count > 1)
        return repeat_collaborations / len(collaboration_pairs)
    
    def _calculate_degree_centralization(self, G: nx.Graph) -> float:
        """Calculate degree centralization."""
        if len(G) <= 2:
            return 0.0
        
        degrees = [d for n, d in G.degree()]
        max_degree = max(degrees) if degrees else 0
        sum_diff = sum(max_degree - d for d in degrees)
        max_possible = (len(G) - 1) * (len(G) - 2)
        
        return sum_diff / max_possible if max_possible > 0 else 0.0
    
    def _calculate_modularity(self, G: nx.Graph) -> float:
        """Calculate modularity using community detection."""
        if len(G) < 3 or G.number_of_edges() == 0:
            return 0.0
        
        try:
            # Try to use community detection
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                return community_louvain.modularity(partition, G)
            except ImportError:
                # Fallback to NetworkX community detection
                communities = list(nx.community.greedy_modularity_communities(G))
                return nx.community.modularity(G, communities)
        except:
            return 0.0
    
    def _calculate_small_world_coefficient_fixed(self, G: nx.Graph, topic_id: int) -> float:
        """
        FIXED small-world coefficient calculation.
        
        Key fix: Uses largest connected component (G_cc) characteristics for both 
        actual AND expected values, eliminating the critical inconsistency.
        """
        # Initialize diagnostic dict for this topic
        diagnostics = {
            'topic_id': topic_id,
            'original_num_nodes': len(G),
            'original_num_edges': G.number_of_edges(),
            'num_components': 0,
            'largest_cc_size': 0,
            'largest_cc_fraction': 0.0,
            'actual_clustering': None,
            'expected_clustering': None,
            'actual_path_length': None,
            'expected_path_length': None,
            'avg_degree_original': None,
            'avg_degree_cc': None,
            'edge_probability_cc': None,
            'clustering_ratio': None,
            'path_length_ratio': None,
            'failure_reason': None,
            'sw_coefficient': 0.0,
            'calculation_method': None
        }
        
        # Early exit conditions with diagnostics
        if len(G) < 10:
            diagnostics['failure_reason'] = f'Network too small: {len(G)} nodes < 10'
            if self.enable_diagnostics:
                self.small_world_diagnostics[topic_id] = diagnostics
            return 0.0
        
        if G.number_of_edges() < 5:
            diagnostics['failure_reason'] = f'Too few edges: {G.number_of_edges()} < 5'
            if self.enable_diagnostics:
                self.small_world_diagnostics[topic_id] = diagnostics
            return 0.0
        
        try:
            # Get connected components
            components = list(nx.connected_components(G))
            diagnostics['num_components'] = len(components)
            diagnostics['component_sizes'] = sorted([len(c) for c in components], reverse=True)
            
            if not components:
                diagnostics['failure_reason'] = 'No connected components found'
                if self.enable_diagnostics:
                    self.small_world_diagnostics[topic_id] = diagnostics
                return 0.0
            
            # Get largest connected component
            largest_cc = max(components, key=len)
            G_cc = G.subgraph(largest_cc)
            
            # Update diagnostics with LCC info
            diagnostics['largest_cc_size'] = len(G_cc)
            diagnostics['largest_cc_fraction'] = len(G_cc) / len(G)
            
            # Minimum size check for LCC
            if len(G_cc) < 5:
                diagnostics['failure_reason'] = f'Largest component too small: {len(G_cc)} nodes < 5'
                if self.enable_diagnostics:
                    self.small_world_diagnostics[topic_id] = diagnostics
                return 0.0
            
            # CRITICAL FIX: Calculate network properties based on G_cc, not G
            n_cc = len(G_cc)
            m_cc = G_cc.number_of_edges()
            avg_degree_cc = 2 * m_cc / n_cc if n_cc > 0 else 0
            
            # Store both original and LCC degree info for diagnostics
            diagnostics['avg_degree_original'] = 2 * G.number_of_edges() / len(G) if len(G) > 0 else 0
            diagnostics['avg_degree_cc'] = avg_degree_cc
            
            # Check if LCC has sufficient connectivity
            if avg_degree_cc <= 1.0:
                diagnostics['failure_reason'] = f'LCC average degree too low: {avg_degree_cc:.3f} <= 1.0'
                if self.enable_diagnostics:
                    self.small_world_diagnostics[topic_id] = diagnostics
                return 0.0
            
            # Calculate actual clustering on G_cc
            actual_clustering = nx.average_clustering(G_cc)
            diagnostics['actual_clustering'] = actual_clustering
            
            # Calculate actual path length on G_cc
            avg_path_length = self._calculate_average_path_length(G_cc, diagnostics)
            if avg_path_length is None:
                # Error details already in diagnostics
                if self.enable_diagnostics:
                    self.small_world_diagnostics[topic_id] = diagnostics
                return 0.0
            
            diagnostics['actual_path_length'] = avg_path_length
            
            # CRITICAL FIX: Calculate expected values based on G_cc characteristics
            # Edge probability for random graph with same n_cc and m_cc
            p_cc = 2 * m_cc / (n_cc * (n_cc - 1)) if n_cc > 1 else 0
            diagnostics['edge_probability_cc'] = p_cc
            
            # Expected clustering for Erdős-Rényi random graph
            expected_clustering = p_cc
            diagnostics['expected_clustering'] = expected_clustering
            
            # Expected path length for Erdős-Rényi random graph
            if avg_degree_cc > 1:
                expected_path_length = np.log(n_cc) / np.log(avg_degree_cc)
            else:
                diagnostics['failure_reason'] = f'LCC average degree too low for path length: {avg_degree_cc}'
                if self.enable_diagnostics:
                    self.small_world_diagnostics[topic_id] = diagnostics
                return 0.0
            
            diagnostics['expected_path_length'] = expected_path_length
            
            # Calculate small world coefficient
            if expected_clustering > 0 and expected_path_length > 0 and avg_path_length > 0:
                clustering_ratio = actual_clustering / expected_clustering
                path_length_ratio = avg_path_length / expected_path_length
                
                diagnostics['clustering_ratio'] = clustering_ratio
                diagnostics['path_length_ratio'] = path_length_ratio
                
                if path_length_ratio > 0:
                    sw_coefficient = clustering_ratio / path_length_ratio
                    diagnostics['sw_coefficient'] = sw_coefficient
                    diagnostics['calculation_method'] = 'erdos_renyi_null_model_lcc'
                    
                    if self.enable_diagnostics:
                        self.small_world_diagnostics[topic_id] = diagnostics
                    
                    return sw_coefficient
                else:
                    diagnostics['failure_reason'] = 'Path length ratio is zero'
            else:
                reasons = []
                if expected_clustering <= 0:
                    reasons.append(f'expected_clustering={expected_clustering:.6f}')
                if expected_path_length <= 0:
                    reasons.append(f'expected_path_length={expected_path_length:.6f}')
                if avg_path_length <= 0:
                    reasons.append(f'avg_path_length={avg_path_length:.6f}')
                diagnostics['failure_reason'] = f'Invalid expected values: {", ".join(reasons)}'
            
            if self.enable_diagnostics:
                self.small_world_diagnostics[topic_id] = diagnostics
            
            return 0.0
            
        except Exception as e:
            diagnostics['failure_reason'] = f'Unexpected error: {str(e)}'
            if self.enable_diagnostics:
                self.small_world_diagnostics[topic_id] = diagnostics
            return 0.0
    
    def _calculate_average_path_length(self, G_cc: nx.Graph, diagnostics: dict) -> float:
        """
        Calculate average shortest path length for the largest connected component.
        
        Args:
            G_cc: Largest connected component (guaranteed to be connected)
            diagnostics: Dictionary to store diagnostic information
        
        Returns:
            Average path length or None if calculation fails
        """
        try:
            # Double-check connectivity (should always be true for LCC)
            if not nx.is_connected(G_cc):
                diagnostics['failure_reason'] = 'LCC not connected (unexpected error)'
                return None
            
            # For small networks, calculate exactly
            if len(G_cc) <= 200:  # Increased threshold for exact calculation
                avg_path_length = nx.average_shortest_path_length(G_cc)
                diagnostics['path_calculation_method'] = f'exact_calculation_{len(G_cc)}_nodes'
                return avg_path_length
            
            # For large networks, use sampling approach
            else:
                # Use larger sample size for better accuracy
                sample_size = min(100, len(G_cc) // 2)  # At least 100 nodes or half the network
                
                # Multiple sampling attempts for stability
                path_estimates = []
                max_attempts = 5
                
                for attempt in range(max_attempts):
                    try:
                        sampled_nodes = random.sample(list(G_cc.nodes()), sample_size)
                        G_sample = G_cc.subgraph(sampled_nodes)
                        
                        # Ensure sample is connected
                        if nx.is_connected(G_sample):
                            path_estimate = nx.average_shortest_path_length(G_sample)
                            path_estimates.append(path_estimate)
                        else:
                            # Get largest component of sample
                            sample_components = list(nx.connected_components(G_sample))
                            if sample_components:
                                largest_sample_cc = max(sample_components, key=len)
                                if len(largest_sample_cc) >= 5:  # Minimum viable sample
                                    G_sample_cc = G_sample.subgraph(largest_sample_cc)
                                    path_estimate = nx.average_shortest_path_length(G_sample_cc)
                                    path_estimates.append(path_estimate)
                    except Exception:
                        continue  # Try next sample
                
                if path_estimates:
                    avg_path_length = np.mean(path_estimates)
                    diagnostics['path_calculation_method'] = f'sampled_{len(path_estimates)}_estimates_size_{sample_size}'
                    diagnostics['path_estimate_std'] = np.std(path_estimates) if len(path_estimates) > 1 else 0
                    return avg_path_length
                else:
                    diagnostics['failure_reason'] = 'All sampling attempts failed for path length calculation'
                    return None
        
        except Exception as e:
            diagnostics['failure_reason'] = f'Path length calculation error: {str(e)}'
            return None
    
    def _calculate_coreness_ratio(self, G: nx.Graph) -> float:
        """Calculate coreness as ratio of max k-core nodes to total nodes."""
        if len(G) < 3:
            return 0.0
        
        try:
            core_numbers = nx.core_number(G)
            max_core = max(core_numbers.values()) if core_numbers else 0
            core_nodes = [n for n, k in core_numbers.items() if k == max_core]
            return len(core_nodes) / len(G)
        except:
            return 0.0
    
    def _calculate_robustness_ratio(self, G: nx.Graph) -> float:
        """Calculate robustness ratio: targeted vs random removal resilience."""
        if len(G) < 10:
            return 1.0
        
        try:
            # Original giant component size
            components = list(nx.connected_components(G))
            if not components:
                return 1.0
            
            original_gcc = len(max(components, key=len))
            
            # Number of nodes to remove (10%)
            num_remove = max(1, int(len(G) * 0.1))
            
            # Random removal
            G_random = G.copy()
            nodes_to_remove = random.sample(list(G.nodes()), min(num_remove, len(G)))
            G_random.remove_nodes_from(nodes_to_remove)
            
            if len(G_random) > 0:
                random_components = list(nx.connected_components(G_random))
                random_gcc = len(max(random_components, key=len)) if random_components else 0
                random_resilience = random_gcc / original_gcc if original_gcc > 0 else 0
            else:
                random_resilience = 0
            
            # Targeted removal (highest degree nodes)
            G_targeted = G.copy()
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:num_remove]
            G_targeted.remove_nodes_from([n for n, d in top_nodes])
            
            if len(G_targeted) > 0:
                targeted_components = list(nx.connected_components(G_targeted))
                targeted_gcc = len(max(targeted_components, key=len)) if targeted_components else 0
                targeted_resilience = targeted_gcc / original_gcc if original_gcc > 0 else 0
            else:
                targeted_resilience = 0
            
            return targeted_resilience / random_resilience if random_resilience > 0 else 0.0
        except:
            return 1.0
    
    def _calculate_avg_constraint(self, G: nx.Graph) -> float:
        """Calculate average structural holes constraint."""
        if len(G) < 3:
            return 0.0
        
        try:
            constraint = nx.constraint(G)
            return np.mean(list(constraint.values())) if constraint else 0.0
        except:
            return 0.0
    
    def _calculate_avg_effective_size(self, G: nx.Graph) -> float:
        """Calculate average effective network size."""
        if len(G) < 3:
            return 0.0
        
        try:
            effective_size = nx.effective_size(G)
            return np.mean(list(effective_size.values())) if effective_size else 0.0
        except:
            return 0.0
    
    def save_small_world_diagnostics(self, timestamp: str):
        """Save the small-world diagnostics to a separate file."""
        if not self.small_world_diagnostics:
            logger.warning("No small-world diagnostics to save")
            return
        
        diagnostics_file = self.results_dir / f"small_world_diagnostics_fixed_{timestamp}.json"
        
        # Convert to serializable format
        serializable_diagnostics = {}
        for topic_id, diag in self.small_world_diagnostics.items():
            serializable_diagnostics[str(topic_id)] = self._convert_numpy_types(diag)
        
        with open(diagnostics_file, 'w') as f:
            json.dump(serializable_diagnostics, f, indent=2)
        
        logger.info(f"Saved small-world diagnostics to {diagnostics_file}")
        
        # Also create a summary CSV for easy analysis
        summary_rows = []
        for topic_id, diag in self.small_world_diagnostics.items():
            summary_rows.append({
                'topic_id': topic_id,
                'original_num_nodes': diag.get('original_num_nodes', 0),
                'original_num_edges': diag.get('original_num_edges', 0),
                'num_components': diag.get('num_components', 0),
                'largest_cc_size': diag.get('largest_cc_size', 0),
                'largest_cc_fraction': diag.get('largest_cc_fraction', 0),
                'avg_degree_original': diag.get('avg_degree_original', 0),
                'avg_degree_cc': diag.get('avg_degree_cc', 0),
                'actual_clustering': diag.get('actual_clustering'),
                'expected_clustering': diag.get('expected_clustering'),
                'actual_path_length': diag.get('actual_path_length'),
                'expected_path_length': diag.get('expected_path_length'),
                'sw_coefficient': diag.get('sw_coefficient', 0),
                'failure_reason': diag.get('failure_reason', ''),
                'calculation_method': diag.get('calculation_method', '')
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = self.results_dir / f"small_world_diagnostics_summary_fixed_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved diagnostics summary to {summary_file}")
    
    def analyze_small_world_failures(self):
        """Analyze patterns in small-world coefficient failures."""
        if not self.small_world_diagnostics:
            logger.warning("No diagnostics available for analysis")
            return
        
        # Count failure reasons
        failure_counts = Counter()
        success_count = 0
        
        for diag in self.small_world_diagnostics.values():
            if diag.get('failure_reason'):
                # Categorize failure reasons
                reason = diag['failure_reason']
                if 'too small' in reason.lower():
                    failure_counts['Network too small'] += 1
                elif 'too few edges' in reason.lower():
                    failure_counts['Too few edges'] += 1
                elif 'component too small' in reason.lower():
                    failure_counts['Largest component too small'] += 1
                elif 'degree too low' in reason.lower():
                    failure_counts['Average degree too low'] += 1
                elif 'path length calculation failed' in reason.lower():
                    failure_counts['Path length calculation failed'] += 1
                else:
                    failure_counts['Other'] += 1
            else:
                success_count += 1
        
        logger.info("\n=== FIXED Small-World Coefficient Analysis ===")
        logger.info(f"Total topics analyzed: {len(self.small_world_diagnostics)}")
        logger.info(f"Successful calculations: {success_count}")
        logger.info(f"Failed calculations: {sum(failure_counts.values())}")
        logger.info("\nFailure reasons:")
        for reason, count in failure_counts.most_common():
            logger.info(f"  {reason}: {count}")
        
        # Analyze network characteristics of failures vs successes
        failed_topics = [diag for diag in self.small_world_diagnostics.values() 
                        if diag.get('failure_reason')]
        success_topics = [diag for diag in self.small_world_diagnostics.values() 
                         if not diag.get('failure_reason')]
        
        if failed_topics:
            avg_nodes_failed = np.mean([d['original_num_nodes'] for d in failed_topics])
            avg_edges_failed = np.mean([d['original_num_edges'] for d in failed_topics])
            
            logger.info(f"\nFailed networks characteristics:")
            logger.info(f"  Average nodes: {avg_nodes_failed:.1f}")
            logger.info(f"  Average edges: {avg_edges_failed:.1f}")
        
        if success_topics:
            avg_nodes_success = np.mean([d['original_num_nodes'] for d in success_topics])
            avg_edges_success = np.mean([d['original_num_edges'] for d in success_topics])
            avg_sw_success = np.mean([d['sw_coefficient'] for d in success_topics])
            
            logger.info(f"\nSuccessful networks characteristics:")
            logger.info(f"  Average nodes: {avg_nodes_success:.1f}")
            logger.info(f"  Average edges: {avg_edges_success:.1f}")
            logger.info(f"  Average SW coefficient: {avg_sw_success:.3f}")
    
    def compare_popular_vs_niche_topics(self, topic_results: dict, percentile_cutoff: float = 0.2) -> dict:
        """Compare the 10 core metrics between popular and niche topics."""
        logger.info("Comparing popular vs niche topics using 10 core metrics...")
        
        # Sort topics by paper count
        topic_sizes = [(tid, data['total_papers']) for tid, data in topic_results.items()]
        topic_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Identify popular and niche topics
        cutoff_index = int(len(topic_sizes) * percentile_cutoff)
        popular_topics = [tid for tid, _ in topic_sizes[:cutoff_index]]
        niche_topics = [tid for tid, _ in topic_sizes[-cutoff_index:]]
        
        # Core metrics to compare
        core_metrics = [
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
        
        # Calculate group statistics
        popular_stats = self._calculate_group_stats([topic_results[tid] for tid in popular_topics], core_metrics)
        niche_stats = self._calculate_group_stats([topic_results[tid] for tid in niche_topics], core_metrics)
        
        # Statistical comparison
        comparison_results = self._statistical_comparison_core_metrics(
            [topic_results[tid] for tid in popular_topics],
            [topic_results[tid] for tid in niche_topics],
            core_metrics
        )
        
        return {
            'popular_topics': popular_topics[:10],
            'niche_topics': niche_topics[:10],
            'popular_stats': popular_stats,
            'niche_stats': niche_stats,
            'comparison_results': comparison_results,
            'core_metrics': core_metrics
        }
    
    def _calculate_group_stats(self, topic_data: list, metrics: list) -> dict:
        """Calculate descriptive statistics for a group of topics."""
        group_stats = {}
        
        for metric in metrics:
            values = [topic[metric] for topic in topic_data if metric in topic and not np.isnan(topic[metric])]
            
            if values:
                group_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                group_stats[metric] = {
                    'mean': 0, 'std': 0, 'median': 0,
                    'min': 0, 'max': 0, 'count': 0
                }
        
        return group_stats
    
    def _statistical_comparison_core_metrics(self, popular_data: list, niche_data: list, metrics: list) -> dict:
        """Perform statistical tests comparing popular and niche topics on core metrics."""
        comparison_results = {}
        
        for metric in metrics:
            popular_values = [topic[metric] for topic in popular_data if metric in topic and not np.isnan(topic[metric])]
            niche_values = [topic[metric] for topic in niche_data if metric in topic and not np.isnan(topic[metric])]
           
            if len(popular_values) > 3 and len(niche_values) > 3:
                try:
                    # Check for normality
                    pop_shapiro = stats.shapiro(popular_values) if len(popular_values) <= 5000 else (None, 0.01)
                    niche_shapiro = stats.shapiro(niche_values) if len(niche_values) <= 5000 else (None, 0.01)
                   
                    # Choose appropriate test
                    if pop_shapiro[1] > 0.05 and niche_shapiro[1] > 0.05:
                       # Normal distributions - use t-test
                       statistic, p_value = stats.ttest_ind(popular_values, niche_values)
                       test_used = 't-test'
                    else:
                       # Non-normal distributions - use Mann-Whitney U
                       statistic, p_value = stats.mannwhitneyu(popular_values, niche_values, alternative='two-sided')
                       test_used = 'Mann-Whitney U'
                   
                    # Effect size (Cohen's d)
                    mean_diff = np.mean(popular_values) - np.mean(niche_values)
                    pooled_std = np.sqrt(
                       ((len(popular_values) - 1) * np.std(popular_values)**2 + 
                        (len(niche_values) - 1) * np.std(niche_values)**2) / 
                       (len(popular_values) + len(niche_values) - 2)
                    )
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                   
                    comparison_results[metric] = {
                       'test_used': test_used,
                       'statistic': float(statistic),
                       'p_value': float(p_value),
                       'significant': p_value < 0.05,
                       'significant_bonferroni': p_value < (0.05 / len(metrics)),
                       'effect_size_cohens_d': float(cohens_d),
                       'popular_mean': float(np.mean(popular_values)),
                       'niche_mean': float(np.mean(niche_values)),
                       'popular_std': float(np.std(popular_values)),
                       'niche_std': float(np.std(niche_values)),
                       'direction': 'popular_higher' if mean_diff > 0 else 'niche_higher',
                       'sample_sizes': {'popular': len(popular_values), 'niche': len(niche_values)}
                   }
                except Exception as e:
                    comparison_results[metric] = {
                       'error': str(e),
                       'test_used': 'failed',
                       'significant': False
                   }
            else:
                comparison_results[metric] = {
                   'test_used': 'insufficient_data',
                   'significant': False,
                   'note': f'Insufficient data: popular n={len(popular_values)}, niche n={len(niche_values)}'
               }
       
        return comparison_results
   
    def generate_summary_statistics(self, topic_results: dict) -> dict:
        """Generate summary statistics for all analyzed topics."""
        logger.info("Generating summary statistics...")
       
        # Convert to DataFrame for easy analysis
        topic_summaries = []
        for topic_id, results in topic_results.items():
           summary = {
               'topic_id': topic_id,
               'total_papers': results['total_papers'],
               'collaboration_papers': results['collaboration_papers'],
               'collaboration_rate': results['collaboration_rate'],
               'repeated_collaboration_rate': results['repeated_collaboration_rate'],
               'degree_centralization': results['degree_centralization'],
               'degree_assortativity': results['degree_assortativity'],
               'modularity': results['modularity'],
               'small_world_coefficient': results['small_world_coefficient'],
               'coreness_ratio': results['coreness_ratio'],
               'robustness_ratio': results['robustness_ratio'],
               'avg_constraint': results['avg_constraint'],
               'avg_effective_size': results['avg_effective_size']
           }
           topic_summaries.append(summary)
       
        topic_summary_df = pd.DataFrame(topic_summaries)
       
        # Overall statistics
        overall_stats = {
            'total_topics_analyzed': len(topic_results),
            'topics_with_collaborations': len(topic_summary_df[topic_summary_df['collaboration_papers'] > 0]),
            'mean_collaboration_rate': topic_summary_df['collaboration_rate'].mean(),
            'mean_repeated_collaboration_rate': topic_summary_df['repeated_collaboration_rate'].mean(),
            'mean_degree_centralization': topic_summary_df['degree_centralization'].mean(),
            'mean_degree_assortativity': topic_summary_df['degree_assortativity'].mean(),
            'mean_modularity': topic_summary_df['modularity'].mean(),
            'mean_small_world_coefficient': topic_summary_df['small_world_coefficient'].mean(),
            'topics_with_small_world': (topic_summary_df['small_world_coefficient'] > 1).sum(),
            'topics_with_zero_sw': (topic_summary_df['small_world_coefficient'] == 0).sum(),
            'mean_coreness_ratio': topic_summary_df['coreness_ratio'].mean(),
            'mean_robustness_ratio': topic_summary_df['robustness_ratio'].mean(),
            'mean_constraint': topic_summary_df['avg_constraint'].mean(),
            'mean_effective_size': topic_summary_df['avg_effective_size'].mean()
        }
       
        return {
           'topic_summaries': topic_summary_df,
           'overall_statistics': overall_stats
        }
   
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
   
    def save_results(self, topic_results: dict, summary_stats: dict, comparison_results: dict = None):
        """Save analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       
        # Convert numpy types to native Python types
        topic_results_clean = self._convert_numpy_types(topic_results)
        summary_stats_clean = self._convert_numpy_types(summary_stats['overall_statistics'])
       
        # Save topic results
        topic_results_file = self.results_dir / f"topic_analysis_10metrics_fixed_{timestamp}.json"
        with open(topic_results_file, 'w') as f:
            json.dump(topic_results_clean, f, indent=2)
       
        # Save summary statistics
        summary_file = self.results_dir / f"summary_statistics_10metrics_fixed_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats_clean, f, indent=2)
       
        # Save topic summaries as CSV
        topic_summaries_file = self.results_dir / f"topic_summaries_10metrics_fixed_{timestamp}.csv"
        summary_stats['topic_summaries'].to_csv(topic_summaries_file, index=False)
       
        # Save comparison results if available
        if comparison_results:
            comparison_clean = self._convert_numpy_types(comparison_results)
            comparison_file = self.results_dir / f"popular_vs_niche_10metrics_fixed_{timestamp}.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison_clean, f, indent=2)
       
        # Save small-world diagnostics if available
        if self.enable_diagnostics and self.small_world_diagnostics:
            self.save_small_world_diagnostics(timestamp)
            self.analyze_small_world_failures()
       
        logger.info(f"Results saved with timestamp: {timestamp}")
        return timestamp


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="10-Metric Collaboration Network Analysis V5 with Fixed SWC")
    # parser.add_argument("--data-path", default="data/cleaned/author_topic_networks_disambiguated_v4.csv", 
    #                     help="Path to author-topic networks CSV")
    parser.add_argument("--sample-topics", type=str, default=None,
                        help="Comma-separated list of topics to analyze (default: all)")
    parser.add_argument("--test-run", action="store_true",
                        help="Run on first 10 topics only")
    parser.add_argument("--compare-popular-niche", action="store_true",
                        help="Perform popular vs niche topic comparison")
    parser.add_argument("--enable-diagnostics", action="store_true", default=True,
                        help="Enable detailed small-world diagnostics (default: True)")
   
    args = parser.parse_args()

    config = ConfigManager()
    input_path = config.get_path('disambiguated_authors_path')
   
    # Initialize analyzer
    analyzer = CollaborationNetworkAnalyzer(input_path)
   
    # Load data
    try:
        df = analyzer.load_and_validate_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
   
    # Determine which topics to analyze
    if args.test_run:
        sample_topics = sorted(df['topic'].unique())[:10]
        logger.info(f"Test run: analyzing first 10 topics: {sample_topics}")
    elif args.sample_topics:
        sample_topics = [int(x.strip()) for x in args.sample_topics.split(',')]
        logger.info(f"Analyzing specified topics: {sample_topics}")
    else:
        sample_topics = None
        logger.info("Analyzing all topics")
   
    # Run 10-metric analysis with fixed diagnostics
    topic_results = analyzer.analyze_topic_networks(df, sample_topics, enable_diagnostics=args.enable_diagnostics)
   
    # Generate summary statistics
    summary_stats = analyzer.generate_summary_statistics(topic_results)
   
    # Compare popular vs niche topics if requested
    comparison_results = None
    if args.compare_popular_niche and len(topic_results) > 20:
        comparison_results = analyzer.compare_popular_vs_niche_topics(topic_results)
   
    # Save results
    timestamp = analyzer.save_results(topic_results, summary_stats, comparison_results)
    
    output_filename = f"topic_analysis_10metrics_fixed_{timestamp}.json"
    output_path = analyzer.results_dir / output_filename
    config.update_path('network_metrics_path', str(output_path))
    
    # Print summary
    print("\n" + "="*70)
    print("10-METRIC COLLABORATION NETWORK ANALYSIS SUMMARY (V5 - FIXED SWC)")
    print("="*70)
   
    overall_stats = summary_stats['overall_statistics']
    print(f"Topics analyzed: {overall_stats['total_topics_analyzed']}")
    print(f"Topics with collaborations: {overall_stats['topics_with_collaborations']}")
   
    print(f"\n📊 CORE NETWORK METRICS (FIXED SWC):")
    print(f"1. Mean collaboration rate: {overall_stats['mean_collaboration_rate']:.3f}")
    print(f"2. Mean repeated collaboration rate: {overall_stats['mean_repeated_collaboration_rate']:.3f}")
    print(f"3. Mean degree centralization: {overall_stats['mean_degree_centralization']:.3f}")
    print(f"4. Mean degree assortativity: {overall_stats['mean_degree_assortativity']:.3f}")
    print(f"5. Mean modularity: {overall_stats['mean_modularity']:.3f}")
    print(f"6. Mean small world coefficient (FIXED): {overall_stats['mean_small_world_coefficient']:.3f}")
    print(f"   Topics with small-world properties (SW > 1): {overall_stats['topics_with_small_world']}")
    print(f"   Topics with SW = 0 (calculation failed): {overall_stats['topics_with_zero_sw']}")
    print(f"7. Mean coreness ratio: {overall_stats['mean_coreness_ratio']:.3f}")
    print(f"8. Mean robustness ratio: {overall_stats['mean_robustness_ratio']:.3f}")
    print(f"9. Mean structural holes constraint: {overall_stats['mean_constraint']:.3f}")
    print(f"10. Mean effective network size: {overall_stats['mean_effective_size']:.3f}")
   
    if comparison_results:
        print("\n" + "-"*70)
        print("🔬 POPULAR VS NICHE TOPIC COMPARISON (FIXED SWC)")
        print("-"*70)
       
        comp_results = comparison_results['comparison_results']
        significant_diffs = []
        bonferroni_significant = []
       
        for metric, stats in comp_results.items():
            if stats.get('significant', False):
                significant_diffs.append((metric, stats))
            if stats.get('significant_bonferroni', False):
                bonferroni_significant.append((metric, stats))
       
        print(f"Significant differences (p < 0.05): {len(significant_diffs)}/10 metrics")
        print(f"Bonferroni-corrected significant (p < 0.005): {len(bonferroni_significant)}/10 metrics")
       
        if significant_diffs:
            print(f"\n📈 SIGNIFICANT DIFFERENCES:")
            for metric, stats in significant_diffs[:5]:  # Show top 5
                direction = "↑" if stats['direction'] == 'popular_higher' else "↓"
                effect_size = stats['effect_size_cohens_d']
                effect_interpretation = (
                    "large" if abs(effect_size) >= 0.8 else
                    "medium" if abs(effect_size) >= 0.5 else
                     "small"
                )
                print(f"   {metric}: Popular {direction} (p={stats['p_value']:.4f}, "
                    f"Cohen's d={effect_size:.3f} [{effect_interpretation}])")
   
    print(f"\n💾 Results saved with timestamp: {timestamp}")
    print(f"📁 Check the results/collaboration_analysis/ directory for detailed outputs:")
    print(f"   - topic_analysis_10metrics_fixed_{timestamp}.json")
    print(f"   - topic_summaries_10metrics_fixed_{timestamp}.csv")
    if args.enable_diagnostics:
        print(f"   - small_world_diagnostics_fixed_{timestamp}.json")
        print(f"   - small_world_diagnostics_summary_fixed_{timestamp}.csv")
    if comparison_results:
        print(f"   - popular_vs_niche_10metrics_fixed_{timestamp}.json")
   
    print(f"\n🎉 Analysis complete! Fixed SWC calculation implemented.")
    print(f"🔧 Key fix: SWC now uses LCC characteristics for both actual and expected values.")


if __name__ == "__main__":
   main()
            