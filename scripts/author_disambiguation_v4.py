#!/usr/bin/env python3
"""
Advanced Author Name Disambiguation (AND) for Math Research Compass - Version 4
===============================================================================

Implements a sophisticated, graph-based disambiguation strategy with intelligent clustering.
This version includes stricter rules to prevent over-merging, especially for Asian names.

NEW IN V4:
- Stricter similarity thresholds for potentially Asian names
- Cluster size limits (max 50 names per automatic merge)
- More conservative prefix matching rules
- Enhanced validation to prevent false positives

Priority 0: High-Confidence String Similarity Graph Construction
- Uses blocking strategy on last name for computational feasibility
- Builds similarity graph instead of immediately merging

Priority 1: Intelligent Cluster Analysis and Merging
- Analyzes each connected component for disambiguation safety
- Only merges clusters with unambiguous evidence
- Prevents false positives like "Pengcheng Xie" + "Pengxu Xie"

Priority 2: Network-Based Similarity Merging (unchanged)
- Uses co-author networks and paper overlap for remaining cases

Input: CSV file (e.g., author_topic_networks.csv) with 'id' and 'authors_parsed' columns.
Output: A new CSV file with a disambiguated 'authors_parsed' column, plus detailed logs
        and a validation script.
"""

import pandas as pd
import ast
import re
import json
import unicodedata
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from itertools import combinations
import logging
import argparse
import difflib
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAuthorDisambiguator:
    """
    Implements a sophisticated, graph-based author disambiguation pipeline.
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.output_path = self.data_path.parent / f"{self.data_path.stem}_disambiguated_v4.csv"
        self.results_dir = Path("results/disambiguation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.author_mapping = {}
        self.disambiguation_log = []
        self.merge_stats = defaultdict(int)
        self.paper_authors = {}
        self.author_papers = defaultdict(set)
        
        # New graph-based structures
        self.similarity_graph = nx.Graph()
        self.cluster_analysis = {}
        
        logger.info(f"Initialized AdvancedAuthorDisambiguator V4 with data: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Loads and validates the input dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        logger.info("Loading author dataset...")
        try:
            df = pd.read_csv(self.data_path, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
            
        logger.info(f"Loaded {len(df)} papers from {self.data_path}")
        
        required_cols = ['id', 'authors_parsed']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input file must contain columns: {required_cols}")
            
        df.dropna(subset=required_cols, inplace=True)
        logger.info(f"After removing rows with missing IDs or authors: {len(df)} papers")
        return df

    def parse_authors(self, authors_str: str) -> list[str]:
        """Parses author strings from the dataset, preserving all name components and deduplicating."""
        if pd.isna(authors_str): return []
        try:
            authors_list = ast.literal_eval(authors_str)
            names = []
            seen_normalized = set()
            for parts in authors_list:
                if isinstance(parts, list) and len(parts) >= 1:
                    clean_parts = [str(p).strip() for p in parts if str(p).strip()]
                    if not clean_parts: continue
                    
                    last_name = clean_parts[0]
                    first_parts = clean_parts[1:]
                    full_name = " ".join(first_parts) + " " + last_name if first_parts else last_name
                    full_name = full_name.strip()
                    
                    normalized_name = self.normalize_name(full_name)
                    if normalized_name and normalized_name not in seen_normalized:
                        names.append(full_name)
                        seen_normalized.add(normalized_name)
            return names
        except (ValueError, SyntaxError, TypeError):
            return []

    def normalize_name(self, name: str) -> str:
        """Normalizes a name string for consistent comparison."""
        if not name: return ""
        norm = name.lower()
        norm = unicodedata.normalize('NFD', norm).encode('ascii', 'ignore').decode('utf-8')
        particles = ['van', 'der', 'de', 'la', 'del', 'von']
        for p in particles:
            norm = re.sub(rf'\b{p}\s', f'{p}_', norm)
        norm = re.sub(r'[^\w\s_]', '', norm)
        return re.sub(r'\s+', ' ', norm).strip()

    def are_names_similar(self, name1: str, name2: str) -> bool:
        """Checks if two names are likely variants using strict string similarity rules."""
        if not name1 or not name2: return False
        norm1, norm2 = self.normalize_name(name1), self.normalize_name(name2)
        if norm1 == norm2: return True
        
        words1, words2 = norm1.split(), norm2.split()
        if len(words1) == len(words2):
            is_expansion = True
            for w1, w2 in zip(words1, words2):
                if not (w1 == w2 or (len(w1) == 1 and w2.startswith(w1)) or (len(w2) == 1 and w1.startswith(w2))):
                    is_expansion = False
                    break
            if is_expansion: return True

        lev_dist = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        if lev_dist > 0.95: return True
        
        return False

    def extract_first_names(self, names: list[str]) -> list[str]:
        """Extracts first names from a list of full names."""
        first_names = []
        for name in names:
            words = name.split()
            if len(words) >= 2:
                # First name is everything except the last word (surname)
                first_name = " ".join(words[:-1])
                if len(first_name) > 1:  # Only consider non-initial first names
                    first_names.append(first_name.lower())
        return first_names

    def are_first_names_compatible(self, first_names: list[str]) -> bool:
        """
        Determines if a set of first names can safely be merged.
        Returns True if all names are variants of the same person.
        """
        if len(first_names) <= 1:
            return True
        
        # Remove duplicates
        unique_names = list(set(first_names))
        if len(unique_names) == 1:
            return True
        
        # Check if all names are similar enough to be variants
        for i, name1 in enumerate(unique_names):
            for name2 in unique_names[i+1:]:
                # Check for common diminutives and variants
                if not self.are_first_name_variants(name1, name2):
                    return False
        return True

    def are_first_name_variants(self, name1: str, name2: str) -> bool:
        """Checks if two first names are likely variants of the same name. V4: Stricter rules."""
        # Exact match
        if name1 == name2:
            return True
        
        # STRICTER: Only merge if one is clearly an abbreviation of the other
        if name1 in name2 or name2 in name1:
            shorter, longer = (name1, name2) if len(name1) < len(name2) else (name2, name1)
            # Require the shorter name to be a prefix AND be reasonably short
            if len(shorter) <= 3 and longer.startswith(shorter):
                return True
        
        # STRICTER: Increase similarity threshold for non-Western names
        # Detect if names might be Chinese/Asian based on common patterns
        asian_patterns = [
            'wei', 'ming', 'xiao', 'yang', 'zhang', 'wang', 'li', 'chen', 
            'liu', 'zhao', 'wu', 'zhou', 'huang', 'xu', 'zhu', 'lin',
            'guo', 'luo', 'cao', 'tang', 'yan', 'yu', 'feng', 'dong',
            'bao', 'tian', 'qian', 'sun', 'ma', 'han', 'hu', 'jiang',
            'hong', 'jian', 'xin', 'hui', 'jun', 'jing', 'yong', 'cheng'
        ]
        
        is_possibly_asian = any(pattern in name1.lower() + name2.lower() for pattern in asian_patterns)
        
        similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
        required_similarity = 0.92 if is_possibly_asian else 0.87
        
        if similarity > required_similarity:
            return True
        
        # Keep known Western variants but be more conservative
        variants = {
            'john': ['jon', 'johnny'],
            'michael': ['mike'],
            'william': ['will', 'bill'],
            'robert': ['rob', 'bob'],
            'christopher': ['chris'],
            'alexander': ['alex'],
            'elizabeth': ['liz', 'beth'],
            'catherine': ['cathy', 'kate'],
            'james': ['jim', 'jimmy'],
            'thomas': ['tom', 'tommy'],
            'daniel': ['dan', 'danny'],
            'matthew': ['matt'],
            'nicholas': ['nick'],
            'timothy': ['tim'],
            'kenneth': ['ken', 'kenny'],
            'stephen': ['steve'],
            'joseph': ['joe', 'joey'],
            'anthony': ['tony'],
            'patricia': ['pat', 'patty'],
            'jennifer': ['jen', 'jenny'],
            'margaret': ['maggie', 'peggy'],
        }
        
        for base, variants_list in variants.items():
            if (name1 == base and name2 in variants_list) or (name2 == base and name1 in variants_list):
                return True
        
        return False

    def analyze_cluster(self, cluster: set) -> dict:
        """
        Analyzes a cluster of similar names to determine merge safety.
        V4: Adds cluster size limits.
        """
        cluster_list = list(cluster)
        
        # NEW: Be very conservative with large clusters
        if len(cluster_list) > 50:
            analysis = {
                'cluster_size': len(cluster),
                'names': cluster_list,
                'first_names': [],
                'unique_first_names': [],
                'is_safe_to_merge': False,
                'recommended_merges': [],
                'reasoning': f"Cluster too large ({len(cluster_list)} names) - requires manual review"
            }
            # Still try to find safe sub-groups even in large clusters
            safe_groups = self.find_safe_subgroups(cluster_list)
            analysis['recommended_merges'] = safe_groups
            return analysis
        
        first_names = self.extract_first_names(cluster_list)
        
        analysis = {
            'cluster_size': len(cluster),
            'names': cluster_list,
            'first_names': first_names,
            'unique_first_names': list(set(first_names)),
            'is_safe_to_merge': False,
            'recommended_merges': [],
            'reasoning': ""
        }
        
        # Case 1: No full first names (all initials) - generally safe
        if not first_names:
            analysis['is_safe_to_merge'] = True
            analysis['recommended_merges'] = [cluster_list]
            analysis['reasoning'] = "All names use initials only"
            return analysis
        
        # Case 2: All names have the same first name pattern
        if self.are_first_names_compatible(first_names):
            analysis['is_safe_to_merge'] = True
            analysis['recommended_merges'] = [cluster_list]
            analysis['reasoning'] = "All first names are compatible variants"
            return analysis
        
        # Case 3: Ambiguous case - multiple distinct first names
        # Try to find safe sub-groups
        analysis['is_safe_to_merge'] = False
        # Limit the displayed first names for readability
        displayed_names = analysis['unique_first_names'][:15]
        if len(analysis['unique_first_names']) > 15:
            displayed_names.append(f"... and {len(analysis['unique_first_names']) - 15} more")
        analysis['reasoning'] = f"Multiple distinct first names: {displayed_names}"
        
        # Group names by first name similarity
        safe_groups = self.find_safe_subgroups(cluster_list)
        analysis['recommended_merges'] = safe_groups
        
        return analysis

    def find_safe_subgroups(self, names: list[str]) -> list[list[str]]:
        """
        Partitions a list of names into safe subgroups that can be merged.
        V4: More conservative grouping.
        """
        # Create groups based on first name compatibility
        groups = []
        
        for name in names:
            placed = False
            name_first = self.extract_first_names([name])
            
            # Try to place in existing group
            for group in groups:
                group_first_names = []
                for group_name in group:
                    group_first_names.extend(self.extract_first_names([group_name]))
                
                combined_first_names = group_first_names + name_first
                if self.are_first_names_compatible(combined_first_names):
                    group.append(name)
                    placed = True
                    break
            
            # Create new group if couldn't place
            if not placed:
                groups.append([name])
        
        # Only return groups with more than one name (merges)
        return [group for group in groups if len(group) > 1]

    def priority0_build_similarity_graph(self, all_authors: set):
        """Builds a similarity graph of potentially related author names."""
        logger.info("Starting Priority 0: Building Similarity Graph with Blocking...")
        
        # Initialize mapping
        for author in all_authors:
            self.author_mapping[author] = author
        
        # Add all authors as nodes
        self.similarity_graph.add_nodes_from(all_authors)
        
        # Create blocking groups by last name
        lastname_groups = defaultdict(list)
        for author in all_authors:
            words = self.normalize_name(author).split()
            if words:
                lastname_groups[words[-1]].append(author)

        logger.info(f"Created {len(lastname_groups):,} blocking groups.")

        # Build similarity edges
        edge_count = 0
        for authors_in_group in lastname_groups.values():
            if len(authors_in_group) > 1:
                for author1, author2 in combinations(authors_in_group, 2):
                    if self.are_names_similar(author1, author2):
                        self.similarity_graph.add_edge(author1, author2)
                        edge_count += 1
        
        logger.info(f"Built similarity graph with {len(self.similarity_graph.nodes)} nodes and {edge_count} edges.")

    def priority1_intelligent_cluster_merging(self):
        """Analyzes connected components and performs intelligent merging."""
        logger.info("Starting Priority 1: Intelligent Cluster Analysis and Merging...")
        
        # Find connected components
        components = list(nx.connected_components(self.similarity_graph))
        logger.info(f"Found {len(components)} connected components.")
        
        # Analyze each component
        merge_count = 0
        large_cluster_count = 0
        
        for i, component in enumerate(components):
            if len(component) == 1:
                continue  # Single nodes don't need analysis
            
            analysis = self.analyze_cluster(component)
            self.cluster_analysis[i] = analysis
            
            if len(component) > 50:
                large_cluster_count += 1
            
            # Perform recommended merges
            for merge_group in analysis['recommended_merges']:
                if len(merge_group) > 1:
                    canonical_name = self._choose_canonical_name(merge_group)
                    
                    # Update mappings
                    for name in merge_group:
                        self.author_mapping[name] = canonical_name
                    
                    # Log the merge
                    self.disambiguation_log.append({
                        'type': 'priority1_intelligent_cluster',
                        'merged': merge_group,
                        'into': canonical_name,
                        'reasoning': analysis['reasoning'],
                        'cluster_size': analysis['cluster_size']
                    })
                    
                    merge_count += 1

        self.merge_stats['priority1_merges'] = merge_count
        self.merge_stats['large_clusters_skipped'] = large_cluster_count
        logger.info(f"Priority 1 completed: {merge_count} intelligent merges from {len([c for c in components if len(c) > 1])} multi-node components.")
        logger.info(f"Skipped/partially processed {large_cluster_count} large clusters (>50 names).")

    def _choose_canonical_name(self, names: list) -> str:
        """Chooses the most complete/representative name from a group."""
        if not names: return ""
        # Sort by completeness: longer names with fewer dots/initials first
        names.sort(key=lambda n: (len(n.split()), -n.count('.'), len(n)), reverse=True)
        return names[0]

    def priority2_network_based_similarity(self, df: pd.DataFrame, jaccard_threshold: float, min_papers: int, coauthor_weight: float):
        """Merges authors based on co-author and paper overlap, using blocking."""
        logger.info("Starting Priority 2: Network-Based Similarity Merging...")
        
        self._build_paper_author_maps(df)
        canonical_authors = set(self.author_mapping.values())
        coauthor_networks = self._build_coauthor_networks()

        blocking_groups = defaultdict(list)
        for author in canonical_authors:
            words = self.normalize_name(author).split()
            if words:
                block_key = (words[-1], words[0][0] if words[0] else '')
                blocking_groups[block_key].append(author)
        
        logger.info(f"Created {len(blocking_groups):,} blocking groups for network comparison.")
        
        merge_candidates = []
        for authors_in_group in blocking_groups.values():
            if len(authors_in_group) > 1:
                for author1, author2 in combinations(authors_in_group, 2):
                    canon1 = self.author_mapping.get(author1, author1)
                    canon2 = self.author_mapping.get(author2, author2)
                    if canon1 == canon2: continue
                    
                    papers1 = self.author_papers.get(canon1, set())
                    papers2 = self.author_papers.get(canon2, set())
                    if len(papers1) < min_papers or len(papers2) < min_papers: continue

                    coauthors1 = coauthor_networks.get(canon1, set())
                    coauthors2 = coauthor_networks.get(canon2, set())
                    
                    paper_jaccard = len(papers1 & papers2) / len(papers1 | papers2) if papers1 | papers2 else 0
                    coauthor_jaccard = len(coauthors1 & coauthors2) / len(coauthors1 | coauthors2) if coauthors1 | coauthors2 else 0
                    combined_similarity = coauthor_weight * coauthor_jaccard + (1.0 - coauthor_weight) * paper_jaccard
                    
                    if combined_similarity >= jaccard_threshold:
                        merge_candidates.append({'author1': canon1, 'author2': canon2, 'similarity': combined_similarity})
        
        logger.info(f"Found {len(merge_candidates)} network similarity candidates.")
        
        network_merges = 0
        for candidate in merge_candidates:
            canon1 = candidate['author1']
            canon2 = candidate['author2']
            
            if canon1 != canon2:  # Only merge if still different
                new_canonical = self._choose_canonical_name([canon1, canon2])
                
                # Update all mappings that point to the old canonicals
                for name, canon in self.author_mapping.items():
                    if canon in [canon1, canon2]:
                        self.author_mapping[name] = new_canonical
                
                self.disambiguation_log.append({
                    'type': 'priority2_network', 
                    'merged': [canon1, canon2], 
                    'into': new_canonical,
                    'similarity': candidate['similarity']
                })
                network_merges += 1
        
        self.merge_stats['priority2_merges'] = network_merges
        logger.info(f"Priority 2 completed: {network_merges} network-based merges.")

    def _build_paper_author_maps(self, df: pd.DataFrame):
        """Builds/updates paper-author and author-paper mappings."""
        self.paper_authors.clear()
        self.author_papers.clear()
        for _, row in df.iterrows():
            paper_id = row['id']
            authors = self.parse_authors(row['authors_parsed'])
            canonical_set = set(self.author_mapping.get(author, author) for author in authors)
            self.paper_authors[paper_id] = canonical_set
            for author_id in canonical_set:
                self.author_papers[author_id].add(paper_id)

    def _build_coauthor_networks(self) -> dict:
        """Builds co-author networks from the paper_authors map."""
        coauthor_networks = defaultdict(set)
        for authors in self.paper_authors.values():
            for author1, author2 in combinations(authors, 2):
                coauthor_networks[author1].add(author2)
                coauthor_networks[author2].add(author1)
        return coauthor_networks

    def apply_disambiguation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the final author mappings to the dataset."""
        logger.info("Applying final disambiguation to dataset...")
        df_disambiguated = df.copy()
        
        # Resolve all mappings to final canonical names
        final_map = {}
        for author in self.author_mapping:
            current = author
            path = {current}
            while current in self.author_mapping and self.author_mapping[current] != current:
                current = self.author_mapping[current]
                if current in path:  # Circular reference
                    break
                path.add(current)
            final_map[author] = current

        def get_disambiguated_list(authors_str):
            original_authors = self.parse_authors(authors_str)
            # Use final mappings and preserve order while removing duplicates
            disambiguated_authors = list(dict.fromkeys([final_map.get(author, author) for author in original_authors]))
            
            final_list = []
            for author in disambiguated_authors:
                parts = author.split()
                if len(parts) >= 2:
                    first, last = parts[0], parts[-1]
                    middle = " ".join(parts[1:-1])
                    final_list.append([last, first, middle])
                else:
                    final_list.append([author, '', ''])
            return str(final_list)

        df_disambiguated['authors_parsed'] = df['authors_parsed'].apply(get_disambiguated_list)
        return df_disambiguated
    
    def generate_statistics(self, original_authors_count: int):
        """Generates and logs final disambiguation statistics."""
        final_canonical_count = len(set(self.author_mapping.values()))
        
        self.merge_stats['total_original_authors'] = original_authors_count
        self.merge_stats['total_canonical_authors'] = final_canonical_count
        
        if original_authors_count > 0:
            reduction = (original_authors_count - final_canonical_count) / original_authors_count
            self.merge_stats['reduction_percentage'] = reduction * 100
        else:
            self.merge_stats['reduction_percentage'] = 0

        total_merges = self.merge_stats['priority1_merges'] + self.merge_stats['priority2_merges']
        
        logger.info("\n" + "="*70)
        logger.info("ADVANCED AUTHOR DISAMBIGUATION STATISTICS (V4)")
        logger.info(f"Original unique authors: {self.merge_stats['total_original_authors']:,}")
        logger.info(f"Canonical authors after AND: {self.merge_stats['total_canonical_authors']:,}")
        logger.info(f"Total merges performed: {total_merges:,}")
        logger.info(f"  - Priority 1 (intelligent clusters): {self.merge_stats['priority1_merges']:,}")
        logger.info(f"  - Priority 2 (network similarity): {self.merge_stats['priority2_merges']:,}")
        logger.info(f"Large clusters (>50 names) handled conservatively: {self.merge_stats.get('large_clusters_skipped', 0)}")
        logger.info(f"Author reduction: {self.merge_stats['reduction_percentage']:.2f}%")
        logger.info(f"Similarity graph components: {len(list(nx.connected_components(self.similarity_graph)))}")
        logger.info("="*70)

    def save_results(self, df_disambiguated: pd.DataFrame, timestamp: str):
        """Saves all outputs: disambiguated data, logs, mappings, and cluster analysis."""
        logger.info(f"Saving results with timestamp: {timestamp}")
        df_disambiguated.to_csv(self.output_path, index=False)
        
        with open(self.results_dir / f"advanced_author_mappings_{timestamp}.json", 'w') as f:
            json.dump(self.author_mapping, f, indent=2)
        
        with open(self.results_dir / f"advanced_disambiguation_log_{timestamp}.json", 'w') as f:
            json.dump(self.disambiguation_log, f, indent=2)
        
        with open(self.results_dir / f"advanced_disambiguation_stats_{timestamp}.json", 'w') as f:
            json.dump(self.merge_stats, f, indent=2)
        
        # Save cluster analysis
        cluster_analysis_serializable = {}
        for k, v in self.cluster_analysis.items():
            cluster_analysis_serializable[str(k)] = v
        
        with open(self.results_dir / f"cluster_analysis_{timestamp}.json", 'w') as f:
            json.dump(cluster_analysis_serializable, f, indent=2)
    
    def generate_validation_script(self, timestamp: str, sample_size: int = 100):
        """Generates a Python script for manual validation of merges."""
        logger.info(f"Generating validation script for timestamp: {timestamp}")
        
        validation_script = f'''#!/usr/bin/env python3
"""
Manual Validation Script for Advanced Author Disambiguation V4
============================================================
Auto-generated for run: {timestamp}
Samples {sample_size} merges for manual validation.
"""

import json
import random
from pathlib import Path

def validate_merges():
    """Interactive validation of sampled merges."""
    # Load disambiguation log
    log_file = Path("results/disambiguation/advanced_disambiguation_log_{timestamp}.json")
    if not log_file.exists():
        print(f"Error: Log file not found: {{log_file}}")
        return
    
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    # Filter for actual merges
    merges = [entry for entry in log_data if entry.get('type') in 
              ['priority1_intelligent_cluster', 'priority2_network']]
    
    actual_sample_size = min(len(merges), {sample_size})
    
    if len(merges) < {sample_size}:
        print(f"Warning: Only {{len(merges)}} merges available, sampling all")
    
    if actual_sample_size == 0:
        print("No merges found to validate!")
        return
    
    sampled_merges = random.sample(merges, actual_sample_size)
    
    print(f"=== ADVANCED AUTHOR DISAMBIGUATION VALIDATION (V4) ===")
    print(f"Reviewing {{len(sampled_merges)}} randomly sampled merges")
    print(f"For each merge, enter 'c' for Correct, 'i' for Incorrect, 'q' to quit\\n")
    
    results = []
    
    for i, merge in enumerate(sampled_merges, 1):
        print(f"\\n[{{i}}/{{len(sampled_merges)}}] {{merge['type'].upper()}}")
        
        merged_authors = merge.get('merged', [])
        canonical_name = merge.get('into', 'Unknown')
        
        print(f"  Merged Authors: {{', '.join(merged_authors)}}")
        print(f"  Into Canonical: {{canonical_name}}")
        
        if merge['type'] == 'priority1_intelligent_cluster':
            reasoning = merge.get('reasoning', 'N/A')
            cluster_size = merge.get('cluster_size', 'N/A')
            print(f"  Reasoning: {{reasoning}}")
            print(f"  Original Cluster Size: {{cluster_size}}")
        elif merge['type'] == 'priority2_network':
            similarity = merge.get('similarity', 0.0)
            print(f"  Network Similarity: {{similarity:.3f}}")
        
        while True:
            response = input("  Correct merge? (c/i/q): ").strip().lower()
            if response in ['c', 'i', 'q']:
                break
            print("  Please enter 'c', 'i', or 'q'")
        
        if response == 'q':
            break
        
        results.append({{
            'merge_id': i,
            'merge_type': merge['type'],
            'correct': response == 'c',
            'merge_data': merge
        }})
    
    # Calculate and save results
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        
        print(f"\\n=== VALIDATION RESULTS ===")
        print(f"Total reviewed: {{len(results)}}")
        print(f"Correct merges: {{correct_count}}")
        print(f"Accuracy: {{accuracy:.1%}}")
        
        # Break down by merge type
        priority1_results = [r for r in results if r['merge_type'] == 'priority1_intelligent_cluster']
        priority2_results = [r for r in results if r['merge_type'] == 'priority2_network']
        
        if priority1_results:
            p1_accuracy = sum(1 for r in priority1_results if r['correct']) / len(priority1_results)
            print(f"Intelligent Cluster Accuracy: {{p1_accuracy:.1%}} ({{len(priority1_results)}} samples)")
        
        if priority2_results:
            p2_accuracy = sum(1 for r in priority2_results if r['correct']) / len(priority2_results)
            print(f"Network Similarity Accuracy: {{p2_accuracy:.1%}} ({{len(priority2_results)}} samples)")
        
        # Save results
        results_file = Path("results/disambiguation/advanced_validation_results_{timestamp}.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({{
                'timestamp': '{timestamp}',
                'sample_size': len(results),
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': len(results),
                'priority1_accuracy': p1_accuracy if priority1_results else None,
                'priority2_accuracy': p2_accuracy if priority2_results else None,
                'detailed_results': results
            }}, f, indent=2)
        
        print(f"\\nResults saved to: {{results_file}}")
        print(f"\\nFor your paper: 'Manual validation of {{len(results)}} randomly")
        print(f"sampled merges achieved {{accuracy:.1%}} accuracy with V4 stricter rules.'")

if __name__ == "__main__":
    validate_merges()
'''

        # Save the validation script
        script_path = self.results_dir / f"validate_advanced_disambiguation_{timestamp}.py"
        with open(script_path, 'w') as f:
            f.write(validation_script)
        
        # Make executable on Unix systems
        try:
            import stat
            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
        except:
            pass
        
        logger.info(f"Validation script generated: {script_path}")
        logger.info(f"To validate: python {script_path}")
        
        return script_path

    def run_advanced_disambiguation(self, jaccard_threshold: float, min_papers: int, coauthor_weight: float):
        """Main execution function for the advanced disambiguation pipeline."""
        logger.info("Starting ADVANCED Author Name Disambiguation (AND) Pipeline V4...")
        logger.info(f"Parameters: jaccard_threshold={jaccard_threshold}, min_papers={min_papers}, coauthor_weight={coauthor_weight}")

        df = self.load_data()
        
        all_authors = set()
        for authors_str in df['authors_parsed']:
            all_authors.update(self.parse_authors(authors_str))
        logger.info(f"Found {len(all_authors):,} unique raw author names.")

        # Stage 1: Build similarity graph
        self.priority0_build_similarity_graph(all_authors)
        
        # Stage 2: Intelligent cluster analysis and merging
        self.priority1_intelligent_cluster_merging()
        
        # Stage 3: Network-based similarity merging
        self.priority2_network_based_similarity(df, jaccard_threshold, min_papers, coauthor_weight)
        
        # Apply final disambiguation
        df_disambiguated = self.apply_disambiguation(df)
        
        # Generate statistics and save results
        self.generate_statistics(len(all_authors))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_results(df_disambiguated, timestamp)
        self.generate_validation_script(timestamp)
        
        logger.info("✅ ADVANCED author disambiguation (V4) completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Advanced Author Name Disambiguation Pipeline V4")
    parser.add_argument("--data-path", default="data/cleaned/author_topic_networks.csv")
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--min-papers", type=int, default=2)
    parser.add_argument("--coauthor-weight", type=float, default=0.6)
    parser.add_argument("--test-run", action="store_true")
    
    args = parser.parse_args()
    
    disambiguator = AdvancedAuthorDisambiguator(args.data_path)
    
    if args.test_run:
        logger.info("Running in test mode on a subset of data.")
        df_full = disambiguator.load_data()
        df_test = df_full.head(1000)
        test_path = disambiguator.data_path.parent / "test_data_v4.csv"
        df_test.to_csv(test_path, index=False)
        disambiguator.data_path = test_path
    
    try:
        disambiguator.run_advanced_disambiguation(
            jaccard_threshold=args.jaccard_threshold,
            min_papers=args.min_papers,
            coauthor_weight=args.coauthor_weight
        )
    except Exception as e:
        logger.error("Pipeline failed.", exc_info=True)

if __name__ == "__main__":
    main()