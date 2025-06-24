#!/usr/bin/env python3
"""
Manual Validation Script for Advanced Author Disambiguation V4
============================================================
Auto-generated for run: 20250623_212134
Samples 100 merges for manual validation.
"""

import json
import random
from pathlib import Path

def validate_merges():
    """Interactive validation of sampled merges."""
    # Load disambiguation log
    log_file = Path("results/disambiguation/advanced_disambiguation_log_20250623_212134.json")
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    # Filter for actual merges
    merges = [entry for entry in log_data if entry.get('type') in 
              ['priority1_intelligent_cluster', 'priority2_network']]
    
    actual_sample_size = min(len(merges), 100)
    
    if len(merges) < 100:
        print(f"Warning: Only {len(merges)} merges available, sampling all")
    
    if actual_sample_size == 0:
        print("No merges found to validate!")
        return
    
    sampled_merges = random.sample(merges, actual_sample_size)
    
    print(f"=== ADVANCED AUTHOR DISAMBIGUATION VALIDATION (V4) ===")
    print(f"Reviewing {len(sampled_merges)} randomly sampled merges")
    print(f"For each merge, enter 'c' for Correct, 'i' for Incorrect, 'q' to quit\n")
    
    results = []
    
    for i, merge in enumerate(sampled_merges, 1):
        print(f"\n[{i}/{len(sampled_merges)}] {merge['type'].upper()}")
        
        merged_authors = merge.get('merged', [])
        canonical_name = merge.get('into', 'Unknown')
        
        print(f"  Merged Authors: {', '.join(merged_authors)}")
        print(f"  Into Canonical: {canonical_name}")
        
        if merge['type'] == 'priority1_intelligent_cluster':
            reasoning = merge.get('reasoning', 'N/A')
            cluster_size = merge.get('cluster_size', 'N/A')
            print(f"  Reasoning: {reasoning}")
            print(f"  Original Cluster Size: {cluster_size}")
        elif merge['type'] == 'priority2_network':
            similarity = merge.get('similarity', 0.0)
            print(f"  Network Similarity: {similarity:.3f}")
        
        while True:
            response = input("  Correct merge? (c/i/q): ").strip().lower()
            if response in ['c', 'i', 'q']:
                break
            print("  Please enter 'c', 'i', or 'q'")
        
        if response == 'q':
            break
        
        results.append({
            'merge_id': i,
            'merge_type': merge['type'],
            'correct': response == 'c',
            'merge_data': merge
        })
    
    # Calculate and save results
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        
        print(f"\n=== VALIDATION RESULTS ===")
        print(f"Total reviewed: {len(results)}")
        print(f"Correct merges: {correct_count}")
        print(f"Accuracy: {accuracy:.1%}")
        
        # Break down by merge type
        priority1_results = [r for r in results if r['merge_type'] == 'priority1_intelligent_cluster']
        priority2_results = [r for r in results if r['merge_type'] == 'priority2_network']
        
        if priority1_results:
            p1_accuracy = sum(1 for r in priority1_results if r['correct']) / len(priority1_results)
            print(f"Intelligent Cluster Accuracy: {p1_accuracy:.1%} ({len(priority1_results)} samples)")
        
        if priority2_results:
            p2_accuracy = sum(1 for r in priority2_results if r['correct']) / len(priority2_results)
            print(f"Network Similarity Accuracy: {p2_accuracy:.1%} ({len(priority2_results)} samples)")
        
        # Save results
        results_file = Path("results/disambiguation/advanced_validation_results_20250623_212134.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': '20250623_212134',
                'sample_size': len(results),
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': len(results),
                'priority1_accuracy': p1_accuracy if priority1_results else None,
                'priority2_accuracy': p2_accuracy if priority2_results else None,
                'detailed_results': results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"\nFor your paper: 'Manual validation of {len(results)} randomly")
        print(f"sampled merges achieved {accuracy:.1%} accuracy with V4 stricter rules.'")

if __name__ == "__main__":
    validate_merges()
