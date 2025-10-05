#!/usr/bin/env python3
"""
Analysis script to investigate the relationship between stability and PPMI values
in the CNE (Conceptual-Neural Engine) project.

This script analyzes the observation that higher instability words have lower PPMI values.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_concepts_data():
    """Load the concepts data from the JSON file."""
    concepts_file = PROJECT_ROOT / "concept_ppmi_results" / "concepts_ppmi.json"
    
    if not concepts_file.exists():
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    print("Loading concepts data...")
    with concepts_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    concepts = data.get('concepts', {})
    print(f"Loaded {len(concepts)} concepts")
    return concepts

def analyze_stability_ppmi_relationship():
    """Analyze the relationship between stability and PPMI values."""
    
    concepts = load_concepts_data()
    
    # Extract stability and PPMI data
    stability_values = []
    ppmi_values = []
    mass_values = []
    spread_values = []
    phase_counts = defaultdict(int)
    
    word_data = []
    
    print("Processing concept data...")
    for word, concept_data in concepts.items():
        stability = concept_data.get('stability', 0)
        mass = concept_data.get('mass', 0)
        spread = concept_data.get('spread', 0)
        phase = concept_data.get('phase', 'unknown')
        
        stability_values.append(stability)
        mass_values.append(mass)
        spread_values.append(spread)
        phase_counts[phase] += 1
        
        # Get average PPMI for this word's neighbors
        neighbors = concept_data.get('neighbors', [])
        if neighbors:
            avg_ppmi = np.mean([n['ppmi'] for n in neighbors])
            ppmi_values.append(avg_ppmi)
        else:
            ppmi_values.append(0)
        
        word_data.append({
            'word': word,
            'stability': stability,
            'mass': mass,
            'spread': spread,
            'avg_ppmi': ppmi_values[-1],
            'phase': phase,
            'neighbor_count': len(neighbors)
        })
    
    # Convert to numpy arrays for analysis
    stability_array = np.array(stability_values)
    ppmi_array = np.array(ppmi_values)
    mass_array = np.array(mass_values)
    spread_array = np.array(spread_values)
    
    print("\n=== ANALYSIS RESULTS ===")
    
    # Basic statistics
    print(f"Total concepts analyzed: {len(concepts)}")
    print(f"Stability range: {stability_array.min():.2f} - {stability_array.max():.2f}")
    print(f"PPMI range: {ppmi_array.min():.2f} - {ppmi_array.max():.2f}")
    print(f"Mass range: {mass_array.min():.2f} - {mass_array.max():.2f}")
    print(f"Spread range: {spread_array.min():.2f} - {spread_array.max():.2f}")
    
    # Phase distribution
    print(f"\nPhase distribution:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} ({count/len(concepts)*100:.1f}%)")
    
    # Correlation analysis
    correlation = np.corrcoef(stability_array, ppmi_array)[0, 1]
    print(f"\nCorrelation between stability and average PPMI: {correlation:.4f}")
    
    # Analyze by stability quartiles
    quartiles = np.percentile(stability_array, [25, 50, 75])
    print(f"\nStability quartiles: {quartiles}")
    
    # Group by stability levels
    low_stability_mask = stability_array < quartiles[0]
    mid_stability_mask = (stability_array >= quartiles[0]) & (stability_array < quartiles[2])
    high_stability_mask = stability_array >= quartiles[2]
    
    low_stability = [word_data[i] for i in range(len(word_data)) if low_stability_mask[i]]
    mid_stability = [word_data[i] for i in range(len(word_data)) if mid_stability_mask[i]]
    high_stability = [word_data[i] for i in range(len(word_data)) if high_stability_mask[i]]
    
    print(f"\nLow stability group ({len(low_stability)} words):")
    if low_stability:
        avg_ppmi_low = np.mean([w['avg_ppmi'] for w in low_stability])
        print(f"  Average PPMI: {avg_ppmi_low:.4f}")
        print(f"  Sample words: {[w['word'] for w in sorted(low_stability, key=lambda x: x['avg_ppmi'], reverse=True)[:5]]}")
    
    print(f"\nHigh stability group ({len(high_stability)} words):")
    if high_stability:
        avg_ppmi_high = np.mean([w['avg_ppmi'] for w in high_stability])
        print(f"  Average PPMI: {avg_ppmi_high:.4f}")
        print(f"  Sample words: {[w['word'] for w in sorted(high_stability, key=lambda x: x['stability'], reverse=True)[:5]]}")
    
    # Theoretical analysis
    print(f"\n=== THEORETICAL ANALYSIS ===")
    print("According to CNE theory:")
    print("- Stability = Mass / Spread")
    print("- Mass = sum of PPMI values from neighbors")
    print("- Higher mass words have stronger conceptual associations")
    print("- Higher spread indicates more diffuse conceptual space")
    
    if correlation < 0:
        print(f"\nOBSERVATION CONFIRMED: Negative correlation ({correlation:.4f})")
        print("This suggests that:")
        print("1. Words with higher instability (low stability) tend to have higher PPMI neighbors")
        print("2. This could indicate these words are in 'transitional' conceptual spaces")
        print("3. They may be bridging different conceptual domains")
        print("4. High PPMI neighbors might indicate strong but narrow associations")
    else:
        print(f"\nUNEXPECTED: Positive correlation ({correlation:.4f})")
    
    # Find interesting examples
    print(f"\n=== INTERESTING EXAMPLES ===")
    
    # Words with high instability but high PPMI
    high_instability_high_ppmi = [w for w in word_data if w['stability'] < quartiles[0] and w['avg_ppmi'] > np.percentile(ppmi_array, 75)]
    print(f"High instability, high PPMI words ({len(high_instability_high_ppmi)}):")
    for w in sorted(high_instability_high_ppmi, key=lambda x: x['avg_ppmi'], reverse=True)[:10]:
        print(f"  {w['word']}: stability={w['stability']:.2f}, avg_ppmi={w['avg_ppmi']:.2f}, phase={w['phase']}")
    
    # Words with high stability but low PPMI
    high_stability_low_ppmi = [w for w in word_data if w['stability'] >= quartiles[2] and w['avg_ppmi'] < np.percentile(ppmi_array, 25)]
    print(f"\nHigh stability, low PPMI words ({len(high_stability_low_ppmi)}):")
    for w in sorted(high_stability_low_ppmi, key=lambda x: x['stability'], reverse=True)[:10]:
        print(f"  {w['word']}: stability={w['stability']:.2f}, avg_ppmi={w['avg_ppmi']:.2f}, phase={w['phase']}")
    
    # Analyze by phase
    print(f"\n=== ANALYSIS BY PHASE ===")
    for phase in phase_counts.keys():
        phase_words = [w for w in word_data if w['phase'] == phase]
        if phase_words:
            avg_stability = np.mean([w['stability'] for w in phase_words])
            avg_ppmi = np.mean([w['avg_ppmi'] for w in phase_words])
            print(f"{phase}: {len(phase_words)} words, avg_stability={avg_stability:.2f}, avg_ppmi={avg_ppmi:.4f}")
    
    return word_data, stability_array, ppmi_array, mass_array, spread_array

def create_visualizations(stability_array, ppmi_array, mass_array, spread_array):
    """Create visualizations of the relationships."""
    try:
        plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('CNE Stability vs PPMI Analysis', fontsize=16)
        
        # Stability vs PPMI scatter
        axes[0, 0].scatter(stability_array, ppmi_array, alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Stability')
        axes[0, 0].set_ylabel('Average PPMI')
        axes[0, 0].set_title('Stability vs Average PPMI')
        
        # Mass vs Spread
        axes[0, 1].scatter(mass_array, spread_array, alpha=0.6, s=20)
        axes[0, 1].set_xlabel('Mass')
        axes[0, 1].set_ylabel('Spread')
        axes[0, 1].set_title('Mass vs Spread')
        
        # Stability histogram
        axes[1, 0].hist(stability_array, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Stability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Stability Distribution')
        
        # PPMI histogram
        axes[1, 1].hist(ppmi_array, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Average PPMI')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('PPMI Distribution')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = PROJECT_ROOT / "analysis" / "stability_ppmi_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

def main():
    """Main analysis function."""
    print("=== CNE Stability vs PPMI Analysis ===")
    print("Investigating the observation that higher instability words have lower PPMI values")
    
    try:
        word_data, stability_array, ppmi_array, mass_array, spread_array = analyze_stability_ppmi_relationship()
        
        # Create visualizations
        print("\n=== Creating Visualizations ===")
        create_visualizations(stability_array, ppmi_array, mass_array, spread_array)
        
        # Save detailed results
        results = {
            'summary': {
                'total_concepts': len(word_data),
                'stability_range': [float(stability_array.min()), float(stability_array.max())],
                'ppmi_range': [float(ppmi_array.min()), float(ppmi_array.max())],
                'correlation': float(np.corrcoef(stability_array, ppmi_array)[0, 1])
            },
            'word_data': word_data
        }
        
        results_path = PROJECT_ROOT / "analysis" / "stability_ppmi_results.json"
        with results_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {results_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
