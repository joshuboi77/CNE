#!/usr/bin/env python3
"""
Schwarzschild Radius Testing for CNE
Testing different formulations to find the correct Schwarzschild radius in conceptual space.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

def load_word_data():
    """Load the word data from analysis results."""
    results_file = PROJECT_ROOT / "lab" / "stability_ppmi_results.json"
    
    with results_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['word_data']

def compute_corpus_statistics(word_data: List[Dict]):
    """Compute corpus-wide statistics needed for advanced formulations."""
    masses = np.array([w['mass'] for w in word_data])
    ppmis = np.array([w['avg_ppmi'] for w in word_data])
    spreads = np.array([w['spread'] for w in word_data])
    stabilities = np.array([w['stability'] for w in word_data])
    
    stats = {
        'mass_percentiles': np.percentile(masses, [25, 50, 75, 90, 95]),
        'ppmi_percentiles': np.percentile(ppmis, [10, 25, 50, 75, 90]),
        'ppmi_mean': np.mean(ppmis),
        'ppmi_std': np.std(ppmis),
        'spread_median': np.median(spreads),
        'stability_percentiles': np.percentile(stabilities, [25, 50, 75, 90]),
        'mass_95': np.percentile(masses, 95),
        'ppmi_quantile_func': lambda p: np.percentile(ppmis, p * 100)
    }
    
    # Fit PPMI vs Mass relationship (simple polynomial)
    log_masses = np.log(masses + 1)
    ppmi_mass_fit = np.polyfit(log_masses, ppmis, 2)  # Quadratic fit
    
    def expected_ppmi(mass):
        log_m = np.log(mass + 1)
        return ppmi_mass_fit[0] * log_m**2 + ppmi_mass_fit[1] * log_m + ppmi_mass_fit[2]
    
    stats['expected_ppmi_func'] = expected_ppmi
    
    return stats

def test_schwarzschild_formulations(word_data: List[Dict]):
    """
    Test different Schwarzschild radius formulations against known collapsed words.
    
    Known collapsed words (should be identified as black holes):
    - "the" (mass: 450.1, avg_ppmi: 1.76) - definitely collapsed
    - "and", "of", "in", "to", "a", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "when", "we", "there", "can", "an", "your", "which", "said", "each", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him", "time", "has", "two", "more", "go", "no", "way", "could", "my", "than", "first", "been", "call", "who", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part"
    """
    
    # Known collapsed words (stop words that should be black holes)
    known_collapsed = {
        'the', 'and', 'of', 'in', 'to', 'a', 'is', 'it', 'you', 'that', 'he', 'was', 
        'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 
        'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 
        'all', 'were', 'when', 'we', 'there', 'can', 'an', 'your', 'which', 'said',
        'each', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about',
        'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
        'like', 'into', 'him', 'time', 'has', 'two', 'more', 'go', 'no', 'way',
        'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find',
        'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
    }
    
    # Known stable words (should NOT be collapsed)
    known_stable = {
        'love', 'death', 'freedom', 'justice', 'beauty', 'truth', 'wisdom', 
        'courage', 'honor', 'faith', 'hope', 'dream', 'journey', 'adventure',
        'mystery', 'magic', 'treasure', 'castle', 'forest', 'mountain', 'ocean',
        'heart', 'soul', 'spirit', 'mind', 'body', 'life', 'world', 'earth',
        'sky', 'stars', 'moon', 'sun', 'wind', 'fire', 'water', 'stone'
    }
    
    print("=== SCHWARZSCHILD RADIUS TESTING ===")
    print(f"Testing against {len(known_collapsed)} known collapsed words")
    print(f"Testing against {len(known_stable)} known stable words")
    
    # Extract data and compute corpus statistics
    words_dict = {w['word']: w for w in word_data}
    stats = compute_corpus_statistics(word_data)
    
    print(f"\nCorpus Statistics:")
    print(f"  Spread median: {stats['spread_median']:.3f}")
    print(f"  PPMI mean: {stats['ppmi_mean']:.3f} ± {stats['ppmi_std']:.3f}")
    print(f"  Mass 95th percentile: {stats['mass_95']:.1f}")
    
    # Test different formulations
    formulations = [
        # Original simple formulations
        {
            'name': 'Real Physics: R_s = 2GM/c²',
            'description': 'Using actual Schwarzschild formula with conceptual constants',
            'formula': lambda mass: 2 * mass / (299792458**2)  # c = speed of light
        },
        {
            'name': 'Universal Threshold: R_s = constant',
            'description': 'Fixed threshold based on corpus statistics',
            'formula': lambda mass: 2.5  # Fixed threshold
        },
        
        # Advanced formulations from the specification
        {
            'name': 'A) Power-log mass floor',
            'description': 'R_s = a + b[ln(M+1)]^α - monotone in mass',
            'formula': lambda mass: 2.3 + 0.55 * (np.log(mass + 1) ** 1.0)
        },
        {
            'name': 'B) Mass × spread gate',
            'description': 'Uses spread plateau to avoid over-collapsing tight concepts',
            'formula': lambda mass, spread: (2.2 + 0.5 * np.log(mass + 1)) * ((spread / stats['spread_median']) ** 0.35)
        },
        {
            'name': 'C) Quantile mapping',
            'description': 'Non-parametric mapping of mass to PPMI quantile',
            'formula': lambda mass: stats['ppmi_quantile_func'](
                0.25 + 0.55 * min(1.0, np.log(mass + 1) / np.log(stats['mass_95']))
            )
        },
        {
            'name': 'D) Residual-aware hybrid',
            'description': 'Uses lifecycle curve with residual penalty',
            'formula': lambda mass, spread, ppmi: (
                (2.1 + 0.52 * np.log(mass + 1)) * 
                ((spread / stats['spread_median']) ** 0.25) * 
                (1 - np.tanh(0.6 * (ppmi - stats['expected_ppmi_func'](mass))))
            )
        },
        {
            'name': 'E) Stability-driven boundary',
            'description': 'Directly uses stability metric σ = M/s',
            'formula': lambda mass, spread: 2.2 + 0.35 * (np.log(1 + mass/spread) ** 1.0)
        }
    ]
    
    results = []
    
    for formulation in formulations:
        print(f"\n--- Testing: {formulation['name']} ---")
        print(f"Description: {formulation['description']}")
        
        # Calculate Schwarzschild radius for each word
        word_results = []
        for word_info in word_data:
            mass = word_info['mass']
            ppmi = word_info['avg_ppmi']  # This is our "radius"
            spread = word_info['spread']
            
            # Calculate Schwarzschild radius (handle different parameter counts)
            try:
                if formulation['name'].startswith('B)') or formulation['name'].startswith('D)') or formulation['name'].startswith('E)'):
                    # Multi-parameter formulations
                    if formulation['name'].startswith('D)'):
                        rs = formulation['formula'](mass, spread, ppmi)
                    else:
                        rs = formulation['formula'](mass, spread)
                else:
                    # Single-parameter formulations
                    rs = formulation['formula'](mass)
            except Exception as e:
                # Don't print every error, just set to no collapse
                rs = float('inf')  # Default to no collapse
            
            # Collapse condition: R ≤ R_s (PPMI ≤ R_s)
            collapsed = ppmi <= rs
            
            word_results.append({
                'word': word_info['word'],
                'mass': mass,
                'ppmi': ppmi,
                'spread': spread,
                'rs': rs,
                'collapsed': bool(collapsed)  # Ensure boolean serialization
            })
        
        # Test against known collapsed words
        correct_collapsed = 0
        total_collapsed = 0
        for word in known_collapsed:
            if word in words_dict:
                word_result = next((w for w in word_results if w['word'] == word), None)
                if word_result:
                    total_collapsed += 1
                    if word_result['collapsed']:
                        correct_collapsed += 1
        
        # Test against known stable words
        correct_stable = 0
        total_stable = 0
        for word in known_stable:
            if word in words_dict:
                word_result = next((w for w in word_results if w['word'] == word), None)
                if word_result:
                    total_stable += 1
                    if not word_result['collapsed']:
                        correct_stable += 1
        
        # Calculate accuracy
        total_tests = total_collapsed + total_stable
        correct_tests = correct_collapsed + correct_stable
        accuracy = correct_tests / total_tests if total_tests > 0 else 0
        
        collapsed_accuracy = correct_collapsed / total_collapsed if total_collapsed > 0 else 0
        stable_accuracy = correct_stable / total_stable if total_stable > 0 else 0
        
        print(f"Collapsed words: {correct_collapsed}/{total_collapsed} ({collapsed_accuracy:.2%})")
        print(f"Stable words: {correct_stable}/{total_stable} ({stable_accuracy:.2%})")
        print(f"Overall accuracy: {correct_tests}/{total_tests} ({accuracy:.2%})")
        
        # Show some examples
        print("Examples:")
        for word_result in word_results[:5]:
            status = "COLLAPSED" if word_result['collapsed'] else "STABLE"
            print(f"  {word_result['word']}: mass={word_result['mass']:.1f}, ppmi={word_result['ppmi']:.2f}, rs={word_result['rs']:.2f} → {status}")
        
        results.append({
            'formulation': formulation['name'],
            'accuracy': accuracy,
            'collapsed_accuracy': collapsed_accuracy,
            'stable_accuracy': stable_accuracy,
            'results': word_results[:100]  # Limit results to avoid JSON issues
        })
    
    # Find best formulation
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n=== BEST FORMULATION ===")
    print(f"Winner: {best_result['formulation']}")
    print(f"Accuracy: {best_result['accuracy']:.2%}")
    
    return results

def main():
    """Main testing function."""
    print("=== CNE Schwarzschild Radius Testing ===")
    print("Testing different formulations to find the correct Schwarzschild radius")
    
    try:
        word_data = load_word_data()
        results = test_schwarzschild_formulations(word_data)
        
        # Save results
        output_path = PROJECT_ROOT / "lab" / "schwarzschild_test_results.json"
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
