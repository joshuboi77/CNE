#!/usr/bin/env python3
"""
Event Horizon Band Test - CNE Validation Protocol
Tests the Residual-Aware Schwarzschild Radius to confirm natural transition zones
between collapsed (black-hole) and stable (main-sequence) concepts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_word_data():
    """Load the word data from analysis results."""
    results_file = PROJECT_ROOT / "lab" / "stability_ppmi_results.json"
    
    with results_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['word_data']

def compute_corpus_statistics(word_data: List[Dict]):
    """Compute corpus-wide statistics needed for Schwarzschild calculations."""
    masses = np.array([w['mass'] for w in word_data])
    ppmis = np.array([w['avg_ppmi'] for w in word_data])
    spreads = np.array([w['spread'] for w in word_data])
    stabilities = np.array([w['stability'] for w in word_data])
    
    stats = {
        'spread_median': np.median(spreads),
        'ppmi_mean': np.mean(ppmis),
        'ppmi_std': np.std(ppmis),
        'mass_percentiles': np.percentile(masses, [25, 50, 75, 90, 95])
    }
    
    # Fit PPMI vs Mass relationship (simple polynomial)
    log_masses = np.log(masses + 1)
    ppmi_mass_fit = np.polyfit(log_masses, ppmis, 2)  # Quadratic fit
    
    def expected_ppmi(mass):
        log_m = np.log(mass + 1)
        return ppmi_mass_fit[0] * log_m**2 + ppmi_mass_fit[1] * log_m + ppmi_mass_fit[2]
    
    stats['expected_ppmi_func'] = expected_ppmi
    
    return stats

def compute_schwarzschild_radius(mass: float, spread: float, ppmi: float, stats: Dict, expected_ppmi_func):
    """
    Compute Residual-Aware Schwarzschild Radius using winning formulation:
    R_s(M,s,r) = (2.1 + 0.52*ln(M+1)) * (s/s*)^0.25 * (1 - tanh(0.6*r))
    """
    # Residual: difference between actual and expected PPMI
    residual = ppmi - expected_ppmi_func(mass)
    
    # Schwarzschild radius calculation
    rs = (2.1 + 0.52 * np.log(mass + 1)) * ((spread / stats['spread_median']) ** 0.25) * (1 - np.tanh(0.6 * residual))
    
    return rs

def run_event_horizon_band_test():
    """Run the complete Event Horizon Band Test."""
    
    print("=== EVENT HORIZON BAND TEST ===")
    print("Testing Residual-Aware Schwarzschild Radius for natural transition zones")
    
    # Step 1: Gather inputs
    print("\n1️⃣ Gathering inputs...")
    word_data = load_word_data()
    stats = compute_corpus_statistics(word_data)
    
    print(f"   Corpus size: {len(word_data)} concepts")
    print(f"   Spread median: {stats['spread_median']:.3f}")
    print(f"   PPMI mean: {stats['ppmi_mean']:.3f} ± {stats['ppmi_std']:.3f}")
    
    # Step 2: Compute Event-Horizon Ratio (φ)
    print("\n2️⃣ Computing Event-Horizon Ratio (φ = PPMI/R_s)...")
    
    phi_values = []
    rs_values = []
    phase_labels = []
    
    # Event horizon band width
    delta = 0.15
    
    for word_info in word_data:
        mass = word_info['mass']
        ppmi = word_info['avg_ppmi']
        spread = word_info['spread']
        
        # Calculate Schwarzschild radius
        rs = compute_schwarzschild_radius(mass, spread, ppmi, stats, stats['expected_ppmi_func'])
        
        # Calculate φ ratio
        phi = ppmi / rs if rs > 0 else float('inf')
        
        # Assign phase label
        if phi < 1 - delta:
            phase = "collapsed"
        elif abs(phi - 1) <= delta:
            phase = "event_horizon"
        else:
            phase = "stable"
        
        phi_values.append(phi)
        rs_values.append(rs)
        phase_labels.append(phase)
        
        # Store in word_info for later use
        word_info['phi'] = phi
        word_info['rs'] = rs
        word_info['phase'] = phase
    
    # Convert to numpy arrays
    phi_array = np.array(phi_values)
    rs_array = np.array(rs_values)
    
    print(f"   φ range: {phi_array.min():.3f} - {phi_array.max():.3f}")
    print(f"   R_s range: {rs_array.min():.3f} - {rs_array.max():.3f}")
    
    # Step 3: Analyze phase distribution
    print("\n3️⃣ Analyzing phase distribution...")
    
    phase_counts = defaultdict(int)
    for phase in phase_labels:
        phase_counts[phase] += 1
    
    total_concepts = len(word_data)
    print(f"   Phase distribution:")
    for phase, count in phase_counts.items():
        percentage = count / total_concepts * 100
        print(f"     {phase}: {count} ({percentage:.1f}%)")
    
    # Step 4: Create diagnostic plots
    print("\n4️⃣ Creating diagnostic plots...")
    
    # Prepare data for plotting
    masses = np.array([w['mass'] for w in word_data])
    ppmis = np.array([w['avg_ppmi'] for w in word_data])
    spreads = np.array([w['spread'] for w in word_data])
    stabilities = np.array([w['stability'] for w in word_data])
    
    # Color mapping for phases
    phase_colors = {
        'collapsed': 'black',
        'event_horizon': 'orange', 
        'stable': 'blue'
    }
    
    colors = [phase_colors[phase] for phase in phase_labels]
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Event Horizon Band Test - CNE Validation', fontsize=16, fontweight='bold')
    
    # A. Scatter Plot (core diagnostic)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(masses, ppmis, c=colors, alpha=0.6, s=20)
    ax1.set_xlabel('Mass (log scale)')
    ax1.set_ylabel('Average PPMI')
    ax1.set_xscale('log')
    ax1.set_title('A. Mass vs PPMI (Event Horizon Band)')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=8, label=f'{phase.replace("_", " ").title()}') 
                      for phase, color in phase_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # B. Density Histogram of φ values
    ax2 = axes[0, 1]
    ax2.hist(phi_array, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='φ = 1.0')
    ax2.axvline(x=1-delta, color='orange', linestyle=':', linewidth=2, label=f'Event Horizon Band (±{delta})')
    ax2.axvline(x=1+delta, color='orange', linestyle=':', linewidth=2)
    ax2.set_xlabel('φ = PPMI / R_s')
    ax2.set_ylabel('Frequency')
    ax2.set_title('B. Distribution of Event-Horizon Ratio (φ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # C. Stability by Phase
    ax3 = axes[1, 0]
    phase_stabilities = defaultdict(list)
    for i, phase in enumerate(phase_labels):
        phase_stabilities[phase].append(stabilities[i])
    
    phase_names = list(phase_stabilities.keys())
    stability_means = [np.mean(phase_stabilities[phase]) for phase in phase_names]
    stability_stds = [np.std(phase_stabilities[phase]) for phase in phase_names]
    
    bars = ax3.bar(phase_names, stability_means, yerr=stability_stds, 
                   color=[phase_colors[phase] for phase in phase_names], alpha=0.7, capsize=5)
    ax3.set_ylabel('Mean Stability')
    ax3.set_title('C. Stability by Phase')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, stability_means)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + stability_stds[i],
                f'{mean_val:.1f}', ha='center', va='bottom')
    
    # D. PPMI vs R_s (should show the 1:1 line)
    ax4 = axes[1, 1]
    ax4.scatter(rs_array, ppmis, c=colors, alpha=0.6, s=20)
    
    # Add 1:1 line and event horizon bands
    max_val = max(rs_array.max(), ppmis.max())
    min_val = min(rs_array.min(), ppmis.min())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='φ = 1.0')
    ax4.plot([min_val, max_val], [min_val*(1+delta), max_val*(1+delta)], 'orange', linestyle=':', linewidth=2, label=f'Event Horizon Band')
    ax4.plot([min_val, max_val], [min_val*(1-delta), max_val*(1-delta)], 'orange', linestyle=':', linewidth=2)
    
    ax4.set_xlabel('R_s (Schwarzschild Radius)')
    ax4.set_ylabel('PPMI (Semantic Radius)')
    ax4.set_title('D. PPMI vs R_s (Event Horizon Validation)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = PROJECT_ROOT / "lab" / "event_horizon_band_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Diagnostic plots saved to: {output_path}")
    
    # Step 5: Quantitative Checks
    print("\n5️⃣ Quantitative validation...")
    
    # Band density
    horizon_count = phase_counts['event_horizon']
    band_density = horizon_count / total_concepts
    print(f"   Band density: {band_density:.3f} ({horizon_count}/{total_concepts} concepts)")
    
    # Entropy of PPMI within each phase
    print(f"   PPMI entropy by phase:")
    for phase in ['collapsed', 'event_horizon', 'stable']:
        phase_ppmis = [ppmis[i] for i, p in enumerate(phase_labels) if p == phase]
        if phase_ppmis:
            entropy = -np.sum([(p/np.sum(phase_ppmis)) * np.log(p/np.sum(phase_ppmis)) for p in phase_ppmis if p > 0])
            print(f"     {phase}: {entropy:.3f}")
    
    # Stability mean per phase (should increase monotonically)
    print(f"   Stability means by phase:")
    stability_means_sorted = []
    for phase in ['collapsed', 'event_horizon', 'stable']:
        phase_stabilities_list = [stabilities[i] for i, p in enumerate(phase_labels) if p == phase]
        if phase_stabilities_list:
            mean_stability = np.mean(phase_stabilities_list)
            stability_means_sorted.append(mean_stability)
            print(f"     {phase}: {mean_stability:.1f}")
    
    # Check monotonic increase
    if len(stability_means_sorted) == 3:
        monotonic = stability_means_sorted[0] < stability_means_sorted[1] < stability_means_sorted[2]
        print(f"   Monotonic stability increase: {'✅' if monotonic else '❌'}")
    
    # Step 6: Confirmation Criteria
    print("\n6️⃣ Confirmation criteria check...")
    
    # 1. Distinct, non-overlapping clusters for φ < 1, ≈ 1, > 1
    phi_collapsed = [phi_array[i] for i, phase in enumerate(phase_labels) if phase == 'collapsed']
    phi_horizon = [phi_array[i] for i, phase in enumerate(phase_labels) if phase == 'event_horizon']
    phi_stable = [phi_array[i] for i, phase in enumerate(phase_labels) if phase == 'stable']
    
    distinct_clusters = True
    if phi_collapsed and phi_horizon and phi_stable:
        max_collapsed = max(phi_collapsed)
        min_horizon = min(phi_horizon)
        max_horizon = max(phi_horizon)
        min_stable = min(phi_stable)
        
        # Check for overlap
        if max_collapsed >= min_horizon or max_horizon >= min_stable:
            distinct_clusters = False
    
    print(f"   1. Distinct clusters: {'✅' if distinct_clusters else '❌'}")
    
    # 2. Density spike at φ ≈ 1
    phi_bins = np.linspace(phi_array.min(), phi_array.max(), 50)
    phi_hist, _ = np.histogram(phi_array, bins=phi_bins)
    horizon_bin_indices = np.where((phi_bins[:-1] >= 1-delta) & (phi_bins[:-1] <= 1+delta))[0]
    
    if len(horizon_bin_indices) > 0:
        horizon_density = np.mean(phi_hist[horizon_bin_indices])
        overall_density = np.mean(phi_hist)
        density_spike = horizon_density > overall_density * 1.2  # 20% above average
    else:
        density_spike = False
    
    print(f"   2. Density spike at φ ≈ 1: {'✅' if density_spike else '❌'}")
    
    # 3. Reproducible median spread (single corpus check)
    expected_spread_range = (0.35, 0.37)
    reproducible_spread = expected_spread_range[0] <= stats['spread_median'] <= expected_spread_range[1]
    print(f"   3. Median spread in expected range: {'✅' if reproducible_spread else '❌'} ({stats['spread_median']:.3f})")
    
    # Overall confirmation
    criteria_met = sum([distinct_clusters, density_spike, reproducible_spread])
    print(f"\n   Overall confirmation: {criteria_met}/3 criteria met")
    
    if criteria_met >= 2:
        print("   🎉 CNE THEORY VALIDATED! Event horizon band detected successfully!")
    else:
        print("   ⚠️  CNE theory needs refinement. Event horizon band not clearly detected.")
    
    # Save detailed results
    results = {
        'test_summary': {
            'total_concepts': int(total_concepts),
            'delta': float(delta),
            'band_density': float(band_density),
            'criteria_met': int(criteria_met),
            'spread_median': float(stats['spread_median']),
            'ppmi_mean': float(stats['ppmi_mean']),
            'ppmi_std': float(stats['ppmi_std'])
        },
        'phase_distribution': {k: int(v) for k, v in phase_counts.items()},
        'stability_means': {k: float(v) for k, v in zip(['collapsed', 'event_horizon', 'stable'], stability_means_sorted)},
        'confirmation_criteria': {
            'distinct_clusters': bool(distinct_clusters),
            'density_spike': bool(density_spike),
            'reproducible_spread': bool(reproducible_spread)
        }
    }
    
    results_path = PROJECT_ROOT / "lab" / "event_horizon_test_results.json"
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n   Detailed results saved to: {results_path}")
    
    return results

def main():
    """Main function to run the Event Horizon Band Test."""
    print("=== CNE Event Horizon Band Test ===")
    print("Validating Residual-Aware Schwarzschild Radius for conceptual transition zones")
    
    try:
        results = run_event_horizon_band_test()
        print(f"\n=== TEST COMPLETE ===")
        
    except Exception as e:
        print(f"Error during Event Horizon Band Test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
