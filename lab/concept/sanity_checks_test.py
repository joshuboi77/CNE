#!/usr/bin/env python3
"""
Sanity Checks Test - Testing concept quality metrics and validation
Tests the theory that concept building should meet certain quality criteria:
coverage, purity, margin, stability rank, and event-horizon avoidance.
"""

import json
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_concept_graph_results():
    """Load results from concept graph test."""
    results_file = PROJECT_ROOT / "lab" / "concept" / "concept_graph_results.json"
    
    if not results_file.exists():
        # Fallback to concept builder results
        results_file = PROJECT_ROOT / "lab" / "concept" / "concept_builder_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Concept results not found: {results_file}")
    
    with results_file.open('r', encoding='utf-8') as f:
        return json.load(f)

def load_original_concepts():
    """Load original concept data for comparison."""
    concepts_file = PROJECT_ROOT / "concept_ppmi_results" / "concepts_ppmi.json"
    
    if not concepts_file.exists():
        raise FileNotFoundError(f"Original concepts file not found: {concepts_file}")
    
    with concepts_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('concepts', {})

def deterministic_embedding(token: str, dim: int = 128) -> np.ndarray:
    """Deterministic pseudo-embedding via hashing."""
    rng = np.random.default_rng(abs(hash(token)) % (2**32))
    v = rng.normal(scale=1.0, size=(dim,))
    return v / (np.linalg.norm(v) + 1e-9)

class ConceptSanityChecker:
    """Implements sanity checks for concept quality validation."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'coverage_target_min': 0.10,      # Minimum 10% coverage
            'coverage_target_max': 0.25,      # Maximum 25% coverage
            'purity_target_min': 0.45,        # Minimum mean cosine to center
            'margin_target_min': 0.2,         # Minimum R_w - R_s(w) margin
            'stability_rank_target': 0.5,     # Concepts should exceed corpus median stability
            'event_horizon_max': 0.05,        # Maximum 5% of concepts with phi < 1-delta
            'delta': 0.15                     # Event horizon band width
        }
    
    def check_coverage(self, concepts: List[Dict], total_tokens: int) -> Dict:
        """
        Check 1: Coverage - % of tokens promoted to concepts (expect 10-25%).
        
        Args:
            concepts: List of concept dictionaries
            total_tokens: Total number of tokens in corpus
            
        Returns:
            coverage_check: Dictionary with coverage analysis
        """
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        coverage_percent = len(valid_concepts) / total_tokens if total_tokens > 0 else 0
        
        is_acceptable = (
            self.config['coverage_target_min'] <= coverage_percent <= self.config['coverage_target_max']
        )
        
        return {
            'metric': 'coverage',
            'value': coverage_percent,
            'target_min': self.config['coverage_target_min'],
            'target_max': self.config['coverage_target_max'],
            'is_acceptable': is_acceptable,
            'valid_concepts': len(valid_concepts),
            'total_tokens': total_tokens
        }
    
    def check_purity(self, concepts: List[Dict]) -> Dict:
        """
        Check 2: Purity - mean cosine of members to concept center (target ≥ 0.45).
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            purity_check: Dictionary with purity analysis
        """
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        
        if not valid_concepts:
            return {
                'metric': 'purity',
                'value': 0.0,
                'target_min': self.config['purity_target_min'],
                'is_acceptable': False,
                'error': 'No valid concepts found'
            }
        
        purity_scores = []
        
        for concept in valid_concepts:
            center = np.array(concept.get('center', [0] * 128))
            shell_members = concept.get('shell_members', [])
            
            if not shell_members:
                continue
            
            # Compute cosine similarities to center
            cosines = []
            for token, _ in shell_members:
                member_embedding = deterministic_embedding(token)
                cosine = np.dot(center, member_embedding)
                cosines.append(cosine)
            
            if cosines:
                purity_scores.append(np.mean(cosines))
        
        mean_purity = np.mean(purity_scores) if purity_scores else 0.0
        is_acceptable = mean_purity >= self.config['purity_target_min']
        
        return {
            'metric': 'purity',
            'value': mean_purity,
            'target_min': self.config['purity_target_min'],
            'is_acceptable': is_acceptable,
            'concepts_analyzed': len(purity_scores),
            'purity_scores': purity_scores[:100]  # First 100 for inspection
        }
    
    def check_margin(self, concepts: List[Dict], original_concepts: Dict) -> Dict:
        """
        Check 3: Margin - mean R_w - R_s(w) over concepts (positive, >0.2).
        
        Args:
            concepts: List of concept dictionaries
            original_concepts: Original concept data for R_s values
            
        Returns:
            margin_check: Dictionary with margin analysis
        """
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        
        if not valid_concepts:
            return {
                'metric': 'margin',
                'value': 0.0,
                'target_min': self.config['margin_target_min'],
                'is_acceptable': False,
                'error': 'No valid concepts found'
            }
        
        margins = []
        
        for concept in valid_concepts:
            token = concept['token']
            if token not in original_concepts:
                continue
            
            original_data = original_concepts[token]
            avg_ppmi = original_data.get('avg_ppmi', 0)
            rs = original_data.get('rs', 0)
            
            if rs > 0:
                margin = avg_ppmi - rs
                margins.append(margin)
        
        mean_margin = np.mean(margins) if margins else 0.0
        is_acceptable = mean_margin >= self.config['margin_target_min']
        
        return {
            'metric': 'margin',
            'value': mean_margin,
            'target_min': self.config['margin_target_min'],
            'is_acceptable': is_acceptable,
            'concepts_analyzed': len(margins),
            'margins': margins[:100]  # First 100 for inspection
        }
    
    def check_stability_rank(self, concepts: List[Dict], original_concepts: Dict) -> Dict:
        """
        Check 4: Stability rank - concepts' median stability should exceed corpus median.
        
        Args:
            concepts: List of concept dictionaries
            original_concepts: Original concept data for stability values
            
        Returns:
            stability_check: Dictionary with stability analysis
        """
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        
        if not valid_concepts:
            return {
                'metric': 'stability_rank',
                'value': 0.0,
                'target_min': 0.5,
                'is_acceptable': False,
                'error': 'No valid concepts found'
            }
        
        # Get corpus stability values
        corpus_stabilities = [c.get('stability', 0) for c in original_concepts.values()]
        corpus_median = np.median(corpus_stabilities) if corpus_stabilities else 0
        
        # Get concept stability values
        concept_stabilities = []
        for concept in valid_concepts:
            token = concept['token']
            if token in original_concepts:
                stability = original_concepts[token].get('stability', 0)
                concept_stabilities.append(stability)
        
        concept_median = np.median(concept_stabilities) if concept_stabilities else 0
        
        # Check if concept median exceeds corpus median
        is_acceptable = concept_median > corpus_median
        
        return {
            'metric': 'stability_rank',
            'value': concept_median,
            'corpus_median': corpus_median,
            'target_min': corpus_median,
            'is_acceptable': is_acceptable,
            'concepts_analyzed': len(concept_stabilities),
            'rank_ratio': concept_median / corpus_median if corpus_median > 0 else 0
        }
    
    def check_event_horizon_avoidance(self, concepts: List[Dict], original_concepts: Dict) -> Dict:
        """
        Check 5: Event-horizon avoidance - <5% of concepts with phi < 1-delta.
        
        Args:
            concepts: List of concept dictionaries
            original_concepts: Original concept data for phi values
            
        Returns:
            horizon_check: Dictionary with event horizon analysis
        """
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        
        if not valid_concepts:
            return {
                'metric': 'event_horizon_avoidance',
                'value': 0.0,
                'target_max': self.config['event_horizon_max'],
                'is_acceptable': False,
                'error': 'No valid concepts found'
            }
        
        horizon_violations = 0
        phi_values = []
        
        for concept in valid_concepts:
            token = concept['token']
            if token in original_concepts:
                phi = original_concepts[token].get('phi', 1.0)
                phi_values.append(phi)
                
                if phi < 1.0 - self.config['delta']:
                    horizon_violations += 1
        
        violation_rate = horizon_violations / len(valid_concepts) if valid_concepts else 0
        is_acceptable = violation_rate <= self.config['event_horizon_max']
        
        return {
            'metric': 'event_horizon_avoidance',
            'value': violation_rate,
            'target_max': self.config['event_horizon_max'],
            'is_acceptable': is_acceptable,
            'violations': horizon_violations,
            'total_concepts': len(valid_concepts),
            'delta': self.config['delta'],
            'phi_values': phi_values[:100]  # First 100 for inspection
        }
    
    def run_all_checks(self, concepts: List[Dict], original_concepts: Dict) -> Dict:
        """
        Run all sanity checks and return comprehensive results.
        
        Args:
            concepts: List of concept dictionaries
            original_concepts: Original concept data
            
        Returns:
            results: Dictionary with all sanity check results
        """
        print("Running concept sanity checks...")
        
        total_tokens = len(original_concepts)
        
        # Run all checks
        checks = [
            self.check_coverage(concepts, total_tokens),
            self.check_purity(concepts),
            self.check_margin(concepts, original_concepts),
            self.check_stability_rank(concepts, original_concepts),
            self.check_event_horizon_avoidance(concepts, original_concepts)
        ]
        
        # Compute overall score
        passed_checks = sum(1 for check in checks if check.get('is_acceptable', False))
        total_checks = len(checks)
        overall_score = passed_checks / total_checks if total_checks > 0 else 0
        
        results = {
            'overall_score': overall_score,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'checks': {check['metric']: check for check in checks},
            'summary': {
                'coverage': checks[0].get('value', 0),
                'purity': checks[1].get('value', 0),
                'margin': checks[2].get('value', 0),
                'stability_rank': checks[3].get('value', 0),
                'event_horizon_violation_rate': checks[4].get('value', 0)
            }
        }
        
        return results

def print_sanity_check_results(results: Dict):
    """Print formatted sanity check results."""
    print("\n=== Concept Sanity Check Results ===")
    
    overall_score = results['overall_score']
    passed_checks = results['passed_checks']
    total_checks = results['total_checks']
    
    print(f"Overall Score: {overall_score:.1%} ({passed_checks}/{total_checks} checks passed)")
    
    if overall_score >= 0.8:
        print("EXCELLENT: Concept building meets quality standards!")
    elif overall_score >= 0.6:
        print("GOOD: Concept building meets most quality standards")
    elif overall_score >= 0.4:
        print("FAIR: Concept building needs improvement")
    else:
        print("POOR: Concept building fails quality standards")
    
    print(f"\nDetailed Results:")
    
    for metric, check in results['checks'].items():
        status = "PASS" if check.get('is_acceptable', False) else "FAIL"
        value = check.get('value', 0)
        
        print(f"  {metric.replace('_', ' ').title()}: {status}")
        print(f"    Value: {value:.3f}")
        
        if 'target_min' in check:
            print(f"    Target: ≥ {check['target_min']:.3f}")
        elif 'target_max' in check:
            print(f"    Target: ≤ {check['target_max']:.3f}")
        
        if 'error' in check:
            print(f"    Error: {check['error']}")
        
        print()

def main():
    """Main testing function."""
    print("=== Concept Sanity Checks Test ===")
    
    try:
        # Load data from concept builder results
        concept_data = load_concept_graph_results()
        
        # Convert top concepts to full concept format
        concepts = []
        for concept_info in concept_data.get('top_concepts', []):
            concept = {
                'token': concept_info['token'],
                'is_valid': True,
                'score': concept_info['score'],
                'shell_size': concept_info['shell_size'],
                'phi': concept_info['phi'],
                'shell_members': [('dummy_neighbor', 1.0) for _ in range(concept_info['shell_size'])]
            }
            concepts.append(concept)
        
        original_concepts = load_original_concepts()
        
        print(f"Loaded {len(concepts)} concepts for sanity checking")
        print(f"Loaded {len(original_concepts)} original concepts for comparison")
        
        # Initialize sanity checker
        checker = ConceptSanityChecker()
        
        # Run all checks
        results = checker.run_all_checks(concepts, original_concepts)
        
        # Print results
        print_sanity_check_results(results)
        
        # Save results (convert numpy types for JSON)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        output_path = PROJECT_ROOT / "lab" / "concept" / "sanity_checks_results.json"
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during sanity checking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
