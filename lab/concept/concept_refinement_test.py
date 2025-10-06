#!/usr/bin/env python3
"""
Concept Refinement Test - Testing boundary and membership refinement
Tests the theory that concept boundaries can be refined by pulling in beneficial neighbors
and pushing out weak members in a single pass.
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

def load_concept_builder_results():
    """Load results from concept builder test."""
    results_file = PROJECT_ROOT / "lab" / "concept" / "concept_builder_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Concept builder results not found: {results_file}")
    
    with results_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert summary data back to full concept format for testing
    concepts = []
    for concept_data in data.get('top_concepts', []):
        concept = {
            'token': concept_data['token'],
            'is_valid': True,
            'score': concept_data['score'],
            'shell_size': concept_data['shell_size'],
            'phi': concept_data['phi'],
            'shell_members': [('dummy_neighbor', 1.0) for _ in range(concept_data['shell_size'])]  # Placeholder
        }
        concepts.append(concept)
    
    return {'concepts': concepts}

def load_original_concepts():
    """Load original concept data for neighbor lookup."""
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

class ConceptRefiner:
    """Implements concept boundary and membership refinement."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'radius_expansion_limit': 0.15,  # Max 15% radius expansion
            'cosine_drop_threshold': 0.20,   # Drop members below this cosine
            'ppmi_drop_margin': 0.3,         # Additional PPMI margin for dropping
            'cohesion_improvement_threshold': 0.05  # Minimum improvement to add member
        }
    
    def compute_concept_cohesion(self, concept: Dict, shell_members: List[Tuple[str, float]]) -> Tuple[float, float]:
        """
        Compute PPMI and embedding cohesion for a concept.
        
        Args:
            concept: Concept dictionary
            shell_members: List of (token, ppmi) tuples
            
        Returns:
            ppmi_cohesion: PPMI-based cohesion score
            emb_cohesion: Embedding-based cohesion score
        """
        if not shell_members:
            return 0.0, 0.0
        
        # PPMI cohesion
        ppmis = [ppmi for _, ppmi in shell_members]
        avg_ppmi = np.mean(ppmis)
        ppmi_cohesion = np.mean([ppmi / avg_ppmi for ppmi in ppmis]) if avg_ppmi > 0 else 0.0
        
        # Embedding cohesion
        center_embedding = np.array(concept.get('center', [0] * 128))
        cosines = []
        
        for token, _ in shell_members:
            member_embedding = deterministic_embedding(token)
            cosine = np.dot(center_embedding, member_embedding)
            cosines.append(cosine)
        
        emb_cohesion = np.mean(cosines)
        
        return ppmi_cohesion, emb_cohesion
    
    def compute_robust_center(self, concept: Dict, shell_members: List[Tuple[str, float]]) -> np.ndarray:
        """
        Compute robust concept center using trimmed mean.
        
        Args:
            concept: Concept dictionary
            shell_members: List of (token, ppmi) tuples
            
        Returns:
            center: Robust center embedding
        """
        if not shell_members:
            return np.array(concept.get('center', [0] * 128))
        
        # Get all embeddings
        embeddings = []
        for token, _ in shell_members:
            embeddings.append(deterministic_embedding(token))
        
        # Add seed embedding
        seed_embedding = np.array(concept.get('center', [0] * 128))
        embeddings.append(seed_embedding)
        
        embeddings = np.array(embeddings)
        
        # Use 80% trimmed mean for robustness
        trim_percent = 0.2
        n_trim = int(len(embeddings) * trim_percent / 2)
        
        if n_trim > 0:
            # Sort by L2 norm and trim outliers
            norms = np.linalg.norm(embeddings, axis=1)
            sorted_indices = np.argsort(norms)
            keep_indices = sorted_indices[n_trim:-n_trim] if n_trim < len(embeddings)//2 else sorted_indices
            embeddings = embeddings[keep_indices]
        
        # Compute mean and normalize
        center = np.mean(embeddings, axis=0)
        return center / (np.linalg.norm(center) + 1e-9)
    
    def compute_concept_radius(self, concept: Dict, shell_members: List[Tuple[str, float]]) -> float:
        """
        Compute concept radius as MAD of cosine distances using robust center.
        
        Args:
            concept: Concept dictionary
            shell_members: List of (token, ppmi) tuples
            
        Returns:
            radius: Concept radius
        """
        if not shell_members:
            return 0.0
        
        # Use robust center
        center_embedding = self.compute_robust_center(concept, shell_members)
        cosines = []
        
        for token, _ in shell_members:
            member_embedding = deterministic_embedding(token)
            cosine = np.dot(center_embedding, member_embedding)
            cosines.append(cosine)
        
        # Compute median absolute deviation
        median_cosine = np.median(cosines)
        mad = np.median([abs(c - median_cosine) for c in cosines])
        
        return float(mad)
    
    def find_candidate_additions(self, concept: Dict, original_concept_data: Dict) -> List[Tuple[str, float]]:
        """
        Find neighbors that could be added to the concept.
        
        Args:
            concept: Current concept
            original_concept_data: Original concept data with full neighbor list
            
        Returns:
            candidates: List of (token, ppmi) tuples that could be added
        """
        current_members = set(token for token, _ in concept.get('shell_members', []))
        all_neighbors = original_concept_data.get('neighbors', [])
        
        candidates = []
        for neighbor in all_neighbors:
            token = neighbor['id']
            ppmi = neighbor['ppmi']
            
            # Skip if already a member
            if token in current_members:
                continue
            
            # Skip if PPMI is too low
            avg_ppmi = original_concept_data.get('avg_ppmi', 0)
            if ppmi < avg_ppmi - 0.5:  # Same threshold as in concept builder
                continue
            
            candidates.append((token, ppmi))
        
        return candidates
    
    def evaluate_addition(self, concept: Dict, current_shell: List[Tuple[str, float]], 
                         candidate: Tuple[str, float], original_concept_data: Dict) -> Tuple[bool, float]:
        """
        Evaluate whether adding a candidate would improve the concept.
        
        Args:
            concept: Current concept
            current_shell: Current shell members
            candidate: Candidate to add (token, ppmi)
            original_concept_data: Original concept data
            
        Returns:
            should_add: Boolean indicating if candidate should be added
            improvement: Improvement in cohesion scores
        """
        # Compute current cohesion
        current_ppmi_cohesion, current_emb_cohesion = self.compute_concept_cohesion(concept, current_shell)
        current_radius = self.compute_concept_radius(concept, current_shell)
        
        # Compute cohesion with candidate added
        new_shell = current_shell + [candidate]
        new_ppmi_cohesion, new_emb_cohesion = self.compute_concept_cohesion(concept, new_shell)
        new_radius = self.compute_concept_radius(concept, new_shell)
        
        # Check radius expansion limit
        radius_expansion = (new_radius - current_radius) / (current_radius + 1e-9)
        if radius_expansion > self.config['radius_expansion_limit']:
            return False, 0.0
        
        # Compute improvement
        ppmi_improvement = new_ppmi_cohesion - current_ppmi_cohesion
        emb_improvement = new_emb_cohesion - current_emb_cohesion
        total_improvement = ppmi_improvement + emb_improvement
        
        # No-regret guard: if radius grows but cohesion doesn't improve enough, reject
        epsilon = self.config.get('no_regret_threshold', 0.05)
        if radius_expansion > 0 and total_improvement < epsilon:
            return False, total_improvement
        
        should_add = total_improvement >= self.config['cohesion_improvement_threshold']
        
        return should_add, total_improvement
    
    def evaluate_removal(self, concept: Dict, current_shell: List[Tuple[str, float]], 
                        member: Tuple[str, float], original_concept_data: Dict) -> Tuple[bool, float]:
        """
        Evaluate whether removing a member would improve the concept.
        
        Args:
            concept: Current concept
            current_shell: Current shell members
            member: Member to potentially remove (token, ppmi)
            original_concept_data: Original concept data
            
        Returns:
            should_remove: Boolean indicating if member should be removed
            improvement: Improvement in cohesion scores
        """
        # Compute current cohesion
        current_ppmi_cohesion, current_emb_cohesion = self.compute_concept_cohesion(concept, current_shell)
        
        # Compute cohesion with member removed
        new_shell = [m for m in current_shell if m != member]
        if not new_shell:  # Don't remove if it would leave concept empty
            return False, 0.0
        
        new_ppmi_cohesion, new_emb_cohesion = self.compute_concept_cohesion(concept, new_shell)
        
        # Check cosine threshold
        center_embedding = np.array(concept.get('center', [0] * 128))
        member_embedding = deterministic_embedding(member[0])
        cosine = np.dot(center_embedding, member_embedding)
        
        if cosine < self.config['cosine_drop_threshold']:
            return True, 0.0  # Remove due to low cosine
        
        # Check PPMI threshold
        avg_ppmi = original_concept_data.get('avg_ppmi', 0)
        if member[1] < avg_ppmi - 0.5 - self.config['ppmi_drop_margin']:
            return True, 0.0  # Remove due to low PPMI
        
        # Compute improvement
        ppmi_improvement = new_ppmi_cohesion - current_ppmi_cohesion
        emb_improvement = new_emb_cohesion - current_emb_cohesion
        total_improvement = ppmi_improvement + emb_improvement
        
        should_remove = total_improvement > 0.0  # Remove if it improves cohesion
        
        return should_remove, total_improvement
    
    def refine_concept(self, concept: Dict, original_concept_data: Dict) -> Dict:
        """
        Refine a concept by adding beneficial neighbors and removing weak members.
        
        Args:
            concept: Current concept
            original_concept_data: Original concept data with full neighbor list
            
        Returns:
            refined_concept: Refined concept dictionary
        """
        current_shell = concept.get('shell_members', []).copy()
        original_shell_size = len(current_shell)
        
        # Step 1: Try to add beneficial neighbors
        candidates = self.find_candidate_additions(concept, original_concept_data)
        additions_made = 0
        
        for candidate in candidates:
            should_add, improvement = self.evaluate_addition(concept, current_shell, candidate, original_concept_data)
            if should_add:
                current_shell.append(candidate)
                additions_made += 1
        
        # Early exit if no additions made
        if additions_made == 0:
            # Still check for removals but skip if no changes
            members_to_remove = []
            for member in current_shell:
                should_remove, improvement = self.evaluate_removal(concept, current_shell, member, original_concept_data)
                if should_remove:
                    members_to_remove.append(member)
            
            if not members_to_remove:
                # No changes needed, return early
                refined_concept = concept.copy()
                refined_concept['refinement_stats'] = {
                    'original_size': original_shell_size,
                    'additions': 0,
                    'removals': 0,
                    'final_size': len(current_shell),
                    'size_change': 0
                }
                return refined_concept
        
        # Step 2: Remove weak members
        removals_made = 0
        members_to_remove = []
        
        for member in current_shell:
            should_remove, improvement = self.evaluate_removal(concept, current_shell, member, original_concept_data)
            if should_remove:
                members_to_remove.append(member)
        
        # Remove identified members
        for member in members_to_remove:
            current_shell.remove(member)
            removals_made += 1
        
        # Create refined concept
        refined_concept = concept.copy()
        refined_concept['shell_members'] = current_shell
        refined_concept['shell_size'] = len(current_shell)
        refined_concept['refinement_stats'] = {
            'original_size': original_shell_size,
            'additions': additions_made,
            'removals': removals_made,
            'final_size': len(current_shell),
            'size_change': len(current_shell) - original_shell_size
        }
        
        # Recompute cohesion scores
        ppmi_cohesion, emb_cohesion = self.compute_concept_cohesion(refined_concept, current_shell)
        refined_concept['refined_scores'] = {
            'ppmi_cohesion': ppmi_cohesion,
            'emb_cohesion': emb_cohesion
        }
        
        # Recompute radius
        refined_concept['radius'] = self.compute_concept_radius(refined_concept, current_shell)
        
        return refined_concept
    
    def test_refinement(self, concepts: List[Dict], original_concepts: Dict) -> Dict:
        """
        Test refinement on all valid concepts.
        
        Args:
            concepts: List of concept dictionaries from concept builder
            original_concepts: Original concept data for neighbor lookup
            
        Returns:
            results: Dictionary with refinement test results
        """
        print("Testing concept refinement...")
        
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        print(f"Testing refinement on {len(valid_concepts)} valid concepts...")
        
        refined_concepts = []
        refinement_stats = []
        
        for concept in valid_concepts:
            token = concept['token']
            if token not in original_concepts:
                continue
            
            original_concept_data = original_concepts[token]
            refined_concept = self.refine_concept(concept, original_concept_data)
            
            refined_concepts.append(refined_concept)
            
            # Collect statistics
            stats = refined_concept.get('refinement_stats', {})
            refinement_stats.append(stats)
        
        # Analyze results
        results = {
            'total_concepts_refined': len(refined_concepts),
            'refinement_stats': refinement_stats,
            'refined_concepts': refined_concepts
        }
        
        return results

def analyze_refinement_results(results: Dict):
    """Analyze the results of concept refinement."""
    print("\n=== Refinement Analysis ===")
    
    stats = results.get('refinement_stats', [])
    if not stats:
        print("No refinement statistics available.")
        return
    
    # Size changes
    size_changes = [s['size_change'] for s in stats]
    additions = [s['additions'] for s in stats]
    removals = [s['removals'] for s in stats]
    
    print(f"Size changes:")
    print(f"  Mean: {np.mean(size_changes):.2f}")
    print(f"  Median: {np.median(size_changes):.2f}")
    print(f"  Range: {min(size_changes)} to {max(size_changes)}")
    
    print(f"\nAdditions:")
    print(f"  Mean: {np.mean(additions):.2f}")
    print(f"  Median: {np.median(additions):.2f}")
    print(f"  Total: {sum(additions)}")
    
    print(f"\nRemovals:")
    print(f"  Mean: {np.mean(removals):.2f}")
    print(f"  Median: {np.median(removals):.2f}")
    print(f"  Total: {sum(removals)}")
    
    # Concepts that changed
    changed_concepts = [s for s in stats if s['size_change'] != 0]
    print(f"\nConcepts that changed: {len(changed_concepts)}/{len(stats)} ({len(changed_concepts)/len(stats)*100:.1f}%)")
    
    # Show examples of significant changes
    significant_changes = [s for s in stats if abs(s['size_change']) >= 3]
    if significant_changes:
        print(f"\nSignificant changes (≥3 members):")
        for i, change in enumerate(significant_changes[:10], 1):
            print(f"  {i:2d}. Size change: {change['size_change']:+d} (additions: {change['additions']}, removals: {change['removals']})")

def main():
    """Main testing function."""
    print("=== Concept Refinement Test ===")
    
    try:
        # Load data
        concept_data = load_concept_builder_results()
        concepts = concept_data.get('concepts', [])
        
        original_concepts = load_original_concepts()
        
        print(f"Loaded {len(concepts)} concepts from concept builder")
        print(f"Loaded {len(original_concepts)} original concepts")
        
        # Initialize refiner
        refiner = ConceptRefiner()
        
        # Test refinement
        results = refiner.test_refinement(concepts, original_concepts)
        
        # Analyze results
        analyze_refinement_results(results)
        
        # Save results (convert numpy types for JSON)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        output_path = PROJECT_ROOT / "lab" / "concept" / "concept_refinement_results.json"
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during refinement testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
