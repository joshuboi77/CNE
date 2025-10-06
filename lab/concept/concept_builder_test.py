#!/usr/bin/env python3
"""
Concept Builder v1 (CB-v1) Testing Framework
Tests the new concept building theories with crisp gates, scores, and refinement.
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

def load_concept_data():
    """Load the existing concept data from PPMI pipeline."""
    concepts_file = PROJECT_ROOT / "concept_ppmi_results" / "concepts_ppmi.json"
    
    if not concepts_file.exists():
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    print("Loading concept data...")
    with concepts_file.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('concepts', {})

def load_embeddings():
    """Load trained embeddings if available."""
    emb_path = PROJECT_ROOT / "models" / "embeddings.npy"
    vocab_path = PROJECT_ROOT / "models" / "embedding_vocab.json"
    
    if emb_path.exists() and vocab_path.exists():
        embeddings = np.load(emb_path)
        with vocab_path.open('r', encoding='utf-8') as f:
            vocab_map = json.load(f)
        
        # Create reverse mapping
        token_to_idx = {token: int(idx) for token, idx in vocab_map.items()}
        return embeddings, token_to_idx
    
    return None, None

def deterministic_embedding(token: str, dim: int = 128) -> np.ndarray:
    """Deterministic pseudo-embedding via hashing."""
    rng = np.random.default_rng(abs(hash(token)) % (2**32))
    v = rng.normal(scale=1.0, size=(dim,))
    return v / (np.linalg.norm(v) + 1e-9)

class ConceptBuilder:
    """Concept Builder v1 implementation for testing."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'K': 64,           # Top-K neighbors to consider
            'lambda': 0.5,     # PPMI density threshold
            'tau_c': 0.35,     # Cosine similarity threshold
            'gamma': 0.3,      # Spread penalty exponent
            'kappa': 0.8,      # Residual boost scaling
            'm': 8,            # Minimum shell size
            'T': 1.6,          # Concept score threshold
            'tau_drop': 0.20,  # Drop threshold for refinement
            'delta': 0.15,     # Event horizon band width
            'alpha': 0.7,      # Embedding cohesion weight
            'beta': 0.6,       # Spread penalty weight
            'eta': 0.4,        # Residual boost weight
            'xi': 0.15         # Stability prior weight
        }
        
        self.embeddings = None
        self.token_to_idx = None
        self.corpus_stats = {}
        self.ppmi_fit = self._load_ppmi_fit()
    
    def load_embeddings(self):
        """Load embeddings for geometric calculations."""
        self.embeddings, self.token_to_idx = load_embeddings()
        if self.embeddings is None:
            print("No trained embeddings found, using deterministic embeddings")
    
    def get_embedding(self, token: str) -> np.ndarray:
        """Get embedding for a token."""
        if self.embeddings is not None and token in self.token_to_idx:
            idx = self.token_to_idx[token]
            return self.embeddings[idx]
        else:
            return deterministic_embedding(token)
    
    def compute_corpus_stats(self, concepts: Dict):
        """Compute corpus-wide statistics needed for concept building."""
        spreads = [c.get('spread', 0) for c in concepts.values()]
        masses = [c.get('mass', 0) for c in concepts.values()]
        ppmis = [c.get('avg_ppmi', 0) for c in concepts.values()]
        
        self.corpus_stats = {
            'spread_median': np.median(spreads) if spreads else 1.0,
            'mass_median': np.median(masses) if masses else 1.0,
            'ppmi_median': np.median(ppmis) if ppmis else 1.0
        }
        
        print(f"Corpus stats: spread_median={self.corpus_stats['spread_median']:.3f}")
    
    def candidate_neighborhood(self, token: str, concept_data: Dict) -> List[Tuple[str, float, np.ndarray]]:
        """
        Step 1: Build candidate neighborhood with density and geometry pruning.
        
        Returns: List of (neighbor_token, ppmi, embedding) tuples
        """
        neighbors = concept_data.get('neighbors', [])
        if not neighbors:
            return []
        
        # Get top-K neighbors
        top_neighbors = neighbors[:self.config['K']]
        
        # Get seed embedding
        seed_embedding = self.get_embedding(token)
        avg_ppmi = concept_data.get('avg_ppmi', 0)
        
        candidate_shell = []
        for neighbor in top_neighbors:
            neighbor_token = neighbor['id']
            ppmi = neighbor['ppmi']
            
            # Density pruning: keep items with PPMI >= R_w - lambda
            if ppmi < avg_ppmi - self.config['lambda']:
                continue
            
            # Geometry pruning: keep items with cos(e_w, e_ni) >= tau_c
            neighbor_embedding = self.get_embedding(neighbor_token)
            cosine_sim = np.dot(seed_embedding, neighbor_embedding)
            
            if cosine_sim >= self.config['tau_c']:
                candidate_shell.append((neighbor_token, ppmi, neighbor_embedding))
        
        return candidate_shell
    
    def compute_cohesion_scores(self, token: str, concept_data: Dict, shell: List[Tuple[str, float, np.ndarray]]) -> Dict[str, float]:
        """
        Step 2: Compute cohesion and tightness scores.
        """
        if not shell:
            return {
                'ppmi_cohesion': 0.0,
                'emb_cohesion': 0.0,
                'spread_penalty': 1.0,
                'residual_boost': 0.0,
                'horizon_safety': 0.0
            }
        
        avg_ppmi = concept_data.get('avg_ppmi', 0)
        spread = concept_data.get('spread', 1.0)
        phi = concept_data.get('phi', 1.0)
        
        # PPMI cohesion
        ppmi_cohesion = np.mean([ppmi / avg_ppmi for _, ppmi, _ in shell]) if avg_ppmi > 0 else 0.0
        
        # Embedding cohesion (mean cosine to center)
        seed_embedding = self.get_embedding(token)
        cosines = [np.dot(seed_embedding, emb) for _, _, emb in shell]
        emb_cohesion = np.mean(cosines)
        
        # Spread penalty
        spread_penalty = (spread / self.corpus_stats['spread_median']) ** self.config['gamma']
        
        # Residual boost (sigmoid of residual)
        expected_ppmi = self._expected_ppmi(concept_data.get('mass', 0))
        residual = avg_ppmi - expected_ppmi
        residual_boost = 1.0 / (1.0 + math.exp(-self.config['kappa'] * residual))
        
        # Horizon safety (indicator function)
        horizon_safety = 1.0 if phi >= 1.0 - self.config['delta'] else 0.0
        
        return {
            'ppmi_cohesion': ppmi_cohesion,
            'emb_cohesion': emb_cohesion,
            'spread_penalty': spread_penalty,
            'residual_boost': residual_boost,
            'horizon_safety': horizon_safety
        }
    
    def _load_ppmi_fit(self) -> Dict:
        """Load or create PPMI-mass fit coefficients."""
        fit_file = PROJECT_ROOT / "lab" / "concept" / "ppmi_mass_fit.json"
        
        if fit_file.exists():
            with fit_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create a reasonable quadratic fit based on typical PPMI-mass relationship
            # This is a placeholder - in production, this would be fitted from actual data
            fit = {
                "type": "quadratic",
                "a": -0.1,  # log(mass)^2 coefficient
                "b": 0.8,   # log(mass) coefficient  
                "c": 1.2    # constant term
            }
            
            # Save the fit for consistency
            with fit_file.open('w', encoding='utf-8') as f:
                json.dump(fit, f, indent=2)
            
            return fit
    
    def _expected_ppmi(self, mass: float) -> float:
        """Compute expected PPMI given mass using fitted curve."""
        logm = math.log1p(mass)
        fit = self.ppmi_fit
        
        if fit["type"] == "quadratic":
            return fit["a"] * logm * logm + fit["b"] * logm + fit["c"]
        else:
            # Fallback to simple approximation
            return 2.0 + 0.5 * logm
    
    def concept_score(self, scores: Dict[str, float], concept_data: Dict) -> float:
        """
        Step 3: Compute final concept score and gate.
        """
        stability = concept_data.get('stability', 0)
        
        score = (
            scores['ppmi_cohesion'] +  # semantic density
            self.config['alpha'] * scores['emb_cohesion'] +  # geometric agreement
            -self.config['beta'] * scores['spread_penalty'] +  # entropy penalty
            self.config['eta'] * scores['residual_boost'] +  # lifecycle
            self.config['xi'] * math.log(1 + stability)  # stability prior
        )
        
        return score
    
    def is_valid_concept(self, token: str, concept_data: Dict, shell: List[Tuple[str, float, np.ndarray]], score: float) -> bool:
        """Check if token qualifies as a concept."""
        scores = self.compute_cohesion_scores(token, concept_data, shell)
        
        return (
            scores['horizon_safety'] == 1.0 and  # not deep-collapsed
            len(shell) >= self.config['m'] and  # minimum shell size
            score >= self.config['T']  # score threshold
        )
    
    def build_concept(self, token: str, concept_data: Dict) -> Dict:
        """Build a single concept with all components."""
        # Step 1: Candidate neighborhood
        shell = self.candidate_neighborhood(token, concept_data)
        
        # Step 2: Cohesion scores
        scores = self.compute_cohesion_scores(token, concept_data, shell)
        
        # Step 3: Concept score
        concept_score = self.concept_score(scores, concept_data)
        
        # Step 4: Validation
        is_valid = self.is_valid_concept(token, concept_data, shell, concept_score)
        
        # Compute sanity metrics
        shell_cosines = [np.dot(self.get_embedding(token), emb) for _, _, emb in shell]
        purity = float(np.mean(shell_cosines)) if shell_cosines else 0.0
        margin = float(concept_data.get('avg_ppmi', 0) - concept_data.get('rs', 0))
        horizon_violation = float(concept_data.get('phi', 1.0) < (1 - self.config['delta']))
        
        # Build concept representation
        concept = {
            'token': token,
            'is_valid': is_valid,
            'score': concept_score,
            'shell_size': len(shell),
            'scores': scores,
            'shell_members': [(token, ppmi) for token, ppmi, _ in shell],
            'center': self.get_embedding(token).tolist(),
            'radius': self._compute_radius(token, shell),
            'mass': concept_data.get('mass', 0),
            'spread': concept_data.get('spread', 0),
            'stability': concept_data.get('stability', 0),
            'phi': concept_data.get('phi', 1.0),
            'purity': purity,
            'margin': margin,
            'horizon_violation': horizon_violation
        }
        
        return concept
    
    def _compute_radius(self, token: str, shell: List[Tuple[str, float, np.ndarray]]) -> float:
        """Compute concept radius as MAD of cosine distances."""
        if not shell:
            return 0.0
        
        seed_embedding = self.get_embedding(token)
        cosines = [np.dot(seed_embedding, emb) for _, _, emb in shell]
        
        # Compute median absolute deviation
        median_cosine = np.median(cosines)
        mad = np.median([abs(c - median_cosine) for c in cosines])
        
        return float(mad)
    
    def build_all_concepts(self, concepts: Dict) -> List[Dict]:
        """Build concepts for all tokens."""
        print("Building concepts for all tokens...")
        
        built_concepts = []
        for i, (token, concept_data) in enumerate(concepts.items()):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(concepts)} tokens...")
            
            concept = self.build_concept(token, concept_data)
            built_concepts.append(concept)
        
        return built_concepts

def main():
    """Main testing function."""
    print("=== Concept Builder v1 Testing ===")
    
    # Load data
    concepts = load_concept_data()
    print(f"Loaded {len(concepts)} concept records")
    
    # Initialize builder
    builder = ConceptBuilder()
    builder.load_embeddings()
    builder.compute_corpus_stats(concepts)
    
    # Build concepts
    built_concepts = builder.build_all_concepts(concepts)
    
    # Analyze results
    valid_concepts = [c for c in built_concepts if c['is_valid']]
    print(f"\nResults:")
    print(f"  Total tokens: {len(built_concepts)}")
    print(f"  Valid concepts: {len(valid_concepts)} ({len(valid_concepts)/len(built_concepts)*100:.1f}%)")
    
    if valid_concepts:
        scores = [c['score'] for c in valid_concepts]
        shell_sizes = [c['shell_size'] for c in valid_concepts]
        purities = [c['purity'] for c in valid_concepts]
        margins = [c['margin'] for c in valid_concepts]
        horizon_violations = [c['horizon_violation'] for c in valid_concepts]
        
        print(f"  Concept scores: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"  Shell sizes: {np.mean(shell_sizes):.1f} ± {np.std(shell_sizes):.1f}")
        print(f"  Purity: {np.mean(purities):.3f} ± {np.std(purities):.3f}")
        print(f"  Margin: {np.mean(margins):.3f} ± {np.std(margins):.3f}")
        print(f"  Horizon violations: {np.mean(horizon_violations)*100:.1f}%")
        
        # Show top concepts
        top_concepts = sorted(valid_concepts, key=lambda x: x['score'], reverse=True)[:10]
        print(f"\nTop 10 concepts by score:")
        for i, concept in enumerate(top_concepts, 1):
            print(f"  {i:2d}. {concept['token']:12s} score={concept['score']:.3f} shell={concept['shell_size']:2d} phi={concept['phi']:.3f}")
    
        # Save results (convert numpy types to native Python types for JSON)
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
        
        results = {
            'config': builder.config,
            'corpus_stats': builder.corpus_stats,
            'summary': {
                'total_tokens': len(built_concepts),
                'valid_concepts': len(valid_concepts),
                'coverage_percent': len(valid_concepts)/len(built_concepts)*100 if built_concepts else 0,
                'coverage': len(valid_concepts)/len(built_concepts) if built_concepts else 0,
                'purity': np.mean(purities) if valid_concepts else 0,
                'margin': np.mean(margins) if valid_concepts else 0,
                'horizon_violations': np.mean(horizon_violations) if valid_concepts else 0
            },
            'concepts': convert_numpy_types(built_concepts)
        }
    
    # Save results without the full concepts list to avoid JSON issues
    summary_results = {
        'config': builder.config,
        'corpus_stats': builder.corpus_stats,
        'summary': {
            'total_tokens': len(built_concepts),
            'valid_concepts': len(valid_concepts),
            'coverage_percent': len(valid_concepts)/len(built_concepts)*100 if built_concepts else 0,
            'coverage': len(valid_concepts)/len(built_concepts) if built_concepts else 0,
            'purity': float(np.mean(purities)) if valid_concepts else 0,
            'margin': float(np.mean(margins)) if valid_concepts else 0,
            'horizon_violations': float(np.mean(horizon_violations)) if valid_concepts else 0
        },
        'top_concepts': [{'token': c['token'], 'score': float(c['score']), 'shell_size': c['shell_size'], 'phi': float(c['phi'])} for c in sorted(valid_concepts, key=lambda x: x['score'], reverse=True)[:100]]
    }
    
    output_path = PROJECT_ROOT / "lab" / "concept" / "concept_builder_results.json"
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
