#!/usr/bin/env python3
"""
Polysemy Split Test - Testing multi-modal concept detection
Tests the theory that some concepts should be split into sub-concepts when they have
multi-modal distributions in embedding space (e.g., "bank" -> "bank_river" + "bank_finance").
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

def deterministic_embedding(token: str, dim: int = 128) -> np.ndarray:
    """Deterministic pseudo-embedding via hashing."""
    rng = np.random.default_rng(abs(hash(token)) % (2**32))
    v = rng.normal(scale=1.0, size=(dim,))
    return v / (np.linalg.norm(v) + 1e-9)

class PolysemySplitter:
    """Implements polysemy detection and splitting using spherical k-means."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'silhouette_threshold': 0.15,  # Minimum silhouette score for split
            'max_splits': 3,               # Maximum number of splits per concept
            'min_cluster_size': 3          # Minimum size for a valid cluster
        }
    
    def spherical_kmeans(self, embeddings: np.ndarray, k: int = 2, max_iters: int = 20) -> Tuple[np.ndarray, List[int]]:
        """
        Spherical k-means clustering on unit sphere.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            k: Number of clusters
            max_iters: Maximum iterations
            
        Returns:
            centroids: Array of shape (k, n_features)
            labels: List of cluster assignments
        """
        n_samples, n_features = embeddings.shape
        
        # Initialize centroids randomly
        np.random.seed(42)
        centroids = np.random.randn(k, n_features)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
        for iteration in range(max_iters):
            # Assign points to closest centroid (cosine similarity)
            similarities = np.dot(embeddings, centroids.T)  # (n_samples, k)
            labels = np.argmax(similarities, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = embeddings[labels == i]
                if len(cluster_points) > 0:
                    # Compute mean and normalize
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                    new_centroids[i] = new_centroids[i] / (np.linalg.norm(new_centroids[i]) + 1e-9)
                else:
                    new_centroids[i] = centroids[i]  # Keep old centroid if no points
            
            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
                
            centroids = new_centroids
        
        return centroids, labels.tolist()
    
    def compute_silhouette_score(self, embeddings: np.ndarray, labels: List[int]) -> float:
        """
        Compute silhouette score for clustering quality.
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            labels: List of cluster assignments
            
        Returns:
            silhouette_score: Average silhouette score
        """
        n_samples = len(embeddings)
        if n_samples <= 1:
            return 0.0
        
        unique_labels = list(set(labels))
        if len(unique_labels) <= 1:
            return 0.0
        
        silhouette_scores = []
        
        for i in range(n_samples):
            # Compute intra-cluster distance (a_i)
            cluster_i = labels[i]
            cluster_points = embeddings[labels == cluster_i]
            
            if len(cluster_points) <= 1:
                a_i = 0.0
            else:
                distances = [1 - np.dot(embeddings[i], point) for point in cluster_points if not np.array_equal(embeddings[i], point)]
                a_i = np.mean(distances) if distances else 0.0
            
            # Compute nearest inter-cluster distance (b_i)
            other_clusters = [l for l in unique_labels if l != cluster_i]
            if not other_clusters:
                b_i = 0.0
            else:
                min_inter_distances = []
                for other_cluster in other_clusters:
                    other_points = embeddings[labels == other_cluster]
                    distances = [1 - np.dot(embeddings[i], point) for point in other_points]
                    if distances:
                        min_inter_distances.append(np.mean(distances))
                
                b_i = min(min_inter_distances) if min_inter_distances else 0.0
            
            # Silhouette score for this point
            if max(a_i, b_i) == 0:
                s_i = 0.0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)
            
            silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores)
    
    def should_split_concept(self, concept: Dict) -> Tuple[bool, float, List[List[int]]]:
        """
        Determine if a concept should be split based on its shell members.
        
        Args:
            concept: Concept dictionary with shell_members
            
        Returns:
            should_split: Boolean indicating if split is recommended
            silhouette_score: Quality score of the split
            clusters: List of cluster assignments for each split
        """
        shell_members = concept.get('shell_members', [])
        if len(shell_members) < 6:  # Need minimum members for meaningful split
            return False, 0.0, []
        
        # Get embeddings for shell members
        embeddings = []
        for token, _ in shell_members:
            emb = deterministic_embedding(token)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Try k=2 split
        centroids, labels = self.spherical_kmeans(embeddings, k=2)
        silhouette_score = self.compute_silhouette_score(embeddings, labels)
        
        # Check cluster balance for k=2
        sizes = np.bincount(labels)
        if len(sizes) == 2 and sizes.min() / sizes.max() < 0.2:
            # Unbalanced clusters, try k=3
            centroids_3, labels_3 = self.spherical_kmeans(embeddings, k=3)
            silhouette_score_3 = self.compute_silhouette_score(embeddings, labels_3)
            
            # Check if k=3 is better
            if (silhouette_score_3 > silhouette_score + 0.03 and 
                silhouette_score_3 >= self.config['silhouette_threshold']):
                # Use k=3 split
                labels = labels_3
                silhouette_score = silhouette_score_3
                k = 3
            else:
                # Reject k=2 due to imbalance, keep single concept
                return False, silhouette_score, []
        else:
            k = 2
        
        # Check if split is worthwhile
        should_split = (
            silhouette_score >= self.config['silhouette_threshold'] and
            len(set(labels)) == k and  # All clusters are non-empty
            all(labels.count(label) >= self.config['min_cluster_size'] for label in set(labels))
        )
        
        # Organize clusters
        clusters = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        return should_split, silhouette_score, clusters
    
    def split_concept(self, concept: Dict) -> List[Dict]:
        """
        Split a concept into sub-concepts.
        
        Args:
            concept: Original concept dictionary
            
        Returns:
            sub_concepts: List of split concept dictionaries
        """
        should_split, silhouette_score, clusters = self.should_split_concept(concept)
        
        if not should_split:
            return [concept]  # Return original concept unchanged
        
        sub_concepts = []
        shell_members = concept.get('shell_members', [])
        
        for cluster_idx, cluster_members in enumerate(clusters):
            if len(cluster_members) < self.config['min_cluster_size']:
                continue
            
            # Create sub-concept
            sub_concept = concept.copy()
            sub_concept['token'] = f"{concept['token']}#{cluster_idx + 1}"
            sub_concept['is_split'] = True
            sub_concept['original_token'] = concept['token']
            sub_concept['cluster_idx'] = cluster_idx
            sub_concept['silhouette_score'] = silhouette_score
            sub_concept['k_value'] = len(clusters)  # Store k value for analysis
            
            # Update shell members for this cluster
            sub_shell_members = [shell_members[i] for i in cluster_members]
            sub_concept['shell_members'] = sub_shell_members
            sub_concept['shell_size'] = len(sub_shell_members)
            
            # Recompute center and radius for this cluster
            cluster_embeddings = []
            for token, _ in sub_shell_members:
                emb = deterministic_embedding(token)
                cluster_embeddings.append(emb)
            
            if cluster_embeddings:
                # New center is mean of cluster embeddings
                new_center = np.mean(cluster_embeddings, axis=0)
                new_center = new_center / (np.linalg.norm(new_center) + 1e-9)
                sub_concept['center'] = new_center.tolist()
                
                # New radius is MAD of cosine distances within cluster
                cosines = [np.dot(new_center, emb) for emb in cluster_embeddings]
                median_cosine = np.median(cosines)
                mad = np.median([abs(c - median_cosine) for c in cosines])
                sub_concept['radius'] = float(mad)
            
            sub_concepts.append(sub_concept)
        
        return sub_concepts
    
    def test_polysemy_detection(self, concepts: List[Dict]) -> Dict:
        """
        Test polysemy detection on all valid concepts.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            results: Dictionary with test results
        """
        print("Testing polysemy detection...")
        
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        print(f"Testing {len(valid_concepts)} valid concepts for polysemy...")
        
        split_candidates = []
        split_results = []
        
        for concept in valid_concepts:
            should_split, silhouette_score, clusters = self.should_split_concept(concept)
            
            if should_split:
                split_candidates.append({
                    'token': concept['token'],
                    'silhouette_score': silhouette_score,
                    'shell_size': concept['shell_size'],
                    'clusters': clusters
                })
                
                # Actually perform the split
                sub_concepts = self.split_concept(concept)
                split_results.extend(sub_concepts)
        
        # Analyze results
        results = {
            'total_concepts_tested': len(valid_concepts),
            'split_candidates': len(split_candidates),
            'split_rate': len(split_candidates) / len(valid_concepts) if valid_concepts else 0,
            'total_sub_concepts': len(split_results),
            'candidates': split_candidates,
            'sub_concepts': split_results
        }
        
        return results

def analyze_polysemy_patterns(results: Dict):
    """Analyze patterns in polysemy detection results."""
    print("\n=== Polysemy Analysis ===")
    
    candidates = results.get('candidates', [])
    if not candidates:
        print("No polysemy candidates found.")
        return
    
    print(f"Found {len(candidates)} polysemy candidates:")
    
    # Sort by silhouette score
    candidates.sort(key=lambda x: x['silhouette_score'], reverse=True)
    
    print(f"\nTop 10 polysemy candidates:")
    for i, candidate in enumerate(candidates[:10], 1):
        print(f"  {i:2d}. {candidate['token']:15s} silhouette={candidate['silhouette_score']:.3f} shell={candidate['shell_size']:2d}")
    
    # Analyze by shell size
    shell_sizes = [c['shell_size'] for c in candidates]
    print(f"\nShell size distribution for polysemy candidates:")
    print(f"  Mean: {np.mean(shell_sizes):.1f}")
    print(f"  Median: {np.median(shell_sizes):.1f}")
    print(f"  Range: {min(shell_sizes)} - {max(shell_sizes)}")
    
    # Analyze by silhouette score
    silhouette_scores = [c['silhouette_score'] for c in candidates]
    print(f"\nSilhouette score distribution:")
    print(f"  Mean: {np.mean(silhouette_scores):.3f}")
    print(f"  Median: {np.median(silhouette_scores):.3f}")
    print(f"  Range: {min(silhouette_scores):.3f} - {max(silhouette_scores):.3f}")

def main():
    """Main testing function."""
    print("=== Polysemy Split Test ===")
    
    try:
        # Load concept builder results
        concept_data = load_concept_builder_results()
        concepts = concept_data.get('concepts', [])
        
        print(f"Loaded {len(concepts)} concepts from concept builder")
        
        # Initialize polysemy splitter
        splitter = PolysemySplitter()
        
        # Test polysemy detection
        results = splitter.test_polysemy_detection(concepts)
        
        # Analyze results
        analyze_polysemy_patterns(results)
        
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
        
        output_path = PROJECT_ROOT / "lab" / "concept" / "polysemy_split_results.json"
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during polysemy testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
