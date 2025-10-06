#!/usr/bin/env python3
"""
Concept Graph Test - Testing inter-concept relationships and clustering
Tests the theory that concepts can be connected into a graph structure where
edge weights represent conceptual relationships, enabling galaxy clustering.
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

def load_refined_concepts():
    """Load refined concepts from refinement test."""
    results_file = PROJECT_ROOT / "lab" / "concept" / "concept_refinement_results.json"
    
    if not results_file.exists():
        # Fallback to concept builder results
        results_file = PROJECT_ROOT / "lab" / "concept" / "concept_builder_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Concept results not found: {results_file}")
    
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
            'shell_members': [('dummy_neighbor', 1.0) for _ in range(concept_data['shell_size'])],  # Placeholder
            'center': [0.1] * 128  # Placeholder center
        }
        concepts.append(concept)
    
    return {'concepts': concepts}

def deterministic_embedding(token: str, dim: int = 128) -> np.ndarray:
    """Deterministic pseudo-embedding via hashing."""
    rng = np.random.default_rng(abs(hash(token)) % (2**32))
    v = rng.normal(scale=1.0, size=(dim,))
    return v / (np.linalg.norm(v) + 1e-9)

class ConceptGraphBuilder:
    """Builds and analyzes concept graphs with inter-concept relationships."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_edge_weight': 0.1,      # Minimum edge weight to include
            'max_edges_per_concept': 20,  # Maximum edges per concept
            'jaccard_threshold': 0.1,     # Minimum Jaccard similarity
            'cosine_threshold': 0.3       # Minimum cosine similarity
        }
    
    def compute_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_concept_edge_weight(self, concept1: Dict, concept2: Dict) -> float:
        """
        Compute edge weight between two concepts.
        
        Formula: E(c_i, c_j) = (cos(e_c_i, e_c_j) * Jaccard(S_c_i, S_c_j)) / (1 + Δρ)
        
        Args:
            concept1: First concept dictionary
            concept2: Second concept dictionary
            
        Returns:
            edge_weight: Weight of the edge between concepts
        """
        # Get concept centers
        center1 = np.array(concept1.get('center', [0] * 128))
        center2 = np.array(concept2.get('center', [0] * 128))
        
        # Compute cosine similarity between centers
        cosine_sim = np.dot(center1, center2)
        
        # Get concept member sets
        members1 = set(token for token, _ in concept1.get('shell_members', []))
        members2 = set(token for token, _ in concept2.get('shell_members', []))
        
        # Compute Jaccard similarity
        jaccard_sim = self.compute_jaccard_similarity(members1, members2)
        
        # Get radii
        radius1 = concept1.get('radius', 0.0)
        radius2 = concept2.get('radius', 0.0)
        radius_diff = abs(radius1 - radius2)
        
        # Compute edge weight
        numerator = cosine_sim * jaccard_sim
        denominator = 1.0 + radius_diff
        
        edge_weight = numerator / denominator if denominator > 0 else 0.0
        
        return edge_weight
    
    def build_concept_graph(self, concepts: List[Dict]) -> Tuple[List[Tuple[int, int, float]], Dict[int, Dict]]:
        """
        Build concept graph with edges between related concepts.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            edges: List of (concept_i, concept_j, weight) tuples
            concept_map: Dictionary mapping concept indices to concept data
        """
        print("Building concept graph...")
        
        # Create concept index mapping
        concept_map = {i: concept for i, concept in enumerate(concepts)}
        n_concepts = len(concepts)
        
        print(f"Computing edges for {n_concepts} concepts...")
        
        edges = []
        edge_count = 0
        
        # Compute edges between all pairs of concepts
        for i in range(n_concepts):
            concept1 = concepts[i]
            
            # Track edges for this concept
            concept_edges = []
            
            for j in range(i + 1, n_concepts):
                concept2 = concepts[j]
                
                # Compute edge weight
                edge_weight = self.compute_concept_edge_weight(concept1, concept2)
                
                # Apply thresholds
                if (edge_weight >= self.config['min_edge_weight'] and
                    edge_weight >= self.config['cosine_threshold'] * 0.5):  # Relaxed threshold for initial filtering
                    
                    concept_edges.append((j, edge_weight))
            
            # Sort by weight and keep top edges
            concept_edges.sort(key=lambda x: x[1], reverse=True)
            top_edges = concept_edges[:self.config['max_edges_per_concept']]
            
            # Add edges to graph
            for j, weight in top_edges:
                edges.append((i, j, weight))
                edge_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_concepts} concepts, {edge_count} edges so far...")
        
        # Normalize edge weights to [0,1]
        if edges:
            weights = [edge[2] for edge in edges]
            min_weight = min(weights)
            max_weight = max(weights)
            weight_range = max_weight - min_weight
            
            if weight_range > 0:
                normalized_edges = []
                for i, j, weight in edges:
                    normalized_weight = (weight - min_weight) / weight_range
                    normalized_edges.append((i, j, normalized_weight))
                edges = normalized_edges
        
        print(f"Built concept graph with {len(edges)} edges")
        
        return edges, concept_map
    
    def analyze_graph_structure(self, edges: List[Tuple[int, int, float]], concept_map: Dict[int, Dict]) -> Dict:
        """
        Analyze the structure of the concept graph.
        
        Args:
            edges: List of edges (concept_i, concept_j, weight)
            concept_map: Dictionary mapping concept indices to concept data
            
        Returns:
            analysis: Dictionary with graph analysis results
        """
        print("Analyzing graph structure...")
        
        n_concepts = len(concept_map)
        n_edges = len(edges)
        
        # Build adjacency lists
        adjacency = defaultdict(list)
        for i, j, weight in edges:
            adjacency[i].append((j, weight))
            adjacency[j].append((i, weight))  # Undirected graph
        
        # Compute degree statistics
        degrees = [len(adjacency[i]) for i in range(n_concepts)]
        isolated_nodes = sum(1 for d in degrees if d == 0)
        
        # Compute weight statistics
        weights = [weight for _, _, weight in edges]
        
        # Find connected components (simple DFS)
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            for neighbor, _ in adjacency[node]:
                dfs(neighbor, component)
        
        for i in range(n_concepts):
            if i not in visited:
                component = []
                dfs(i, component)
                if component:
                    components.append(component)
        
        # Analyze components
        component_sizes = [len(comp) for comp in components]
        
        # Find high-degree concepts (hubs)
        high_degree_threshold = np.percentile(degrees, 90) if degrees else 0
        hubs = [i for i, degree in enumerate(degrees) if degree >= high_degree_threshold]
        
        # Find high-weight edges
        high_weight_threshold = np.percentile(weights, 90) if weights else 0
        strong_edges = [(i, j, w) for i, j, w in edges if w >= high_weight_threshold]
        
        analysis = {
            'n_concepts': n_concepts,
            'n_edges': n_edges,
            'density': (2 * n_edges) / (n_concepts * (n_concepts - 1)) if n_concepts > 1 else 0,
            'degree_stats': {
                'mean': np.mean(degrees) if degrees else 0,
                'median': np.median(degrees) if degrees else 0,
                'std': np.std(degrees) if degrees else 0,
                'max': max(degrees) if degrees else 0
            },
            'isolated_nodes': isolated_nodes,
            'isolated_percentage': isolated_nodes / n_concepts * 100 if n_concepts > 0 else 0,
            'weight_stats': {
                'mean': np.mean(weights) if weights else 0,
                'median': np.median(weights) if weights else 0,
                'std': np.std(weights) if weights else 0,
                'max': max(weights) if weights else 0
            },
            'n_components': len(components),
            'component_sizes': component_sizes,
            'largest_component_size': max(component_sizes) if component_sizes else 0,
            'n_hubs': len(hubs),
            'n_strong_edges': len(strong_edges),
            'hubs': hubs[:20],  # Top 20 hubs
            'strong_edges': strong_edges[:20]  # Top 20 strong edges
        }
        
        return analysis
    
    def find_concept_clusters(self, edges: List[Tuple[int, int, float]], concept_map: Dict[int, Dict]) -> List[List[int]]:
        """
        Find concept clusters using simple connected components.
        
        Args:
            edges: List of edges (concept_i, concept_j, weight)
            concept_map: Dictionary mapping concept indices to concept data
            
        Returns:
            clusters: List of clusters, each containing concept indices
        """
        print("Finding concept clusters...")
        
        n_concepts = len(concept_map)
        
        # Build adjacency lists
        adjacency = defaultdict(list)
        for i, j, weight in edges:
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        # Find connected components
        visited = set()
        clusters = []
        
        def dfs(node, cluster):
            if node in visited:
                return
            visited.add(node)
            cluster.append(node)
            for neighbor in adjacency[node]:
                dfs(neighbor, cluster)
        
        for i in range(n_concepts):
            if i not in visited:
                cluster = []
                dfs(i, cluster)
                if cluster:
                    clusters.append(cluster)
        
        # Filter out single-node clusters
        multi_node_clusters = [cluster for cluster in clusters if len(cluster) > 1]
        
        print(f"Found {len(clusters)} total clusters, {len(multi_node_clusters)} multi-node clusters")
        
        return multi_node_clusters
    
    def analyze_clusters(self, clusters: List[List[int]], concept_map: Dict[int, Dict]) -> Dict:
        """
        Analyze the discovered concept clusters.
        
        Args:
            clusters: List of clusters, each containing concept indices
            concept_map: Dictionary mapping concept indices to concept data
            
        Returns:
            analysis: Dictionary with cluster analysis results
        """
        print("Analyzing clusters...")
        
        if not clusters:
            return {'n_clusters': 0}
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        # Analyze cluster composition
        cluster_analyses = []
        for i, cluster in enumerate(clusters):
            cluster_concepts = [concept_map[idx] for idx in cluster]
            
            # Get all unique tokens in cluster
            all_tokens = set()
            for concept in cluster_concepts:
                tokens = set(token for token, _ in concept.get('shell_members', []))
                all_tokens.update(tokens)
            
            # Compute cluster cohesion (average pairwise similarity)
            cohesion_scores = []
            for j in range(len(cluster_concepts)):
                for k in range(j + 1, len(cluster_concepts)):
                    concept1 = cluster_concepts[j]
                    concept2 = cluster_concepts[k]
                    
                    center1 = np.array(concept1.get('center', [0] * 128))
                    center2 = np.array(concept2.get('center', [0] * 128))
                    cosine_sim = np.dot(center1, center2)
                    cohesion_scores.append(cosine_sim)
            
            avg_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.0
            
            cluster_analysis = {
                'cluster_id': i,
                'size': len(cluster),
                'tokens': list(all_tokens)[:20],  # First 20 tokens
                'avg_cohesion': avg_cohesion,
                'concept_tokens': [concept['token'] for concept in cluster_concepts]
            }
            
            cluster_analyses.append(cluster_analysis)
        
        # Sort clusters by size
        cluster_analyses.sort(key=lambda x: x['size'], reverse=True)
        
        analysis = {
            'n_clusters': len(clusters),
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'clusters': cluster_analyses[:20]  # Top 20 clusters
        }
        
        return analysis

def main():
    """Main testing function."""
    print("=== Concept Graph Test ===")
    
    try:
        # Load refined concepts
        concept_data = load_refined_concepts()
        concepts = concept_data.get('refined_concepts', concept_data.get('concepts', []))
        
        # Filter to valid concepts only
        valid_concepts = [c for c in concepts if c.get('is_valid', False)]
        
        print(f"Loaded {len(valid_concepts)} valid concepts")
        
        if not valid_concepts:
            print("No valid concepts found for graph building")
            return
        
        # Initialize graph builder
        graph_builder = ConceptGraphBuilder()
        
        # Build concept graph
        edges, concept_map = graph_builder.build_concept_graph(valid_concepts)
        
        # Analyze graph structure
        graph_analysis = graph_builder.analyze_graph_structure(edges, concept_map)
        
        # Find clusters
        clusters = graph_builder.find_concept_clusters(edges, concept_map)
        
        # Analyze clusters
        cluster_analysis = graph_builder.analyze_clusters(clusters, concept_map)
        
        # Print results
        print(f"\n=== Graph Analysis Results ===")
        print(f"Concepts: {graph_analysis['n_concepts']}")
        print(f"Edges: {graph_analysis['n_edges']}")
        print(f"Density: {graph_analysis['density']:.4f}")
        print(f"Average degree: {graph_analysis['degree_stats']['mean']:.2f}")
        print(f"Average edge weight: {graph_analysis['weight_stats']['mean']:.3f}")
        print(f"Connected components: {graph_analysis['n_components']}")
        print(f"Largest component: {graph_analysis['largest_component_size']} concepts")
        print(f"Hubs: {graph_analysis['n_hubs']}")
        
        print(f"\n=== Cluster Analysis Results ===")
        print(f"Multi-node clusters: {cluster_analysis['n_clusters']}")
        print(f"Average cluster size: {cluster_analysis['avg_cluster_size']:.1f}")
        print(f"Largest cluster: {cluster_analysis['max_cluster_size']} concepts")
        
        if cluster_analysis['clusters']:
            print(f"\nTop 5 clusters:")
            for i, cluster in enumerate(cluster_analysis['clusters'][:5], 1):
                print(f"  {i}. Size: {cluster['size']}, Cohesion: {cluster['avg_cohesion']:.3f}")
                print(f"     Concepts: {', '.join(cluster['concept_tokens'][:5])}")
                if len(cluster['concept_tokens']) > 5:
                    print(f"     ... and {len(cluster['concept_tokens']) - 5} more")
        
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
        
        results = {
            'config': graph_builder.config,
            'graph_analysis': graph_analysis,
            'cluster_analysis': cluster_analysis,
            'edges': edges[:1000],  # Save first 1000 edges to avoid huge files
            'clusters': clusters
        }
        
        output_path = PROJECT_ROOT / "lab" / "concept" / "concept_graph_results.json"
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during graph testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
