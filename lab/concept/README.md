# Concept Building Test Suite

This directory contains tests for the new Concept Builder v1 (CB-v1) theories, separate from the existing Schwarzschild physics tests in the `schwarzschild/` folder.

## Overview

The Concept Builder v1 implements a new approach to building crisp, well-defined concepts from PPMI neighborhoods with:

- **Crisp Gates**: Deterministic criteria for concept acceptance
- **Quality Scores**: Multi-dimensional scoring for concept quality
- **Refinement**: Single-pass boundary and membership optimization
- **Polysemy Detection**: Automatic splitting of multi-modal concepts
- **Graph Structure**: Inter-concept relationships and clustering

## Test Files

### 1. `concept_builder_test.py`
**Core concept building with crisp gates**

Tests the fundamental concept building process:
- Candidate neighborhood selection with density and geometry pruning
- Cohesion and tightness score computation
- Concept score aggregation with configurable weights
- Validation gates (horizon safety, minimum shell size, score threshold)

**Key Metrics:**
- Coverage: % of tokens promoted to concepts (target: 10-25%)
- Concept scores and shell sizes
- Phase distribution analysis

### 2. `polysemy_split_test.py`
**Multi-modal concept detection**

Tests automatic detection and splitting of polysemous concepts:
- Spherical k-means clustering on embedding space
- Silhouette score evaluation for split quality
- Automatic sub-concept generation (e.g., "bank" → "bank#1", "bank#2")

**Key Metrics:**
- Split rate: % of concepts that benefit from splitting
- Silhouette score distribution
- Cluster quality analysis

### 3. `concept_refinement_test.py`
**Boundary and membership refinement**

Tests single-pass refinement of concept boundaries:
- Pulling in beneficial neighbors that improve cohesion
- Pushing out weak members below thresholds
- Radius expansion limits to prevent concept drift

**Key Metrics:**
- Size changes (additions/removals)
- Cohesion improvements
- Refinement effectiveness

### 4. `concept_graph_test.py`
**Inter-concept relationships and clustering**

Tests building a graph structure of concept relationships:
- Edge weight computation using cosine similarity and Jaccard overlap
- Connected component analysis
- Concept clustering and galaxy formation

**Key Metrics:**
- Graph density and connectivity
- Cluster sizes and cohesion
- Hub identification

### 5. `sanity_checks_test.py`
**Quality metrics and validation**

Tests that concept building meets quality standards:
- **Coverage**: 10-25% of tokens become concepts
- **Purity**: Mean cosine ≥ 0.45 to concept centers
- **Margin**: Positive R_w - R_s(w) margin ≥ 0.2
- **Stability Rank**: Concept median stability > corpus median
- **Event Horizon Avoidance**: <5% of concepts with φ < 1-δ

### 6. `run_all_concept_tests.py`
**Master test runner**

Orchestrates the complete test suite:
- Runs all tests in sequence
- Checks prerequisites
- Generates comprehensive summary report
- Provides overall success/failure assessment

## Configuration

Each test uses configurable parameters with sensible defaults:

```python
config = {
    'K': 64,                    # Top-K neighbors to consider
    'lambda': 0.5,              # PPMI density threshold
    'tau_c': 0.35,              # Cosine similarity threshold
    'gamma': 0.3,               # Spread penalty exponent
    'kappa': 0.8,               # Residual boost scaling
    'm': 8,                     # Minimum shell size
    'T': 1.6,                   # Concept score threshold
    'tau_drop': 0.20,           # Drop threshold for refinement
    'delta': 0.15,              # Event horizon band width
    'alpha': 0.7,               # Embedding cohesion weight
    'beta': 0.6,                # Spread penalty weight
    'eta': 0.4,                 # Residual boost weight
    'xi': 0.15                  # Stability prior weight
}
```

## Running the Tests

### Prerequisites
1. Run the concept PPMI pipeline first:
   ```bash
   python service/concept_ppmi_pipeline.py
   ```

2. (Optional) Train embeddings:
   ```bash
   python service/train_concept_net.py
   ```

### Run All Tests
```bash
python lab/concept/run_all_concept_tests.py
```

### Run Individual Tests
```bash
python lab/concept/concept_builder_test.py
python lab/concept/polysemy_split_test.py
python lab/concept/concept_refinement_test.py
python lab/concept/concept_graph_test.py
python lab/concept/sanity_checks_test.py
```

## Expected Results

### Concept Builder v1
- **Coverage**: 10-25% of tokens become valid concepts
- **Quality**: High-scoring concepts with meaningful semantic coherence
- **Distribution**: Reasonable spread across different concept types

### Polysemy Detection
- **Split Rate**: 5-15% of concepts benefit from splitting
- **Quality**: Silhouette scores ≥ 0.15 for successful splits
- **Examples**: Clear cases like "bank", "set", "run" should be detected

### Refinement
- **Improvement**: Net positive changes in concept quality
- **Stability**: Most concepts remain stable or improve
- **Boundaries**: Clearer concept boundaries after refinement

### Graph Structure
- **Connectivity**: Reasonable graph density (0.01-0.05)
- **Clusters**: Meaningful concept clusters with semantic coherence
- **Hubs**: Identification of central concepts

### Sanity Checks
- **Overall Score**: ≥80% for excellent, ≥60% for good
- **All Metrics**: Should meet or exceed target thresholds
- **Quality**: Concepts should be semantically coherent and well-bounded

## Theory Validation

These tests validate the core CB-v1 theories:

1. **Crisp Gates Work**: Deterministic criteria successfully identify high-quality concepts
2. **Multi-Dimensional Scoring**: Combined metrics capture concept quality better than single measures
3. **Refinement Improves Quality**: Single-pass refinement enhances concept boundaries
4. **Polysemy is Detectable**: Multi-modal concepts can be automatically identified and split
5. **Graph Structure Emerges**: Meaningful inter-concept relationships form naturally
6. **Quality Standards Met**: The approach produces concepts meeting linguistic quality criteria

## Future Extensions

- **Domain Adaptation**: Re-scale parameters per domain (news, fiction, tech)
- **Learned Weights**: Replace hand-tuned weights with logistic regression
- **Concept Signatures**: Add human-readable concept names via TF-IDF
- **Incremental Updates**: Support for adding new concepts without full rebuild
- **Multi-Modal Integration**: Extend to visual, spatial, and sensory data

## Troubleshooting

### Common Issues

1. **No valid concepts found**: Check that concept PPMI pipeline ran successfully
2. **Low coverage**: Adjust thresholds (T, m) or increase K
3. **Poor purity**: Increase tau_c or adjust embedding quality
4. **High event horizon violations**: Check Schwarzschild parameters
5. **Memory issues**: Reduce max_vocab or process in batches

### Debug Mode

Add debug prints to individual tests to see intermediate results:
```python
print(f"Debug: {variable_name} = {value}")
```

### Parameter Tuning

If results don't meet expectations, try adjusting:
- **K**: Increase for more neighbors, decrease for speed
- **T**: Lower for more concepts, raise for higher quality
- **tau_c**: Raise for stricter geometric filtering
- **alpha, beta, eta, xi**: Adjust scoring weights based on priorities
