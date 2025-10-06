#!/usr/bin/env python3
"""
Run All Concept Tests - Master script to run all concept building tests in sequence
This script orchestrates the complete concept building test suite:
1. Concept Builder v1 test
2. Polysemy Split test  
3. Concept Refinement test
4. Concept Graph test
5. Sanity Checks test
"""

import subprocess
import sys
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent

def run_test(script_name: str, description: str) -> bool:
    """
    Run a single test script and return success status.
    
    Args:
        script_name: Name of the test script to run
        description: Human-readable description of the test
        
    Returns:
        success: Boolean indicating if test passed
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    script_path = PROJECT_ROOT / "lab" / "concept" / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        start_time = time.time()
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS: {description} completed in {duration:.1f}s")
            if result.stdout:
                print("Output:")
                print(result.stdout[-1000:])  # Last 1000 characters
            return True
        else:
            print(f"FAILED: {description} failed after {duration:.1f}s")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-1000:])  # Last 1000 characters
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {description} failed with exception: {e}")
        return False

def check_prerequisites() -> bool:
    """Check if prerequisites are met before running tests."""
    print("Checking prerequisites...")
    
    # Check if concept data exists
    concepts_file = PROJECT_ROOT / "concept_ppmi_results" / "concepts_ppmi.json"
    if not concepts_file.exists():
        print(f"ERROR: Concept data not found: {concepts_file}")
        print("Please run the concept PPMI pipeline first:")
        print("  python service/concept_ppmi_pipeline.py")
        return False
    
    # Check if embeddings exist (optional)
    embeddings_file = PROJECT_ROOT / "models" / "embeddings.npy"
    if not embeddings_file.exists():
        print("WARNING: Trained embeddings not found, will use deterministic embeddings")
    
    print("Prerequisites check passed")
    return True

def generate_summary_report(results: dict):
    """Generate a summary report of all test results."""
    print(f"\n{'='*60}")
    print("CONCEPT BUILDING TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\nTest Results:")
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
    
    # Overall assessment
    if passed_tests == total_tests:
        print(f"\nALL TESTS PASSED! Concept building theories are working correctly.")
    elif passed_tests >= total_tests * 0.8:
        print(f"\nMOSTLY SUCCESSFUL: {passed_tests}/{total_tests} tests passed. Minor issues to address.")
    elif passed_tests >= total_tests * 0.6:
        print(f"\nPARTIAL SUCCESS: {passed_tests}/{total_tests} tests passed. Several issues need attention.")
    else:
        print(f"\nSIGNIFICANT ISSUES: Only {passed_tests}/{total_tests} tests passed. Major problems detected.")
    
    # Save summary to file
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests/total_tests*100,
        'results': results
    }
    
    summary_path = PROJECT_ROOT / "lab" / "concept" / "test_suite_summary.json"
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")

def generate_dashboard(results: dict):
    """Generate a dashboard JSON with key metrics from all tests."""
    print("\nGenerating dashboard...")
    
    dashboard = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_results": results,
        "metrics": {}
    }
    
    # Try to load metrics from individual test results
    try:
        # Load concept builder results
        cb_results_path = PROJECT_ROOT / "lab" / "concept" / "concept_builder_results.json"
        if cb_results_path.exists():
            with cb_results_path.open('r', encoding='utf-8') as f:
                cb_data = json.load(f)
                dashboard["metrics"]["coverage"] = cb_data.get("summary", {}).get("coverage", 0)
                dashboard["metrics"]["purity"] = cb_data.get("summary", {}).get("purity", 0)
                dashboard["metrics"]["margin"] = cb_data.get("summary", {}).get("margin", 0)
                dashboard["metrics"]["horizon_violations"] = cb_data.get("summary", {}).get("horizon_violations", 0)
        
        # Load sanity check results
        sc_results_path = PROJECT_ROOT / "lab" / "concept" / "sanity_checks_results.json"
        if sc_results_path.exists():
            with sc_results_path.open('r', encoding='utf-8') as f:
                sc_data = json.load(f)
                dashboard["metrics"]["sanity_score"] = sc_data.get("overall_score", 0)
        
        # Load graph results
        graph_results_path = PROJECT_ROOT / "lab" / "concept" / "concept_graph_results.json"
        if graph_results_path.exists():
            with graph_results_path.open('r', encoding='utf-8') as f:
                graph_data = json.load(f)
                analysis = graph_data.get("analysis", {})
                dashboard["metrics"]["avg_degree"] = analysis.get("degree_stats", {}).get("mean", 0)
                dashboard["metrics"]["avg_edge_weight"] = analysis.get("weight_stats", {}).get("mean", 0)
                dashboard["metrics"]["isolated_percentage"] = analysis.get("isolated_percentage", 0)
        
    except Exception as e:
        print(f"Warning: Could not load all metrics: {e}")
    
    # Save dashboard
    dashboard_path = PROJECT_ROOT / "lab" / "concept" / "dashboard.json"
    with dashboard_path.open('w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"Dashboard saved to: {dashboard_path}")

def main():
    """Main function to run all concept building tests."""
    print("Concept Building Test Suite")
    print("Testing CB-v1 theories: crisp gates, scores, and refinement")
    
    # Check prerequisites
    if not check_prerequisites():
        print("Prerequisites not met. Exiting.")
        return 1
    
    # Define test sequence
    tests = [
        ("concept_builder_test.py", "Concept Builder v1 - Core concept building with crisp gates"),
        ("polysemy_split_test.py", "Polysemy Split - Multi-modal concept detection"),
        ("concept_refinement_test.py", "Concept Refinement - Boundary and membership refinement"),
        ("concept_graph_test.py", "Concept Graph - Inter-concept relationships and clustering"),
        ("sanity_checks_test.py", "Sanity Checks - Quality metrics and validation")
    ]
    
    # Run tests
    results = {}
    start_time = time.time()
    
    for script_name, description in tests:
        success = run_test(script_name, description)
        results[description] = success
        
        # If a critical test fails, we might want to stop
        if not success and "Concept Builder v1" in description:
            print(f"\nCritical test failed. Continuing with remaining tests...")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Generate summary
    generate_summary_report(results)
    
    # Generate dashboard
    generate_dashboard(results)
    
    print(f"\nTotal execution time: {total_duration:.1f} seconds")
    
    # Return appropriate exit code
    passed_tests = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    if passed_tests == total_tests:
        return 0  # All tests passed
    elif passed_tests >= total_tests * 0.6:
        return 1  # Most tests passed, minor issues
    else:
        return 2  # Significant issues

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
