#!/usr/bin/env python3
"""
Test script to verify randomization behavior when w_interaction = 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ProXeek_Optimization import ProXeekOptimizer

def test_randomization_with_zero_interaction():
    """Test that substrate assignments are randomized when w_interaction = 0"""
    print("="*60)
    print("TESTING RANDOMIZATION WITH w_interaction = 0")
    print("="*60)
    
    # Create optimizer instance
    optimizer = ProXeekOptimizer()
    
    # Load data
    if not optimizer.load_data():
        print("Failed to load data")
        return
    
    # Set interaction weight to 0
    optimizer.w_interaction = 0.0
    print(f"Set w_interaction = {optimizer.w_interaction}")
    
    # Find substrate objects
    substrate_objects = []
    for i, obj in enumerate(optimizer.virtual_objects):
        if obj.involvement_type == "substrate":
            substrate_objects.append((i, obj.name))
    
    print(f"Found {len(substrate_objects)} substrate objects:")
    for idx, name in substrate_objects:
        print(f"  [{idx}] {name}")
    
    # Run optimization multiple times and collect substrate assignments
    assignments_results = []
    num_runs = 5
    
    for run in range(num_runs):
        print(f"\n--- RUN {run + 1} ---")
        assignment = optimizer.optimize()
        
        if assignment:
            run_results = {}
            for substrate_idx, substrate_name in substrate_objects:
                physical_idx = assignment.virtual_to_physical[substrate_idx]
                physical_name = optimizer.physical_objects[physical_idx].name
                run_results[substrate_name] = physical_name
                print(f"  {substrate_name} -> {physical_name}")
            assignments_results.append(run_results)
        else:
            print("  Optimization failed")
    
    # Analyze results
    print(f"\n{'='*60}")
    print("RANDOMIZATION ANALYSIS")
    print(f"{'='*60}")
    
    if len(assignments_results) > 1:
        # Check if any substrate assignments differ between runs
        randomization_detected = False
        for substrate_name in assignments_results[0].keys():
            assignments_for_substrate = [result[substrate_name] for result in assignments_results]
            unique_assignments = set(assignments_for_substrate)
            
            print(f"\n{substrate_name} assignments across runs:")
            for i, assignment in enumerate(assignments_for_substrate):
                print(f"  Run {i+1}: {assignment}")
            
            if len(unique_assignments) > 1:
                print(f"  ✓ RANDOMIZATION DETECTED - {len(unique_assignments)} different assignments")
                randomization_detected = True
            else:
                print(f"  - Same assignment in all runs")
        
        if randomization_detected:
            print(f"\n✓ SUCCESS: Randomization is working for substrate objects when w_interaction = 0")
        else:
            print(f"\n⚠ INFO: All runs produced same assignments (this can happen by chance or if spatial/realism constraints are very restrictive)")
    else:
        print("Need at least 2 successful runs to test randomization")

if __name__ == "__main__":
    test_randomization_with_zero_interaction()

