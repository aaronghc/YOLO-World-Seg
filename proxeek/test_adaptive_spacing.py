#!/usr/bin/env python3
"""
Test script for adaptive PIN_GRID_SPACING feature

This script demonstrates the adaptive spacing calculation by loading
optimization data and showing the calculated spacing.

The adaptive spacing is calculated as the average of each object's
minimum distance to its closest neighbor (excludes banned objects).
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ProXeek_Optimization import ProXeekOptimizer, ENABLE_ADAPTIVE_PIN_GRID_SPACING, PIN_GRID_SPACING

def test_adaptive_spacing():
    """Test the adaptive spacing calculation"""
    print("="*60)
    print("Testing Adaptive PIN_GRID_SPACING")
    print("="*60)
    print()
    
    # Initialize optimizer
    print("Initializing ProXeekOptimizer...")
    optimizer = ProXeekOptimizer()
    
    # Load data
    print("Loading optimization data...")
    if not optimizer.load_data():
        print("❌ Failed to load data")
        return False
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Physical objects: {len(optimizer.physical_objects)}")
    print(f"  Virtual objects: {len(optimizer.virtual_objects)}")
    
    # Test with no banned objects
    print("\n" + "="*60)
    print("Test 1: No Banned Objects")
    print("="*60)
    
    optimizer.set_banned_physical_objects([])
    avg_distance_no_ban = optimizer.calculate_average_physical_distance()
    
    # Test with some banned objects
    print("\n" + "="*60)
    print("Test 2: With Banned Objects")
    print("="*60)
    
    # Ban first 2 objects as example
    banned_pairs = []
    if len(optimizer.physical_objects) >= 2:
        banned_pairs = [
            (optimizer.physical_objects[0].image_id, optimizer.physical_objects[0].object_id),
            (optimizer.physical_objects[1].image_id, optimizer.physical_objects[1].object_id),
        ]
    
    optimizer.set_banned_physical_objects(banned_pairs)
    avg_distance_with_ban = optimizer.calculate_average_physical_distance()
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"  Default PIN_GRID_SPACING (constant): {PIN_GRID_SPACING}m")
    print(f"  Adaptive spacing (no bans):          {avg_distance_no_ban:.3f}m")
    print(f"  Adaptive spacing (with bans):        {avg_distance_with_ban:.3f}m")
    print(f"  Difference from default:             {abs(avg_distance_no_ban - PIN_GRID_SPACING):.3f}m")
    
    if ENABLE_ADAPTIVE_PIN_GRID_SPACING:
        print(f"\n✓ Adaptive mode is ENABLED")
        print(f"  Optimization will use: {avg_distance_no_ban:.3f}m spacing")
    else:
        print(f"\n✓ Adaptive mode is DISABLED")
        print(f"  Optimization will use: {PIN_GRID_SPACING}m spacing")
    
    # Test pin grid generation with adaptive spacing
    print("\n" + "="*60)
    print("Test 3: Pin Grid Generation")
    print("="*60)
    
    optimizer.set_banned_physical_objects([])
    
    # Generate with default spacing
    pin_points_default = optimizer.generate_pin_grid(spacing=PIN_GRID_SPACING)
    
    # Generate with adaptive spacing
    pin_points_adaptive = optimizer.generate_pin_grid(spacing=avg_distance_no_ban)
    
    print(f"\n  Pin points with default spacing ({PIN_GRID_SPACING}m): {len(pin_points_default)}")
    print(f"  Pin points with adaptive spacing ({avg_distance_no_ban:.3f}m): {len(pin_points_adaptive)}")
    print(f"  Difference: {len(pin_points_default) - len(pin_points_adaptive)} points")
    
    return True


if __name__ == "__main__":
    print("\nProXeek Adaptive Spacing Test\n")
    
    success = test_adaptive_spacing()
    
    if success:
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("❌ Tests failed")
        print("="*60 + "\n")

