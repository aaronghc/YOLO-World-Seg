#!/usr/bin/env python3
"""
ProXeek Global Optimization Module

This module implements the global optimization stage for haptic proxy assignment,
implementing the Multi-Objective Loss Function from the backend specification.
"""

import os
import sys
import json
import math
import uuid
import numpy as np
import pandas as pd
import itertools
import random
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
import time
# Set matplotlib to use non-interactive backend for multiprocessing compatibility
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

# =============================================================================
# CONFIGURATION PARAMETERS - Easy to find and modify
# =============================================================================

# Top-K Filtering Configuration
TOP_K_CONTACT_OBJECTS = 5           # Number of top contact objects to consider (with tie handling)
TOP_K_SUBSTRATE_OBJECTS = 5         # Number of top substrate objects to consider (based on max interaction ratings)

# Adaptive Top-K Configuration
ENABLE_ADAPTIVE_TOP_K = False        # Enable adaptive Top-K that adjusts based on candidate overlap
ADAPTIVE_K_INITIAL = 3              # Initial K value for adaptive Top-K
ADAPTIVE_K_MAX = 5                  # Maximum K value for adaptive Top-K (can exceed due to ties)

# Parallelization Configuration
ENABLE_PARALLEL_EVALUATION = True   # Enable parallel evaluation of assignments
NUM_WORKER_PROCESSES = None         # Number of worker processes (None = auto-detect CPU cores)
BATCH_SIZE_PER_WORKER = 3000         # Number of assignments per batch for each worker
PARALLEL_EVALUATION_THRESHOLD = 6000  # Minimum number of assignments to trigger parallel evaluation
CACHE_ACTIVATION_THRESHOLD = 10000    # Minimum number of assignments to trigger result caching

# Pin & Rotate Configuration
ENABLE_ADAPTIVE_PIN_GRID_SPACING = True  # Enable adaptive spacing based on physical object distances
                                         # When enabled: spacing = average of each object's minimum distance to its closest neighbor
                                         # When disabled: spacing = PIN_GRID_SPACING constant below
PIN_GRID_SPACING = 0.5              # Spacing between pin points in meters (used when adaptive is disabled)
ROTATION_STEP = 10                  # Rotation step in degrees (10° = 36 rotations per full circle)
STANDALONE_SELECTION_RADIUS = 0.3   # Selection pool radius for standalone interactables (meters)

# Adaptive Rotation Step Configuration
ENABLE_ADAPTIVE_ROTATION = True     # Enable adaptive rotation step (coarse → fine)
ADAPTIVE_ROTATION_COARSE_STEP = 45  # Coarse rotation step in degrees (for initial screening)
ADAPTIVE_ROTATION_FINE_STEP = 10    # Fine rotation step in degrees (for promising configs)
ADAPTIVE_ROTATION_THRESHOLD = 1.07   # Switch to fine step if (loss - best_loss) / |best_loss| ≤ threshold - 1.0
                                    # E.g., 1.5 means refine if within 50% worse than best_loss
ADAPTIVE_ROTATION_WARMUP_CONFIGS = 1  # Number of initial configs to evaluate with fine step (for baseline)

# Spatial Constraint Mode Configuration
SPATIAL_POOL_MODE = "relaxed"        # Options: "strict" or "relaxed"
                                    # - "strict": Selection pools limited to pedestal/circular bounds (hard constraint)
                                    # - "relaxed": Selection pools include all objects in overlapping region (soft constraint via L_spatial)

# Occlusion Configuration
ENABLE_PEDESTAL_OCCLUSION = False    # Enable/disable pedestals as occluders
                                    # - True: Pedestals block physical objects (use 3D bounds checking)
                                    # - False: Pedestals do not block physical objects (occlusion checking disabled)

SURROUNDINGS_OCCLUSION_MODE = "2d"   # Options: "2d" or "3d"
                                    # - "2d": Check only XZ plane (ignore height) - physical objects inside surroundings 2D bounds are occluded
                                    # - "3d": Check full 3D bounds (XYZ) - physical objects must be within height AND 2D bounds to be occluded
                                    # Note: This only affects surroundings objects (when ENABLE_PEDESTAL_OCCLUSION is True, pedestals always use 3D checking)

# Progress Tracking Configuration
ENABLE_PROGRESS_TRACKING = True     # Enable progress saving for recovery from interruptions
PROGRESS_SAVE_INTERVAL = 10         # Save progress every N configurations evaluated
PROGRESS_FILE = "optimization_progress.json"  # Progress tracking file name

# =============================================================================

@dataclass
class VirtualObject:
    """Represents a virtual object with its properties"""
    name: str
    index: int  # Index in the virtual objects list
    engagement_level: float  # 0.0-1.0 scale (normalized priority weight for primary role)
    substrate_engagement_level: float  # Priority weight when used as substrate in relationships
    involvement_type: str  # grasp, contact, substrate, pedestal, surroundings
    position: Optional[np.ndarray] = None  # 3D position in virtual space
    bounds_3d: Optional[Dict] = None  # 3D bounds {center, size, min, max}
    bounds_2d: Optional[Dict] = None  # 2D bounds {center, size, min, max, corners}

@dataclass
class PhysicalObject:
    """Represents a physical object with its properties"""
    name: str
    object_id: int
    image_id: int
    index: int  # Index in the physical objects list
    position: Optional[np.ndarray] = None  # 3D position in world space

@dataclass
class PlayArea:
    """Represents the physical play area boundary from Quest"""
    boundary_points: np.ndarray  # Nx3 array of boundary vertices
    center: np.ndarray  # 3D center point
    width: float  # X dimension
    depth: float  # Z dimension
    area: float  # Total area in m²

@dataclass
class VirtualEnvironment:
    """Represents the virtual environment bounds"""
    center: np.ndarray  # 3D center point (user-defined or geometric center)
    bounds_2d: Dict  # 2D bounds {center, size, min, max, corners}
    bounds_3d: Dict  # 3D bounds {center, size, min, max}

@dataclass
class PedestalGroup:
    """Represents a group with a pedestal and associated interactables"""
    pedestal_name: str
    pedestal_index: int  # Index in virtual_objects
    interactable_indices: List[int]  # Indices of interactables in this group
    selection_bounds_2d: Dict  # 2D bounds for selection pool

@dataclass
class Assignment:
    """Represents an assignment of virtual objects to physical objects"""
    assignment_matrix: np.ndarray  # Binary matrix X[i,j] where i=virtual, j=physical
    virtual_to_physical: Dict[int, int]  # Maps virtual object index to physical object index
    total_loss: float
    loss_components: Dict[str, float]  # Pin & rotate configuration (optional)
    pin_point: Optional[np.ndarray] = None
    rotation_angle: Optional[float] = None

# =============================================================================
# Parallel Evaluation Support Functions
# =============================================================================

def _rotate_point_2d_worker(point: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    """
    Rotate a 2D point around a center point (worker version)
    
    Args:
        point: 2D point [x, z]
        angle: Rotation angle in radians (counter-clockwise)
        center: Center of rotation [x, z]
    
    Returns:
        Rotated point [x, z]
    """
    # Translate to origin
    translated = point - center
    
    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Apply rotation
    x_rot = translated[0] * cos_a - translated[1] * sin_a
    z_rot = translated[0] * sin_a + translated[1] * cos_a
    
    # Translate back
    rotated = np.array([x_rot, z_rot]) + center
    
    return rotated

def _point_in_polygon_worker(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if a 2D point is inside a polygon using ray casting algorithm (worker version)
    
    Args:
        point: 2D point [x, z]
        polygon: Nx2 array of polygon vertices
    
    Returns:
        True if point is inside polygon
    """
    x, z = point
    n = len(polygon)
    inside = False
    
    p1x, p1z = polygon[0]
    for i in range(1, n + 1):
        p2x, p2z = polygon[i % n]
        if z > min(p1z, p2z):
            if z <= max(p1z, p2z):
                if x <= max(p1x, p2x):
                    if p1z != p2z:
                        xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1z = p2x, p2z
    
    return inside

def _get_transformed_virtual_env_polygon_worker(pin_point: np.ndarray, rotation_angle: float, virtual_env: dict) -> np.ndarray:
    """
    Get the transformed virtual environment polygon in physical space (worker version)
    
    Args:
        pin_point: 3D pin location
        rotation_angle: Rotation angle in radians
        virtual_env: Virtual environment dict with 'center' and 'bounds_2d'
        
    Returns:
        Nx2 array of transformed polygon corners in XZ plane
    """
    if virtual_env is None:
        return None
    
    virt_center_2d = np.array([virtual_env['center'][0], virtual_env['center'][2]])
    pin_2d = pin_point[[0, 2]]
    translation_2d = pin_2d - virt_center_2d
    
    virt_corners_2d = [np.array(c) for c in virtual_env['bounds_2d']['corners']]
    transformed_corners = []
    for corner in virt_corners_2d:
        rotated = _rotate_point_2d_worker(corner, rotation_angle, virt_center_2d)
        translated = rotated + translation_2d
        transformed_corners.append(translated)
    
    return np.array(transformed_corners)

def _check_occlusion_worker(physical_objects: list, virtual_objects: list, virtual_env: dict, 
                           pin_point: np.ndarray, rotation_angle: float) -> set:
    """
    Check which physical objects are occluded (inside virtual pedestals/surroundings) - worker version
    
    Occlusion behavior:
    - Pedestals: Controlled by ENABLE_PEDESTAL_OCCLUSION
        * True: Check 3D bounds (XYZ) - physical object must be within both height and 2D bounds
        * False: Pedestals do not block physical objects
    - Surroundings: Controlled by SURROUNDINGS_OCCLUSION_MODE
        * "3d" mode: Check full 3D bounds (XYZ) - same as pedestals
        * "2d" mode: Check only 2D bounds (XZ plane) - ignore height
    
    Returns:
        Set of physical object indices that are occluded
    """
    if virtual_env is None:
        return set()
    
    occluded_indices = set()
    virt_center_2d = np.array([virtual_env['center'][0], virtual_env['center'][2]])
    pin_2d = pin_point[[0, 2]]
    translation_2d = pin_2d - virt_center_2d
    
    for p_idx, p_obj in enumerate(physical_objects):
        if p_obj['position'] is None:
            continue
        
        p_pos_2d = np.array([p_obj['position'][0], p_obj['position'][2]])
        p_pos_y = p_obj['position'][1]
        
        # Check against each virtual object's 3D bounds
        for v_obj in virtual_objects:
            # Only check set dressing (pedestals, surroundings) for occlusion
            involvement = v_obj['involvement_type']
            
            # Skip if not set dressing
            if involvement not in ['pedestal', 'surroundings']:
                continue
            
            # Skip pedestal occlusion if disabled
            if involvement == 'pedestal' and not ENABLE_PEDESTAL_OCCLUSION:
                continue
            
            if v_obj.get('bounds_3d') is None:
                continue
            
            # Determine if we should check height based on object type and configuration
            check_height = True
            if v_obj['involvement_type'] == 'surroundings' and SURROUNDINGS_OCCLUSION_MODE == "2d":
                check_height = False  # Skip height check for surroundings in 2D mode
            
            # Y check (height) - only if required
            if check_height:
                v_bounds = v_obj['bounds_3d']
                v_center_y = v_bounds['center'][1]
                half_size_y = v_bounds['size'][1] / 2
                dy = abs(p_pos_y - v_center_y)
                
                if dy > half_size_y:
                    continue  # Not within height bounds
            
            # 2D check (XZ plane)
            if v_obj.get('bounds_2d') and 'corners' in v_obj['bounds_2d']:
                corners = v_obj['bounds_2d']['corners']
                
                if len(corners) >= 3:
                    # Transform corners
                    transformed_corners = []
                    for corner in corners:
                        corner_arr = np.array(corner)
                        rotated = _rotate_point_2d_worker(corner_arr, rotation_angle, virt_center_2d)
                        transformed = rotated + translation_2d
                        transformed_corners.append(transformed)
                    
                    transformed_polygon = np.array(transformed_corners)
                    if _point_in_polygon_worker(p_pos_2d, transformed_polygon):
                        occluded_indices.add(p_idx)
                        break  # No need to check other virtual objects
    
    return occluded_indices

def _get_selection_pool_worker(v_idx: int, pin_point: Optional[np.ndarray], rotation_angle: Optional[float], 
                               optimizer_state: dict) -> list:
    """
    Get selection pool for an interactable (worker version)
    Recreates the selection pool logic for parallel processing
    """
    virtual_objects = optimizer_state['virtual_objects']
    physical_objects = optimizer_state['physical_objects']
    virtual_env = optimizer_state['virtual_env']
    pedestal_groups = optimizer_state['pedestal_groups']
    standalone_interactables = optimizer_state['standalone_interactables']
    play_area = optimizer_state.get('play_area')
    
    v_obj = virtual_objects[v_idx]
    
    # Handle None values for pin_point and rotation_angle (when spatial constraints are disabled)
    if pin_point is None:
        if play_area is not None and 'center' in play_area:
            center = play_area['center']
            # Handle both list and numpy array formats
            if isinstance(center, list):
                pin_point = np.array(center)
            else:
                pin_point = np.array([center[0], center[1], center[2]])
        else:
            pin_point = np.array([0.0, 0.0, 0.0])
    if rotation_angle is None:
        rotation_angle = 0.0
    
    # Get play area and transformed virtual environment boundaries for overlap check
    play_area_boundary = None
    if play_area is not None and 'boundary_points' in play_area:
        boundary_pts = play_area['boundary_points']
        if isinstance(boundary_pts, list):
            play_area_boundary = np.array([[pt[0], pt[2]] for pt in boundary_pts])
        else:
            play_area_boundary = boundary_pts[:, [0, 2]]
    
    transformed_virt_polygon = _get_transformed_virtual_env_polygon_worker(pin_point, rotation_angle, virtual_env)
    
    # RELAXED MODE: Return all objects in overlapping region (ignore pedestal/circular bounds)
    # BUT exclude occluded objects (inside virtual pedestals/surroundings)
    if SPATIAL_POOL_MODE == "relaxed":
        if play_area_boundary is None or transformed_virt_polygon is None:
            # Return empty pool instead of all objects when boundaries are missing
            return []
        
        # Get occluded objects for this configuration
        occluded_indices = _check_occlusion_worker(physical_objects, virtual_objects, virtual_env, 
                                                   pin_point, rotation_angle)
        
        pool = []
        for p_idx, p_obj in enumerate(physical_objects):
            if p_obj['position'] is None:
                continue
            
            # CRITICAL: Exclude occluded objects (inside virtual pedestals/surroundings)
            if p_idx in occluded_indices:
                continue
            
            p_pos_2d = np.array([p_obj['position'][0], p_obj['position'][2]])
            
            in_play_area = _point_in_polygon_worker(p_pos_2d, play_area_boundary)
            in_virtual_env = _point_in_polygon_worker(p_pos_2d, transformed_virt_polygon)
            
            if in_play_area and in_virtual_env:
                pool.append(p_idx)
        return pool
    
    # STRICT MODE: Check pedestal bounds or circular radius AND overlapping region
    # Check if this interactable belongs to a pedestal group
    for pg in pedestal_groups:
        if v_idx in pg['interactable_indices']:
            selection_bounds_2d = pg['selection_bounds_2d']
            
            if selection_bounds_2d is None:
                return list(range(len(physical_objects)))
            
            # Get corners from bounds
            corners = selection_bounds_2d.get('corners', [])
            
            if len(corners) < 3:
                # Fallback: use center and size to create rectangle
                center = np.array(selection_bounds_2d['center'])
                size = selection_bounds_2d['size']
                half_x = size[0] / 2
                half_z = size[1] / 2
                corners = [
                    center + np.array([-half_x, -half_z]),
                    center + np.array([half_x, -half_z]),
                    center + np.array([half_x, half_z]),
                    center + np.array([-half_x, half_z])
                ]
            else:
                # Convert corners from list to numpy arrays
                corners = [np.array(c) for c in corners]
            
            # Transform corners to physical space
            virt_center_2d = np.array([virtual_env['center'][0], virtual_env['center'][2]])
            pin_2d = pin_point[[0, 2]]
            translation_2d = pin_2d - virt_center_2d
            
            transformed_corners = []
            for corner in corners:
                rotated = _rotate_point_2d_worker(corner, rotation_angle, virt_center_2d)
                transformed = rotated + translation_2d
                transformed_corners.append(transformed)
            
            transformed_polygon = np.array(transformed_corners)
            
            # Find physical objects within this polygon
            pool = []
            for p_idx, p_obj in enumerate(physical_objects):
                if p_obj['position'] is None:
                    continue
                p_pos_2d = np.array([p_obj['position'][0], p_obj['position'][2]])
                
                # First check if inside pedestal bounds
                if not _point_in_polygon_worker(p_pos_2d, transformed_polygon):
                    continue
                
                # CRITICAL FIX: Also check if physical object is in the overlapping region
                # of play area AND transformed virtual environment
                if play_area_boundary is not None and transformed_virt_polygon is not None:
                    in_play_area = _point_in_polygon_worker(p_pos_2d, play_area_boundary)
                    in_virtual_env = _point_in_polygon_worker(p_pos_2d, transformed_virt_polygon)
                    
                    # Only add to pool if in BOTH play area AND virtual environment
                    if in_play_area and in_virtual_env:
                        pool.append(p_idx)
                else:
                    # Fallback: if boundaries not available, use pedestal bounds only
                    pool.append(p_idx)
            return pool
    
    # Standalone interactable - use circular selection area
    if v_idx in standalone_interactables:
        # Transform virtual object position to physical space
        v_pos_2d = np.array([v_obj['position'][0], v_obj['position'][2]]) if v_obj['position'] is not None else None
        
        if v_pos_2d is None:
            return list(range(len(physical_objects)))
        
        virt_center_2d = np.array([virtual_env['center'][0], virtual_env['center'][2]])
        pin_2d = pin_point[[0, 2]]
        translation_2d = pin_2d - virt_center_2d
        
        rotated_v_pos = _rotate_point_2d_worker(v_pos_2d, rotation_angle, virt_center_2d)
        transformed_v_pos = rotated_v_pos + translation_2d
        
        # Use STANDALONE_SELECTION_RADIUS
        radius = 0.3  # STANDALONE_SELECTION_RADIUS
        
        pool = []
        for p_idx, p_obj in enumerate(physical_objects):
            if p_obj['position'] is None:
                continue
            p_pos_2d = np.array([p_obj['position'][0], p_obj['position'][2]])
            distance = np.linalg.norm(p_pos_2d - transformed_v_pos)
            
            # First check if within circular radius
            if distance > radius:
                continue
            
            # CRITICAL FIX: Also check if physical object is in the overlapping region
            # of play area AND transformed virtual environment
            if play_area_boundary is not None and transformed_virt_polygon is not None:
                in_play_area = _point_in_polygon_worker(p_pos_2d, play_area_boundary)
                in_virtual_env = _point_in_polygon_worker(p_pos_2d, transformed_virt_polygon)
                
                # Only add to pool if in BOTH play area AND virtual environment
                if in_play_area and in_virtual_env:
                    pool.append(p_idx)
            else:
                # Fallback: if boundaries not available, use circular area only
                pool.append(p_idx)
        
        return pool
    
    # Non-interactable (shouldn't happen, but return empty pool)
    return []

def _calculate_spatial_loss_worker(assignment_matrix: np.ndarray, pin_point: Optional[np.ndarray],
                                   rotation_angle: Optional[float], optimizer_state: dict) -> float:
    """
    Calculate spatial loss in worker process (mode-aware)
    
    STRICT mode: Penalizes assignments where physical objects are outside the selection pool
    RELAXED mode: Penalizes assignments based on distance from pedestal/circular bounds
    """
    virtual_objects = optimizer_state['virtual_objects']
    physical_objects = optimizer_state['physical_objects']
    virtual_env = optimizer_state['virtual_env']
    play_area = optimizer_state.get('play_area')
    pedestal_groups = optimizer_state.get('pedestal_groups', [])
    standalone_interactables = optimizer_state.get('standalone_interactables', [])
    
    if virtual_env is None:
        return 0.0
    
    # Calculate dmax: longest edge of the virtual environment
    bounds_3d = virtual_env.get('bounds_3d', {})
    if not bounds_3d or 'size' not in bounds_3d:
        return 0.0
    
    virt_env_size = bounds_3d['size']
    if isinstance(virt_env_size, (list, tuple)):
        dmax = max(virt_env_size)
    elif isinstance(virt_env_size, np.ndarray):
        dmax = float(np.max(virt_env_size))
    else:
        return 0.0
    
    if dmax == 0:
        return 0.0
    
    # Handle None values for pin_point and rotation_angle (when spatial constraints are disabled)
    if pin_point is None:
        if play_area is not None and 'center' in play_area:
            center = play_area['center']
            # Handle both list and numpy array formats
            if isinstance(center, list):
                pin_point = np.array(center)
            else:
                pin_point = np.array([center[0], center[1], center[2]])
        else:
            pin_point = np.array([0.0, 0.0, 0.0])
    if rotation_angle is None:
        rotation_angle = 0.0
    
    loss = 0.0
    
    # Get assignments
    assigned_physical = np.argmax(assignment_matrix, axis=1)
    
    # Check each interactable's assignment
    for v_idx, v_obj in enumerate(virtual_objects):
        # Only apply to interactables (grasp, contact, substrate)
        if v_obj['involvement_type'] not in ['grasp', 'contact', 'substrate']:
            continue
        
        # Get the assigned physical object
        p_idx = assigned_physical[v_idx]
        p_obj = physical_objects[p_idx]
        
        if p_obj['position'] is None or v_obj['position'] is None:
            continue
        
        p_pos_2d = np.array([p_obj['position'][0], p_obj['position'][2]])
        distance_to_target = None
        
        # MODE-AWARE PENALTY CALCULATION
        if SPATIAL_POOL_MODE == "strict":
            # STRICT MODE: Penalize if outside selection pool
            selection_pool = _get_selection_pool_worker(v_idx, pin_point, rotation_angle, optimizer_state)
            
            if p_idx in selection_pool:
                continue  # No penalty
            
            # Calculate distance to transformed virtual position
            v_pos_2d = np.array([v_obj['position'][0], v_obj['position'][2]])
            virt_center_2d = np.array([virtual_env['center'][0], virtual_env['center'][2]])
            pin_2d = pin_point[[0, 2]]
            translation_2d = pin_2d - virt_center_2d
            
            rotated_v_pos = _rotate_point_2d_worker(v_pos_2d, rotation_angle, virt_center_2d)
            transformed_v_pos = rotated_v_pos + translation_2d
            distance_to_target = np.linalg.norm(p_pos_2d - transformed_v_pos)
        
        else:  # SPATIAL_POOL_MODE == "relaxed"
            # RELAXED MODE: Check if outside pedestal/circular bounds
            is_in_pedestal_group = False
            for pg in pedestal_groups:
                if v_idx in pg['interactable_indices']:
                    is_in_pedestal_group = True
                    # Check if within pedestal bounds
                    if pg['selection_bounds_2d'] is not None:
                        bounds_2d = pg['selection_bounds_2d']
                        corners = bounds_2d.get('corners', [])
                        
                        if len(corners) < 3:
                            center = np.array(bounds_2d['center'])
                            size = bounds_2d['size']
                            half_x = size[0] / 2
                            half_z = size[1] / 2
                            corners = [
                                center + np.array([-half_x, -half_z]),
                                center + np.array([half_x, -half_z]),
                                center + np.array([half_x, half_z]),
                                center + np.array([-half_x, half_z])
                            ]
                        else:
                            corners = [np.array(c) for c in corners]
                        
                        # Transform corners
                        virt_center_2d = np.array([virtual_env['center'][0], virtual_env['center'][2]])
                        pin_2d = pin_point[[0, 2]]
                        translation_2d = pin_2d - virt_center_2d
                        
                        transformed_corners = []
                        for corner in corners:
                            rotated = _rotate_point_2d_worker(corner, rotation_angle, virt_center_2d)
                            transformed = rotated + translation_2d
                            transformed_corners.append(transformed)
                        
                        transformed_polygon = np.array(transformed_corners)
                        
                        # Check if inside pedestal bounds
                        if _point_in_polygon_worker(p_pos_2d, transformed_polygon):
                            continue  # No penalty
                        
                        # Calculate distance to pedestal center
                        pedestal_center = np.mean(transformed_polygon, axis=0)
                        distance_to_target = np.linalg.norm(p_pos_2d - pedestal_center)
                    break
            
            if not is_in_pedestal_group and v_idx in standalone_interactables:
                # Standalone interactable - check circular radius
                v_pos_2d = np.array([v_obj['position'][0], v_obj['position'][2]])
                virt_center_2d = np.array([virtual_env['center'][0], virtual_env['center'][2]])
                pin_2d = pin_point[[0, 2]]
                translation_2d = pin_2d - virt_center_2d
                
                rotated_v_pos = _rotate_point_2d_worker(v_pos_2d, rotation_angle, virt_center_2d)
                transformed_v_pos = rotated_v_pos + translation_2d
                distance_to_target = np.linalg.norm(p_pos_2d - transformed_v_pos)
                
                if distance_to_target <= 0.3:  # STANDALONE_SELECTION_RADIUS
                    continue  # No penalty
            
            if distance_to_target is None:
                continue  # No penalty for non-spatial objects
        
        if distance_to_target is None:
            continue
        
        # Apply exponential distance-based penalty: exp(priority_weight * di / dmax)
        enable_priority_weighting = optimizer_state.get('enable_priority_weighting', True)
        if enable_priority_weighting:
            priority_weight = v_obj['engagement_level']
        else:
            priority_weight = 1.0
        penalty = np.exp(priority_weight * distance_to_target / dmax)
        loss += penalty
    
    return float(loss)

def _evaluate_assignment_batch(args):
    """
    Worker function for parallel evaluation of assignment batches.
    Must be at module level for multiprocessing to pickle it.
    
    Args:
        args: Tuple of (batch_assignments, optimizer_state, pin_point, rotation_angle, batch_idx)
        pin_point and rotation_angle can be None when spatial constraints are disabled
    
    Returns:
        List of tuples: [(loss, loss_components, assignment_matrix, is_valid), ...]
    """
    batch_assignments, optimizer_state, pin_point, rotation_angle, batch_idx = args
    
    results = []
    for assignment_matrix in batch_assignments:
        # Validate assignment
        is_valid = _validate_assignment(assignment_matrix, optimizer_state)
        if not is_valid:
            results.append((float('inf'), {}, assignment_matrix, False))
            continue
        
        # Calculate loss
        total_loss, loss_components = _calculate_loss_standalone(
            assignment_matrix, optimizer_state, pin_point, rotation_angle
        )
        results.append((total_loss, loss_components, assignment_matrix, True))
    
    return results

def _evaluate_cached_assignment_batch(args):
    """
    Worker function for parallel evaluation of CACHED assignment batches.
    Only recalculates L_spatial, reuses L_realism and L_interaction from cache.
    
    Args:
        args: Tuple of (batch_cached_assignments, optimizer_state, pin_point, rotation_angle, 
                       w_realism, w_interaction, w_spatial, batch_idx)
    
    Returns:
        List of tuples: [(loss, loss_components, assignment_matrix, is_valid), ...]
    """
    (batch_cached_assignments, optimizer_state, pin_point, rotation_angle, 
     w_realism, w_interaction, w_spatial, batch_idx) = args
    
    results = []
    
    for cached_assignment in batch_cached_assignments:
        assignment_matrix = cached_assignment['assignment_matrix']
        l_realism = cached_assignment['L_realism']
        l_interaction = cached_assignment['L_interaction']
        
        # Calculate only L_spatial for this configuration
        if w_spatial > 0 and optimizer_state.get('virtual_env') is not None:
            l_spatial = _calculate_spatial_loss_worker(assignment_matrix, pin_point, rotation_angle, optimizer_state)
        else:
            l_spatial = 0.0
        
        # Calculate total loss
        total_loss = (w_realism * l_realism + 
                     w_interaction * l_interaction +
                     w_spatial * l_spatial)
        
        loss_components = {
            "L_realism": l_realism,
            "L_interaction": l_interaction,
            "L_spatial": l_spatial,
            "total": total_loss
        }
        
        results.append((total_loss, loss_components, assignment_matrix, True))
    
    return results

def _validate_assignment(assignment_matrix: np.ndarray, optimizer_state: Dict) -> bool:
    """Validate an assignment matrix"""
    assignable_indices = optimizer_state['assignable_indices']
    
    # Check each assignable virtual object gets exactly one proxy
    for i in assignable_indices:
        if not np.isclose(np.sum(assignment_matrix[i, :]), 1.0):
            return False
    
    # Check exclusivity if enabled
    if optimizer_state['enable_exclusivity']:
        for j in range(assignment_matrix.shape[1]):
            if np.sum(assignment_matrix[:, j]) > 1:
                return False
    
    return True

def _calculate_loss_standalone(assignment_matrix: np.ndarray, optimizer_state: Dict,
                               pin_point: Optional[np.ndarray] = None, rotation_angle: Optional[float] = None) -> Tuple[float, Dict]:
    """
    Calculate total loss for an assignment (standalone version for workers).
    This recreates the loss calculation logic without needing the full optimizer object.
    """
    # Extract state
    realism_matrix = optimizer_state['realism_matrix']
    interaction_matrix_3d = optimizer_state['interaction_matrix_3d']
    virtual_relationship_pairs = optimizer_state['virtual_relationship_pairs']
    interaction_exists = optimizer_state['interaction_exists']
    virtual_objects = optimizer_state['virtual_objects']
    physical_objects = optimizer_state['physical_objects']
    w_realism = optimizer_state['w_realism']
    w_interaction = optimizer_state['w_interaction']
    w_spatial = optimizer_state['w_spatial']
    enable_priority_weighting = optimizer_state['enable_priority_weighting']
    enable_spatial_constraint = optimizer_state['enable_spatial_constraint']

    # Calculate realism loss
    l_realism = 0.0
    grasp_contact_count = 0
    if realism_matrix is not None:
        for i in range(len(virtual_objects)):
            if virtual_objects[i]['involvement_type'] == "substrate":
                continue
            
            # Count grasp and contact objects (exclude substrate)
            if virtual_objects[i]['involvement_type'] in ['grasp', 'contact']:
                grasp_contact_count += 1
            
            priority_weight = virtual_objects[i]['engagement_level'] if enable_priority_weighting else 1.0
            
            for j in range(len(physical_objects)):
                if assignment_matrix[i, j] > 0:
                    l_realism -= 2 * priority_weight * realism_matrix[i, j]
    
    # Normalize by the count of grasp and contact objects
    if grasp_contact_count > 0:
        l_realism = l_realism / grasp_contact_count
    
    # Calculate interaction loss
    l_interaction = 0.0
    if interaction_matrix_3d is not None and virtual_relationship_pairs is not None:
        interaction_count = 0
        for rel_idx, (contact_v_idx, substrate_v_idx) in enumerate(virtual_relationship_pairs):
            if interaction_exists[contact_v_idx, substrate_v_idx] > 0:
                proxy_contact = np.argmax(assignment_matrix[contact_v_idx, :])
                proxy_substrate = np.argmax(assignment_matrix[substrate_v_idx, :])
                
                interaction_rating = interaction_matrix_3d[rel_idx, proxy_contact, proxy_substrate]
                
                if enable_priority_weighting:
                    # Contact object uses its primary engagement_level
                    contact_priority = virtual_objects[contact_v_idx]['engagement_level']
                    # Substrate object uses its substrate-specific engagement_level
                    # This allows dual-role objects (e.g., contact type used as substrate) 
                    # to have different priorities in their substrate role
                    substrate_priority = virtual_objects[substrate_v_idx]['substrate_engagement_level']
                else:
                    contact_priority = 1.0
                    substrate_priority = 1.0
                combined_priority = contact_priority + substrate_priority
                
                l_interaction -= interaction_rating * combined_priority
                interaction_count += 1
        
        if interaction_count > 0:
            l_interaction = l_interaction / interaction_count
    
    # Calculate spatial loss
    l_spatial = 0.0
    # NOTE: Changed from enable_spatial_constraint to w_spatial > 0
    if w_spatial > 0 and optimizer_state.get('virtual_env') is not None:
        l_spatial = _calculate_spatial_loss_worker(
            assignment_matrix, pin_point, rotation_angle, optimizer_state
        )
    
    total_loss = w_realism * l_realism + w_interaction * l_interaction + w_spatial * l_spatial
    
    loss_components = {
        "L_realism": l_realism,
        "L_interaction": l_interaction,
        "L_spatial": l_spatial,
        "total": total_loss
    }
    
    return total_loss, loss_components

# =============================================================================

class ProXeekOptimizer:
    """Global optimization for haptic proxy assignment
    
    HARD CONSTRAINTS:
    1. Assignment completeness: Each assignable virtual object (grasp, contact, substrate) gets exactly one proxy
    2. No proxy for set dressing: Pedestal and surroundings objects receive no proxies
    3. Exclusivity (when enabled): Each physical object used as proxy at most once
    4. Banned objects: User-specified physical objects are excluded from consideration
    5. Occlusion constraint (pin & rotate only): Physical objects inside virtual set dressing 3D bounds are banned
    6. Early termination (pin & rotate only): Skip configuration if insufficient non-occluded physical objects 
       in intersection of play area and virtual environment
    
    SOFT CONSTRAINTS (Multi-Objective Loss Functions):
    
    Both Default Mode and Pin & Rotate Mode use the same three loss functions:
    
    - L_realism = (1/N_grasp_contact) × -∑ᵢ∑ⱼ (2 × priority_weight[i] × realism_rating[i,j] × X[i,j])
      Applied only to grasp and contact objects
      
    - L_interaction = (1/N_relationships) × -∑ᵣ (interaction_rating[proxy_contact, proxy_substrate] × combined_priority_weight)
      Uses 3D interaction matrix indexed by virtual relationship
      
    - L_spatial = ∑ᵢ [exp(priority_weight[i] × dᵢ / dₘₐₓ)]
      Penalizes assignments outside selection pools (pedestal bounds or circular areas)
      where dᵢ is distance to transformed virtual position, dₘₐₓ is longest edge of virtual environment
      In default mode: uses default pin location (play area center or origin) with 0° rotation
      In pin & rotate mode: optimizes over all pin points and rotation angles
    
    Where:
    - priority_weight[i] = 4 / (1 + exp(-k × x_i)) from symmetric sigmoid ranking (k=1)
    - combined_priority_weight = priority_weight[contact] + substrate_priority_weight[substrate]
      Note: Dual-role objects (e.g., involvementType="contact" but used as substrate in relationships)
      have separate priority values for their primary role and substrate role
    - Top-K filtering: Only top-K physical objects (by realism) considered for grasp/contact
    - Tie-breaker: Average proxy distance used when multiple configurations have same loss
    """
    
    def __init__(self, data_dir: str = r"C:\Users\aaron\Documents\GitHub\YOLO-World-Seg\proxeek\output"):
        self.data_dir = data_dir
        self.virtual_objects: List[VirtualObject] = []
        self.physical_objects: List[PhysicalObject] = []
        self.realism_matrix: Optional[np.ndarray] = None  # realism_rating[i,j]

        self.interaction_matrix: Optional[np.ndarray] = None  # interaction_rating[j,k] for physical objects
        self.interaction_exists: Optional[np.ndarray] = None  # interaction_exists[i,k] for virtual objects
        
        # Loss function weights
        self.w_realism = 1.0      # Weight for semantic similarity loss
        self.w_interaction = 1.0  # Weight for interaction relationship loss
        self.w_spatial = 1.0      # Weight for spatial constraint loss (pedestal boundaries)
        
        # Priority control
        self.enable_priority_weighting = True  # Control whether to apply priority weights
        
        # Matrices for spatial calculations (OLD - distance-based)
        self.spatial_group_matrix: Optional[np.ndarray] = None
        self.virtual_distance_matrix: Optional[np.ndarray] = None
        self.virtual_angle_matrix: Optional[np.ndarray] = None
        self.physical_distance_matrix: Optional[np.ndarray] = None
        self.physical_angle_matrix: Optional[np.ndarray] = None
        
        # NEW: Spatial constraint system (Pin & Rotate)
        # IMPORTANT DISTINCTION:
        # - enable_spatial_constraint: Controls whether to use Pin & Rotate OPTIMIZATION (search for best placement)
        # - w_spatial: Controls whether to PENALIZE objects outside pedestal boundaries in loss function
        # These are independent! You can have w_spatial > 0 without enable_spatial_constraint to
        # penalize boundary violations even when not using pin & rotate optimization.
        self.enable_spatial_constraint = True  # Enable pin & rotate optimization
        self.play_area: Optional[PlayArea] = None
        self.virtual_env: Optional[VirtualEnvironment] = None
        self.pedestal_groups: List[PedestalGroup] = []
        self.standalone_interactables: List[int] = []  # Indices of interactables without pedestals
        self.pin_points: Optional[np.ndarray] = None  # Grid of pin points
        self.rotation_angles: Optional[np.ndarray] = None  # Array of rotation angles
        self.occluded_physical_indices: Set[int] = set()  # Banned due to occlusion
        
        # Constraints
        self.enable_exclusivity = True  # Each physical object used at most once
        
        # Banned physical objects (by (image_id, object_id)) and their indices
        self.banned_physical_pairs: Set[Tuple[int, int]] = set()
        self.banned_physical_indices: Set[int] = set()
        
    def load_data(self) -> bool:
        """Load all required data files"""
        try:
            # Load haptic annotation data - locate the most recent file in the output directory
            haptic_files = [f for f in os.listdir(self.data_dir) if f.startswith("haptic_annotation") and f.endswith(".json")]
            
            if not haptic_files:
                raise FileNotFoundError(f"No haptic annotation files found in {self.data_dir}")
            
            # Use the most recent file (sorted alphabetically, which works for timestamp format)
            haptic_file = os.path.join(self.data_dir, sorted(haptic_files)[-1])
            print(f"Loading haptic annotation file: {haptic_file}")
            
            with open(haptic_file, 'r') as f:
                haptic_data = json.load(f)
            
            # Load physical object database
            physical_file = os.path.join(self.data_dir, "physical_object_database.json")
            print(f"DEBUG: Loading physical objects from: {physical_file}")
            with open(physical_file, 'r') as f:
                physical_data_raw = json.load(f)
            
            # Handle new format: check if data has 'data' and 'playArea' fields
            if isinstance(physical_data_raw, dict) and 'data' in physical_data_raw:
                # New format with play area
                print("DEBUG: Detected new format with play area data")
                physical_data = physical_data_raw  # Keep full structure for play area processing
            else:
                # Old format: direct dictionary of image IDs to object lists
                print("DEBUG: Detected old format without play area data")
                physical_data = physical_data_raw
            
            # Debug: Show sample physical objects
            sample_objects = []
            # Extract the actual object data (handle both formats)
            objects_dict = physical_data.get('data', physical_data) if isinstance(physical_data, dict) else physical_data
            
            for image_id, objects in objects_dict.items():
                if image_id in ['playArea', 'action', 'timestamp', 'total_objects']:
                    continue  # Skip metadata fields
                for obj in objects[:2]:  # First 2 objects
                    if isinstance(obj, dict):
                        sample_objects.append(obj.get('object', 'Unknown'))
                    if len(sample_objects) >= 3:
                        break
                if len(sample_objects) >= 3:
                    break
            print(f"DEBUG: Sample physical objects loaded: {sample_objects}")
            
            # Load proxy matching results (for realism ratings)
            proxy_file = os.path.join(self.data_dir, "proxy_matching_results.json")
            print(f"DEBUG: Loading proxy matching from: {proxy_file}")
            with open(proxy_file, 'r') as f:
                proxy_data = json.load(f)
            
            # Store proxy data for later use (e.g., utilization methods)
            self.proxy_data = proxy_data
            
            # Load relationship rating results (for interaction ratings)
            relationship_file = os.path.join(self.data_dir, "relationship_rating_by_dimension.json")
            with open(relationship_file, 'r') as f:
                relationship_data = json.load(f)
            
            # NEW: Load virtual object database (for positions)
            virtual_file = os.path.join(self.data_dir, "virtual_object_database.json")
            with open(virtual_file, 'r') as f:
                virtual_data = json.load(f)
            
            # Process the loaded data
            self._process_virtual_objects(haptic_data)
            self._process_physical_objects(physical_data)
            # Assign positions to virtual objects from haptic data (not from virtual object database)
            self._assign_virtual_positions(haptic_data.get("nodeAnnotations", []))
            
            # NEW: Process spatial constraint data
            self._process_play_area(physical_data)
            self._process_virtual_environment(haptic_data)
            self._process_pedestal_groups(haptic_data)
            
            # Build auxiliary matrices
            self._build_realism_matrix(proxy_data)
            self._build_interaction_matrices(haptic_data, relationship_data)
            self._build_distance_matrices()  # for OLD spatial loss
            
            # Recompute banned indices if any pairs were set prior to load
            self.refresh_banned_indices_after_load()
            
            print(f"Loaded data successfully:")
            print(f"  Virtual objects: {len(self.virtual_objects)}")
            print(f"  Physical objects: {len(self.physical_objects)}")
            if self.realism_matrix is not None:
                print(f"  Realism matrix shape: {self.realism_matrix.shape}")
            if self.interaction_matrix is not None:
                print(f"  Interaction matrix shape: {self.interaction_matrix.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_data_from_memory(self, haptic_annotation_json: str, physical_object_database: Dict, 
                             virtual_objects: List[Dict], proxy_matching_results: List[Dict], 
                             relationship_rating_results) -> bool:
        """Load data from in-memory objects instead of files
        
        Args:
            relationship_rating_results: Either List[Dict] (old format) or Dict (new dimension-based format)
                                       with keys "harmony", "expressivity", "realism"
        """
        try:
            print(f"DEBUG: load_data_from_memory received:")
            print(f"  Physical DB keys: {list(physical_object_database.keys()) if physical_object_database else 'None'}")
            if physical_object_database:
                sample_objects = []
                for image_id, objects in physical_object_database.items():
                    sample_objects.extend([obj.get('object', 'Unknown') for obj in objects[:2]])
                    if len(sample_objects) >= 3:
                        break
                print(f"  Sample physical objects: {sample_objects[:3]}")
            print(f"  Proxy matching results count: {len(proxy_matching_results) if proxy_matching_results else 0}")
            
            # Detect relationship data format and convert if needed
            if isinstance(relationship_rating_results, dict) and "harmony" in relationship_rating_results:
                print("  Relationship data format: dimension-based (new format)")
                relationship_data = relationship_rating_results
            elif isinstance(relationship_rating_results, list):
                print("  Relationship data format: flat list (old format) - converting to dimension-based")
                # Convert old format to new format for compatibility
                relationship_data = {"harmony": [], "expressivity": [], "realism": []}
                for result in relationship_rating_results:
                    # Create base entry
                    base_entry = {
                        "virtualContactObject": result.get("virtualContactObject", ""),
                        "virtualSubstrateObject": result.get("virtualSubstrateObject", ""),
                        "physicalContactObject": result.get("physicalContactObject", ""),
                        "physicalSubstrateObject": result.get("physicalSubstrateObject", ""),
                        "contactObject_id": result.get("contactObject_id", ""),
                        "contactImage_id": result.get("contactImage_id", ""),
                        "substrateObject_id": result.get("substrateObject_id", ""),
                        "substrateImage_id": result.get("substrateImage_id", ""),
                        "contactUtilizationMethod": result.get("contactUtilizationMethod", ""),
                        "substrateUtilizationMethod": result.get("substrateUtilizationMethod", ""),
                        "group_index": result.get("group_index", ""),
                        "expectedHapticFeedback": result.get("expectedHapticFeedback", "")
                    }
                    
                    # Add dimension-specific entries
                    if "harmony_rating" in result:
                        harmony_entry = base_entry.copy()
                        harmony_entry.update({
                            "dimension": "harmony",
                            "rating": result.get("harmony_rating", 0),
                            "explanation": result.get("harmony_explanation", "")
                        })
                        relationship_data["harmony"].append(harmony_entry)
                    
                    if "expressivity_rating" in result:
                        expressivity_entry = base_entry.copy()
                        expressivity_entry.update({
                            "dimension": "expressivity",
                            "rating": result.get("expressivity_rating", 0),
                            "explanation": result.get("expressivity_explanation", "")
                        })
                        relationship_data["expressivity"].append(expressivity_entry)
                    
                    if "realism_rating" in result:
                        realism_entry = base_entry.copy()
                        realism_entry.update({
                            "dimension": "realism",
                            "rating": result.get("realism_rating", 0),
                            "explanation": result.get("realism_explanation", "")
                        })
                        relationship_data["realism"].append(realism_entry)
            else:
                print("  Warning: Unknown relationship data format, using empty data")
                relationship_data = {"harmony": [], "expressivity": [], "realism": []}
            
            # Parse haptic annotation JSON
            haptic_data = json.loads(haptic_annotation_json)
            
            # Store haptic data for spatial group processing
            self.haptic_data = haptic_data
            
            # Process the in-memory data
            self._process_virtual_objects(haptic_data)
            self._process_physical_objects(physical_object_database)
            # Assign positions to virtual objects from haptic data (not from separate virtual_objects parameter)
            self._assign_virtual_positions(haptic_data.get("nodeAnnotations", []))
            # Build auxiliary matrices
            self._build_realism_matrix(proxy_matching_results)
            self._build_interaction_matrices(haptic_data, relationship_data)
            self._build_distance_matrices()  # for spatial loss
            
            # Recompute banned indices if any pairs were set prior to load
            self.refresh_banned_indices_after_load()
            
            print(f"Loaded in-memory data successfully:")
            print(f"  Virtual objects: {len(self.virtual_objects)}")
            print(f"  Physical objects: {len(self.physical_objects)}")
            if self.realism_matrix is not None:
                print(f"  Realism matrix shape: {self.realism_matrix.shape}")
                print(f"  Non-zero entries in realism matrix: {np.count_nonzero(self.realism_matrix)}")
            if self.interaction_matrix is not None:
                print(f"  Interaction matrix shape: {self.interaction_matrix.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading in-memory data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_virtual_objects(self, haptic_data: Dict) -> None:
        """Process virtual objects from haptic annotation data"""
        node_annotations = haptic_data.get("nodeAnnotations", [])
        
        # NEW: Use single unified priority order list (proxyPriorityOrder)
        # This replaces the old highEngagementOrder, mediumEngagementOrder, lowEngagementOrder
        complete_priority_order = haptic_data.get("proxyPriorityOrder", [])
        
        print(f"DEBUG: Processing {len(node_annotations)} node annotations")
        print(f"DEBUG: Available involvement types: {list(set(obj.get('involvementType', '') for obj in node_annotations))}")
        print(f"DEBUG: Proxy priority order: {complete_priority_order}")
        
        # Include ALL object types
        # Note: pedestal and surroundings are included for spatial constraint (occlusion, selection pools)
        # but they won't be assigned proxies (no priority weights calculated for them)
        filtered_objects = [obj for obj in node_annotations 
                          if obj.get("involvementType") in ["grasp", "contact", "substrate", "pedestal", "surroundings"]]
        
        print(f"DEBUG: Filtered to {len(filtered_objects)} objects")
        for obj in filtered_objects:
            print(f"  DEBUG: {obj.get('objectName', 'Unknown')} - Type: {obj.get('involvementType', 'Unknown')}")
        
        # Create separate priority orders for different object types
        # Note: Set dressing (pedestal, surroundings) are excluded from proxyPriorityOrder in Unity
        # They don't need proxies, so no priority weights are calculated for them
        grasp_contact_priority_order = []
        substrate_priority_order = []
        
        for obj_name in complete_priority_order:
            # Find the object in filtered_objects to check its type
            for obj in filtered_objects:
                if obj.get("objectName") == obj_name:
                    if obj.get("involvementType") in ["grasp", "contact"]:
                        grasp_contact_priority_order.append(obj_name)
                    elif obj.get("involvementType") == "substrate":
                        substrate_priority_order.append(obj_name)
                    # Note: pedestal and surroundings should NOT be in proxyPriorityOrder
                    # but if they are, we ignore them (they don't need proxies)
                    break
        
        print(f"DEBUG: Priority order for grasp/contact objects: {grasp_contact_priority_order}")
        print(f"DEBUG: Priority order for substrate objects: {substrate_priority_order}")
        
        # IMPORTANT: Identify which objects appear as substrates in relationships
        # This is needed BEFORE creating VirtualObject instances
        relationship_annotations = haptic_data.get("relationshipAnnotations", [])
        substrate_role_objects = set()  # Objects that appear as substrates in any relationship
        for rel in relationship_annotations:
            substrate_name = rel.get("substrateObject", "")
            if substrate_name:
                substrate_role_objects.add(substrate_name)
        
        print(f"DEBUG: Objects appearing as substrates in relationships: {substrate_role_objects}")
        
        # NEW: Create substrate role priority order
        # This is the priority order for objects appearing as substrates in relationships,
        # ordered by their position in complete_priority_order
        # This ensures substrate priorities are calculated relative to other substrate-role objects
        substrate_role_priority_order = [name for name in complete_priority_order 
                                          if name in substrate_role_objects]
        
        print(f"DEBUG: Substrate role priority order: {substrate_role_priority_order}")
        
        for i, obj in enumerate(filtered_objects):
            name = obj.get("objectName", "")
            involvement_type = obj.get("involvementType", "")
            
            # Calculate PRIMARY engagement level based on object type and ranking
            engagement_level = 0.0  # default for unranked objects
            
            if involvement_type in ["grasp", "contact"]:
                # Grasp and contact objects get priority weights from ranking
                if name in grasp_contact_priority_order:
                    # Sigmoid priority weighting (scaled by 2)
                    priority_rank = grasp_contact_priority_order.index(name)
                    N = len(grasp_contact_priority_order)
                    if N > 0:  # Only calculate if we have ranked objects
                        center = (N - 1) / 2.0
                        x_i = center - priority_rank          # evenly spaced, centred at 0
                        k = 1                                # slope parameter for sigmoid
                        engagement_level = 4 / (1 + math.exp(-k * x_i))
                    else:
                        engagement_level = 1.0  # Default if no ranking available
                else:
                    engagement_level = 1.0  # Default for unranked grasp/contact objects
            elif involvement_type == "substrate":
                # Substrate objects now get priority weights from ranking similar to grasp/contact objects
                if name in substrate_priority_order:
                    # Sigmoid priority weighting (scaled by 2)
                    priority_rank = substrate_priority_order.index(name)
                    N = len(substrate_priority_order)
                    if N > 0:  # Only calculate if we have ranked objects
                        center = (N - 1) / 2.0
                        x_i = center - priority_rank          # evenly spaced, centred at 0
                        k = 1                              # slope parameter for sigmoid
                        engagement_level = 4 / (1 + math.exp(-k * x_i))
                    else:
                        engagement_level = 1.0  # Default if no ranking available
                else:
                    engagement_level = 1.0  # Default for unranked substrate objects
            elif involvement_type in ["pedestal", "surroundings"]:
                # Set dressing objects don't need proxies - set priority to 0
                # They are only used for spatial constraints (occlusion, selection pools)
                engagement_level = 0.0
            
            # Calculate SUBSTRATE-SPECIFIC engagement level for objects appearing as substrates
            # This applies to ALL objects that appear as substrates in relationships,
            # regardless of their primary involvementType
            substrate_engagement_level = 0.0  # default for objects not used as substrates
            
            if name in substrate_role_objects:
                # This object appears as a substrate in relationships
                # Calculate its substrate priority based on position in substrate_role_priority_order
                # IMPORTANT: This ranks against OTHER substrate-role objects, not all objects
                if name in substrate_role_priority_order:
                    substrate_priority_rank = substrate_role_priority_order.index(name)
                    N_substrate_roles = len(substrate_role_priority_order)
                    if N_substrate_roles > 0:
                        center = (N_substrate_roles - 1) / 2.0
                        x_i = center - substrate_priority_rank
                        k = 1
                        substrate_engagement_level = 4 / (1 + math.exp(-k * x_i))
                    else:
                        substrate_engagement_level = 1.0
                else:
                    substrate_engagement_level = 1.0  # Default for unranked objects
            
            # For objects with involvementType="substrate", substrate_engagement_level should match primary
            if involvement_type == "substrate" and name in substrate_role_objects:
                substrate_engagement_level = engagement_level
            
            # Parse bounds data if available
            bounds_3d = obj.get("bounds3D", None)
            bounds_2d = obj.get("bounds2D", None)
            
            # Convert bounds to dict format if present
            bounds_3d_dict = None
            if bounds_3d and isinstance(bounds_3d, dict):
                try:
                    bounds_3d_dict = {
                        'center': np.array([
                            bounds_3d.get('center', {}).get('x', 0.0),
                            bounds_3d.get('center', {}).get('y', 0.0),
                            bounds_3d.get('center', {}).get('z', 0.0)
                        ]),
                        'size': np.array([
                            bounds_3d.get('size', {}).get('x', 0.0),
                            bounds_3d.get('size', {}).get('y', 0.0),
                            bounds_3d.get('size', {}).get('z', 0.0)
                        ]),
                        'min': np.array([
                            bounds_3d.get('min', {}).get('x', 0.0),
                            bounds_3d.get('min', {}).get('y', 0.0),
                            bounds_3d.get('min', {}).get('z', 0.0)
                        ]),
                        'max': np.array([
                            bounds_3d.get('max', {}).get('x', 0.0),
                            bounds_3d.get('max', {}).get('y', 0.0),
                            bounds_3d.get('max', {}).get('z', 0.0)
                        ])
                    }
                except Exception as e:
                    print(f"  WARNING: Failed to parse 3D bounds for {name}: {e}")
                    bounds_3d_dict = None
            
            bounds_2d_dict = None
            if bounds_2d and isinstance(bounds_2d, dict):
                try:
                    # Parse corners from Unity export
                    # Unity exports both min/max (for calculations) AND oriented corners (for visualization)
                    # We must use the oriented corners for correct visualization!
                    corners_list = []
                    if 'corners' in bounds_2d and bounds_2d['corners']:
                        corners_list = [
                            np.array([corner.get('x', 0.0), corner.get('z', 0.0)]) 
                            for corner in bounds_2d['corners']
                            if isinstance(corner, dict)
                        ]
                    
                    # Handle flat format (minX, maxX, minZ, maxZ) from Unity export
                    if 'minX' in bounds_2d:
                        min_x = bounds_2d.get('minX', 0.0)
                        max_x = bounds_2d.get('maxX', 0.0)
                        min_z = bounds_2d.get('minZ', 0.0)
                        max_z = bounds_2d.get('maxZ', 0.0)
                        center_x = bounds_2d.get('centerX', (min_x + max_x) / 2)
                        center_z = bounds_2d.get('centerZ', (min_z + max_z) / 2)
                        width = bounds_2d.get('width', max_x - min_x)
                        depth = bounds_2d.get('depth', max_z - min_z)
                        
                        bounds_2d_dict = {
                            'center': np.array([center_x, center_z]),
                            'size': np.array([width, depth]),
                            'min': np.array([min_x, min_z]),
                            'max': np.array([max_x, max_z]),
                            'corners': corners_list  # Use oriented corners from Unity, not reconstructed!
                        }
                    # Handle nested format (min/max objects with x/z fields)
                    else:
                        bounds_2d_dict = {
                            'center': np.array([
                                bounds_2d.get('center', {}).get('x', 0.0),
                                bounds_2d.get('center', {}).get('z', 0.0)
                            ]),
                            'size': np.array([
                                bounds_2d.get('size', {}).get('x', 0.0),
                                bounds_2d.get('size', {}).get('z', 0.0)
                            ]),
                            'min': np.array([
                                bounds_2d.get('min', {}).get('x', 0.0),
                                bounds_2d.get('min', {}).get('z', 0.0)
                            ]),
                            'max': np.array([
                                bounds_2d.get('max', {}).get('x', 0.0),
                                bounds_2d.get('max', {}).get('z', 0.0)
                            ]),
                            'corners': corners_list  # Use oriented corners from Unity
                        }
                except Exception as e:
                    print(f"  WARNING: Failed to parse 2D bounds for {name}: {e}")
                    bounds_2d_dict = None
            
            virtual_obj = VirtualObject(
                name=name,
                index=i,
                engagement_level=engagement_level,
                substrate_engagement_level=substrate_engagement_level,
                involvement_type=involvement_type,
                bounds_3d=bounds_3d_dict,
                bounds_2d=bounds_2d_dict
            )
            self.virtual_objects.append(virtual_obj)
        
        # Print priority assignments for debugging
        print(f"Priority weights assigned:")
        for virtual_obj in self.virtual_objects:
            if virtual_obj.substrate_engagement_level > 0:
                print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): primary={virtual_obj.engagement_level:.3f}, substrate={virtual_obj.substrate_engagement_level:.3f}")
            else:
                print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): {virtual_obj.engagement_level:.3f}")
        
        print(f"DEBUG: Created {len(self.virtual_objects)} virtual objects:")
        for i, obj in enumerate(self.virtual_objects):
            print(f"  DEBUG: [{i}] {obj.name} - Type: {obj.involvement_type}")
    
    def _process_physical_objects(self, physical_data: Dict) -> None:
        """Process physical objects from database"""
        # Handle both old and new formats
        # New format: {'data': {image_id: [objects]}, 'playArea': {...}}
        # Old format: {image_id: [objects]}
        objects_dict = physical_data.get('data', physical_data) if 'data' in physical_data else physical_data
        
        index = 0
        for image_id_str, objects in objects_dict.items():
            # Skip metadata fields in new format
            if image_id_str in ['playArea', 'action', 'timestamp', 'total_objects']:
                continue
            
            if not isinstance(objects, list):
                continue
            
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                
                # Extract 3D position if available
                world_pos = obj.get("worldposition", {})
                pos_arr = None
                if world_pos:
                    pos_arr = np.array([
                        float(world_pos.get("x", 0.0)),
                        float(world_pos.get("y", 0.0)),
                        float(world_pos.get("z", 0.0))
                    ])
                physical_obj = PhysicalObject(
                    name=obj.get("object", ""),
                    object_id=obj.get("object_id", -1),
                    image_id=obj.get("image_id", int(image_id_str)),
                    index=index,
                    position=pos_arr
                )
                self.physical_objects.append(physical_obj)
                index += 1
    
    def _assign_virtual_positions(self, virtual_data: List[Dict]) -> None:
        """Assign 3D positions to previously created VirtualObject instances"""
        name_to_pos = {}
        positions_found = 0
        
        # Extract positions from the haptic annotation data
        print(f"DEBUG: Looking for positions in {len(virtual_data)} entries")
        for entry in virtual_data:
            if "objectName" in entry and "globalPosition" in entry:
                gp = entry["globalPosition"]
                name_to_pos[entry["objectName"]] = np.array([
                    float(gp.get("x", 0.0)),
                    float(gp.get("y", 0.0)),
                    float(gp.get("z", 0.0))
                ])
                positions_found += 1
                print(f"  DEBUG: Found position for '{entry['objectName']}' at {entry['globalPosition']}")
            else:
                missing_fields = []
                if "objectName" not in entry:
                    missing_fields.append("objectName")
                if "globalPosition" not in entry:
                    missing_fields.append("globalPosition")
                print(f"  DEBUG: Entry missing fields: {missing_fields}")
                if "objectName" in entry:
                    print(f"    DEBUG: Entry has objectName: '{entry['objectName']}'")
        
        print(f"Found positions for {positions_found} objects in haptic annotation data")
        
        # Assign positions to virtual objects
        positions_assigned = 0
        for v_obj in self.virtual_objects:
            if v_obj.name in name_to_pos:
                v_obj.position = name_to_pos[v_obj.name]
                positions_assigned += 1
            else:
                print(f"WARNING: No position found for virtual object '{v_obj.name}'")
        
        print(f"Assigned positions to {positions_assigned}/{len(self.virtual_objects)} virtual objects")
        
        # Verify all objects have positions
        missing_positions = [obj.name for obj in self.virtual_objects if obj.position is None]
        if missing_positions:
            print(f"Objects still missing positions: {missing_positions}")
        else:
            print("All virtual objects now have positions assigned!")
    
    def _process_play_area(self, physical_data: Dict) -> None:
        """Process play area data from physical object database"""
        play_area_data = physical_data.get('playArea', None)
        
        if not play_area_data:
            print("No play area data found in physical database (boundary not configured on Quest)")
            self.play_area = None
            return
        
        try:
            # Extract boundary points
            boundary_points_data = play_area_data.get('boundaryPoints', [])
            if not boundary_points_data:
                print("Play area data exists but no boundary points found")
                self.play_area = None
                return
            
            # Convert to numpy array
            boundary_points = np.array([
                [pt['x'], pt['y'], pt['z']] 
                for pt in boundary_points_data
            ])
            
            # Extract other play area properties
            center_data = play_area_data.get('center', {'x': 0, 'y': 0, 'z': 0})
            center = np.array([center_data['x'], center_data['y'], center_data['z']])
            
            width = float(play_area_data.get('width', 0.0))
            depth = float(play_area_data.get('depth', 0.0))
            area = float(play_area_data.get('area', 0.0))
            
            self.play_area = PlayArea(
                boundary_points=boundary_points,
                center=center,
                width=width,
                depth=depth,
                area=area
            )
            
            print(f"Play area loaded: {len(boundary_points)} boundary points, "
                  f"Size: {width:.2f}m × {depth:.2f}m, Area: {area:.2f}m²")
            
            # Save play area back to physical_object_database.json for persistence
            self._save_play_area_to_database(play_area_data)
            
        except Exception as e:
            print(f"Error processing play area data: {e}")
            import traceback
            traceback.print_exc()
            self.play_area = None
    
    def _save_play_area_to_database(self, play_area_data: Dict) -> None:
        """Save play area data to physical_object_database.json for persistence"""
        try:
            physical_db_path = os.path.join(self.data_dir, "physical_object_database.json")
            
            # Load current database
            with open(physical_db_path, 'r') as f:
                db_data = json.load(f)
            
            # Ensure it's in the new format
            if 'data' not in db_data:
                # Convert old format to new format
                db_data = {
                    "action": "update_bounding_boxes",
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    "total_objects": sum(len(v) if isinstance(v, list) else 0 for v in db_data.values()),
                    "data": db_data,
                    "playArea": play_area_data
                }
            else:
                # Already new format, just update playArea
                db_data['playArea'] = play_area_data
            
            # Save back to file
            with open(physical_db_path, 'w') as f:
                json.dump(db_data, f, indent=2)
            
            print(f"✓ Play area data persisted to: {physical_db_path}")
            
        except Exception as e:
            print(f"Warning: Could not save play area to database: {e}")
    
    def _process_virtual_environment(self, haptic_data: Dict) -> None:
        """Process virtual environment bounds from haptic annotation data"""
        try:
            # Extract virtual environment center (user-defined rotation center)
            virt_env_center_data = haptic_data.get('virtualEnvironmentCenter', None)
            if virt_env_center_data:
                virt_env_center = np.array([
                    virt_env_center_data['x'],
                    virt_env_center_data['y'],
                    virt_env_center_data['z']
                ])
            else:
                print("WARNING: No virtualEnvironmentCenter found in haptic data")
                virt_env_center = np.array([0.0, 0.0, 0.0])
            
            # Extract virtual environment bounds
            virt_env_bounds_data = haptic_data.get('virtualEnvironmentBounds', None)
            if not virt_env_bounds_data:
                print("WARNING: No virtualEnvironmentBounds found in haptic data")
                self.virtual_env = None
                return
            
            # Parse 2D bounds - Unity exports as 'footprint2D' with flat structure
            footprint_2d_data = virt_env_bounds_data.get('footprint2D', {})
            if not footprint_2d_data:
                # Try old format with bounds2D
                bounds_2d_data = virt_env_bounds_data.get('bounds2D', {})
                if not bounds_2d_data:
                    print("WARNING: No bounds2D or footprint2D in virtualEnvironmentBounds")
                    self.virtual_env = None
                    return
                
                # Old format parsing
                bounds_2d = {
                    'center': np.array([
                        bounds_2d_data.get('center', {}).get('x', 0.0),
                        bounds_2d_data.get('center', {}).get('z', 0.0)
                    ]),
                    'size': np.array([
                        bounds_2d_data.get('size', {}).get('x', 0.0),
                        bounds_2d_data.get('size', {}).get('z', 0.0)
                    ]),
                    'min': np.array([
                        bounds_2d_data.get('min', {}).get('x', 0.0),
                        bounds_2d_data.get('min', {}).get('z', 0.0)
                    ]),
                    'max': np.array([
                        bounds_2d_data.get('max', {}).get('x', 0.0),
                        bounds_2d_data.get('max', {}).get('z', 0.0)
                    ]),
                    'corners': [
                        np.array([corner.get('x', 0.0), corner.get('z', 0.0)]) 
                        for corner in bounds_2d_data.get('corners', [])
                        if isinstance(corner, dict)
                    ]
                }
            else:
                # New format parsing - flat structure from Unity
                center_x = float(footprint_2d_data.get('centerX', 0.0))
                center_z = float(footprint_2d_data.get('centerZ', 0.0))
                width = float(footprint_2d_data.get('width', 0.0))
                depth = float(footprint_2d_data.get('depth', 0.0))
                min_x = float(footprint_2d_data.get('minX', 0.0))
                max_x = float(footprint_2d_data.get('maxX', 0.0))
                min_z = float(footprint_2d_data.get('minZ', 0.0))
                max_z = float(footprint_2d_data.get('maxZ', 0.0))
                
                # Calculate corners from min/max
                corners = [
                    np.array([min_x, min_z]),  # Bottom-left
                    np.array([max_x, min_z]),  # Bottom-right
                    np.array([max_x, max_z]),  # Top-right
                    np.array([min_x, max_z])   # Top-left
                ]
                
                bounds_2d = {
                    'center': np.array([center_x, center_z]),
                    'size': np.array([width, depth]),
                    'min': np.array([min_x, min_z]),
                    'max': np.array([max_x, max_z]),
                    'corners': corners
                }
                
                print(f"Virtual environment 2D bounds loaded: "
                      f"Center=({center_x:.2f}, {center_z:.2f}), "
                      f"Size={width:.2f}m × {depth:.2f}m")
            
            # Parse 3D bounds - Unity exports at the same level as footprint2D
            # Try to get from top level first (new format)
            if 'center' in virt_env_bounds_data and 'size' in virt_env_bounds_data:
                bounds_3d_data = virt_env_bounds_data
            else:
                # Try old nested format
                bounds_3d_data = virt_env_bounds_data.get('bounds3D', {})
            
            if not bounds_3d_data or 'center' not in bounds_3d_data:
                print("WARNING: No valid 3D bounds in virtualEnvironmentBounds")
                self.virtual_env = None
                return
            
            bounds_3d = {
                'center': np.array([
                    bounds_3d_data.get('center', {}).get('x', 0.0),
                    bounds_3d_data.get('center', {}).get('y', 0.0),
                    bounds_3d_data.get('center', {}).get('z', 0.0)
                ]),
                'size': np.array([
                    bounds_3d_data.get('size', {}).get('x', 0.0),
                    bounds_3d_data.get('size', {}).get('y', 0.0),
                    bounds_3d_data.get('size', {}).get('z', 0.0)
                ]),
                'min': np.array([
                    bounds_3d_data.get('min', {}).get('x', 0.0),
                    bounds_3d_data.get('min', {}).get('y', 0.0),
                    bounds_3d_data.get('min', {}).get('z', 0.0)
                ]),
                'max': np.array([
                    bounds_3d_data.get('max', {}).get('x', 0.0),
                    bounds_3d_data.get('max', {}).get('y', 0.0),
                    bounds_3d_data.get('max', {}).get('z', 0.0)
                ])
            }
            
            self.virtual_env = VirtualEnvironment(
                center=virt_env_center,
                bounds_2d=bounds_2d,
                bounds_3d=bounds_3d
            )
            
            print(f"Virtual environment loaded: Center={virt_env_center}, "
                  f"2D Size={bounds_2d['size']}, 3D Size={bounds_3d['size']}")
            
        except Exception as e:
            print(f"Error processing virtual environment data: {e}")
            import traceback
            traceback.print_exc()
            self.virtual_env = None
    
    def _process_pedestal_groups(self, haptic_data: Dict) -> None:
        """Process pedestal groups and identify standalone interactables"""
        try:
            groups_data = haptic_data.get('groups', [])
            
            self.pedestal_groups = []
            grouped_interactables = set()
            
            for group in groups_data:
                pedestal_name = group.get('pedestal', None)
                object_names = group.get('objectNames', [])
                
                if not pedestal_name or not object_names:
                    continue
                
                # Find pedestal index
                pedestal_idx = None
                for idx, v_obj in enumerate(self.virtual_objects):
                    if v_obj.name == pedestal_name:
                        pedestal_idx = idx
                        break
                
                if pedestal_idx is None:
                    print(f"WARNING: Pedestal '{pedestal_name}' not found in virtual objects")
                    continue
                
                # Find interactable indices (exclude pedestal itself)
                interactable_indices = []
                for obj_name in object_names:
                    if obj_name == pedestal_name:
                        continue
                    for idx, v_obj in enumerate(self.virtual_objects):
                        if v_obj.name == obj_name:
                            interactable_indices.append(idx)
                            grouped_interactables.add(idx)
                            break
                
                # Get pedestal's 2D bounds for selection area
                pedestal_obj = self.virtual_objects[pedestal_idx]
                if pedestal_obj.bounds_2d:
                    selection_bounds = pedestal_obj.bounds_2d
                else:
                    print(f"WARNING: Pedestal '{pedestal_name}' has no 2D bounds")
                    selection_bounds = None
                
                pedestal_group = PedestalGroup(
                    pedestal_name=pedestal_name,
                    pedestal_index=pedestal_idx,
                    interactable_indices=interactable_indices,
                    selection_bounds_2d=selection_bounds
                )
                
                self.pedestal_groups.append(pedestal_group)
                print(f"Pedestal group: '{pedestal_name}' with {len(interactable_indices)} interactables")
            
            # Identify standalone interactables (not in any group, and are interactables)
            self.standalone_interactables = []
            for idx, v_obj in enumerate(self.virtual_objects):
                if (idx not in grouped_interactables and 
                    v_obj.involvement_type in ['grasp', 'contact', 'substrate']):
                    self.standalone_interactables.append(idx)
            
            print(f"Found {len(self.pedestal_groups)} pedestal groups and "
                  f"{len(self.standalone_interactables)} standalone interactables")
            
        except Exception as e:
            print(f"Error processing pedestal groups: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_average_physical_distance(self) -> float:
        """
        Calculate the average minimum distance between physical objects
        For each object, finds its closest neighbor, then averages these minimum distances
        Excludes banned physical objects from the calculation
        
        Returns:
            Average minimum distance in meters between physical objects and their closest neighbors
        """
        if len(self.physical_objects) < 2:
            print("WARNING: Less than 2 physical objects available for distance calculation")
            return PIN_GRID_SPACING  # Return default spacing
        
        # Get non-banned physical objects with valid positions
        valid_objects = []
        for p_idx, phys_obj in enumerate(self.physical_objects):
            if p_idx in self.banned_physical_indices:
                continue  # Skip banned objects
            if phys_obj.position is not None:
                valid_objects.append(phys_obj)
        
        if len(valid_objects) < 2:
            print(f"WARNING: Only {len(valid_objects)} non-banned physical object(s) with valid positions")
            return PIN_GRID_SPACING  # Return default spacing
        
        # For each object, find its minimum distance to any other object
        min_distances = []
        
        for i in range(len(valid_objects)):
            pos_i = valid_objects[i].position
            min_dist = float('inf')
            closest_neighbor = None
            
            # Find the closest neighbor to this object
            for j in range(len(valid_objects)):
                if i == j:
                    continue  # Skip self
                
                pos_j = valid_objects[j].position
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < min_dist:
                    min_dist = distance
                    closest_neighbor = valid_objects[j].name
            
            if min_dist != float('inf'):
                min_distances.append(min_dist)
        
        if len(min_distances) == 0:
            print("WARNING: No valid minimum distances calculated")
            return PIN_GRID_SPACING
        
        # Calculate average of minimum distances
        average_min_distance = sum(min_distances) / len(min_distances)
        
        # Calculate statistics for reporting
        max_min_dist = max(min_distances)
        min_min_dist = min(min_distances)
        
        print(f"\nAdaptive Pin Grid Spacing Calculation:")
        print(f"  Total physical objects: {len(self.physical_objects)}")
        print(f"  Banned physical objects: {len(self.banned_physical_indices)}")
        print(f"  Valid objects for calculation: {len(valid_objects)}")
        print(f"  Method: Average of each object's minimum distance to its closest neighbor")
        print(f"  Min distance range: {min_min_dist:.3f}m - {max_min_dist:.3f}m")
        print(f"  Average minimum distance: {average_min_distance:.3f}m")
        
        return average_min_distance
    
    def generate_pin_grid(self, spacing: float = PIN_GRID_SPACING) -> np.ndarray:
        """
        Generate pin points on the play area with specified spacing
        
        Args:
            spacing: Distance between pin points in meters
            
        Returns:
            Nx3 array of pin point positions
        """
        if self.play_area is None:
            print("WARNING: No play area available for pin grid generation")
            return np.array([[0, 0, 0]])  # Default to origin
        
        center = self.play_area.center
        width = self.play_area.width
        depth = self.play_area.depth
        
        # Generate grid from center outward
        # X direction (left-right)
        x_min = center[0] - width / 2
        x_max = center[0] + width / 2
        # Z direction (forward-backward)
        z_min = center[2] - depth / 2
        z_max = center[2] + depth / 2
        
        # Create grid points
        x_points = np.arange(center[0], x_max + spacing/2, spacing)
        x_points = np.concatenate([np.arange(center[0] - spacing, x_min - spacing/2, -spacing)[::-1], x_points])
        
        z_points = np.arange(center[2], z_max + spacing/2, spacing)
        z_points = np.concatenate([np.arange(center[2] - spacing, z_min - spacing/2, -spacing)[::-1], z_points])
        
        # Create meshgrid
        xx, zz = np.meshgrid(x_points, z_points)
        
        # Flatten to list of points
        pin_points = np.stack([xx.flatten(), np.full(xx.size, center[1]), zz.flatten()], axis=1)
        
        # Filter points to only those inside play area boundary (2D polygon test on XZ plane)
        if len(self.play_area.boundary_points) >= 3:
            boundary_2d = self.play_area.boundary_points[:, [0, 2]]  # X, Z only
            valid_pins = []
            for pin in pin_points:
                pin_2d = pin[[0, 2]]
                if self._point_in_polygon(pin_2d, boundary_2d):
                    valid_pins.append(pin)
            pin_points = np.array(valid_pins) if valid_pins else pin_points
        
        print(f"Generated {len(pin_points)} pin points with {spacing}m spacing")
        self.pin_points = pin_points
        return pin_points
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if a 2D point is inside a 2D polygon using ray casting"""
        x, z = point
        n = len(polygon)
        inside = False
        
        p1x, p1z = polygon[0]
        for i in range(1, n + 1):
            p2x, p2z = polygon[i % n]
            if z > min(p1z, p2z):
                if z <= max(p1z, p2z):
                    if x <= max(p1x, p2x):
                        if p1z != p2z:
                            xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1z = p2x, p2z
        
        return inside
    
    def _get_transformed_virtual_env_polygon(self, pin_point: np.ndarray, rotation_angle: float) -> np.ndarray:
        """
        Get the transformed virtual environment polygon in physical space
        
        Args:
            pin_point: 3D pin location
            rotation_angle: Rotation angle in radians
            
        Returns:
            Nx2 array of transformed polygon corners in XZ plane
        """
        if self.virtual_env is None:
            return None
        
        virt_center_2d = self.virtual_env.center[[0, 2]]
        pin_2d = pin_point[[0, 2]]
        translation_2d = pin_2d - virt_center_2d
        
        virt_corners_2d = np.array(self.virtual_env.bounds_2d['corners'])
        transformed_corners = []
        for corner in virt_corners_2d:
            rotated = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
            translated = rotated + translation_2d
            transformed_corners.append(translated)
        
        return np.array(transformed_corners)
    
    def generate_rotation_angles(self, step: float = ROTATION_STEP) -> np.ndarray:
        """
        Generate rotation angles from 0 to 360 degrees
        
        Args:
            step: Rotation step in degrees
            
        Returns:
            Array of rotation angles in radians
        """
        angles_deg = np.arange(0, 360, step)
        angles_rad = np.deg2rad(angles_deg)
        
        print(f"Generated {len(angles_rad)} rotation angles with {step}° step")
        self.rotation_angles = angles_rad
        return angles_rad
    
    def rotate_point_2d(self, point: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
        """
        Rotate a 2D point around a center
        
        Args:
            point: 2D point [x, z]
            angle: Rotation angle in radians
            center: 2D rotation center [x, z]
            
        Returns:
            Rotated 2D point [x, z]
        """
        # Translate to origin
        p = point - center
        
        # Rotate
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotated = np.array([
            p[0] * cos_a - p[1] * sin_a,
            p[0] * sin_a + p[1] * cos_a
        ])
        
        # Translate back
        return rotated + center
    
    def _get_objects_outside_intersection(self, pin_point: np.ndarray, rotation_angle: float) -> Set[int]:
        """
        Get physical objects that are outside the intersection of play area and virtual environment
        
        Args:
            pin_point: 3D pin location where virtual env center is placed
            rotation_angle: Rotation angle in radians
            
        Returns:
            Set of physical object indices that are outside the intersection
        """
        if self.play_area is None or self.virtual_env is None:
            return set()
        
        outside_indices = set()
        
        # Get play area boundary
        play_area_boundary = self.play_area.boundary_points[:, [0, 2]]
        
        # Get transformed virtual environment boundary
        transformed_virt_polygon = self._get_transformed_virtual_env_polygon(pin_point, rotation_angle)
        
        if transformed_virt_polygon is None:
            return set()
        
        # Check each physical object
        for p_idx, p_obj in enumerate(self.physical_objects):
            if p_obj.position is None:
                continue
            
            p_pos_2d = p_obj.position[[0, 2]]
            
            # Check if in BOTH play area AND virtual environment
            in_play_area = self._point_in_polygon(p_pos_2d, play_area_boundary)
            in_virtual_env = self._point_in_polygon(p_pos_2d, transformed_virt_polygon)
            
            # If NOT in both, it's outside the intersection
            if not (in_play_area and in_virtual_env):
                outside_indices.add(p_idx)
        
        return outside_indices
    
    def check_occlusion_for_config(self, pin_point: np.ndarray, rotation_angle: float) -> Set[int]:
        """
        Check which physical objects are occluded for a given pin & rotation configuration
        
        Occlusion behavior:
        - Pedestals: Controlled by ENABLE_PEDESTAL_OCCLUSION
            * True: Check 3D bounds (XYZ) - physical object must be within both height and 2D bounds
            * False: Pedestals do not block physical objects
        - Surroundings: Controlled by SURROUNDINGS_OCCLUSION_MODE
            * "3d" mode: Check full 3D bounds (XYZ) - same as pedestals
            * "2d" mode: Check only 2D bounds (XZ plane) - ignore height
        
        Args:
            pin_point: 3D pin location where virtual env center is placed
            rotation_angle: Rotation angle in radians
            
        Returns:
            Set of physical object indices that are occluded (inside virtual objects' bounds)
        """
        if self.virtual_env is None:
            return set()
        
        occluded_indices = set()
        
        # Calculate translation vector (from virtual env center to pin point)
        virt_center_2d = self.virtual_env.center[[0, 2]]  # X, Z only
        pin_2d = pin_point[[0, 2]]
        translation_2d = pin_2d - virt_center_2d
        
        # Check each physical object against all virtual objects' 3D bounds
        for p_idx, p_obj in enumerate(self.physical_objects):
            if p_obj.position is None:
                continue
            
            p_pos_2d = p_obj.position[[0, 2]]  # X, Z only for rotation
            p_pos_y = p_obj.position[1]  # Y coordinate for 3D check
            
            # Check against each virtual object's 3D bounds
            for v_obj in self.virtual_objects:
                # Only check set dressing (pedestals, surroundings) for occlusion
                involvement = v_obj.involvement_type
                
                # Skip if not set dressing
                if involvement not in ['pedestal', 'surroundings']:
                    continue
                
                # Skip pedestal occlusion if disabled
                if involvement == 'pedestal' and not ENABLE_PEDESTAL_OCCLUSION:
                    continue
                
                if v_obj.bounds_3d is None:
                    continue
                
                # Determine if we should check height based on object type and configuration
                check_height = True
                if v_obj.involvement_type == 'surroundings' and SURROUNDINGS_OCCLUSION_MODE == "2d":
                    check_height = False  # Skip height check for surroundings in 2D mode
                
                # Y check (height) - only if required
                if check_height:
                    v_bounds = v_obj.bounds_3d
                    v_center_y = v_bounds['center'][1]
                    half_size_y = v_bounds['size'][1] / 2
                    dy = abs(p_pos_y - v_center_y)
                    
                    if dy > half_size_y:
                        continue  # Not within height bounds
                
                # 2D check (XZ plane) - use rotated corners for accurate containment
                if v_obj.bounds_2d and 'corners' in v_obj.bounds_2d:
                    corners = v_obj.bounds_2d['corners']
                    
                    if len(corners) >= 3:
                        # Transform corners with rotation and translation
                        transformed_corners = []
                        for corner in corners:
                            rotated = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
                            transformed = rotated + translation_2d
                            transformed_corners.append(transformed)
                        
                        # Point-in-polygon test
                        polygon = MplPath(transformed_corners)
                        if polygon.contains_point(p_pos_2d):
                            occluded_indices.add(p_idx)
                            break  # No need to check other virtual objects
        
        return occluded_indices
    
    def get_selection_pool_for_interactable(self, v_idx: int, pin_point: np.ndarray, 
                                           rotation_angle: float) -> Set[int]:
        """
        Get the selection pool (allowed physical objects) for a virtual interactable
        
        Args:
            v_idx: Virtual object index
            pin_point: 3D pin location
            rotation_angle: Rotation angle in radians
            
        Returns:
            Set of physical object indices in the selection pool
        """
        # Check if this interactable is in a pedestal group
        for group in self.pedestal_groups:
            if v_idx in group.interactable_indices:
                # Use pedestal's 2D bounds as selection area
                return self._get_pool_from_pedestal_bounds(group, pin_point, rotation_angle)
        
        # Check if it's a standalone interactable
        if v_idx in self.standalone_interactables:
            # Use circular selection area
            return self._get_pool_from_circular_area(v_idx, pin_point, rotation_angle)
        
        # Not an interactable or no selection pool defined - return all physical objects
        return set(range(len(self.physical_objects)))
    
    def _get_pool_from_pedestal_bounds(self, group: PedestalGroup, pin_point: np.ndarray, 
                                      rotation_angle: float) -> Set[int]:
        """Get physical objects within pedestal's 2D bounds AND in the overlapping play area/virtual env region
        
        Behavior depends on SPATIAL_POOL_MODE:
        - "strict": Objects must be in pedestal bounds AND overlapping region (hard constraint)
        - "relaxed": Objects only need to be in overlapping region (soft constraint via L_spatial)
        """
        if group.selection_bounds_2d is None:
            return set(range(len(self.physical_objects)))
        
        pool = set()
        
        # Get play area and transformed virtual environment boundaries for overlap check
        play_area_boundary = self.play_area.boundary_points[:, [0, 2]] if self.play_area is not None else None
        transformed_virt_polygon = self._get_transformed_virtual_env_polygon(pin_point, rotation_angle)
        
        # RELAXED MODE: Return all objects in overlapping region (ignore pedestal bounds)
        # BUT exclude occluded objects (inside virtual pedestals/surroundings)
        if SPATIAL_POOL_MODE == "relaxed":
            if play_area_boundary is None or transformed_virt_polygon is None:
                print(f"WARNING: Boundaries not available in relaxed mode for pedestal group!")
                print(f"  play_area_boundary: {play_area_boundary is not None}")
                print(f"  transformed_virt_polygon: {transformed_virt_polygon is not None}")
                # Return empty pool instead of all objects when boundaries are missing
                return set()
            
            # Get occluded objects for this configuration
            occluded_indices = self.check_occlusion_for_config(pin_point, rotation_angle)
            
            for p_idx, p_obj in enumerate(self.physical_objects):
                if p_obj.position is None:
                    continue
                
                # CRITICAL: Exclude occluded objects (inside virtual pedestals/surroundings)
                if p_idx in occluded_indices:
                    continue
                
                p_pos_2d = p_obj.position[[0, 2]]
                
                in_play_area = self._point_in_polygon(p_pos_2d, play_area_boundary)
                in_virtual_env = self._point_in_polygon(p_pos_2d, transformed_virt_polygon)
                
                if in_play_area and in_virtual_env:
                    pool.add(p_idx)
            return pool
        
        # STRICT MODE: Check pedestal bounds AND overlapping region (original implementation)
        # Get pedestal bounds
        bounds_2d = group.selection_bounds_2d
        corners = bounds_2d.get('corners', [])
        
        if len(corners) < 3:
            # Fallback: use center and size to create rectangle
            center = bounds_2d['center']
            size = bounds_2d['size']
            half_x = size[0] / 2
            half_z = size[1] / 2
            corners = [
                center + np.array([-half_x, -half_z]),
                center + np.array([half_x, -half_z]),
                center + np.array([half_x, half_z]),
                center + np.array([-half_x, half_z])
            ]
        
        # Transform corners to physical space
        virt_center_2d = self.virtual_env.center[[0, 2]]
        pin_2d = pin_point[[0, 2]]
        translation_2d = pin_2d - virt_center_2d
        
        transformed_corners = []
        for corner in corners:
            rotated = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
            transformed = rotated + translation_2d
            transformed_corners.append(transformed)
        
        transformed_polygon = np.array(transformed_corners)
        
        # Check which physical objects are inside this polygon
        for p_idx, p_obj in enumerate(self.physical_objects):
            if p_obj.position is None:
                continue
            p_pos_2d = p_obj.position[[0, 2]]
            
            # First check if inside pedestal bounds
            if not self._point_in_polygon(p_pos_2d, transformed_polygon):
                continue
            
            # CRITICAL FIX: Also check if physical object is in the overlapping region
            # of play area AND transformed virtual environment
            if play_area_boundary is not None and transformed_virt_polygon is not None:
                in_play_area = self._point_in_polygon(p_pos_2d, play_area_boundary)
                in_virtual_env = self._point_in_polygon(p_pos_2d, transformed_virt_polygon)
                
                # Only add to pool if in BOTH play area AND virtual environment
                if in_play_area and in_virtual_env:
                    pool.add(p_idx)
            else:
                # WARNING: Boundaries not available - this shouldn't happen in strict mode
                # Falling back to pedestal bounds only (may include objects outside overlapping region)
                if play_area_boundary is None or transformed_virt_polygon is None:
                    print(f"WARNING: Boundaries not fully available in strict mode (pedestal group)")
                pool.add(p_idx)
        
        return pool
    
    def _get_pool_from_circular_area(self, v_idx: int, pin_point: np.ndarray, 
                                    rotation_angle: float, radius: float = STANDALONE_SELECTION_RADIUS) -> Set[int]:
        """Get physical objects within circular area around virtual object AND in the overlapping play area/virtual env region
        
        Behavior depends on SPATIAL_POOL_MODE:
        - "strict": Objects must be within circular radius AND overlapping region (hard constraint)
        - "relaxed": Objects only need to be in overlapping region (soft constraint via L_spatial)
        """
        v_obj = self.virtual_objects[v_idx]
        if v_obj.position is None:
            return set(range(len(self.physical_objects)))
        
        pool = set()
        
        # Get play area and transformed virtual environment boundaries for overlap check
        play_area_boundary = self.play_area.boundary_points[:, [0, 2]] if self.play_area is not None else None
        transformed_virt_polygon = self._get_transformed_virtual_env_polygon(pin_point, rotation_angle)
        
        # RELAXED MODE: Return all objects in overlapping region (ignore circular radius)
        # BUT exclude occluded objects (inside virtual pedestals/surroundings)
        if SPATIAL_POOL_MODE == "relaxed":
            if play_area_boundary is None or transformed_virt_polygon is None:
                print(f"WARNING: Boundaries not available in relaxed mode for standalone interactable!")
                print(f"  play_area_boundary: {play_area_boundary is not None}")
                print(f"  transformed_virt_polygon: {transformed_virt_polygon is not None}")
                # Return empty pool instead of all objects when boundaries are missing
                return set()
            
            # Get occluded objects for this configuration
            occluded_indices = self.check_occlusion_for_config(pin_point, rotation_angle)
            
            for p_idx, p_obj in enumerate(self.physical_objects):
                if p_obj.position is None:
                    continue
                
                # CRITICAL: Exclude occluded objects (inside virtual pedestals/surroundings)
                if p_idx in occluded_indices:
                    continue
                
                p_pos_2d = p_obj.position[[0, 2]]
                
                in_play_area = self._point_in_polygon(p_pos_2d, play_area_boundary)
                in_virtual_env = self._point_in_polygon(p_pos_2d, transformed_virt_polygon)
                
                if in_play_area and in_virtual_env:
                    pool.add(p_idx)
            return pool
        
        # STRICT MODE: Check circular radius AND overlapping region (original implementation)
        # Transform virtual object position to physical space
        v_pos_2d = v_obj.position[[0, 2]]
        virt_center_2d = self.virtual_env.center[[0, 2]]
        pin_2d = pin_point[[0, 2]]
        translation_2d = pin_2d - virt_center_2d
        
        rotated_v_pos = self.rotate_point_2d(v_pos_2d, rotation_angle, virt_center_2d)
        transformed_v_pos = rotated_v_pos + translation_2d
        
        # Check which physical objects are within radius
        for p_idx, p_obj in enumerate(self.physical_objects):
            if p_obj.position is None:
                continue
            p_pos_2d = p_obj.position[[0, 2]]
            distance = np.linalg.norm(p_pos_2d - transformed_v_pos)
            
            # First check if within circular radius
            if distance > radius:
                continue
            
            # CRITICAL FIX: Also check if physical object is in the overlapping region
            # of play area AND transformed virtual environment
            if play_area_boundary is not None and transformed_virt_polygon is not None:
                in_play_area = self._point_in_polygon(p_pos_2d, play_area_boundary)
                in_virtual_env = self._point_in_polygon(p_pos_2d, transformed_virt_polygon)
                
                # Only add to pool if in BOTH play area AND virtual environment
                if in_play_area and in_virtual_env:
                    pool.add(p_idx)
            else:
                # WARNING: Boundaries not available - this shouldn't happen in strict mode
                # Falling back to circular area only (may include objects outside overlapping region)
                if play_area_boundary is None or transformed_virt_polygon is None:
                    print(f"WARNING: Boundaries not fully available in strict mode (standalone interactable)")
                pool.add(p_idx)
        
        return pool
    
    def _build_realism_matrix(self, proxy_data: List[Dict]) -> None:
        """Build the realism rating matrix from proxy matching results"""
        n_virtual = len(self.virtual_objects)
        n_physical = len(self.physical_objects)
        self.realism_matrix = np.zeros((n_virtual, n_physical))
        
        # Create mapping from virtual object name to index
        virtual_name_to_index = {obj.name: obj.index for obj in self.virtual_objects}
        
        # Create mapping from (object_id, image_id) to physical object index
        physical_id_to_index = {(obj.object_id, obj.image_id): obj.index 
                               for obj in self.physical_objects}
        
        for proxy_result in proxy_data:
            virtual_name = proxy_result.get("virtualObject", "")
            object_id = proxy_result.get("object_id", -1)
            image_id = proxy_result.get("image_id", -1)
            rating_score = proxy_result.get("rating_score", 0.0)
            
            if virtual_name in virtual_name_to_index:
                virtual_idx = virtual_name_to_index[virtual_name]
                
                if (object_id, image_id) in physical_id_to_index:
                    physical_idx = physical_id_to_index[(object_id, image_id)]
                    self.realism_matrix[virtual_idx, physical_idx] = rating_score
    
    def _build_interaction_matrices(self, haptic_data: Dict, relationship_data: List[Dict]) -> None:
        """Build interaction matrices from relationship rating data"""
        n_virtual = len(self.virtual_objects)
        n_physical = len(self.physical_objects)
        
        # Initialize interaction_exists matrix for virtual objects
        self.interaction_exists = np.zeros((n_virtual, n_virtual))
        
        # Build virtual object name to index mapping
        virtual_name_to_index = {obj.name: obj.index for obj in self.virtual_objects}
        
        # Mark which virtual objects have interactions and create relationship mapping
        relationship_annotations = haptic_data.get("relationshipAnnotations", [])
        virtual_relationship_pairs = []  # Store (contact_idx, substrate_idx) tuples
        
        for rel in relationship_annotations:
            contact_name = rel.get("contactObject", "")
            substrate_name = rel.get("substrateObject", "")
            
            if (contact_name in virtual_name_to_index and 
                substrate_name in virtual_name_to_index):
                contact_idx = virtual_name_to_index[contact_name]
                substrate_idx = virtual_name_to_index[substrate_name]
                self.interaction_exists[contact_idx, substrate_idx] = 1.0
                virtual_relationship_pairs.append((contact_idx, substrate_idx))
                # Note: This is directional (contact -> substrate)
        
        n_relationships = len(virtual_relationship_pairs)
        print(f"Found {n_relationships} virtual object relationships")
        
        # Initialize 3D interaction rating matrix: [relationship_idx, contact_phys, substrate_phys]
        self.interaction_matrix_3d = np.zeros((n_relationships, n_physical, n_physical))
        self.virtual_relationship_pairs = virtual_relationship_pairs  # Store for later use
        
        # Build physical object (object_id, image_id) to index mapping
        physical_id_to_index = {(obj.object_id, obj.image_id): obj.index 
                               for obj in self.physical_objects}
        
        # Build virtual relationship name pair to index mapping
        virtual_relationship_name_to_index = {}
        for rel_idx, (contact_idx, substrate_idx) in enumerate(virtual_relationship_pairs):
            contact_name = self.virtual_objects[contact_idx].name
            substrate_name = self.virtual_objects[substrate_idx].name
            virtual_relationship_name_to_index[(contact_name, substrate_name)] = rel_idx
        
        # Process relationship rating data to build 3D interaction matrix
        # The data structure is now organized by dimension: {"harmony": [...], "expressivity": [...], "realism": [...]}
        ratings_processed = 0
        
        # Create a mapping to collect ratings by (virtual_pair, contact_phys, substrate_phys)
        rating_combinations = {}
        
        # Process each dimension separately
        for dimension in ["harmony", "expressivity", "realism"]:
            if dimension not in relationship_data:
                continue
                
            for rel_result in relationship_data[dimension]:
                virtual_contact = rel_result.get("virtualContactObject", "")
                virtual_substrate = rel_result.get("virtualSubstrateObject", "")
                contact_obj_id = rel_result.get("contactObject_id", -1)
                contact_img_id = rel_result.get("contactImage_id", -1)
                substrate_obj_id = rel_result.get("substrateObject_id", -1)
                substrate_img_id = rel_result.get("substrateImage_id", -1)
                rating = rel_result.get("rating", 0)
                
                # Find the relationship index
                virtual_pair_key = (virtual_contact, virtual_substrate)
                if virtual_pair_key not in virtual_relationship_name_to_index:
                    continue
                    
                rel_idx = virtual_relationship_name_to_index[virtual_pair_key]
                
                # Map to physical object indices
                contact_key = (contact_obj_id, contact_img_id)
                substrate_key = (substrate_obj_id, substrate_img_id)
                
                if (contact_key in physical_id_to_index and 
                    substrate_key in physical_id_to_index):
                    contact_phys_idx = physical_id_to_index[contact_key]
                    substrate_phys_idx = physical_id_to_index[substrate_key]
                    
                    # Create key for this specific combination
                    combination_key = (rel_idx, contact_phys_idx, substrate_phys_idx)
                    
                    # Initialize if not exists
                    if combination_key not in rating_combinations:
                        rating_combinations[combination_key] = {"harmony": 0, "expressivity": 0, "realism": 0}
                    
                    # Store the rating for this dimension
                    rating_combinations[combination_key][dimension] = rating
        
        # Now calculate combined ratings and populate the 3D matrix
        for combination_key, ratings in rating_combinations.items():
            rel_idx, contact_phys_idx, substrate_phys_idx = combination_key
            
            harmony_rating = ratings["harmony"]
            expressivity_rating = ratings["expressivity"]
            realism_rating = ratings["realism"]
            
            # Use geometric mean (cube root of product) instead of arithmetic sum
            # This ensures all three dimensions must be good for a high combined rating
            if harmony_rating > 0 and expressivity_rating > 0 and realism_rating > 0:
                combined_rating = (harmony_rating * expressivity_rating * realism_rating) ** (1/3)
            else:
                # If any rating is 0, the combined rating is 0
                combined_rating = 0.0
            
            self.interaction_matrix_3d[rel_idx, contact_phys_idx, substrate_phys_idx] = combined_rating
            ratings_processed += 1
        
        print(f"Processed {ratings_processed} interaction ratings into 3D matrix")
    
    def _build_distance_matrices(self) -> None:
        """Pre-compute distance and angle matrices for spatial loss"""
        # Virtual objects
        n_virtual = len(self.virtual_objects)
        self.virtual_distance_matrix = np.zeros((n_virtual, n_virtual))
        self.virtual_angle_matrix = np.zeros((n_virtual, n_virtual))
        
        print(f"Building distance matrices for {n_virtual} virtual objects...")
        
        # Check position status before building matrices
        objects_with_positions = []
        objects_without_positions = []
        for i, obj in enumerate(self.virtual_objects):
            if obj.position is not None:
                objects_with_positions.append(f"{obj.name} (index {i})")
            else:
                objects_without_positions.append(f"{obj.name} (index {i})")
        
        if objects_with_positions:
            print(f"  Objects WITH positions ({len(objects_with_positions)}): {', '.join(objects_with_positions)}")
        if objects_without_positions:
            print(f"  Objects WITHOUT positions ({len(objects_without_positions)}): {', '.join(objects_without_positions)}")
        
        # Build distance matrices
        distances_calculated = 0
        for i in range(n_virtual):
            pos_i = self.virtual_objects[i].position
            if pos_i is None:
                print(f"  Skipping {self.virtual_objects[i].name} (index {i}) - no position")
                continue
            for k in range(n_virtual):
                pos_k = self.virtual_objects[k].position
                if pos_k is None:
                    continue
                diff = pos_k - pos_i
                self.virtual_distance_matrix[i, k] = float(np.linalg.norm(diff))
                self.virtual_angle_matrix[i, k] = float(np.arctan2(diff[2], diff[0]))
                if i != k:  # Don't count diagonal (self-distance)
                    distances_calculated += 1
        
        print(f"  Calculated {distances_calculated} non-zero distances")
        print(f"  Expected: {n_virtual} × {n_virtual} - {n_virtual} = {(n_virtual * n_virtual) - n_virtual}")
        
        # Physical objects
        n_physical = len(self.physical_objects)
        self.physical_distance_matrix = np.zeros((n_physical, n_physical))
        self.physical_angle_matrix = np.zeros((n_physical, n_physical))
        for i in range(n_physical):
            pos_i = self.physical_objects[i].position
            if pos_i is None:
                continue
            for k in range(n_physical):
                pos_k = self.physical_objects[k].position
                if pos_k is None:
                    continue
                diff = pos_k - pos_i
                self.physical_distance_matrix[i, k] = float(np.linalg.norm(diff))
                self.physical_angle_matrix[i, k] = float(np.arctan2(diff[2], diff[0]))
        
        # Spatial group matrix: default no spatial relationships (all zeros)
        # Only explicit spatial groups should have non-zero values
        self.spatial_group_matrix = np.zeros((n_virtual, n_virtual))
        
        # Process spatial groups from haptic annotation data if available
        # We need to access the haptic data that was loaded earlier
        if hasattr(self, 'haptic_data') and self.haptic_data is not None:
            self._process_spatial_groups(self.haptic_data)
        else:
            # Try to find haptic data from the output directory
            self._try_load_haptic_data_for_spatial_groups()
    
    def _try_load_haptic_data_for_spatial_groups(self) -> None:
        """Try to load haptic annotation data to process spatial groups"""
        try:
            # Look for haptic annotation files in the output directory
            haptic_files = [f for f in os.listdir(self.data_dir) if f.startswith("haptic_annotation") and f.endswith(".json")]
            
            if haptic_files:
                # Use the most recent file
                haptic_file = os.path.join(self.data_dir, sorted(haptic_files)[-1])
                print(f"Loading haptic annotation for spatial groups: {haptic_file}")
                
                with open(haptic_file, 'r') as f:
                    haptic_data = json.load(f)
                
                self._process_spatial_groups(haptic_data)
            else:
                print("No haptic annotation files found for spatial group processing")
        except Exception as e:
            print(f"Error loading haptic data for spatial groups: {e}")
    
    def _process_spatial_groups(self, haptic_data: Dict) -> None:
        """Process spatial groups from haptic annotation data to populate spatial_group_matrix"""
        groups = haptic_data.get("groups", [])
        if not groups:
            print("No spatial groups found in haptic annotation data")
            return
        
        # Create mapping from virtual object name to index
        virtual_name_to_index = {obj.name: obj.index for obj in self.virtual_objects}
        
        spatial_relationships_found = 0
        
        for group in groups:
            group_title = group.get("title", "Unknown")
            object_names = group.get("objectNames", [])
            object_vectors = group.get("objectVectors", [])
            
            # Count valid objects and relationships for this group
            valid_objects = []
            group_relationships = 0
            
            # Mark all objects in the group as spatially related to each other
            for i, obj_name_i in enumerate(object_names):
                if obj_name_i not in virtual_name_to_index:
                    print(f"  Warning: Virtual object '{obj_name_i}' not found in virtual objects list")
                    continue
                idx_i = virtual_name_to_index[obj_name_i]
                valid_objects.append(obj_name_i)
                
                for j, obj_name_j in enumerate(object_names):
                    if obj_name_j not in virtual_name_to_index:
                        continue
                    idx_j = virtual_name_to_index[obj_name_j]
                    
                    # Mark spatial relationship (symmetric)
                    if idx_i != idx_j:
                        self.spatial_group_matrix[idx_i, idx_j] = 1.0
                        self.spatial_group_matrix[idx_j, idx_i] = 1.0
                        spatial_relationships_found += 1
                        group_relationships += 1
            
            # Print group summary
            print(f"Spatial group '{group_title}': {len(valid_objects)} objects, {group_relationships} relationships")
            print(f"  Objects: {', '.join(valid_objects)}")
            
            # Process object vectors for distance validation (optional)
            for vector_info in object_vectors:
                obj_a = vector_info.get("objectA", "")
                obj_b = vector_info.get("objectB", "")
                expected_distance = vector_info.get("distance", 0.0)
                
                if obj_a in virtual_name_to_index and obj_b in virtual_name_to_index:
                    idx_a = virtual_name_to_index[obj_a]
                    idx_b = virtual_name_to_index[obj_b]
                    
                    # Verify that the computed distance matches the expected distance
                    if (self.virtual_objects[idx_a].position is not None and 
                        self.virtual_objects[idx_b].position is not None):
                        computed_distance = self.virtual_distance_matrix[idx_a, idx_b]
                        distance_diff = abs(computed_distance - expected_distance)
                        
                        if distance_diff > 0.01:  # 1cm tolerance
                            print(f"Warning: Distance mismatch for {obj_a}->{obj_b}: "
                                  f"computed={computed_distance:.3f}, expected={expected_distance:.3f}")
        
        print(f"Found {spatial_relationships_found} spatial relationships in {len(groups)} groups")
        
        # Debug: show which virtual objects have spatial relationships
        spatial_objects = set()
        for i in range(len(self.virtual_objects)):
            for j in range(len(self.virtual_objects)):
                if self.spatial_group_matrix[i, j] > 0:
                    spatial_objects.add(self.virtual_objects[i].name)
                    spatial_objects.add(self.virtual_objects[j].name)
        
        if spatial_objects:
            print(f"Spatially related virtual objects: {', '.join(sorted(spatial_objects))}")
        else:
            print("No spatial relationships found - L_spatial will be 0")
    
    def calculate_realism_loss(self, assignment_matrix: np.ndarray) -> float:
        """Calculate combined L_realism = -∑ᵢ∑ⱼ (2 × priority_weight[i] × realism_rating[i,j] × X[i,j]) for grasp and contact objects only"""
        if self.realism_matrix is None:
            return 0.0
            
        loss = 0.0
        n_virtual = len(self.virtual_objects)
        
        # Count grasp and contact objects (exclude substrate objects)
        grasp_contact_count = 0
        
        for i in range(n_virtual):
            virtual_obj = self.virtual_objects[i]
            
            # Skip substrate objects - only calculate realism loss for grasp and contact objects
            if virtual_obj.involvement_type == "substrate":
                continue
            
            # Get priority weight for this virtual object
            if self.enable_priority_weighting:
                priority_weight = virtual_obj.engagement_level
            else:
                priority_weight = 1.0  # Equal priority when disabled
            
            # Calculate realism loss for this virtual object
            for j in range(len(self.physical_objects)):
                if assignment_matrix[i, j] > 0:  # If virtual object i is assigned to physical object j
                    realism_rating = self.realism_matrix[i, j]
                    # Apply 2 × priority weight to realism rating
                    weighted_loss = 2 * priority_weight * realism_rating
                    loss -= weighted_loss
                    
                    # # Debug: show first few realism calculations
                    # if grasp_contact_count < 3:
                    #     physical_obj_name = self.physical_objects[j].name
                    #     print(f"DEBUG: Realism {virtual_obj.name} -> {physical_obj_name}: "
                    #           f"priority={priority_weight:.3f}, rating={realism_rating:.3f}, "
                    #           f"2×weighted={weighted_loss:.3f}")
            
            # Count grasp and contact objects (exclude substrate)
            if virtual_obj.involvement_type in ['grasp', 'contact']:
                grasp_contact_count += 1
        
        # Normalize by the count of grasp and contact objects
        if grasp_contact_count > 0:
            loss = loss / grasp_contact_count
        
        return float(loss)
    
    def calculate_interaction_loss(self, assignment_matrix: np.ndarray, verbose: bool = False) -> float:
        """Calculate L_interaction = -∑ᵢ∑ₖ (interaction_exists[i,k] × interaction_rating[proxy_assigned[i], proxy_assigned[k]] × combined_priority_weight[i,k])"""
        if self.interaction_exists is None:
            return 0.0
            
        # Use 3D matrix for accurate interaction calculation
        if hasattr(self, 'interaction_matrix_3d') and hasattr(self, 'virtual_relationship_pairs'):
            return self._calculate_interaction_loss_3d(assignment_matrix, verbose)
        else:
            return 0.0
    
    def _calculate_interaction_loss_3d(self, assignment_matrix: np.ndarray, verbose: bool = False) -> float:
        """Calculate interaction loss using 3D interaction matrix"""
        if not hasattr(self, 'interaction_matrix_3d') or not hasattr(self, 'virtual_relationship_pairs'):
            return 0.0
        if self.interaction_matrix_3d is None or self.virtual_relationship_pairs is None or self.interaction_exists is None:
            return 0.0
            
        loss = 0.0
        n_virtual = len(self.virtual_objects)
        interaction_relationships_count = 0
        
        if verbose:
            print("\n" + "="*60)
            print("L_INTERACTION CALCULATION DETAILS")
            print("="*60)
            print("Formula: L_interaction = -∑ᵢ∑ₖ (interaction_exists[i,k] × interaction_rating[proxy_assigned[i], proxy_assigned[k]] × combined_priority_weight[i,k])")
            print("Where combined_priority_weight[i,k] = priority_weight[i] + priority_weight[k]")
            print()
        
        # Iterate through each virtual relationship
        for rel_idx, (contact_virtual_idx, substrate_virtual_idx) in enumerate(self.virtual_relationship_pairs):
            if self.interaction_exists[contact_virtual_idx, substrate_virtual_idx] > 0:
                # Find assigned physical objects for this virtual relationship
                proxy_contact = np.argmax(assignment_matrix[contact_virtual_idx, :])
                proxy_substrate = np.argmax(assignment_matrix[substrate_virtual_idx, :])
                
                # Get interaction rating from 3D matrix for this specific relationship
                interaction_rating = self.interaction_matrix_3d[rel_idx, proxy_contact, proxy_substrate]
                
                # Calculate the sum of contact and substrate virtual objects' priority weights
                if self.enable_priority_weighting:
                    # Contact object uses its primary engagement_level
                    contact_priority_weight = self.virtual_objects[contact_virtual_idx].engagement_level
                    # Substrate object uses its substrate-specific engagement_level
                    # This allows dual-role objects (e.g., contact type used as substrate) 
                    # to have different priorities in their substrate role
                    substrate_priority_weight = self.virtual_objects[substrate_virtual_idx].substrate_engagement_level
                else:
                    contact_priority_weight = 1.0
                    substrate_priority_weight = 1.0
                combined_priority_weight = contact_priority_weight + substrate_priority_weight
                
                # Calculate contribution to loss for this relationship
                relationship_loss = interaction_rating * combined_priority_weight
                loss -= relationship_loss
                interaction_relationships_count += 1
                
                if verbose:
                    contact_name = self.virtual_objects[contact_virtual_idx].name
                    substrate_name = self.virtual_objects[substrate_virtual_idx].name
                    contact_phys_name = self.physical_objects[proxy_contact].name
                    substrate_phys_name = self.physical_objects[proxy_substrate].name
                    
                    print(f"Relationship {rel_idx}: {contact_name} -> {substrate_name}")
                    print(f"  Virtual Objects: {contact_name} (contact) -> {substrate_name} (substrate)")
                    print(f"  Assigned Physical Objects: {contact_phys_name} -> {substrate_phys_name}")
                    print(f"  Contact Priority Weight: {contact_priority_weight:.3f}")
                    print(f"  Substrate Priority Weight: {substrate_priority_weight:.3f}")
                    print(f"  Combined Priority Weight: {combined_priority_weight:.3f}")
                    print(f"  Interaction Rating: {interaction_rating:.3f}")
                    print(f"  Relationship Loss Contribution: {relationship_loss:.3f}")
                    print(f"  Running Total Loss: {loss:.3f}")
                    print()
        
        # Normalize by the count of interaction relationships
        if interaction_relationships_count > 0:
            loss = loss / interaction_relationships_count
            if verbose:
                print(f"Normalization: Divided by {interaction_relationships_count} relationships")
                print(f"Final L_interaction: {loss:.3f}")
                print("="*60)
        
        return loss
    
    def calculate_spatial_loss(self, assignment_matrix: np.ndarray) -> float:
        """[DEPRECATED] Old distance-based spatial loss - no longer used.
        
        This function has been replaced by calculate_spatial_loss_with_pool() which uses
        selection pool constraints instead of distance distortions.
        
        Kept for reference only - not called by any optimization methods.
        """
        if (self.spatial_group_matrix is None or
            self.virtual_distance_matrix is None or self.physical_distance_matrix is None):
            print("DEBUG: Spatial loss returning 0 - missing required matrices")
            return 0.0
        
        # Cast to local vars for clarity
        virtual_dist_mat = self.virtual_distance_matrix
        physical_dist_mat = self.physical_distance_matrix
        n_virtual = len(self.virtual_objects)
        assigned_physical = np.argmax(assignment_matrix, axis=1)
        loss = 0.0
        relationships_processed = 0
        
        for i in range(n_virtual):
            for k in range(n_virtual):
                if i == k or self.spatial_group_matrix[i, k] == 0:
                    continue
                
                v_dist = virtual_dist_mat[i, k]
                p_dist = physical_dist_mat[assigned_physical[i], assigned_physical[k]]
                dist_diff = v_dist - p_dist
                
                # Calculate the combined priority weight for this spatial relationship
                # Similar to interaction loss: combined_priority_weight[i,k] = priority_weight[i] + priority_weight[k]
                priority_weight_i = self.virtual_objects[i].engagement_level
                priority_weight_k = self.virtual_objects[k].engagement_level
                combined_priority_weight = priority_weight_i + priority_weight_k
                
                # Apply combined priority weight to the squared distance difference
                weighted_loss = (dist_diff ** 2) * combined_priority_weight
                loss += weighted_loss
                relationships_processed += 1
                
                # # Debug: show first few relationships
                # if relationships_processed <= 3:
                #     print(f"DEBUG: Spatial relationship {self.virtual_objects[i].name}->{self.virtual_objects[k].name}: "
                #           f"virtual_dist={v_dist:.3f}, physical_dist={p_dist:.3f}, diff={dist_diff:.3f}, "
                #           f"priority_i={priority_weight_i:.3f}, priority_k={priority_weight_k:.3f}, "
                #           f"combined_priority={combined_priority_weight:.3f}, weighted_loss={weighted_loss:.6f}")
        
        # Normalize by the count of spatial relationships
        if relationships_processed > 0:
            loss = loss / relationships_processed
        
        # print(f"DEBUG: Processed {relationships_processed} spatial relationships, total loss: {loss:.6f}")
        return float(loss)
    
    def calculate_spatial_loss_with_pool(self, assignment_matrix: np.ndarray, pin_point: np.ndarray, 
                                        rotation_angle: float) -> float:
        """
        Calculate NEW spatial loss based on selection pool constraint (mode-aware)
        
        STRICT mode: Penalizes assignments where physical objects are outside the selection pool
        RELAXED mode: Penalizes assignments based on distance from pedestal/circular bounds
        
        Penalty: exp(priority_weight * di / dmax)
        where di is the distance to the transformed virtual position, dmax is the longest
        edge of the virtual environment, and priority_weight is the engagement level of the object.
        
        Args:
            assignment_matrix: Current assignment matrix
            pin_point: Current pin location
            rotation_angle: Current rotation angle
            
        Returns:
            Spatial loss value (0 = all assignments within bounds, higher = more violations)
        """
        # NOTE: Removed enable_spatial_constraint check - spatial loss should be calculated
        # independently of whether pin & rotate optimization is used
        # It is controlled by w_spatial weight instead
        
        if self.virtual_env is None:
            return 0.0
        
        # Calculate dmax: longest edge of the virtual environment
        if 'size' not in self.virtual_env.bounds_3d:
            return 0.0
        
        virt_env_size = self.virtual_env.bounds_3d['size']
        if isinstance(virt_env_size, np.ndarray):
            dmax = float(np.max(virt_env_size))
        elif isinstance(virt_env_size, (list, tuple)):
            dmax = max(virt_env_size)
        else:
            return 0.0
        
        if dmax == 0:
            return 0.0
        
        loss = 0.0
        
        # Get assignments
        assigned_physical = np.argmax(assignment_matrix, axis=1)
        
        # Check each interactable's assignment
        for v_idx, v_obj in enumerate(self.virtual_objects):
            # Only apply to interactables (grasp, contact, substrate)
            if v_obj.involvement_type not in ['grasp', 'contact', 'substrate']:
                continue
            
            # Get the assigned physical object
            p_idx = assigned_physical[v_idx]
            p_obj = self.physical_objects[p_idx]
            
            if p_obj.position is None or v_obj.position is None:
                continue
            
            p_pos_2d = p_obj.position[[0, 2]]
            distance_to_target = None  # Initialize for this iteration
            
            # MODE-AWARE PENALTY CALCULATION
            if SPATIAL_POOL_MODE == "strict":
                # STRICT MODE: Penalize if outside selection pool (hard constraint already enforced)
                selection_pool = self.get_selection_pool_for_interactable(v_idx, pin_point, rotation_angle)
                
                # If assigned physical object is in the pool, no penalty
                if p_idx in selection_pool:
                    continue
                
                # Outside pool - calculate distance penalty
                v_pos_2d = v_obj.position[[0, 2]]
                virt_center_2d = self.virtual_env.center[[0, 2]]
                pin_2d = pin_point[[0, 2]]
                translation_2d = pin_2d - virt_center_2d
                
                rotated_v_pos = self.rotate_point_2d(v_pos_2d, rotation_angle, virt_center_2d)
                transformed_v_pos = rotated_v_pos + translation_2d
                
                distance_to_target = np.linalg.norm(p_pos_2d - transformed_v_pos)
            
            else:  # SPATIAL_POOL_MODE == "relaxed"
                # RELAXED MODE: Check if outside pedestal/circular bounds and penalize based on distance
                # NOTE: Selection pool already ensures object is in overlapping region (hard constraint)
                
                # Check if this interactable belongs to a pedestal group
                is_in_pedestal_group = False
                for group in self.pedestal_groups:
                    if v_idx in group.interactable_indices:
                        is_in_pedestal_group = True
                        # Check if physical object is within pedestal bounds
                        if group.selection_bounds_2d is not None:
                            bounds_2d = group.selection_bounds_2d
                            corners = bounds_2d.get('corners', [])
                            
                            if len(corners) < 3:
                                # Fallback: use center and size to create rectangle
                                center = bounds_2d['center']
                                size = bounds_2d['size']
                                half_x = size[0] / 2
                                half_z = size[1] / 2
                                corners = [
                                    center + np.array([-half_x, -half_z]),
                                    center + np.array([half_x, -half_z]),
                                    center + np.array([half_x, half_z]),
                                    center + np.array([-half_x, half_z])
                                ]
                            
                            # Transform corners to physical space
                            virt_center_2d = self.virtual_env.center[[0, 2]]
                            pin_2d = pin_point[[0, 2]]
                            translation_2d = pin_2d - virt_center_2d
                            
                            transformed_corners = []
                            for corner in corners:
                                rotated = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
                                transformed = rotated + translation_2d
                                transformed_corners.append(transformed)
                            
                            transformed_polygon = np.array(transformed_corners)
                            
                            # Check if physical object is inside pedestal bounds
                            if self._point_in_polygon(p_pos_2d, transformed_polygon):
                                # Inside pedestal bounds - no penalty
                                continue
                            
                            # Outside pedestal bounds - calculate distance to pedestal center
                            pedestal_center = np.mean(transformed_polygon, axis=0)
                            distance_to_target = np.linalg.norm(p_pos_2d - pedestal_center)
                        else:
                            continue  # No bounds defined, no penalty
                        break
                
                if not is_in_pedestal_group:
                    # Check if it's a standalone interactable
                    if v_idx in self.standalone_interactables:
                        # Check if within circular radius
                        v_pos_2d = v_obj.position[[0, 2]]
                        virt_center_2d = self.virtual_env.center[[0, 2]]
                        pin_2d = pin_point[[0, 2]]
                        translation_2d = pin_2d - virt_center_2d
                        
                        rotated_v_pos = self.rotate_point_2d(v_pos_2d, rotation_angle, virt_center_2d)
                        transformed_v_pos = rotated_v_pos + translation_2d
                        
                        distance_to_target = np.linalg.norm(p_pos_2d - transformed_v_pos)
                        
                        # If within radius, no penalty
                        if distance_to_target <= STANDALONE_SELECTION_RADIUS:
                            continue
                    else:
                        # Not an interactable with spatial bounds - no penalty
                        continue
            
            # Safety check: ensure distance_to_target was calculated
            if distance_to_target is None:
                continue  # No penalty if distance wasn't calculated
            
            # Apply exponential distance-based penalty: exp(priority_weight * di / dmax)
            if self.enable_priority_weighting:
                priority_weight = v_obj.engagement_level
            else:
                priority_weight = 1.0
            penalty = np.exp(priority_weight * distance_to_target / dmax)
            loss += penalty
        
        return float(loss)
    
    def calculate_average_proxy_distance(self, assignment_matrix: np.ndarray, pin_point: np.ndarray,
                                         rotation_angle: float) -> float:
        """
        Calculate the average distance between virtual objects and their assigned physical proxies.
        Used as a tie-breaker when multiple configurations have the same loss.
        
        Args:
            assignment_matrix: Current assignment matrix
            pin_point: Current pin location
            rotation_angle: Current rotation angle
            
        Returns:
            Average 3D distance between virtual objects and their assigned physical proxies
        """
        if self.virtual_env is None:
            return 0.0
        
        total_distance = 0.0
        num_assignments = 0
        
        # Get assignments for assignable virtual objects only
        assignable_indices = self.get_assignable_virtual_indices()
        assigned_physical = np.argmax(assignment_matrix, axis=1)
        
        for v_idx in assignable_indices:
            v_obj = self.virtual_objects[v_idx]
            if v_obj.position is None:
                continue
            
            # Get the assigned physical object
            p_idx = assigned_physical[v_idx]
            p_obj = self.physical_objects[p_idx]
            
            if p_obj.position is None:
                continue
            
            # Transform virtual object position to physical space
            v_pos = v_obj.position.copy()
            v_pos_2d = v_pos[[0, 2]]
            v_pos_y = v_pos[1]
            
            virt_center_2d = self.virtual_env.center[[0, 2]]
            pin_2d = pin_point[[0, 2]]
            translation_2d = pin_2d - virt_center_2d
            
            # Apply rotation and translation
            rotated_v_pos_2d = self.rotate_point_2d(v_pos_2d, rotation_angle, virt_center_2d)
            transformed_v_pos_2d = rotated_v_pos_2d + translation_2d
            
            # Reconstruct 3D position (Y coordinate remains unchanged)
            transformed_v_pos_3d = np.array([transformed_v_pos_2d[0], v_pos_y, transformed_v_pos_2d[1]])
            
            # Calculate 3D distance to assigned physical object
            distance = np.linalg.norm(p_obj.position - transformed_v_pos_3d)
            total_distance += distance
            num_assignments += 1
        
        # Return average distance
        if num_assignments > 0:
            return total_distance / num_assignments
        else:
            return 0.0
    
    def check_early_termination(self, pin_point: np.ndarray, rotation_angle: float) -> Tuple[bool, str]:
        """
        Check if this pin & rotate configuration should be terminated early
        
        Early termination occurs if the number of non-occluded physical objects in the 
        intersection of play area boundary and virtual environment boundary is less than 
        the number of virtual interactables
        
        Args:
            pin_point: Current pin location
            rotation_angle: Current rotation angle
            
        Returns:
            Tuple of (should_terminate: bool, reason: str)
        """
        if not self.enable_spatial_constraint:
            return (False, "")
        
        if self.play_area is None or self.virtual_env is None:
            return (False, "Missing play area or virtual environment data")
        
        # Check occlusion for this configuration
        occluded_indices = self.check_occlusion_for_config(pin_point, rotation_angle)
        
        # Transform virtual environment bounds to physical space
        virt_center_2d = self.virtual_env.center[[0, 2]]
        pin_2d = pin_point[[0, 2]]
        translation_2d = pin_2d - virt_center_2d
        
        # Rotate and translate virtual env corners
        virt_corners_2d = np.array(self.virtual_env.bounds_2d['corners'])
        transformed_virt_corners = []
        for corner in virt_corners_2d:
            rotated = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
            translated = rotated + translation_2d
            transformed_virt_corners.append(translated)
        transformed_virt_polygon = np.array(transformed_virt_corners)
        
        # Count non-occluded physical objects in the INTERSECTION of play area AND virtual env
        non_occluded_in_intersection = 0
        play_area_boundary = self.play_area.boundary_points[:, [0, 2]]
        
        for p_idx, p_obj in enumerate(self.physical_objects):
            if p_idx in occluded_indices:
                continue
            if p_obj.position is None:
                continue
            
            p_pos_2d = p_obj.position[[0, 2]]
            
            # Check if physical object is in BOTH play area AND virtual environment
            in_play_area = self._point_in_polygon(p_pos_2d, play_area_boundary)
            in_virtual_env = self._point_in_polygon(p_pos_2d, transformed_virt_polygon)
            
            if in_play_area and in_virtual_env:
                non_occluded_in_intersection += 1
        
        # Count virtual interactables (including substrate)
        num_interactables = sum(1 for v_obj in self.virtual_objects 
                               if v_obj.involvement_type in ['grasp', 'contact', 'substrate'])
        
        # Early termination if not enough physical objects in the intersection
        if non_occluded_in_intersection < num_interactables:
            reason = f"Insufficient physical objects in intersection: {non_occluded_in_intersection} available, {num_interactables} needed"
            return (True, reason)
        
        return (False, "")
    
    def calculate_total_loss(self, assignment_matrix: np.ndarray, verbose: bool = False, 
                           pin_point: Optional[np.ndarray] = None, 
                           rotation_angle: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """Calculate the total multi-objective loss function
        
        Args:
            assignment_matrix: Current assignment matrix
            verbose: If True, print detailed loss calculation
            pin_point: Pin location for spatial constraint (if None, uses default)
            rotation_angle: Rotation angle for spatial constraint (if None, uses 0)
        """
        l_realism = self.calculate_realism_loss(assignment_matrix)
        l_interaction = self.calculate_interaction_loss(assignment_matrix, verbose)
        
        # Calculate spatial loss using selection pool constraint
        # Spatial loss penalizes objects outside pedestal boundaries
        # NOTE: Changed from enable_spatial_constraint to w_spatial > 0
        # This allows spatial loss to be calculated independently of pin & rotate optimization
        if self.w_spatial > 0 and self.virtual_env is not None:
            # If pin_point not provided, use default location (play area center or origin)
            if pin_point is None:
                # Default: use play area center if available, otherwise origin
                if self.play_area is not None:
                    pin_point = self.play_area.center
                else:
                    pin_point = np.array([0.0, 0.0, 0.0])
            if rotation_angle is None:
                rotation_angle = 0.0
            
            l_spatial = self.calculate_spatial_loss_with_pool(assignment_matrix, pin_point, rotation_angle)
        else:
            l_spatial = 0.0
        
        total_loss = (self.w_realism * l_realism + 
                     self.w_interaction * l_interaction +
                     self.w_spatial * l_spatial)
        
        loss_components = {
            "L_realism": l_realism,
            "L_interaction": l_interaction,
            "L_spatial": l_spatial,
            "total": total_loss
        }
        
        return total_loss, loss_components
    
    def is_valid_assignment(self, assignment_matrix: np.ndarray) -> bool:
        """Check if assignment satisfies hard constraints"""
        # Hard constraint 1: Each ASSIGNABLE virtual object gets exactly one proxy
        # Set dressing objects (pedestal, surroundings) should have row sum = 0
        assignable_indices = self.get_assignable_virtual_indices()
        
        for i in range(len(self.virtual_objects)):
            row_sum = np.sum(assignment_matrix[i, :])
            
            if i in assignable_indices:
                # Assignable objects must have exactly one proxy
                if not np.isclose(row_sum, 1.0):
                    return False
            else:
                # Set dressing objects must have zero proxies
                if not np.isclose(row_sum, 0.0):
                    return False
        
        # Hard constraint 2: Exclusivity (if enabled)
        if self.enable_exclusivity:
            col_sums = np.sum(assignment_matrix, axis=0)
            if np.any(col_sums > 1.0):
                return False
        
        return True
    
    def set_priority_weighting(self, enabled: bool) -> None:
        """Enable or disable priority weighting in the realism loss calculation
        
        Args:
            enabled: If True, use engagement_level as priority weights. If False, use equal weights (1.0).
        """
        self.enable_priority_weighting = enabled
        if enabled:
            print("Priority weighting enabled - using engagement_level as priority weights")
        else:
            print("Priority weighting disabled - using equal weights for all objects")
    
    def compute_adaptive_k_values(self, assignable_indices: List[int], initial_k: int = ADAPTIVE_K_INITIAL, 
                                  max_k: int = ADAPTIVE_K_MAX, verbose: bool = False) -> Dict[int, int]:
        """
        Compute adaptive K values for each grasp/contact virtual object based on candidate overlap
        
        Logic:
        - Start with K=initial_k for all grasp/contact objects
        - If virtual objects share n candidates in their top-K lists, increase their K by n
        - Cap K at max_k (but actual candidates can exceed due to tie handling)
        
        Tie Handling:
        - When selecting top-K, ALL candidates with score >= K-th score are included
        - This means if K=5 and candidates 5,6,7 are tied, all 7 are selected
        - The K value itself is capped at max_k, but actual candidate count can exceed it
        
        Args:
            assignable_indices: Indices of assignable virtual objects
            initial_k: Initial K value to start with (default: 3)
            max_k: Maximum K value allowed (default: 5, but can exceed due to ties)
            verbose: If True, print detailed information including tie info
            
        Returns:
            Dictionary mapping virtual object index to its adaptive K value
        """
        if self.realism_matrix is None:
            return {idx: max_k for idx in assignable_indices}
        
        n_physical = len(self.physical_objects)
        initial_k = min(initial_k, n_physical)
        max_k = min(max_k, n_physical)
        
        # Initialize K values
        k_values = {}
        for v_idx in assignable_indices:
            v_obj = self.virtual_objects[v_idx]
            if v_obj.involvement_type in ['grasp', 'contact']:
                k_values[v_idx] = initial_k
            else:
                # Substrate objects don't use this adaptive K (they have their own logic)
                k_values[v_idx] = max_k
        
        # Get initial top-K candidates for each grasp/contact object
        grasp_contact_indices = [idx for idx in assignable_indices 
                                if self.virtual_objects[idx].involvement_type in ['grasp', 'contact']]
        
        if len(grasp_contact_indices) <= 1:
            # No overlap possible with 0 or 1 object
            return k_values
        
        # Get initial candidates for each object
        initial_candidates = {}
        for v_idx in grasp_contact_indices:
            realism_scores = self.realism_matrix[v_idx, :]
            allowed_indices = [idx for idx in range(len(realism_scores)) 
                             if idx not in self.banned_physical_indices]
            
            if len(allowed_indices) <= initial_k:
                top_k_indices = allowed_indices
            else:
                sorted_allowed = sorted(allowed_indices, key=lambda idx: realism_scores[idx], reverse=True)
                kth_score = realism_scores[sorted_allowed[initial_k-1]]
                top_k_indices = [idx for idx in sorted_allowed if realism_scores[idx] >= kth_score]
            
            initial_candidates[v_idx] = set(top_k_indices)
        
        # Iteratively adjust K values based on overlaps
        max_iterations = 3  # Prevent infinite loops
        for iteration in range(max_iterations):
            adjustments_made = False
            
            # Check all pairs of virtual objects
            for i, v_idx1 in enumerate(grasp_contact_indices):
                for v_idx2 in grasp_contact_indices[i+1:]:
                    # Find overlapping candidates
                    overlap = initial_candidates[v_idx1].intersection(initial_candidates[v_idx2])
                    n_overlap = len(overlap)
                    
                    if n_overlap > 0:
                        # Both objects need more candidates
                        old_k1 = k_values[v_idx1]
                        old_k2 = k_values[v_idx2]
                        
                        k_values[v_idx1] = min(k_values[v_idx1] + n_overlap, max_k)
                        k_values[v_idx2] = min(k_values[v_idx2] + n_overlap, max_k)
                        
                        if k_values[v_idx1] != old_k1 or k_values[v_idx2] != old_k2:
                            adjustments_made = True
                            
                            # Expand candidates for objects whose K increased
                            if k_values[v_idx1] != old_k1:
                                initial_candidates[v_idx1] = self._get_top_k_candidates(
                                    v_idx1, k_values[v_idx1]
                                )
                            if k_values[v_idx2] != old_k2:
                                initial_candidates[v_idx2] = self._get_top_k_candidates(
                                    v_idx2, k_values[v_idx2]
                                )
            
            # Check multi-way overlaps (>2 objects sharing candidates)
            if len(grasp_contact_indices) > 2:
                for v_idx in grasp_contact_indices:
                    # Find how many candidates this object shares with others
                    candidates = initial_candidates[v_idx]
                    shared_count = {}
                    
                    for p_idx in candidates:
                        # Count how many other objects also have this candidate
                        sharing_objects = [other_v_idx for other_v_idx in grasp_contact_indices 
                                         if other_v_idx != v_idx and p_idx in initial_candidates[other_v_idx]]
                        if sharing_objects:
                            shared_count[p_idx] = len(sharing_objects)
                    
                    # If many candidates are highly contested, increase K
                    if shared_count:
                        max_contention = max(shared_count.values())
                        if max_contention >= 2:  # At least 3 objects want same candidate
                            old_k = k_values[v_idx]
                            k_values[v_idx] = min(k_values[v_idx] + max_contention, max_k)
                            
                            if k_values[v_idx] != old_k:
                                adjustments_made = True
                                initial_candidates[v_idx] = self._get_top_k_candidates(
                                    v_idx, k_values[v_idx]
                                )
            
            if not adjustments_made:
                break  # Converged
        
        if verbose:
            print("\nAdaptive Top-K Results:")
            for v_idx in grasp_contact_indices:
                v_obj = self.virtual_objects[v_idx]
                num_candidates = len(initial_candidates[v_idx])
                k_val = k_values[v_idx]
                
                # Check if ties caused exceeding K
                if num_candidates > k_val:
                    tie_info = f" [+{num_candidates - k_val} tied]"
                else:
                    tie_info = ""
                
                print(f"  {v_obj.name}: K={k_val}, actual={num_candidates} candidates{tie_info}")
        
        return k_values
    
    def _get_top_k_candidates(self, v_idx: int, k: int) -> Set[int]:
        """
        Helper to get top-K candidates for a virtual object WITH TIE HANDLING
        
        Important: This method includes ALL candidates with scores >= the K-th score,
        which means it can return MORE than K candidates when there are ties.
        
        Example: If K=5 and candidates 5, 6, 7 all have score 0.85, all 7 are returned.
        
        Args:
            v_idx: Virtual object index
            k: Target number of candidates (actual count may exceed due to ties)
            
        Returns:
            Set of physical object indices (size >= k if ties exist)
        """
        realism_scores = self.realism_matrix[v_idx, :]
        allowed_indices = [idx for idx in range(len(realism_scores)) 
                         if idx not in self.banned_physical_indices]
        
        if len(allowed_indices) <= k:
            return set(allowed_indices)
        
        sorted_allowed = sorted(allowed_indices, key=lambda idx: realism_scores[idx], reverse=True)
        kth_score = realism_scores[sorted_allowed[k-1]]
        
        # Include ALL candidates with scores >= K-th score (handles ties)
        top_k_indices = [idx for idx in sorted_allowed if realism_scores[idx] >= kth_score]
        
        return set(top_k_indices)
    
    def filter_by_top_k_realism(self, k: Optional[int] = None, assignable_indices: Optional[List[int]] = None, 
                                verbose: bool = False, k_values: Optional[Dict[int, int]] = None) -> List[np.ndarray]:
        """Generate assignments using only top-K realism scores for grasp and contact virtual objects
        
        Args:
            k: Number of top physical objects to consider for each virtual object
            assignable_indices: Indices of virtual objects that need proxy assignment (excludes set dressing)
            verbose: If True, print detailed top-K candidate information
        """
        # If assignable_indices not provided, use all virtual objects (backward compatibility)
        if assignable_indices is None:
            assignable_indices = list(range(len(self.virtual_objects)))
        
        n_virtual = len(self.virtual_objects)  # Total including set dressing
        n_assignable = len(assignable_indices)  # Only those needing proxies
        n_physical = len(self.physical_objects)
        
        # Set default K to fixed value
        if k is None:
            k = TOP_K_CONTACT_OBJECTS
            print(f"Setting K = {k}")
        
        # Ensure K doesn't exceed available physical objects
        k = min(k, n_physical)
        
        # print(f"Applying Top-K filtering with K = {k}")
        
        # Check if realism matrix is available
        if self.realism_matrix is None:
            print("Warning: Realism matrix not available, falling back to exhaustive generation")
            # Fall back to exhaustive generation (only for assignable objects)
            valid_assignments = []
            if self.enable_exclusivity:
                for perm in itertools.permutations(range(n_physical), n_assignable):
                    assignment_matrix = np.zeros((n_virtual, n_physical))
                    for i, j in enumerate(perm):
                        v_idx = assignable_indices[i]  # Map to actual virtual object index
                        assignment_matrix[v_idx, j] = 1.0
                    valid_assignments.append(assignment_matrix)
            else:
                for assignment_tuple in itertools.product(range(n_physical), repeat=n_assignable):
                    assignment_matrix = np.zeros((n_virtual, n_physical))
                    for i, j in enumerate(assignment_tuple):
                        v_idx = assignable_indices[i]  # Map to actual virtual object index
                        assignment_matrix[v_idx, j] = 1.0
                    valid_assignments.append(assignment_matrix)
            return valid_assignments
        
        # Get top-K physical objects for each ASSIGNABLE virtual object only
        top_k_assignments = {}
        for v_idx in assignable_indices:  # Only process assignable objects
            virtual_obj = self.virtual_objects[v_idx]
            
            # Only apply Top-K filtering to grasp and contact objects
            if virtual_obj.involvement_type in ["grasp", "contact"]:
                # Use per-object K value if provided (adaptive), otherwise use global K
                object_k = k_values[v_idx] if k_values is not None and v_idx in k_values else k
                
                realism_scores = self.realism_matrix[v_idx, :]

                # Exclude banned indices first
                allowed_indices = [idx for idx in range(len(realism_scores)) if idx not in self.banned_physical_indices]

                # Handle ties: include all objects with the same score as the k-th object among allowed only
                if len(allowed_indices) <= object_k:
                    # If we have k or fewer allowed objects, use all of them
                    top_k_indices = allowed_indices
                else:
                    # Get allowed indices sorted by realism score (highest first)
                    sorted_allowed = sorted(allowed_indices, key=lambda idx: realism_scores[idx], reverse=True)

                    # Find the k-th highest score among allowed
                    kth_score = realism_scores[sorted_allowed[object_k-1]]

                    # Include all allowed objects with scores >= k-th score (tie handling)
                    top_k_indices = []
                    for idx in sorted_allowed:
                        if realism_scores[idx] >= kth_score:
                            top_k_indices.append(idx)
                        else:
                            break  # Since list is sorted, we can break early
                
                top_k_assignments[v_idx] = top_k_indices
                
                if verbose:
                    k_label = f"K={object_k}" if k_values is not None else f"Top-{object_k}"
                    num_selected = len(top_k_indices)
                    
                    # Show if ties caused exceeding K
                    if num_selected > object_k:
                        tie_info = f" [+{num_selected - object_k} tied at K-th score]"
                    else:
                        tie_info = ""
                    
                    print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): {k_label}, selected={num_selected}{tie_info}")
                    print(f"    Physical objects:")
                    for idx in top_k_indices:
                        physical_obj_name = self.physical_objects[idx].name
                        score = realism_scores[idx]
                        print(f"      - {physical_obj_name}: {score:.3f}")
            else:
                # For substrate objects, apply Top-K filtering per relationship, then UNION
                # This ensures we keep candidates that are good for ANY relationship
                
                has_interaction_data = False
                all_top_k_sets = []  # List of sets, one per relationship
                relationship_info = []  # For verbose output
                
                if (hasattr(self, 'interaction_matrix_3d') and 
                    hasattr(self, 'virtual_relationship_pairs') and
                    self.interaction_matrix_3d is not None):
                    
                    # Find all relationships involving this substrate
                    for rel_idx, (contact_v_idx, sub_v_idx) in enumerate(self.virtual_relationship_pairs):
                        if sub_v_idx == v_idx:  # This substrate is involved in this relationship
                            has_interaction_data = True
                            
                            # Calculate scores for each physical substrate for THIS relationship only
                            relationship_scores = {}
                            
                            for p_sub_idx in range(n_physical):
                                if p_sub_idx in self.banned_physical_indices:
                                    continue
                                
                                # Calculate MAX interaction rating across all contact assignments for THIS relationship
                                # interaction_matrix_3d shape: [relationship_idx, contact_physical, substrate_physical]
                                max_rating = np.max(self.interaction_matrix_3d[rel_idx, :, p_sub_idx])
                                relationship_scores[p_sub_idx] = max_rating
                            
                            # Get top-K for THIS relationship
                            k_substrate = TOP_K_SUBSTRATE_OBJECTS
                            k_substrate = min(k_substrate, len(relationship_scores))
                            
                            if relationship_scores:
                                # Sort by score (descending) and get top-K
                                sorted_substrates = sorted(relationship_scores.items(), key=lambda x: x[1], reverse=True)
                                
                                # Handle ties: include all with same score as k-th element
                                if len(sorted_substrates) <= k_substrate:
                                    top_k_for_rel = [idx for idx, _ in sorted_substrates]
                                else:
                                    kth_score = sorted_substrates[k_substrate-1][1]
                                    top_k_for_rel = [idx for idx, score in sorted_substrates if score >= kth_score]
                                
                                all_top_k_sets.append(set(top_k_for_rel))
                                
                                # Store info for verbose output
                                contact_name = self.virtual_objects[contact_v_idx].name
                                relationship_info.append({
                                    'contact': contact_name,
                                    'top_k': top_k_for_rel,
                                    'scores': relationship_scores
                                })
                
                if has_interaction_data and all_top_k_sets:
                    # Take UNION of all top-K sets
                    final_candidates = set()
                    for top_k_set in all_top_k_sets:
                        final_candidates.update(top_k_set)
                    
                    top_k_indices = list(final_candidates)
                    top_k_assignments[v_idx] = top_k_indices
                    
                    if verbose:
                        print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): Top-{k_substrate} per relationship, UNION")
                        print(f"    {len(relationship_info)} relationships → {len(top_k_indices)} total candidates")
                        for rel_info in relationship_info:
                            print(f"      {rel_info['contact']} → {virtual_obj.name}: {len(rel_info['top_k'])} candidates")
                        print(f"    Final union candidates:")
                        # Show top candidates by max score across all relationships
                        max_scores = {}
                        for rel_info in relationship_info:
                            for idx in top_k_indices:
                                if idx in rel_info['scores']:
                                    if idx not in max_scores:
                                        max_scores[idx] = rel_info['scores'][idx]
                                    else:
                                        max_scores[idx] = max(max_scores[idx], rel_info['scores'][idx])
                        sorted_final = sorted(max_scores.items(), key=lambda x: x[1], reverse=True)
                        for idx, score in sorted_final[:5]:
                            physical_obj_name = self.physical_objects[idx].name
                            print(f"      - {physical_obj_name}: {score:.3f}")
                        if len(sorted_final) > 5:
                            print(f"      ... and {len(sorted_final) - 5} more")
                else:
                    # Fallback: no interaction data, use all available physical objects
                    substrate_indices = [idx for idx in range(n_physical) if idx not in self.banned_physical_indices]
                    
                    # Add randomization for substrate objects when interaction weight is 0
                    if self.w_interaction == 0:
                        random.shuffle(substrate_indices)
                    
                    top_k_assignments[v_idx] = substrate_indices
                    
                    if verbose:
                        status = "all available (no interaction data)"
                        print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): Using {status} ({len(substrate_indices)} total)")
        
        # Generate assignments using only top-K combinations (only for assignable objects)
        valid_assignments = []
        assignment_count = 0
        
        # print(f"Generating assignments from filtered combinations...")
        
        for assignment_tuple in itertools.product(*[top_k_assignments[i] for i in assignable_indices]):
            # Check exclusivity constraint if enabled
            if self.enable_exclusivity and len(set(assignment_tuple)) != len(assignment_tuple):
                continue
                
            assignment_matrix = np.zeros((n_virtual, n_physical))
            for i, j in enumerate(assignment_tuple):
                v_idx = assignable_indices[i]  # Map to actual virtual object index
                assignment_matrix[v_idx, j] = 1.0
            valid_assignments.append(assignment_matrix)
            assignment_count += 1
            
            # Progress indicator for large searches
            if assignment_count % 10000 == 0:
                print(f"    Generated {assignment_count} assignments...")
        
        # print(f"Top-K filtering generated {len(valid_assignments)} assignments")
        
        # Add additional randomization when interaction weight is 0 to ensure varied results
        if self.w_interaction == 0:
            random.shuffle(valid_assignments)
            print("Assignment order randomized due to w_interaction=0")
        
        return valid_assignments

    def set_banned_physical_objects(self, banned_pairs: List[Tuple[int, int]]) -> None:
        """Set banned physical objects by (image_id, object_id) pairs and compute their indices.
        
        Args:
            banned_pairs: List of (image_id, object_id) pairs to exclude from consideration.
        """
        # Store canonicalized set
        self.banned_physical_pairs = set((int(img_id), int(obj_id)) for img_id, obj_id in banned_pairs)

        # Map to indices if physical objects already loaded
        self.banned_physical_indices = set()
        if self.physical_objects:
            for phys in self.physical_objects:
                key = (int(phys.image_id), int(phys.object_id))
                if key in self.banned_physical_pairs:
                    self.banned_physical_indices.add(phys.index)

        if self.banned_physical_pairs:
            print(f"Banned physical objects set: {len(self.banned_physical_pairs)} pairs")
            # Print a few examples
            examples = list(self.banned_physical_pairs)[:5]
            for (img_id, obj_id) in examples:
                print(f"  - (image_id={img_id}, object_id={obj_id})")
        if self.banned_physical_indices:
            print(f"Banned physical indices resolved: {sorted(self.banned_physical_indices)}")

    def refresh_banned_indices_after_load(self) -> None:
        """Recompute banned indices after data load, preserving existing banned pair list."""
        if not self.banned_physical_pairs:
            return
        self.set_banned_physical_objects(list(self.banned_physical_pairs))
    
    def get_assignable_virtual_objects(self) -> List[VirtualObject]:
        """
        Get only virtual objects that need proxy assignment
        Excludes set dressing (pedestal, surroundings) as they don't need proxies
        """
        return [obj for obj in self.virtual_objects 
                if obj.involvement_type not in ['pedestal', 'surroundings']]
    
    def get_assignable_virtual_indices(self) -> List[int]:
        """Get indices of virtual objects that need proxy assignment"""
        return [i for i, obj in enumerate(self.virtual_objects)
                if obj.involvement_type not in ['pedestal', 'surroundings']]
    
    def generate_all_assignments(self) -> List[np.ndarray]:
        """Generate all valid assignment permutations with optional Top-K filtering
        
        Only generates assignments for assignable virtual objects (excludes set dressing)
        """
        # Get only virtual objects that need proxies (exclude pedestals and surroundings)
        assignable_indices = self.get_assignable_virtual_indices()
        n_virtual = len(self.virtual_objects)  # Total including set dressing
        n_assignable = len(assignable_indices)  # Only those needing proxies
        n_physical = len(self.physical_objects)
        
        # print(f"Total virtual objects: {n_virtual}")
        # print(f"Assignable virtual objects (excluding set dressing): {n_assignable}")
        # print(f"Physical objects: {n_physical}")
        
        if n_physical < n_assignable:
            print(f"Error: Not enough physical objects ({n_physical}) for assignable virtual objects ({n_assignable})")
            return []
        
        # Calculate theoretical assignment count (for informational purposes)
        if self.enable_exclusivity:
            theoretical_count = math.factorial(n_physical) // math.factorial(n_physical - n_assignable)
        else:
            theoretical_count = n_physical ** n_assignable
        
        # print(f"Theoretical assignment count: {theoretical_count}")
        
        # Always use Top-K filtering for consistent behavior
        # Use adaptive K if enabled and available, otherwise use fixed K
        if ENABLE_ADAPTIVE_TOP_K and hasattr(self, 'adaptive_k_values') and self.adaptive_k_values is not None:
            return self.filter_by_top_k_realism(k_values=self.adaptive_k_values, assignable_indices=assignable_indices)
        else:
            k = TOP_K_CONTACT_OBJECTS
            return self.filter_by_top_k_realism(k=k, assignable_indices=assignable_indices)
    
    def print_debug_matrices(self) -> None:
        """Print all matrices for debugging purposes"""
        print("\n" + "="*60)
        print("DEBUG: MATRICES INFORMATION")
        print("="*60)
        
        # Print virtual objects info
        print(f"\nVirtual Objects ({len(self.virtual_objects)}):")
        print("-" * 40)
        for i, obj in enumerate(self.virtual_objects):
            print(f"[{i}] {obj.name} - Type: {obj.involvement_type}, Engagement: {obj.engagement_level}")
        
        # Print physical objects info
        print(f"\nPhysical Objects ({len(self.physical_objects)}):")
        print("-" * 40)
        for i, obj in enumerate(self.physical_objects):
            print(f"[{i}] {obj.name} - ID: {obj.object_id}, Image: {obj.image_id}")
        
        # Print realism matrix
        if self.realism_matrix is not None:
            print(f"\nRealism Matrix ({self.realism_matrix.shape[0]}x{self.realism_matrix.shape[1]}):")
            print("-" * 40)
            print("Rows = Virtual Objects, Columns = Physical Objects")
            print("Matrix values (showing first 10x10 if larger):")
            display_rows = min(10, self.realism_matrix.shape[0])
            display_cols = min(10, self.realism_matrix.shape[1])
            
            # Print column headers
            print("       ", end="")
            for j in range(display_cols):
                print(f"P{j:2d}   ", end="")
            print()
            
            # Print matrix with row headers
            for i in range(display_rows):
                print(f"V{i:2d}  ", end="")
                for j in range(display_cols):
                    print(f"{self.realism_matrix[i,j]:5.2f}", end=" ")
                print()
            
            if self.realism_matrix.shape[0] > 10 or self.realism_matrix.shape[1] > 10:
                print("... (truncated for display)")
        
        # Print interaction_exists matrix
        if self.interaction_exists is not None:
            print(f"\nInteraction Exists Matrix ({self.interaction_exists.shape[0]}x{self.interaction_exists.shape[1]}):")
            print("-" * 40)
            print("1.0 = interaction exists, 0.0 = no interaction")
            print("Rows = Contact Virtual Objects, Columns = Substrate Virtual Objects")
            
            # Show which interactions exist
            interactions_found = []
            for i in range(self.interaction_exists.shape[0]):
                for j in range(self.interaction_exists.shape[1]):
                    if self.interaction_exists[i, j] > 0:
                        contact_name = self.virtual_objects[i].name
                        substrate_name = self.virtual_objects[j].name
                        interactions_found.append(f"{contact_name} -> {substrate_name}")
            
            if interactions_found:
                print("Interactions found:")
                for interaction in interactions_found:
                    print(f"  {interaction}")
            else:
                print("No interactions found")
        
        # Print interaction_matrix matrix
        if self.interaction_matrix is not None:
            print(f"\nInteraction Rating Matrix ({self.interaction_matrix.shape[0]}x{self.interaction_matrix.shape[1]}):")
            print("-" * 40)
            print("Rows = Contact Physical Objects, Columns = Substrate Physical Objects")
            
            # Count non-zero entries
            non_zero_count = np.count_nonzero(self.interaction_matrix)
            print(f"Non-zero entries: {non_zero_count}/{self.interaction_matrix.size}")
            
            if non_zero_count > 0:
                print("Sample non-zero interaction ratings:")
                count = 0
                for i in range(self.interaction_matrix.shape[0]):
                    for j in range(self.interaction_matrix.shape[1]):
                        if self.interaction_matrix[i, j] > 0 and count < 10:
                            contact_name = self.physical_objects[i].name
                            substrate_name = self.physical_objects[j].name
                            rating = self.interaction_matrix[i, j]
                            print(f"  {contact_name} -> {substrate_name}: {rating:.3f}")
                            count += 1
                if non_zero_count > 10:
                    print(f"  ... and {non_zero_count - 10} more")
        
        # Print 3D interaction matrix information
        if hasattr(self, 'interaction_matrix_3d') and self.interaction_matrix_3d is not None:
            print(f"\n3D Interaction Rating Matrix ({self.interaction_matrix_3d.shape}):")
            print("-" * 40)
            print("Dimensions: [relationship_idx, contact_physical, substrate_physical]")
            
            if hasattr(self, 'virtual_relationship_pairs') and self.virtual_relationship_pairs:
                print("Virtual Relationships:")
                for rel_idx, (contact_idx, substrate_idx) in enumerate(self.virtual_relationship_pairs):
                    contact_name = self.virtual_objects[contact_idx].name
                    substrate_name = self.virtual_objects[substrate_idx].name
                    
                    # Count non-zero entries for this relationship
                    rel_matrix = self.interaction_matrix_3d[rel_idx, :, :]
                    non_zero_count = np.count_nonzero(rel_matrix)
                    
                    print(f"  [{rel_idx}] {contact_name} -> {substrate_name}: {non_zero_count} ratings")
                    
                    # Show a few sample ratings
                    if non_zero_count > 0:
                        sample_count = 0
                        for i in range(rel_matrix.shape[0]):
                            for j in range(rel_matrix.shape[1]):
                                if rel_matrix[i, j] > 0 and sample_count < 3:
                                    contact_phys_name = self.physical_objects[i].name
                                    substrate_phys_name = self.physical_objects[j].name
                                    rating = rel_matrix[i, j]
                                    print(f"    {contact_phys_name} -> {substrate_phys_name}: {rating:.3f}")
                                    sample_count += 1
                        if non_zero_count > 3:
                            print(f"    ... and {non_zero_count - 3} more")
        
        # Print spatial constraint information (NEW system uses pedestal groups and selection pools)
        print(f"\nSpatial Constraint Information:")
        print("-" * 40)
        
        # Show pedestal groups
        if self.pedestal_groups:
            print(f"Pedestal Groups: {len(self.pedestal_groups)}")
            for group in self.pedestal_groups:
                print(f"  '{group.pedestal_name}': {len(group.interactable_indices)} interactables")
        else:
            print("Pedestal Groups: None")
        
        # Show standalone interactables
        if self.standalone_interactables:
            standalone_names = [self.virtual_objects[idx].name for idx in self.standalone_interactables]
            print(f"Standalone Interactables: {len(self.standalone_interactables)}")
            print(f"  {', '.join(standalone_names)}")
        else:
            print("Standalone Interactables: None")
        
        # Note about deprecated matrices (kept for backward compatibility but not used in optimization)
        print("\nNote: Distance-based spatial matrices are deprecated.")
        print("Current spatial constraint uses selection pools based on pedestal bounds.")
        
        print("="*60)

    def optimize(self, start_pin_idx: int = 0, start_rot_idx: int = 0, 
                auto_resume: bool = True) -> Optional[Assignment]:
        """
        Find the optimal assignment with minimum loss
        
        If spatial constraint is enabled and play area data is available, uses pin & rotate.
        Otherwise, uses default optimization.
        
        Args:
            start_pin_idx: Starting pin index (0-based). Only used for pin & rotate mode.
            start_rot_idx: Starting rotation index (0-based). Only used for pin & rotate mode.
            auto_resume: If True, automatically load and resume from saved progress if available.
        """
        if self.enable_spatial_constraint and self.play_area is not None and self.virtual_env is not None:
            print("Running optimization with pin & rotate spatial constraint...")
            return self.optimize_with_pin_and_rotate(start_pin_idx, start_rot_idx, auto_resume)
        else:
            print("Running optimization without spatial constraint...")
            return self.optimize_default()
    
    def optimize_default(self) -> Optional[Assignment]:
        """Find the optimal assignment with minimum loss (original method without pin & rotate)"""
        print("Starting global optimization (default mode)...")
        
        # Print debug information about matrices
        self.print_debug_matrices()
        
        # Compute adaptive K values if enabled
        assignable_indices = self.get_assignable_virtual_indices()
        if ENABLE_ADAPTIVE_TOP_K:
            print(f"\nComputing Adaptive Top-K values (initial={ADAPTIVE_K_INITIAL}, max={ADAPTIVE_K_MAX})...")
            self.adaptive_k_values = self.compute_adaptive_k_values(
                assignable_indices, 
                initial_k=ADAPTIVE_K_INITIAL,
                max_k=ADAPTIVE_K_MAX,
                verbose=True
            )
            print("\nTop-K Proxy Candidates (Adaptive):")
            print(f"  Grasp/Contact: Adaptive K (see above), Substrate: K={TOP_K_SUBSTRATE_OBJECTS}")
        else:
            self.adaptive_k_values = None
            print(f"\nTop-K Proxy Candidates (Fixed):")
            print(f"  Grasp/Contact: K={TOP_K_CONTACT_OBJECTS}, Substrate: K={TOP_K_SUBSTRATE_OBJECTS}")
        
        print("-" * 60)
        self.filter_by_top_k_realism(
            k=TOP_K_CONTACT_OBJECTS if not ENABLE_ADAPTIVE_TOP_K else None,
            k_values=self.adaptive_k_values if ENABLE_ADAPTIVE_TOP_K else None,
            assignable_indices=assignable_indices, 
            verbose=True
        )
        print("="*60)
        
        # Generate all possible assignments
        all_assignments = self.generate_all_assignments()
        if not all_assignments:
            return None
        
        best_assignments = []  # Store all assignments with minimum loss
        best_loss = float('inf')
        best_components = None
        
        print(f"Evaluating {len(all_assignments)} assignments...")
        start_time = time.time()
        
        # Use parallel evaluation for large batches (same threshold as pin & rotate)
        total_to_evaluate = len(all_assignments)
        if ENABLE_PARALLEL_EVALUATION and total_to_evaluate > PARALLEL_EVALUATION_THRESHOLD:
            # Use parallel evaluation for large batches
            evaluation_results = self._evaluate_assignments_parallel(
                all_assignments, pin_point=None, rotation_angle=None
            )
            
            # Process results and update best assignments
            for total_loss, loss_components, assignment_matrix, is_valid in evaluation_results:
                if not is_valid:
                    continue
                
                # Update best assignments
                if total_loss < best_loss:
                    # Found a better assignment, reset the list
                    best_loss = total_loss
                    best_components = loss_components
                    best_assignments = [assignment_matrix.copy()]
                elif total_loss == best_loss:
                    # Found an assignment with equal loss, add to list
                    best_assignments.append(assignment_matrix.copy())
        else:
            # Use sequential evaluation for small batches
            for i, assignment_matrix in enumerate(all_assignments):
                if i % 1000 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {i}/{len(all_assignments)} assignments ({elapsed:.1f}s)")
                
                # Verify assignment is valid (should be by construction, but double-check)
                if not self.is_valid_assignment(assignment_matrix):
                    continue
                
                # Calculate loss
                total_loss, loss_components = self.calculate_total_loss(assignment_matrix)
                
                # Update best assignments
                if total_loss < best_loss:
                    # Found a better assignment, reset the list
                    best_loss = total_loss
                    best_components = loss_components
                    best_assignments = [assignment_matrix.copy()]
                elif total_loss == best_loss:
                    # Found an assignment with equal loss, add to list
                    best_assignments.append(assignment_matrix.copy())
        
        # Select final assignment
        if not best_assignments:
            return None
            
        # If interaction weight is 0 and we have multiple optimal assignments, randomize selection
        if self.w_interaction == 0 and len(best_assignments) > 1:
            selected_assignment_matrix = random.choice(best_assignments)
            print(f"Randomly selected 1 assignment from {len(best_assignments)} optimal assignments (w_interaction=0)")
        else:
            selected_assignment_matrix = best_assignments[0]
            if len(best_assignments) > 1:
                print(f"Selected first of {len(best_assignments)} optimal assignments")
        
        # Create virtual-to-physical mapping for the selected assignment
        # Only create mappings for assignable virtual objects (exclude set dressing)
        virtual_to_physical = {}
        assignable_indices = self.get_assignable_virtual_indices()
        for virtual_idx in assignable_indices:
            physical_idx = np.argmax(selected_assignment_matrix[virtual_idx, :])
            virtual_to_physical[virtual_idx] = physical_idx
        
        best_assignment = Assignment(
            assignment_matrix=selected_assignment_matrix,
            virtual_to_physical=virtual_to_physical,
            total_loss=best_loss,
            loss_components=best_components
        )
        
        elapsed = time.time() - start_time
        print(f"Optimization completed in {elapsed:.2f}s")
        print(f"Best total loss: {best_loss:.4f}")
        print(f"Loss components: {best_components}")
        
        return best_assignment
    
    def _prepare_optimizer_state(self) -> Dict:
        """
        Prepare a serializable state dict for worker processes.
        This extracts only the data needed for loss calculation.
        """
        # Convert virtual_objects to serializable dict format
        virtual_objects_dict = []
        for v_obj in self.virtual_objects:
            virtual_objects_dict.append({
                'name': v_obj.name,
                'involvement_type': v_obj.involvement_type,
                'engagement_level': v_obj.engagement_level,
                'substrate_engagement_level': v_obj.substrate_engagement_level,
                'index': v_obj.index,
                'position': v_obj.position.tolist() if v_obj.position is not None else None,
                'bounds_2d': v_obj.bounds_2d if v_obj.bounds_2d else None
            })
        
        # Convert physical_objects to serializable dict format
        physical_objects_dict = []
        for p_obj in self.physical_objects:
            physical_objects_dict.append({
                'name': p_obj.name,
                'index': p_obj.index,
                'position': p_obj.position.tolist() if p_obj.position is not None else None
            })
        
        # Serialize VirtualEnvironment for spatial loss calculation
        virtual_env_dict = None
        if self.virtual_env is not None:
            virtual_env_dict = {
                'center': self.virtual_env.center.tolist(),
                'bounds_2d': self.virtual_env.bounds_2d,
                'bounds_3d': self.virtual_env.bounds_3d
            }
        
        # Serialize PedestalGroups for spatial loss calculation
        pedestal_groups_dict = []
        for pg in self.pedestal_groups:
            # Convert selection_bounds_2d to serializable format
            selection_bounds_serialized = None
            if pg.selection_bounds_2d is not None:
                selection_bounds_serialized = {}
                for key, value in pg.selection_bounds_2d.items():
                    if isinstance(value, np.ndarray):
                        selection_bounds_serialized[key] = value.tolist()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        selection_bounds_serialized[key] = [v.tolist() for v in value]
                    else:
                        selection_bounds_serialized[key] = value
            
            pedestal_groups_dict.append({
                'pedestal_index': pg.pedestal_index,
                'interactable_indices': pg.interactable_indices,
                'selection_bounds_2d': selection_bounds_serialized
            })
        
        # Serialize PlayArea for spatial loss calculation
        play_area_dict = None
        if self.play_area is not None:
            play_area_dict = {
                'width': self.play_area.width,
                'depth': self.play_area.depth,
                'center': self.play_area.center.tolist() if self.play_area.center is not None else [0.0, 0.0, 0.0]
            }
        
        optimizer_state = {
            'realism_matrix': self.realism_matrix,
            'interaction_matrix_3d': self.interaction_matrix_3d,
            'virtual_relationship_pairs': self.virtual_relationship_pairs,
            'interaction_exists': self.interaction_exists,
            'virtual_objects': virtual_objects_dict,
            'physical_objects': physical_objects_dict,
            'w_realism': self.w_realism,
            'w_interaction': self.w_interaction,
            'w_spatial': self.w_spatial,
            'enable_priority_weighting': self.enable_priority_weighting,
            'enable_spatial_constraint': self.enable_spatial_constraint,
            'enable_exclusivity': self.enable_exclusivity,
            'assignable_indices': self.get_assignable_virtual_indices(),
            'virtual_env': virtual_env_dict,
            'play_area': play_area_dict,
            'pedestal_groups': pedestal_groups_dict,
            'standalone_interactables': self.standalone_interactables.copy()
        }
        
        return optimizer_state
    
    def _evaluate_assignments_parallel(self, assignments: List[np.ndarray], 
                                      pin_point: Optional[np.ndarray] = None, 
                                      rotation_angle: Optional[float] = None) -> List[Tuple]:
        """Evaluate assignments in parallel using multiprocessing"""
        num_workers = NUM_WORKER_PROCESSES if NUM_WORKER_PROCESSES else cpu_count()
        batch_size = BATCH_SIZE_PER_WORKER
        
        print(f"    Using parallel evaluation: {num_workers} workers, batch size={batch_size}")
        
        # Prepare optimizer state once
        optimizer_state = self._prepare_optimizer_state()
        
        # Split assignments into batches
        batches = []
        for i in range(0, len(assignments), batch_size):
            batch = assignments[i:i+batch_size]
            batches.append((batch, optimizer_state, pin_point, rotation_angle, len(batches)))
        
        # Evaluate batches in parallel
        all_results = []
        with Pool(processes=num_workers) as pool:
            batch_results = pool.map(_evaluate_assignment_batch, batches)
            
            # Flatten results
            for batch_result in batch_results:
                all_results.extend(batch_result)
        
        print(f"    Parallel evaluation complete: {len(all_results)} results")
        return all_results
    
    def _evaluate_assignments_sequential(self, assignments: List[np.ndarray],
                                        pin_point: Optional[np.ndarray] = None, 
                                        rotation_angle: Optional[float] = None) -> List[Tuple]:
        """Evaluate assignments sequentially (fallback for small batches)"""
        results = []
        for i, assignment_matrix in enumerate(assignments):
            if not self.is_valid_assignment(assignment_matrix):
                results.append((float('inf'), {}, assignment_matrix, False))
                continue
            
            # Progress indicator
            if (i + 1) % 50000 == 0:
                print(f"    Evaluated {i+1}/{len(assignments)}...")
            
            total_loss, loss_components = self.calculate_total_loss(
                assignment_matrix, 
                verbose=False, 
                pin_point=pin_point, 
                rotation_angle=rotation_angle
            )
            results.append((total_loss, loss_components, assignment_matrix, True))
        
        return results
    
    def _cache_assignment_results(self, cache_key: frozenset, evaluation_results: List[Tuple], 
                                   assignment_cache: Dict) -> None:
        """
        Cache assignment results for reuse when same candidate set appears
        
        Stores assignments with their L_realism and L_interaction values (which don't change
        across different pin/rotation configs with same candidates)
        
        NOTE: This method should only be called for configs with > CACHE_ACTIVATION_THRESHOLD assignments to
        reduce memory overhead for small configs.
        
        Args:
            cache_key: Frozen set of banned physical indices
            evaluation_results: List of (total_loss, loss_components, assignment_matrix, is_valid)
            assignment_cache: Cache dictionary to update
        """
        cached_assignments = []
        
        for total_loss, loss_components, assignment_matrix, is_valid in evaluation_results:
            if not is_valid:
                continue
            
            # Store assignment with its realism and interaction losses
            cached_assignments.append({
                'assignment_matrix': assignment_matrix.copy(),
                'L_realism': loss_components.get('L_realism', 0.0),
                'L_interaction': loss_components.get('L_interaction', 0.0)
            })
        
        # Store in cache
        assignment_cache[cache_key] = {
            'assignments': cached_assignments
        }
    
    def _evaluate_cached_assignments(self, cached_data: Dict, pin_point: np.ndarray, 
                                     rotation_angle: float) -> List[Tuple]:
        """
        Evaluate cached assignments by only recalculating L_spatial
        
        Reuses L_realism and L_interaction from cache, only calculates L_spatial
        for the new pin_point and rotation_angle. Uses parallel processing for large batches.
        
        Args:
            cached_data: Dictionary with cached assignments
            pin_point: Current pin location
            rotation_angle: Current rotation angle
            
        Returns:
            List of (total_loss, loss_components, assignment_matrix, is_valid)
        """
        cached_assignments = cached_data['assignments']
        num_assignments = len(cached_assignments)
        
        # Use parallel evaluation for large batches
        if ENABLE_PARALLEL_EVALUATION and num_assignments > PARALLEL_EVALUATION_THRESHOLD:
            return self._evaluate_cached_assignments_parallel(cached_assignments, pin_point, rotation_angle)
        else:
            return self._evaluate_cached_assignments_sequential(cached_assignments, pin_point, rotation_angle)
    
    def _evaluate_cached_assignments_parallel(self, cached_assignments: List[Dict], 
                                             pin_point: np.ndarray, 
                                             rotation_angle: float) -> List[Tuple]:
        """Evaluate cached assignments in parallel using multiprocessing"""
        num_workers = NUM_WORKER_PROCESSES if NUM_WORKER_PROCESSES else cpu_count()
        batch_size = BATCH_SIZE_PER_WORKER
        
        print(f"      Using parallel evaluation for cached assignments: {num_workers} workers")
        
        # Prepare optimizer state once
        optimizer_state = self._prepare_optimizer_state()
        
        # Split cached assignments into batches
        batches = []
        for i in range(0, len(cached_assignments), batch_size):
            batch = cached_assignments[i:i+batch_size]
            batches.append((batch, optimizer_state, pin_point, rotation_angle, 
                          self.w_realism, self.w_interaction, self.w_spatial, len(batches)))
        
        # Evaluate batches in parallel
        all_results = []
        with Pool(processes=num_workers) as pool:
            batch_results = pool.map(_evaluate_cached_assignment_batch, batches)
            
            # Flatten results
            for batch_result in batch_results:
                all_results.extend(batch_result)
        
        return all_results
    
    def _evaluate_cached_assignments_sequential(self, cached_assignments: List[Dict], 
                                               pin_point: np.ndarray, 
                                               rotation_angle: float) -> List[Tuple]:
        """Evaluate cached assignments sequentially (for small batches)"""
        results = []
        
        for cached_assignment in cached_assignments:
            assignment_matrix = cached_assignment['assignment_matrix']
            l_realism = cached_assignment['L_realism']
            l_interaction = cached_assignment['L_interaction']
            
            # Calculate only L_spatial for this configuration
            if self.w_spatial > 0 and self.virtual_env is not None:
                l_spatial = self.calculate_spatial_loss_with_pool(assignment_matrix, pin_point, rotation_angle)
            else:
                l_spatial = 0.0
            
            # Calculate total loss
            total_loss = (self.w_realism * l_realism + 
                         self.w_interaction * l_interaction +
                         self.w_spatial * l_spatial)
            
            loss_components = {
                "L_realism": l_realism,
                "L_interaction": l_interaction,
                "L_spatial": l_spatial,
                "total": total_loss
            }
            
            results.append((total_loss, loss_components, assignment_matrix, True))
        
        return results
    
    def save_progress(self, pin_idx: int, rot_idx: int, best_loss: float, 
                     best_assignments: List[Dict], configs_evaluated: int, 
                     configs_terminated: int, evaluation_phase: str = "fine") -> None:
        """Save optimization progress to allow recovery from interruptions
        
        Args:
            evaluation_phase: Which phase we're in - "warmup", "coarse", or "fine"
        """
        if not ENABLE_PROGRESS_TRACKING:
            return
        
        progress_data = {
            'last_pin_idx': pin_idx,
            'last_rot_idx': rot_idx,
            'evaluation_phase': evaluation_phase,  # Track which phase we're in
            'best_loss': float(best_loss) if best_loss != float('inf') else None,
            'num_best_assignments': len(best_assignments),
            'configs_evaluated': configs_evaluated,
            'configs_terminated': configs_terminated,
            'timestamp': time.time()
        }
        
        # Save best assignment info if available
        if best_assignments:
            best = best_assignments[0]
            progress_data['best_assignment_info'] = {
                'pin_idx': best['config'][0],
                'rot_idx': best['config'][1],
                'pin_point': best['config'][2].tolist(),
                'rotation_angle': float(best['config'][3]),
                'distance': float(best['distance']),
                'loss': float(best['assignment'].total_loss),
                'loss_components': {k: float(v) for k, v in best['assignment'].loss_components.items()}
            }
        
        progress_path = os.path.join(self.data_dir, PROGRESS_FILE)
        try:
            with open(progress_path, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save progress: {e}")
    
    def load_progress(self) -> Optional[Dict]:
        """Load optimization progress from file if it exists"""
        if not ENABLE_PROGRESS_TRACKING:
            return None
        
        progress_path = os.path.join(self.data_dir, PROGRESS_FILE)
        if not os.path.exists(progress_path):
            return None
        
        try:
            with open(progress_path, 'r') as f:
                progress_data = json.load(f)
            
            # Print progress info
            print("\n" + "="*60)
            print("FOUND PREVIOUS PROGRESS")
            print("="*60)
            eval_phase = progress_data.get('evaluation_phase', 'unknown')
            # Display pin and rotation with 1-based numbering (idx+1) to match terminal output
            print(f"Last evaluated: Pin[{progress_data['last_pin_idx']+1}], Rotation[{progress_data['last_rot_idx']+1}], Phase[{eval_phase}]")
            print(f"  (Internal indices: pin_idx={progress_data['last_pin_idx']}, rot_idx={progress_data['last_rot_idx']})")
            print(f"Configs evaluated: {progress_data['configs_evaluated']}")
            print(f"Configs terminated: {progress_data['configs_terminated']}")
            if progress_data.get('best_loss') is not None:
                print(f"Best loss found: {progress_data['best_loss']:.4f}")
                if 'best_assignment_info' in progress_data:
                    info = progress_data['best_assignment_info']
                    # Display with 1-based numbering to match terminal output
                    print(f"  Config: Pin[{info['pin_idx']+1}], Rotation[{info['rot_idx']+1}]")
                    print(f"  Distance: {info['distance']:.4f}m")
            print("="*60)
            
            return progress_data
        except Exception as e:
            print(f"Warning: Failed to load progress file: {e}")
            return None
    
    def delete_progress(self) -> None:
        """Delete the progress file after successful completion"""
        if not ENABLE_PROGRESS_TRACKING:
            return
        
        progress_path = os.path.join(self.data_dir, PROGRESS_FILE)
        if os.path.exists(progress_path):
            try:
                os.remove(progress_path)
                print(f"Deleted progress file: {PROGRESS_FILE}")
            except Exception as e:
                print(f"Warning: Failed to delete progress file: {e}")
    
    def optimize_with_pin_and_rotate(self, start_pin_idx: int = 0, start_rot_idx: int = 0, 
                                     auto_resume: bool = True) -> Optional[Assignment]:
        """
        Find the optimal assignment using pin & rotate spatial constraint
        
        Iterates through all pin points and rotation angles, applying occlusion (hard) and
        selection pool (soft) constraints
        
        Args:
            start_pin_idx: Starting pin index (0-based). Use to resume or start from specific point.
            start_rot_idx: Starting rotation index (0-based). Use to resume or start from specific point.
            auto_resume: If True, automatically load and resume from saved progress if available.
        """
        print("Starting pin & rotate optimization...")
        
        # Check for saved progress and optionally resume
        resume_from_progress = False
        previous_best_loss = None
        resume_phase = None  # Initialize to None (only used when resuming)
        if auto_resume:
            progress = self.load_progress()
            if progress is not None:
                user_input = input("\nDo you want to resume from this progress? (yes/no): ").strip().lower()
                if user_input in ['yes', 'y']:
                    resume_from_progress = True
                    # Resume from the NEXT configuration after the last completed one
                    start_pin_idx = progress['last_pin_idx']
                    start_rot_idx = progress['last_rot_idx'] + 1  # Move to next rotation
                    resume_phase = progress.get('evaluation_phase', 'fine')  # Which phase to resume from
                    
                    # Load previous best loss to avoid overwriting better results
                    previous_best_loss = progress.get('best_loss')
                    # Display with 1-based numbering for consistency
                    print(f"\nResuming from: Pin[{start_pin_idx+1}], Rotation[{start_rot_idx+1}], Phase[{resume_phase}]")
                    print(f"  (Internal indices: pin_idx={start_pin_idx}, rot_idx={start_rot_idx})")
                    if previous_best_loss is not None:
                        print(f"Previous best loss: {previous_best_loss:.4f} (will not overwrite unless improved)")
                else:
                    print("Starting fresh optimization...")
                    self.delete_progress()  # Clean up old progress file
        
        # Display starting point if not from beginning
        if start_pin_idx > 0 or start_rot_idx > 0:
            print(f"Starting from custom position: Pin[{start_pin_idx+1}], Rotation[{start_rot_idx+1}] (indices: {start_pin_idx}, {start_rot_idx})")
        
        # Calculate adaptive grid spacing if enabled
        if ENABLE_ADAPTIVE_PIN_GRID_SPACING:
            grid_spacing = self.calculate_average_physical_distance()
            print(f"  Grid spacing mode: ADAPTIVE (calculated: {grid_spacing:.3f}m)")
        else:
            grid_spacing = PIN_GRID_SPACING
            print(f"  Grid spacing mode: FIXED ({grid_spacing}m)")
        
        if ENABLE_ADAPTIVE_ROTATION:
            print(f"  Rotation step: ADAPTIVE")
            print(f"    → Coarse step: {ADAPTIVE_ROTATION_COARSE_STEP}° (initial screening)")
            print(f"    → Fine step: {ADAPTIVE_ROTATION_FINE_STEP}° (promising configs)")
            print(f"    → Threshold: {ADAPTIVE_ROTATION_THRESHOLD}× best_loss")
            print(f"    → Warmup: First {ADAPTIVE_ROTATION_WARMUP_CONFIGS} configs use fine step")
        else:
            print(f"  Rotation step: {ROTATION_STEP}°")
        
        print(f"  Spatial pool mode: {SPATIAL_POOL_MODE.upper()}")
        if SPATIAL_POOL_MODE == "relaxed":
            print("    → Selection pools = overlapping region only (soft constraint via L_spatial)")
        else:
            print("    → Selection pools = pedestal/circular bounds + overlapping region (hard constraint)")
        
        # Generate pin grid
        pin_points = self.generate_pin_grid(spacing=grid_spacing)
        
        # Generate rotation angles based on adaptive mode
        if ENABLE_ADAPTIVE_ROTATION:
            # For adaptive mode, we'll generate angles dynamically per pin config
            # Start with coarse angles for estimation
            coarse_rotation_angles = self.generate_rotation_angles(step=ADAPTIVE_ROTATION_COARSE_STEP)
            fine_rotation_angles = self.generate_rotation_angles(step=ADAPTIVE_ROTATION_FINE_STEP)
            
            # Estimate total configs (worst case: all pins need fine step)
            estimated_min_configs = len(pin_points) * len(coarse_rotation_angles)
            estimated_max_configs = len(pin_points) * len(fine_rotation_angles)
            print(f"Total configurations to evaluate: {estimated_min_configs}-{estimated_max_configs} "
                  f"({len(pin_points)} pins × {len(coarse_rotation_angles)}-{len(fine_rotation_angles)} rotations)")
            # Use max estimate for progress tracking
            total_configs = estimated_max_configs
        else:
            # Non-adaptive mode: use fixed rotation step
            rotation_angles = self.generate_rotation_angles()
            total_configs = len(pin_points) * len(rotation_angles)
            print(f"Total configurations to evaluate: {total_configs} ({len(pin_points)} pins × {len(rotation_angles)} rotations)")
        
        # Print debug information about matrices
        self.print_debug_matrices()
        
        # Compute adaptive K values if enabled
        assignable_indices = self.get_assignable_virtual_indices()
        if ENABLE_ADAPTIVE_TOP_K:
            print(f"\nComputing Adaptive Top-K values (initial={ADAPTIVE_K_INITIAL}, max={ADAPTIVE_K_MAX})...")
            self.adaptive_k_values = self.compute_adaptive_k_values(
                assignable_indices, 
                initial_k=ADAPTIVE_K_INITIAL,
                max_k=ADAPTIVE_K_MAX,
                verbose=True
            )
            print("\nTop-K Proxy Candidates (Adaptive):")
            print(f"  Grasp/Contact: Adaptive K (see above), Substrate: K={TOP_K_SUBSTRATE_OBJECTS}")
        else:
            self.adaptive_k_values = None
            print(f"\nTop-K Proxy Candidates (Fixed):")
            print(f"  Grasp/Contact: K={TOP_K_CONTACT_OBJECTS}, Substrate: K={TOP_K_SUBSTRATE_OBJECTS}")
        
        print("-" * 60)
        self.filter_by_top_k_realism(
            k=TOP_K_CONTACT_OBJECTS if not ENABLE_ADAPTIVE_TOP_K else None,
            k_values=self.adaptive_k_values if ENABLE_ADAPTIVE_TOP_K else None,
            assignable_indices=assignable_indices, 
            verbose=True
        )
        print("="*60)
        
        best_assignments = []  # List to track all configurations with best loss
        # Initialize best_loss with previous best if resuming, otherwise start from infinity
        if resume_from_progress and previous_best_loss is not None:
            best_loss = previous_best_loss
            print(f"\nInitialized with previous best loss: {best_loss:.4f}")
            print("Note: Output files will only be updated if a better loss is found")
        else:
            best_loss = float('inf')
        configs_evaluated = 0
        configs_terminated = 0
        
        # Cache for assignments with same candidate sets
        # Key: frozenset of banned indices, Value: dict with assignments and their realism/interaction losses
        assignment_cache = {}
        cache_hits = 0
        cache_misses = 0
        
        # Adaptive rotation tracking
        configs_needing_refinement = 0  # Count of pin configs that needed fine-grained rotation
        
        start_time = time.time()
        
        for pin_idx, pin_point in enumerate(pin_points):
            # Adaptive Rotation Strategy:
            # 1. Warmup configs (first N): Use fine step for all rotations
            # 2. After warmup with coarse step first:
            #    a. Evaluate all rotations with coarse step
            #    b. If any result is promising (≤ threshold × best_loss), re-evaluate with fine step
            # 3. Non-adaptive mode: Use fixed rotation step
            
            # Track which evaluation phase we're in (for progress saving)
            current_eval_phase = "fine"  # Default to fine
            
            if ENABLE_ADAPTIVE_ROTATION:
                is_warmup = pin_idx < ADAPTIVE_ROTATION_WARMUP_CONFIGS
                
                if is_warmup:
                    # Warmup mode: use fine step directly
                    rotation_angles_to_evaluate = fine_rotation_angles
                    current_eval_phase = "warmup"
                    if pin_idx == 0:
                        print(f"\n[WARMUP MODE] First {ADAPTIVE_ROTATION_WARMUP_CONFIGS} pin config(s) using fine rotation step ({ADAPTIVE_ROTATION_FINE_STEP}°)")
                    print(f"  Pin {pin_idx+1}: WARMUP - using fine step directly")
                else:
                    print(f"  Pin {pin_idx+1}: Starting COARSE evaluation ({ADAPTIVE_ROTATION_COARSE_STEP}° step, {len(coarse_rotation_angles)} angles)")

                    # Non-warmup adaptive mode: evaluate coarse first, then decide
                    min_loss_coarse = float('inf')
                    
                    # Phase 1: Quick screening with coarse rotation step
                    # Note: If resuming from "fine" phase, skip all coarse evaluation for this pin
                    # (it was already completed in a previous run)
                    if resume_from_progress and resume_phase == 'fine' and pin_idx <= start_pin_idx:
                        # This pin's coarse evaluation was already completed, skip to fine phase
                        print(f"  Pin {pin_idx+1}: Skipping COARSE (already completed in previous run)")
                        print(f"  Pin {pin_idx+1}: Resuming FINE evaluation from rotation {start_rot_idx+1}/{len(fine_rotation_angles)}")
                        # Directly proceed to fine evaluation, bypass Phase 2 decision logic
                        rotation_angles_to_evaluate = fine_rotation_angles
                        current_eval_phase = "fine"
                    else:
                        for rot_idx_coarse, rotation_angle_coarse in enumerate(coarse_rotation_angles):
                            # Skip configurations before the starting point (only for coarse phase resumption)
                            if resume_from_progress and resume_phase == 'coarse':
                                if pin_idx < start_pin_idx or (pin_idx == start_pin_idx and rot_idx_coarse < start_rot_idx):
                                    continue
                            
                            # Check early termination
                            should_terminate, reason = self.check_early_termination(pin_point, rotation_angle_coarse)
                            if should_terminate:
                                configs_terminated += 1
                                continue
                            
                            # Get banned indices for this configuration
                            occluded_indices = self.check_occlusion_for_config(pin_point, rotation_angle_coarse)
                            outside_intersection_indices = self._get_objects_outside_intersection(pin_point, rotation_angle_coarse)
                            
                            # Temporarily ban occluded and outside objects
                            original_banned = self.banned_physical_indices.copy()
                            self.banned_physical_indices.update(occluded_indices)
                            self.banned_physical_indices.update(outside_intersection_indices)
                            
                            # Create cache key
                            cache_key = frozenset(self.banned_physical_indices)
                            
                            # Quick evaluation to find minimum loss
                            if cache_key in assignment_cache:
                                cached_data = assignment_cache[cache_key]
                                if len(cached_data['assignments']) > CACHE_ACTIVATION_THRESHOLD:
                                    cache_hits += 1
                                    evaluation_results = self._evaluate_cached_assignments(cached_data, pin_point, rotation_angle_coarse)
                                else:
                                    valid_assignments = self.generate_all_assignments()
                                    if not valid_assignments:
                                        self.banned_physical_indices = original_banned
                                        continue
                                    total_to_evaluate = len(valid_assignments)
                                    if ENABLE_PARALLEL_EVALUATION and total_to_evaluate > PARALLEL_EVALUATION_THRESHOLD:
                                        evaluation_results = self._evaluate_assignments_parallel(valid_assignments, pin_point, rotation_angle_coarse)
                                    else:
                                        evaluation_results = self._evaluate_assignments_sequential(valid_assignments, pin_point, rotation_angle_coarse)
                            else:
                                valid_assignments = self.generate_all_assignments()
                                if not valid_assignments:
                                    self.banned_physical_indices = original_banned
                                    continue
                                total_to_evaluate = len(valid_assignments)
                                if total_to_evaluate > CACHE_ACTIVATION_THRESHOLD:
                                    cache_misses += 1
                                if ENABLE_PARALLEL_EVALUATION and total_to_evaluate > PARALLEL_EVALUATION_THRESHOLD:
                                    evaluation_results = self._evaluate_assignments_parallel(valid_assignments, pin_point, rotation_angle_coarse)
                                else:
                                    evaluation_results = self._evaluate_assignments_sequential(valid_assignments, pin_point, rotation_angle_coarse)
                                if total_to_evaluate > CACHE_ACTIVATION_THRESHOLD:
                                    self._cache_assignment_results(cache_key, evaluation_results, assignment_cache)
                            
                            # Track minimum loss from coarse evaluation  
                            num_valid_in_this_rotation = 0
                            for total_loss, loss_components, assignment_matrix, is_valid in evaluation_results:
                                if is_valid:
                                    num_valid_in_this_rotation += 1
                                    if total_loss < min_loss_coarse:
                                        min_loss_coarse = total_loss
                            
                            # Debug: print for first few pins
                            if pin_idx < 3:
                                print(f"    Coarse rot {rot_idx_coarse+1}/{len(coarse_rotation_angles)} ({np.rad2deg(rotation_angle_coarse):.0f}°): {num_valid_in_this_rotation} valid, min_loss_so_far={min_loss_coarse:.4f}")
                            
                            # Restore banned indices
                            self.banned_physical_indices = original_banned
                            configs_evaluated += 1
                        
                        # Phase 2: Decide if we need fine-grained evaluation
                        # (Only runs if coarse evaluation actually happened)
                        # Use relative difference to handle both positive and negative losses correctly
                        # Refine if coarse result is within threshold% of best_loss
                        if pin_idx < 5:  # Debug output for first few pins
                            print(f"  Pin {pin_idx+1}: Coarse phase complete. min_loss_coarse={min_loss_coarse:.4f}, best_loss={best_loss:.4f}")
                        
                        if best_loss < float('inf') and abs(best_loss) > 1e-9:
                            # Calculate relative difference: (coarse - best) / |best|
                            # Positive difference = worse loss, negative = better loss
                            relative_diff = (min_loss_coarse - best_loss) / abs(best_loss)
                            threshold_diff = ADAPTIVE_ROTATION_THRESHOLD - 1.0  # e.g., 1.5 - 1.0 = 0.5 (50% worse)
                            
                            if relative_diff <= threshold_diff:
                                # Promising: coarse result is within threshold% of best (or better)
                                configs_needing_refinement += 1
                                percent_diff = relative_diff * 100
                                print(f"  Pin {pin_idx+1}: PROMISING (loss={min_loss_coarse:.4f}, {percent_diff:+.1f}% vs best={best_loss:.4f})")
                                print(f"               → Re-evaluating with FINE rotation step ({ADAPTIVE_ROTATION_FINE_STEP}°)")
                                rotation_angles_to_evaluate = fine_rotation_angles
                                current_eval_phase = "fine"
                            else:
                                # Not promising - skip fine evaluation and move to next pin
                                if pin_idx % 10 == 0:  # Print progress occasionally
                                    percent_diff = relative_diff * 100
                                    print(f"  Pin {pin_idx+1}: Not promising (loss={min_loss_coarse:.4f}, {percent_diff:+.1f}% vs best={best_loss:.4f}), skipping")
                                continue
                        else:
                            # First few evals or best_loss is zero - always refine
                            configs_needing_refinement += 1
                            print(f"  Pin {pin_idx+1}: No baseline yet, using FINE rotation step")
                            rotation_angles_to_evaluate = fine_rotation_angles
                            current_eval_phase = "fine"
            else:
                # Non-adaptive mode
                rotation_angles_to_evaluate = rotation_angles
            
            # Main evaluation loop for this pin configuration
            for rot_idx, rotation_angle in enumerate(rotation_angles_to_evaluate):
                # Skip configurations before the starting point
                # Only skip if we're in the same phase as the one we're resuming from
                if resume_from_progress and current_eval_phase == resume_phase:
                    if pin_idx < start_pin_idx or (pin_idx == start_pin_idx and rot_idx < start_rot_idx):
                        continue
                elif resume_from_progress and pin_idx < start_pin_idx:
                    # If we're in a different phase but haven't reached the pin yet, skip
                    continue
                
                # Start timing for this config
                config_start_time = time.time()
                
                # Check early termination
                should_terminate, reason = self.check_early_termination(pin_point, rotation_angle)
                if should_terminate:
                    configs_terminated += 1
                    if configs_terminated % 100 == 0:
                        print(f"  Early terminated: {configs_terminated} configs")
                    continue
                
                # Get occluded physical indices for this configuration
                occluded_indices = self.check_occlusion_for_config(pin_point, rotation_angle)
                
                # Get physical objects outside the overlapping region (play area ∩ virtual env)
                outside_intersection_indices = self._get_objects_outside_intersection(pin_point, rotation_angle)
                
                # Debug output for first few configs
                if configs_evaluated < 3:
                    print(f"  Config ({pin_idx+1}, {rot_idx+1}): Occluded: {len(occluded_indices)}, Outside intersection: {len(outside_intersection_indices)}")
                
                # Temporarily ban occluded physical objects AND objects outside intersection
                original_banned = self.banned_physical_indices.copy()
                self.banned_physical_indices.update(occluded_indices)
                self.banned_physical_indices.update(outside_intersection_indices)
                
                # Create cache key from current banned indices
                cache_key = frozenset(self.banned_physical_indices)
                
                # Check if we've evaluated this candidate set before
                if cache_key in assignment_cache:
                    # Potential cache hit - check if this cached config has > CACHE_ACTIVATION_THRESHOLD assignments
                    cached_data = assignment_cache[cache_key]
                    total_to_evaluate = len(cached_data['assignments'])
                    
                    if total_to_evaluate > CACHE_ACTIVATION_THRESHOLD:
                        # Cache hit for large config! Reuse assignments and their L_realism/L_interaction
                        cache_hits += 1
                        print(f"  Config ({pin_idx+1}, {rot_idx+1}): Cache HIT! Reusing {total_to_evaluate} assignments, recalculating L_spatial only...")
                        
                        # Recalculate only L_spatial for this pin/rotation
                        evaluation_results = self._evaluate_cached_assignments(
                            cached_data, pin_point, rotation_angle
                        )
                    else:
                        # Cached config has <= CACHE_ACTIVATION_THRESHOLD assignments, don't use cache (treat as normal evaluation)
                        print(f"  Config ({pin_idx+1}, {rot_idx+1}): Config in cache but only {total_to_evaluate} assignments (<= {CACHE_ACTIVATION_THRESHOLD}). Evaluating normally...")
                        
                        # Generate valid assignments for this configuration
                        valid_assignments = self.generate_all_assignments()
                        
                        if not valid_assignments:
                            # Restore banned list
                            self.banned_physical_indices = original_banned
                            continue
                        
                        # Evaluate each assignment (parallel or sequential)
                        total_to_evaluate = len(valid_assignments)
                        
                        if ENABLE_PARALLEL_EVALUATION and total_to_evaluate > PARALLEL_EVALUATION_THRESHOLD:
                            # Use parallel evaluation for large batches
                            evaluation_results = self._evaluate_assignments_parallel(
                                valid_assignments, pin_point, rotation_angle
                            )
                        else:
                            # Use sequential evaluation for small batches
                            evaluation_results = self._evaluate_assignments_sequential(
                                valid_assignments, pin_point, rotation_angle
                            )
                else:
                    # Cache miss - need full evaluation
                    # Generate valid assignments for this configuration
                    valid_assignments = self.generate_all_assignments()
                    
                    if not valid_assignments:
                        # Restore banned list
                        self.banned_physical_indices = original_banned
                        continue
                    
                    # Evaluate each assignment (parallel or sequential)
                    total_to_evaluate = len(valid_assignments)
                    
                    # Only track cache miss and cache results if > CACHE_ACTIVATION_THRESHOLD assignments
                    if total_to_evaluate > CACHE_ACTIVATION_THRESHOLD:
                        cache_misses += 1
                        print(f"  Config ({pin_idx+1}, {rot_idx+1}): Cache MISS. Evaluating {total_to_evaluate} assignments...")
                    else:
                        print(f"  Config ({pin_idx+1}, {rot_idx+1}): Evaluating {total_to_evaluate} assignments (not cached, <= {CACHE_ACTIVATION_THRESHOLD})...")
                    
                    if ENABLE_PARALLEL_EVALUATION and total_to_evaluate > PARALLEL_EVALUATION_THRESHOLD:
                        # Use parallel evaluation for large batches
                        evaluation_results = self._evaluate_assignments_parallel(
                            valid_assignments, pin_point, rotation_angle
                        )
                    else:
                        # Use sequential evaluation for small batches
                        evaluation_results = self._evaluate_assignments_sequential(
                            valid_assignments, pin_point, rotation_angle
                        )
                    
                    # Cache the results only if > CACHE_ACTIVATION_THRESHOLD assignments
                    if total_to_evaluate > CACHE_ACTIVATION_THRESHOLD:
                        self._cache_assignment_results(cache_key, evaluation_results, assignment_cache)
                
                # Process results and update best configurations
                # NOTE: Results are exported immediately when a new best is found (lower loss or same loss with better distance)
                evaluation_count = len(evaluation_results)
                for total_loss, loss_components, assignment_matrix, is_valid in evaluation_results:
                    if not is_valid:
                        continue
                    
                    # Update best configurations
                    if total_loss < best_loss:
                        # Found a better configuration - reset list
                        best_loss = total_loss
                        best_assignments = []
                        
                        virtual_to_physical = {}
                        # Only create mappings for assignable virtual objects (exclude set dressing)
                        assignable_indices = self.get_assignable_virtual_indices()
                        for v_idx in assignable_indices:
                            p_idx = np.argmax(assignment_matrix[v_idx, :])
                            virtual_to_physical[v_idx] = p_idx
                        
                        assignment = Assignment(
                            assignment_matrix=assignment_matrix.copy(),
                            virtual_to_physical=virtual_to_physical,
                            total_loss=total_loss,
                            loss_components=loss_components,
                            pin_point=pin_point.copy(),
                            rotation_angle=rotation_angle
                        )
                        
                        # Calculate distance immediately for this best assignment
                        avg_distance = self.calculate_average_proxy_distance(
                            assignment.assignment_matrix,
                            assignment.pin_point,
                            assignment.rotation_angle
                        )
                        
                        best_assignments.append({
                            'assignment': assignment,
                            'config': (pin_idx, rot_idx, pin_point.copy(), rotation_angle),
                            'distance': avg_distance
                        })
                        
                        # Export results immediately when new best loss is found
                        improvement_msg = ""
                        if resume_from_progress and previous_best_loss is not None:
                            improvement = previous_best_loss - best_loss
                            improvement_msg = f" [IMPROVED from {previous_best_loss:.4f}, Δ={improvement:.4f}]"
                        print(f"      *** NEW BEST LOSS: {best_loss:.4f} (distance: {avg_distance:.4f}m){improvement_msg} - Exporting results...")
                        self.save_results(assignment, "optimization_results.json")
                        self.save_results_for_quest(assignment, "optimization_results_for_quest.json")
                        self.export_pin_rotate_visualization(assignment, "final_pin_rotate_config.png")
                        
                    elif abs(total_loss - best_loss) < 1e-9:  # Same loss (within floating point tolerance)
                        # Found another configuration with the same best loss
                        virtual_to_physical = {}
                        assignable_indices = self.get_assignable_virtual_indices()
                        for v_idx in assignable_indices:
                            p_idx = np.argmax(assignment_matrix[v_idx, :])
                            virtual_to_physical[v_idx] = p_idx
                        
                        assignment = Assignment(
                            assignment_matrix=assignment_matrix.copy(),
                            virtual_to_physical=virtual_to_physical,
                            total_loss=total_loss,
                            loss_components=loss_components,
                            pin_point=pin_point.copy(),
                            rotation_angle=rotation_angle
                        )
                        
                        # Calculate distance immediately for tie-breaking
                        avg_distance = self.calculate_average_proxy_distance(
                            assignment.assignment_matrix,
                            assignment.pin_point,
                            assignment.rotation_angle
                        )
                        
                        # Check if this is the first assignment with this loss or if it has better distance
                        if not best_assignments:
                            # First assignment with this loss (e.g., when resuming from previous best)
                            print(f"      *** FIRST ASSIGNMENT WITH LOSS: {best_loss:.4f} (distance: {avg_distance:.4f}m) - Exporting results...")
                            best_assignments = [{
                                'assignment': assignment,
                                'config': (pin_idx, rot_idx, pin_point.copy(), rotation_angle),
                                'distance': avg_distance
                            }]
                            
                            # Export results for this first config matching the resumed best loss
                            self.save_results(assignment, "optimization_results.json")
                            self.save_results_for_quest(assignment, "optimization_results_for_quest.json")
                            self.export_pin_rotate_visualization(assignment, "final_pin_rotate_config.png")
                        else:
                            # Check if this config has better distance than current best
                            current_best_distance = best_assignments[0]['distance']
                            if avg_distance < current_best_distance:
                                # This config has same loss but better distance - replace the best
                                print(f"      *** BETTER DISTANCE: {avg_distance:.4f}m < {current_best_distance:.4f}m (same loss: {best_loss:.4f}) - Exporting results...")
                                best_assignments = [{
                                    'assignment': assignment,
                                    'config': (pin_idx, rot_idx, pin_point.copy(), rotation_angle),
                                    'distance': avg_distance
                                }]
                                
                                # Export results for this better distance config
                                self.save_results(assignment, "optimization_results.json")
                                self.save_results_for_quest(assignment, "optimization_results_for_quest.json")
                                self.export_pin_rotate_visualization(assignment, "final_pin_rotate_config.png")
                            elif abs(avg_distance - current_best_distance) < 1e-6:
                                # Same distance - add to list of ties
                                best_assignments.append({
                                    'assignment': assignment,
                                    'config': (pin_idx, rot_idx, pin_point.copy(), rotation_angle),
                                    'distance': avg_distance
                                })
                            # else: worse distance, don't add to list
                
                # Config evaluation complete
                config_elapsed = time.time() - config_start_time
                print(f"    Completed: {evaluation_count} evaluated, best loss so far: {best_loss:.4f}, time: {config_elapsed:.2f}s")
                
                # Restore banned list
                self.banned_physical_indices = original_banned
                
                configs_evaluated += 1
                
                # Save progress periodically
                if ENABLE_PROGRESS_TRACKING and configs_evaluated % PROGRESS_SAVE_INTERVAL == 0:
                    self.save_progress(pin_idx, rot_idx, best_loss, best_assignments, 
                                      configs_evaluated, configs_terminated, current_eval_phase)
                
                if configs_evaluated % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Evaluated: {configs_evaluated}/{total_configs} configs, "
                          f"Terminated: {configs_terminated}, Best loss: {best_loss:.4f} ({elapsed:.1f}s)")
        
        elapsed = time.time() - start_time
        print(f"\nPin & rotate optimization completed in {elapsed:.2f}s")
        print(f"  Configurations evaluated: {configs_evaluated}")
        print(f"  Configurations terminated early: {configs_terminated}")
        if ENABLE_ADAPTIVE_ROTATION:
            print(f"  Adaptive rotation statistics:")
            print(f"    Pin configs needing fine-grained rotation: {configs_needing_refinement}")
            print(f"    Pin configs skipped (not promising): {len(pin_points) - ADAPTIVE_ROTATION_WARMUP_CONFIGS - configs_needing_refinement}")
        print(f"  Cache statistics (configs with > {CACHE_ACTIVATION_THRESHOLD} assignments only):")
        print(f"    Cache hits: {cache_hits} ({100*cache_hits/(cache_hits+cache_misses):.1f}%)" if (cache_hits+cache_misses) > 0 else "    Cache hits: 0")
        print(f"    Cache misses: {cache_misses}")
        
        # Select best configuration
        best_assignment = None
        best_config = None
        
        if best_assignments:
            if len(best_assignments) == 1:
                # Only one best configuration
                best_assignment = best_assignments[0]['assignment']
                best_config = best_assignments[0]['config']
                pin_idx, rot_idx, pin_point, rotation_angle = best_config
                print(f"  Best configuration: Pin[{pin_idx}]={pin_point}, Rotation[{rot_idx}]={np.rad2deg(rotation_angle):.1f}°")
                print(f"  Best total loss: {best_loss:.4f}")
                print(f"  Average proxy distance: {best_assignments[0]['distance']:.4f}m")
                print(f"  Loss components: {best_assignment.loss_components}")
            else:
                # Multiple configurations with same best loss AND same distance (exact tie)
                print(f"  Found {len(best_assignments)} configurations with same best loss ({best_loss:.4f}) AND same distance ({best_assignments[0]['distance']:.4f}m)")
                print(f"  Selecting first configuration...")
                
                # All configs in the list have same loss and distance, just pick the first
                best_config_data = best_assignments[0]
                best_assignment = best_config_data['assignment']
                best_config = best_config_data['config']
                
                pin_idx, rot_idx, pin_point, rotation_angle = best_config
                print(f"  Best configuration: Pin[{pin_idx}]={pin_point}, Rotation[{rot_idx}]={np.rad2deg(rotation_angle):.1f}°")
                print(f"  Best total loss: {best_loss:.4f}")
                print(f"  Average proxy distance: {best_config_data['distance']:.4f}m")
                print(f"  Loss components: {best_assignment.loss_components}")
        elif resume_from_progress and previous_best_loss is not None:
            # Resumed but didn't find anything better than previous best
            print(f"\n  No improvement found in this run.")
            print(f"  Previous best loss ({previous_best_loss:.4f}) remains the best result.")
            print(f"  Output files were NOT overwritten (previous best results preserved).")
        
        # Delete progress file after successful completion
        if best_assignment is not None:
            self.delete_progress()
        elif resume_from_progress:
            # If resumed but no improvement, also delete progress since we completed the search
            self.delete_progress()
        
        return best_assignment
    
    def print_assignment_details(self, assignment: Assignment) -> None:
        """Print detailed information about an assignment"""
        print("\n" + "="*60)
        print("OPTIMAL HAPTIC PROXY ASSIGNMENT")
        print("="*60)
        
        print(f"\nTotal Loss: {assignment.total_loss:.4f}")
        print("Loss Components:")
        for component, value in assignment.loss_components.items():
            print(f"  {component}: {value:.4f}")
        
        print(f"\nVirtual Object Assignments:")
        print("-" * 40)
        for virtual_idx, physical_idx in assignment.virtual_to_physical.items():
            virtual_obj = self.virtual_objects[virtual_idx]
            physical_obj = self.physical_objects[physical_idx]
            
            if self.realism_matrix is not None:
                realism_score = self.realism_matrix[virtual_idx, physical_idx]
            else:
                realism_score = 0.0
                
            # Show effective priority weight used in optimization
            if self.enable_priority_weighting:
                effective_priority = virtual_obj.engagement_level
            else:
                effective_priority = 1.0  # Equal priority when disabled
            
            print(f"{virtual_obj.name} -> {physical_obj.name}")
            print(f"  Priority: {effective_priority:.3f}")
            print(f"  Realism Score: {realism_score:.3f}")
            print(f"  Physical ID: {physical_obj.object_id}, Image: {physical_obj.image_id}")
            print()
    
    def print_detailed_loss_calculation(self, assignment: Assignment) -> None:
        """Print detailed calculation process for all loss components including L_interaction"""
        print("\n" + "="*60)
        print("DETAILED LOSS CALCULATION PROCESS")
        print("="*60)
        
        # Recalculate all losses with verbose output using unified function
        print("\n1. L_REALISM CALCULATION:")
        print("-" * 40)
        l_realism = self.calculate_realism_loss(assignment.assignment_matrix)
        print(f"Final L_realism: {l_realism:.4f}")
        
        print("\n2. L_INTERACTION CALCULATION:")
        print("-" * 40)
        l_interaction = self.calculate_interaction_loss(assignment.assignment_matrix, verbose=True)
        
        print("\n3. L_SPATIAL CALCULATION:")
        print("-" * 40)
        # Use pin_point and rotation_angle from assignment if available
        pin_point = assignment.pin_point if assignment.pin_point is not None else None
        rotation_angle = assignment.rotation_angle if assignment.rotation_angle is not None else None
        
        # IMPORTANT: Show what was stored in the assignment
        print(f"Stored L_spatial (from optimization): {assignment.loss_components.get('L_spatial', 'N/A'):.4f}")
        
        # NOTE: Changed from enable_spatial_constraint to w_spatial > 0
        # Spatial loss should be calculated independently of pin & rotate optimization
        if self.w_spatial > 0 and self.virtual_env is not None:
            if pin_point is None:
                # Default: use play area center if available, otherwise origin
                if self.play_area is not None:
                    pin_point = self.play_area.center
                else:
                    pin_point = np.array([0.0, 0.0, 0.0])
            if rotation_angle is None:
                rotation_angle = 0.0
            
            l_spatial = self.calculate_spatial_loss_with_pool(assignment.assignment_matrix, pin_point, rotation_angle)
            print(f"Using selection pool constraint with pin={pin_point}, rotation={np.rad2deg(rotation_angle):.1f}°")
            print(f"Current mode: {SPATIAL_POOL_MODE}")
        else:
            l_spatial = 0.0
            if self.w_spatial == 0:
                print("Spatial loss disabled (w_spatial = 0)")
            else:
                print("Spatial constraint disabled or virtual environment not available")
        
        print(f"Recalculated L_spatial: {l_spatial:.4f}")
        if abs(l_spatial - assignment.loss_components.get('L_spatial', 0)) > 0.01:
            print(f"⚠️  WARNING: Mismatch detected! Difference: {abs(l_spatial - assignment.loss_components.get('L_spatial', 0)):.4f}")
        
        print("\n4. TOTAL LOSS CALCULATION:")
        print("-" * 40)
        total_loss = (self.w_realism * l_realism + 
                     self.w_interaction * l_interaction +
                     self.w_spatial * l_spatial)
        print(f"Weights: w_realism={self.w_realism}, w_interaction={self.w_interaction}, w_spatial={self.w_spatial}")
        print(f"Components: {l_realism:.4f} + {l_interaction:.4f} + {l_spatial:.4f}")
        print(f"Final Total Loss: {total_loss:.4f}")
        print("="*60)
    
    def save_results(self, assignment: Assignment, output_file: str = "optimization_results.json") -> None:
        """Save optimization results to JSON file"""
        results = {
            "optimization_summary": {
                "total_loss": assignment.total_loss,
                "loss_components": assignment.loss_components,
                "num_virtual_objects": len(self.virtual_objects),
                "num_physical_objects": len(self.physical_objects),
                "exclusivity_enabled": self.enable_exclusivity,
                "loss_weights": {
                    "w_realism": self.w_realism,
                    "w_interaction": self.w_interaction,
                    "w_spatial": self.w_spatial
                },
                "priority_weighting_enabled": self.enable_priority_weighting
            },
            "assignments": []
        }
        
        # Add detailed assignment information
        for virtual_idx, physical_idx in assignment.virtual_to_physical.items():
            virtual_obj = self.virtual_objects[virtual_idx]
            physical_obj = self.physical_objects[physical_idx]
            
            if self.realism_matrix is not None:
                realism_score = self.realism_matrix[virtual_idx, physical_idx]
            else:
                realism_score = 0.0
                
            # Use effective priority weight from optimization
            if self.enable_priority_weighting:
                priority_weight = float(virtual_obj.engagement_level)
            else:
                priority_weight = 1.0  # Equal priority when disabled
            
            assignment_info = {
                "virtual_object": {
                    "name": virtual_obj.name,
                    "index": virtual_obj.index,
                    "engagement_level": virtual_obj.engagement_level,
                    "involvement_type": virtual_obj.involvement_type,
                    "priority_weight": priority_weight
                },
                "physical_object": {
                    "name": physical_obj.name,
                    "object_id": physical_obj.object_id,
                    "image_id": physical_obj.image_id,
                    "index": physical_obj.index
                },
                "realism_score": float(realism_score),
                # "assignment_matrix_row": assignment.assignment_matrix[virtual_idx, :].tolist()
            }

            # Lookup utilization method from proxy matching results if available
            util_method = None
            if hasattr(self, "proxy_data") and self.proxy_data:
                for entry in self.proxy_data:
                    if (entry.get("virtualObject") == virtual_obj.name and
                        entry.get("object_id") == physical_obj.object_id and
                        entry.get("image_id") == physical_obj.image_id):
                        util_method = entry.get("utilizationMethod") or entry.get("utilization_method")
                        break
            
            # If no utilization method found and this is a substrate object, 
            # look for substrate utilization method from relationship rating results
            if not util_method and virtual_obj.involvement_type == "substrate":
                # For substrate objects, we need to find the method that matches both the substrate 
                # assignment AND the contact object assignment from the same virtual relationship
                
                # First, find which virtual contact object interacts with this substrate object
                contact_virtual_obj = None
                if hasattr(self, 'virtual_relationship_pairs') and self.virtual_relationship_pairs:
                    for contact_idx, substrate_idx in self.virtual_relationship_pairs:
                        if substrate_idx == virtual_obj.index:  # This substrate is part of a relationship
                            contact_virtual_obj = self.virtual_objects[contact_idx]
                            break
                
                if contact_virtual_obj:
                    # Find the physical object assigned to the contact virtual object
                    contact_physical_obj = None
                    for contact_v_idx, contact_p_idx in assignment.virtual_to_physical.items():
                        if contact_v_idx == contact_virtual_obj.index:
                            contact_physical_obj = self.physical_objects[contact_p_idx]
                            break
                    
                    if contact_physical_obj:
                        # Load relationship rating results to find substrate utilization method
                        relationship_file = os.path.join(self.data_dir, "relationship_rating_by_dimension.json")
                        if os.path.exists(relationship_file):
                            try:
                                with open(relationship_file, 'r') as f:
                                    relationship_data = json.load(f)
                                
                                # Search for matching substrate object in dimension-based relationship data
                                # Now match BOTH contact and substrate objects to find the correct method
                                for dimension in ["harmony", "expressivity", "realism"]:
                                    if dimension not in relationship_data:
                                        continue
                                    for rel_entry in relationship_data[dimension]:
                                        if (rel_entry.get("virtualContactObject") == contact_virtual_obj.name and
                                            rel_entry.get("virtualSubstrateObject") == virtual_obj.name and
                                            rel_entry.get("contactObject_id") == contact_physical_obj.object_id and
                                            rel_entry.get("contactImage_id") == contact_physical_obj.image_id and
                                            rel_entry.get("substrateObject_id") == physical_obj.object_id and
                                            rel_entry.get("substrateImage_id") == physical_obj.image_id):
                                            util_method = rel_entry.get("substrateUtilizationMethod")
                                            print(f"Found matching substrate method for {contact_virtual_obj.name} -> {virtual_obj.name}: {contact_physical_obj.name} -> {physical_obj.name}")
                                            break
                                    if util_method:  # Break out of dimension loop if found
                                        break
                            except Exception as e:
                                print(f"Warning: Could not load relationship data for substrate utilization: {e}")
                        
                        # If still no method found, try substrate utilization results file directly
                        if not util_method:
                            substrate_file = os.path.join(self.data_dir, "substrate_utilization_results.json")
                            if os.path.exists(substrate_file):
                                try:
                                    with open(substrate_file, 'r') as f:
                                        substrate_data = json.load(f)
                                    
                                    # Search for matching relationship in substrate utilization results
                                    for sub_result in substrate_data:
                                        if (sub_result.get("virtualContactObject") == contact_virtual_obj.name and
                                            sub_result.get("virtualSubstrateObject") == virtual_obj.name and
                                            sub_result.get("contactObject_id") == contact_physical_obj.object_id and
                                            sub_result.get("contactImage_id") == contact_physical_obj.image_id and
                                            sub_result.get("substrateObject_id") == physical_obj.object_id and
                                            sub_result.get("substrateImage_id") == physical_obj.image_id):
                                            util_method = sub_result.get("substrateUtilizationMethod")
                                            print(f"Found matching substrate method in substrate_utilization_results.json")
                                            break
                                except Exception as e:
                                    print(f"Warning: Could not load substrate utilization results: {e}")
                    else:
                        print(f"Warning: Could not find physical contact object for virtual contact '{contact_virtual_obj.name}'")
                else:
                    print(f"Warning: Could not find virtual contact object for substrate '{virtual_obj.name}'")
            
            if util_method:
                assignment_info["utilization_method"] = util_method

            results["assignments"].append(assignment_info)
        
        # Save to file
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")

    def save_results_for_quest(self, assignment: Assignment, output_file: str = "optimization_results_for_quest.json") -> None:
        """Save optimization results in Quest-compatible format
        
        This generates the same format as ProXeek.py's send_optimization_results_to_quest function.
        Format includes: action, data (assignments), pinPoint, rotationAngle, timestamp, total_assignments
        """
        quest_results = []
        
        # Process each assignment
        for virtual_idx, physical_idx in assignment.virtual_to_physical.items():
            virtual_obj = self.virtual_objects[virtual_idx]
            physical_obj = self.physical_objects[physical_idx]
            
            # Lookup utilization method from proxy matching results
            util_method = "No utilization method available"
            if hasattr(self, "proxy_data") and self.proxy_data:
                for entry in self.proxy_data:
                    if (entry.get("virtualObject") == virtual_obj.name and
                        entry.get("object_id") == physical_obj.object_id and
                        entry.get("image_id") == physical_obj.image_id):
                        util_method = entry.get("utilizationMethod") or entry.get("utilization_method")
                        break
            
            # If no utilization method found and this is a substrate object,
            # look for substrate utilization method from relationship rating results
            if util_method == "No utilization method available" and virtual_obj.involvement_type == "substrate":
                # Find which virtual contact object interacts with this substrate object
                contact_virtual_obj = None
                if hasattr(self, 'virtual_relationship_pairs') and self.virtual_relationship_pairs:
                    for contact_idx, substrate_idx in self.virtual_relationship_pairs:
                        if substrate_idx == virtual_obj.index:
                            contact_virtual_obj = self.virtual_objects[contact_idx]
                            break
                
                if contact_virtual_obj:
                    # Find the physical object assigned to the contact virtual object
                    contact_physical_obj = None
                    for contact_v_idx, contact_p_idx in assignment.virtual_to_physical.items():
                        if contact_v_idx == contact_virtual_obj.index:
                            contact_physical_obj = self.physical_objects[contact_p_idx]
                            break
                    
                    if contact_physical_obj:
                        # Load relationship rating results to find substrate utilization method
                        relationship_file = os.path.join(self.data_dir, "relationship_rating_by_dimension.json")
                        if os.path.exists(relationship_file):
                            try:
                                with open(relationship_file, 'r') as f:
                                    relationship_data = json.load(f)
                                
                                # Search for matching substrate object in dimension-based relationship data
                                for dimension in ["harmony", "expressivity", "realism"]:
                                    if dimension not in relationship_data:
                                        continue
                                    for rel_entry in relationship_data[dimension]:
                                        if (rel_entry.get("virtualContactObject") == contact_virtual_obj.name and
                                            rel_entry.get("virtualSubstrateObject") == virtual_obj.name and
                                            rel_entry.get("contactObject_id") == contact_physical_obj.object_id and
                                            rel_entry.get("contactImage_id") == contact_physical_obj.image_id and
                                            rel_entry.get("substrateObject_id") == physical_obj.object_id and
                                            rel_entry.get("substrateImage_id") == physical_obj.image_id):
                                            util_method = rel_entry.get("substrateUtilizationMethod")
                                            break
                                    if util_method != "No utilization method available":
                                        break
                            except Exception as e:
                                print(f"Warning: Could not load relationship data for substrate utilization: {e}")
            
            quest_result = {
                "virtualObjectName": virtual_obj.name,
                "proxyObjectName": physical_obj.name,
                "utilizationMethod": util_method
            }
            quest_results.append(quest_result)
        
        # Extract pin & rotate configuration if available
        pin_point = None
        rotation_angle_deg = None
        if assignment.pin_point is not None:
            # Convert numpy array to list for JSON serialization
            pin_point = assignment.pin_point.tolist()
        if assignment.rotation_angle is not None:
            # Convert radians to degrees for Quest
            # IMPORTANT: Negate the angle to match Unity's rotation direction
            # Python optimization: counter-clockwise positive (standard math convention)
            # Unity Y-axis rotation: clockwise positive (left-handed coordinate system)
            rotation_angle_deg = float(-np.rad2deg(assignment.rotation_angle))
        
        # Prepare the payload for Quest
        quest_payload = {
            "action": "optimization_results",
            "data": quest_results,
            "pinPoint": pin_point,
            "rotationAngle": rotation_angle_deg,
            "timestamp": str(uuid.uuid4()),
            "total_assignments": len(quest_results)
        }
        
        # Save to file
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(quest_payload, f, indent=2)
        
        print(f"\nQuest-compatible results saved to: {output_path}")

    def export_realism_matrix_to_csv(self, output_filename: str = "realism_matrix.csv") -> None:
        """Export the full realism matrix to a CSV file"""
        if self.realism_matrix is None:
            print("Warning: Realism matrix not available for export")
            return
        
        # Create row labels (virtual object names)
        row_labels = [f"V{i}_{obj.name}" for i, obj in enumerate(self.virtual_objects)]
        
        # Create column labels (physical object names)
        col_labels = [f"P{i}_{obj.name}" for i, obj in enumerate(self.physical_objects)]
        
        # Create DataFrame with labels - use explicit pd.Index to avoid type issues
        df = pd.DataFrame(
            self.realism_matrix,
            index=pd.Index(row_labels),
            columns=pd.Index(col_labels)
        )
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, output_filename)
        df.to_csv(output_path, float_format='%.4f')
        
        print(f"Realism matrix exported to: {output_path}")
        print(f"Matrix shape: {self.realism_matrix.shape}")
        print(f"Total non-zero entries: {np.count_nonzero(self.realism_matrix)}")

    def export_pin_rotate_visualization(self, assignment: Assignment, output_filename: str = "final_pin_rotate_config.png") -> None:
        """
        Export a visualization of the final assignment configuration.
        
        If spatial constraint was enabled (pin & rotate), shows the transformed configuration.
        Otherwise, shows a simplified visualization with just the assignments.
        
        NOTE: This function is safe to call during multiprocessing evaluation because 
        matplotlib uses the 'Agg' non-interactive backend (configured at module import).
        
        Args:
            assignment: The optimal assignment with pin_point and rotation_angle (optional)
            output_filename: Name of the output image file
        """
        # Check if we have pin & rotate data
        has_pin_rotate = (assignment.pin_point is not None and 
                         assignment.rotation_angle is not None and
                         self.play_area is not None and 
                         self.virtual_env is not None)
        
        if has_pin_rotate:
            # Full spatial constraint visualization
            self._export_pin_rotate_full_visualization(assignment, output_filename)
        else:
            # Simplified assignment visualization without transformation
            self._export_simplified_assignment_visualization(assignment, output_filename)
    
    def _export_pin_rotate_full_visualization(self, assignment: Assignment, output_filename: str) -> None:
        """
        Export full pin & rotate visualization with spatial transformation
        """
        print(f"\nGenerating pin & rotate visualization...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Extract configuration
        pin_point = assignment.pin_point
        rotation_angle = assignment.rotation_angle
        rotation_deg = np.rad2deg(rotation_angle)
        
        # Plot play area
        play_area_2d = self.play_area.boundary_points[:, [0, 2]]  # X, Z only
        play_poly = MplPolygon(play_area_2d, fill=True, alpha=0.2,
                              facecolor='lightblue', edgecolor='blue', linewidth=2, label='Play Area')
        ax.add_patch(play_poly)
        
        # Plot play area center
        play_center_2d = self.play_area.center[[0, 2]]
        ax.plot(play_center_2d[0], play_center_2d[1], 'b^', markersize=8, label='Play Area Center')
        
        # Transform virtual environment
        virt_center_2d = self.virtual_env.center[[0, 2]]
        pin_2d = pin_point[[0, 2]]
        translation_2d = pin_2d - virt_center_2d
        
        # Rotate and translate virtual env corners
        virt_corners_2d = np.array(self.virtual_env.bounds_2d['corners'])
        rotated_corners = []
        for corner in virt_corners_2d:
            rotated = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
            translated = rotated + translation_2d
            rotated_corners.append(translated)
        
        # Plot virtual environment
        virt_poly = MplPolygon(rotated_corners, fill=True, alpha=0.2,
                              facecolor='lightgreen', edgecolor='green', linewidth=2,
                              linestyle='--', label='Virtual Environment')
        ax.add_patch(virt_poly)
        
        # Plot pin point (virtual center location)
        ax.plot(pin_2d[0], pin_2d[1], 'r*', markersize=15, label='Pin Point (Virtual Center)')
        
        # Plot virtual objects (transformed)
        for v_idx, v_obj in enumerate(self.virtual_objects):
            if v_obj.position is None:
                continue
            
            v_pos_2d = v_obj.position[[0, 2]]
            rotated_pos = self.rotate_point_2d(v_pos_2d, rotation_angle, virt_center_2d)
            transformed_pos = rotated_pos + translation_2d
            
            # Different markers for different types
            if v_obj.involvement_type in ['grasp', 'contact', 'substrate']:
                # Interactables - green circles
                ax.plot(transformed_pos[0], transformed_pos[1], 'go', markersize=6,
                       markeredgecolor='darkgreen', markeredgewidth=1.5)
                ax.text(transformed_pos[0], transformed_pos[1] + 0.1, v_obj.name,
                       fontsize=7, ha='center', color='darkgreen', fontweight='bold')
                
                # Draw bounds if available
                # NOTE: bounds_2d from Unity export contains pre-computed oriented corners
                # DO NOT reconstruct from min/max as that creates axis-aligned boxes
                if v_obj.bounds_2d and 'corners' in v_obj.bounds_2d:
                    corners = v_obj.bounds_2d['corners']
                    
                    if corners and len(corners) >= 3:
                        rotated_bounds = []
                        for corner in corners:
                            rot_corner = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
                            trans_corner = rot_corner + translation_2d
                            rotated_bounds.append(trans_corner)
                        bounds_poly = MplPolygon(rotated_bounds, fill=False,
                                               edgecolor='darkgreen', linewidth=1.5, alpha=0.7)
                        ax.add_patch(bounds_poly)
                        
            elif v_obj.involvement_type == 'pedestal':
                # Pedestals - yellow squares with bounds
                ax.plot(transformed_pos[0], transformed_pos[1], 'ys', markersize=8,
                       markeredgecolor='darkorange', markeredgewidth=2)
                ax.text(transformed_pos[0], transformed_pos[1] + 0.1, v_obj.name,
                       fontsize=7, ha='center', color='darkorange', fontweight='bold')
                
                # Draw bounds
                # NOTE: bounds_2d from Unity export contains pre-computed oriented corners
                if v_obj.bounds_2d and 'corners' in v_obj.bounds_2d:
                    corners = v_obj.bounds_2d['corners']
                    
                    if corners and len(corners) >= 3:
                        rotated_bounds = []
                        for corner in corners:
                            rot_corner = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
                            trans_corner = rot_corner + translation_2d
                            rotated_bounds.append(trans_corner)
                        bounds_poly = MplPolygon(rotated_bounds, fill=True, alpha=0.15,
                                               facecolor='orange', edgecolor='orange', linewidth=2)
                        ax.add_patch(bounds_poly)
                        
            elif v_obj.involvement_type == 'surroundings':
                # Surroundings - purple diamonds with bounds
                ax.plot(transformed_pos[0], transformed_pos[1], 'md', markersize=5,
                       markeredgecolor='purple', markeredgewidth=1)
                
                # Draw bounds
                # NOTE: bounds_2d from Unity export contains pre-computed oriented corners
                if v_obj.bounds_2d and 'corners' in v_obj.bounds_2d:
                    corners = v_obj.bounds_2d['corners']
                    
                    if corners and len(corners) >= 3:
                        rotated_bounds = []
                        for corner in corners:
                            rot_corner = self.rotate_point_2d(corner, rotation_angle, virt_center_2d)
                            trans_corner = rot_corner + translation_2d
                            rotated_bounds.append(trans_corner)
                        bounds_poly = MplPolygon(rotated_bounds, fill=True, alpha=0.1,
                                               facecolor='purple', edgecolor='purple',
                                               linewidth=1.5, linestyle='--')
                        ax.add_patch(bounds_poly)
        
        # Plot physical objects
        for p_idx, p_obj in enumerate(self.physical_objects):
            if p_obj.position is None:
                continue
            
            p_pos_2d = p_obj.position[[0, 2]]
            
            # Check if this physical object is assigned
            is_assigned = False
            assigned_virtual_name = None
            for v_idx, assigned_p_idx in assignment.virtual_to_physical.items():
                if assigned_p_idx == p_idx:
                    is_assigned = True
                    assigned_virtual_name = self.virtual_objects[v_idx].name
                    break
            
            if is_assigned:
                # Assigned physical objects - red X with larger marker
                ax.plot(p_pos_2d[0], p_pos_2d[1], 'rx', markersize=8,
                       markeredgewidth=2.5, alpha=1.0, label='Assigned Physical' if p_idx == 0 else '')
                ax.text(p_pos_2d[0] + 0.05, p_pos_2d[1] + 0.05,
                       f"{p_obj.name[:15]}\n→ {assigned_virtual_name}",
                       fontsize=6, ha='left', color='red', fontweight='bold')
            else:
                # Unassigned physical objects - gray X (skip banned objects)
                if p_idx in self.banned_physical_indices:
                    continue
                ax.plot(p_pos_2d[0], p_pos_2d[1], 'x', color='gray', markersize=5,
                       markeredgewidth=1.5, alpha=0.5, label='Unassigned Physical' if p_idx == 0 else '')
        
        # Add grid and formatting
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Z (meters)', fontsize=10)
        
        # Title with configuration details
        title = f"Final Pin & Rotate Configuration\n"
        title += f"Pin: [{pin_2d[0]:.2f}, {pin_2d[1]:.2f}], Rotation: {rotation_deg:.1f}°\n"
        title += f"Total Loss: {assignment.total_loss:.4f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add text box with loss components
        loss_text = "Loss Components:\n"
        for component, value in assignment.loss_components.items():
            if component != 'total':
                loss_text += f"  {component}: {value:.4f}\n"
        ax.text(0.02, 0.98, loss_text,
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Legend (avoid duplicate labels)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        # Save figure
        output_path = os.path.join(self.data_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Pin & rotate visualization saved to: {output_path}")
    
    def _export_simplified_assignment_visualization(self, assignment: Assignment, output_filename: str) -> None:
        """
        Export simplified assignment visualization without spatial transformation.
        Shows virtual and physical objects in their original positions with assignment connections.
        Uses the same visual style as the pin & rotate visualization.
        """
        print(f"\nGenerating assignment visualization (no spatial constraint)...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Plot play area if available
        if self.play_area is not None:
            play_area_2d = self.play_area.boundary_points[:, [0, 2]]  # X, Z only
            play_poly = MplPolygon(play_area_2d, fill=True, alpha=0.2,
                                  facecolor='lightblue', edgecolor='blue', linewidth=2, label='Play Area')
            ax.add_patch(play_poly)
            
            # Plot play area center
            play_center_2d = self.play_area.center[[0, 2]]
            ax.plot(play_center_2d[0], play_center_2d[1], 'b^', markersize=8, label='Play Area Center')
        
        # Plot virtual environment bounds if available
        if self.virtual_env is not None:
            virt_corners_2d = np.array(self.virtual_env.bounds_2d['corners'])
            virt_poly = MplPolygon(virt_corners_2d, fill=True, alpha=0.2,
                                  facecolor='lightgreen', edgecolor='green', linewidth=2,
                                  linestyle='--', label='Virtual Environment')
            ax.add_patch(virt_poly)
            
            # Plot virtual environment center
            virt_center_2d = self.virtual_env.center[[0, 2]]
            ax.plot(virt_center_2d[0], virt_center_2d[1], 'g^', markersize=8, label='Virtual Center')
        
        # Plot virtual objects (same style as pin & rotate visualization)
        for v_idx, v_obj in enumerate(self.virtual_objects):
            if v_obj.position is None:
                continue
            
            v_pos_2d = v_obj.position[[0, 2]]
            
            # Different markers for different types (matching full visualization)
            if v_obj.involvement_type in ['grasp', 'contact', 'substrate']:
                # Interactables - green circles
                ax.plot(v_pos_2d[0], v_pos_2d[1], 'go', markersize=6,
                       markeredgecolor='darkgreen', markeredgewidth=1.5)
                ax.text(v_pos_2d[0], v_pos_2d[1] + 0.1, v_obj.name,
                       fontsize=7, ha='center', color='darkgreen', fontweight='bold')
                
                # Draw bounds if available
                if v_obj.bounds_2d and 'corners' in v_obj.bounds_2d:
                    corners = v_obj.bounds_2d['corners']
                    if corners and len(corners) >= 3:
                        bounds_poly = MplPolygon(corners, fill=False,
                                               edgecolor='darkgreen', linewidth=1.5, alpha=0.7)
                        ax.add_patch(bounds_poly)
                        
            elif v_obj.involvement_type == 'pedestal':
                # Pedestals - yellow squares with bounds
                ax.plot(v_pos_2d[0], v_pos_2d[1], 'ys', markersize=8,
                       markeredgecolor='darkorange', markeredgewidth=2)
                ax.text(v_pos_2d[0], v_pos_2d[1] + 0.1, v_obj.name,
                       fontsize=7, ha='center', color='darkorange', fontweight='bold')
                
                # Draw bounds
                if v_obj.bounds_2d and 'corners' in v_obj.bounds_2d:
                    corners = v_obj.bounds_2d['corners']
                    if corners and len(corners) >= 3:
                        bounds_poly = MplPolygon(corners, fill=True, alpha=0.15,
                                               facecolor='orange', edgecolor='orange', linewidth=2)
                        ax.add_patch(bounds_poly)
                        
            elif v_obj.involvement_type == 'surroundings':
                # Surroundings - purple diamonds with bounds
                ax.plot(v_pos_2d[0], v_pos_2d[1], 'md', markersize=5,
                       markeredgecolor='purple', markeredgewidth=1)
                
                # Draw bounds
                if v_obj.bounds_2d and 'corners' in v_obj.bounds_2d:
                    corners = v_obj.bounds_2d['corners']
                    if corners and len(corners) >= 3:
                        bounds_poly = MplPolygon(corners, fill=True, alpha=0.1,
                                               facecolor='purple', edgecolor='purple',
                                               linewidth=1.5, linestyle='--')
                        ax.add_patch(bounds_poly)
        
        # Plot physical objects (same style as pin & rotate visualization)
        for p_idx, p_obj in enumerate(self.physical_objects):
            if p_obj.position is None:
                continue
            
            p_pos_2d = p_obj.position[[0, 2]]
            
            # Check if this physical object is assigned
            is_assigned = False
            assigned_virtual_name = None
            for v_idx, assigned_p_idx in assignment.virtual_to_physical.items():
                if assigned_p_idx == p_idx:
                    is_assigned = True
                    assigned_virtual_name = self.virtual_objects[v_idx].name
                    break
            
            if is_assigned:
                # Assigned physical objects - red X with larger marker
                ax.plot(p_pos_2d[0], p_pos_2d[1], 'rx', markersize=8,
                       markeredgewidth=2.5, alpha=1.0, label='Assigned Physical' if p_idx == 0 else '')
                ax.text(p_pos_2d[0] + 0.05, p_pos_2d[1] + 0.05,
                       f"{p_obj.name[:15]}\n→ {assigned_virtual_name}",
                       fontsize=6, ha='left', color='red', fontweight='bold')
            else:
                # Unassigned physical objects - gray X (skip banned objects)
                if p_idx in self.banned_physical_indices:
                    continue
                ax.plot(p_pos_2d[0], p_pos_2d[1], 'x', color='gray', markersize=5,
                       markeredgewidth=1.5, alpha=0.5, label='Unassigned Physical' if p_idx == 0 else '')
        
        # Add grid and formatting
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Z (meters)', fontsize=10)
        
        # Title with configuration details
        title = f"Final Assignment Configuration (No Spatial Constraint)\n"
        title += f"Total Loss: {assignment.total_loss:.4f}\n"
        title += f"Assignments: {len(assignment.virtual_to_physical)}/{len(self.virtual_objects)}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add text box with loss components
        loss_text = "Loss Components:\n"
        for component, value in assignment.loss_components.items():
            if component != 'total':
                loss_text += f"  {component}: {value:.4f}\n"
        ax.text(0.02, 0.98, loss_text,
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Legend (avoid duplicate labels)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        # Save figure
        output_path = os.path.join(self.data_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Assignment visualization saved to: {output_path}")
    
    def export_interaction_matrices_to_csv(self) -> None:
        """Export the interaction matrices to CSV files with detailed dimension ratings"""
        
        # Export 2D interaction matrix if it exists
        if hasattr(self, 'interaction_matrix') and self.interaction_matrix is not None:
            # Create physical object labels
            phys_labels = [f"P{i}_{obj.name}" for i, obj in enumerate(self.physical_objects)]
            
            # Create DataFrame for 2D interaction matrix
            df_2d = pd.DataFrame(
                self.interaction_matrix,
                index=pd.Index(phys_labels),  # Contact physical objects
                columns=pd.Index(phys_labels)  # Substrate physical objects
            )
            
            # Save 2D interaction matrix
            output_path_2d = os.path.join(self.data_dir, "interaction_matrix_2d.csv")
            df_2d.to_csv(output_path_2d, float_format='%.4f')
            
            print(f"2D Interaction matrix exported to: {output_path_2d}")
            print(f"2D Matrix shape: {self.interaction_matrix.shape}")
            print(f"2D Total non-zero entries: {np.count_nonzero(self.interaction_matrix)}")
        
        # Export 3D interaction matrix if it exists
        if (hasattr(self, 'interaction_matrix_3d') and self.interaction_matrix_3d is not None and
            hasattr(self, 'virtual_relationship_pairs') and self.virtual_relationship_pairs is not None):
            
            # Create physical object labels
            phys_labels = [f"P{i}_{obj.name}" for i, obj in enumerate(self.physical_objects)]
            
            # Export each relationship as a separate CSV with detailed ratings
            for rel_idx, (contact_idx, substrate_idx) in enumerate(self.virtual_relationship_pairs):
                contact_name = self.virtual_objects[contact_idx].name
                substrate_name = self.virtual_objects[substrate_idx].name
                
                # Get the 2D slice for this relationship
                rel_matrix = self.interaction_matrix_3d[rel_idx, :, :]
                
                # Create detailed matrix with dimension breakdown
                detailed_matrix = np.empty(rel_matrix.shape, dtype=object)
                
                # Load relationship rating results to get individual dimension ratings
                relationship_file = os.path.join(self.data_dir, "relationship_rating_by_dimension.json")
                if os.path.exists(relationship_file):
                    with open(relationship_file, 'r') as f:
                        relationship_data = json.load(f)
                    
                    # Create mapping from (contact_id, contact_img, substrate_id, substrate_img) to ratings
                    rating_map = {}
                    
                    # Process dimension-based data format
                    for dimension in ["harmony", "expressivity", "realism"]:
                        if dimension not in relationship_data:
                            continue
                            
                        for rel_result in relationship_data[dimension]:
                            if (rel_result.get("virtualContactObject") == contact_name and 
                                rel_result.get("virtualSubstrateObject") == substrate_name):
                                
                                contact_obj_id = rel_result.get("contactObject_id", -1)
                                contact_img_id = rel_result.get("contactImage_id", -1)
                                substrate_obj_id = rel_result.get("substrateObject_id", -1)
                                substrate_img_id = rel_result.get("substrateImage_id", -1)
                                rating = rel_result.get("rating", 0)
                                
                                # Create key for mapping
                                rating_key = (contact_obj_id, contact_img_id, substrate_obj_id, substrate_img_id)
                                
                                # Initialize if not exists
                                if rating_key not in rating_map:
                                    rating_map[rating_key] = {'harmony': 0, 'expressivity': 0, 'realism': 0}
                                
                                # Store the rating for this dimension
                                rating_map[rating_key][dimension] = rating
                    
                    # Calculate combined ratings for each key
                    for rating_key in rating_map:
                        ratings = rating_map[rating_key]
                        harmony_rating = ratings['harmony']
                        expressivity_rating = ratings['expressivity']
                        realism_rating = ratings['realism']
                        
                        # Use geometric mean (same as in _build_interaction_matrices)
                        if harmony_rating > 0 and expressivity_rating > 0 and realism_rating > 0:
                            combined_rating = (harmony_rating * expressivity_rating * realism_rating) ** (1/3)
                        else:
                            combined_rating = 0.0
                        
                        rating_map[rating_key]['combined'] = combined_rating
                    
                    # Fill detailed matrix
                    for i, contact_obj in enumerate(self.physical_objects):
                        for j, substrate_obj in enumerate(self.physical_objects):
                            rating_key = (contact_obj.object_id, contact_obj.image_id, 
                                        substrate_obj.object_id, substrate_obj.image_id)
                            
                            if rating_key in rating_map:
                                ratings = rating_map[rating_key]
                                detailed_matrix[i, j] = f"{ratings['combined']}({ratings['harmony']},{ratings['expressivity']},{ratings['realism']})"
                            else:
                                detailed_matrix[i, j] = f"{rel_matrix[i, j]:.0f}(0,0,0)"
                else:
                    # Fallback: just show combined ratings
                    for i in range(rel_matrix.shape[0]):
                        for j in range(rel_matrix.shape[1]):
                            detailed_matrix[i, j] = f"{rel_matrix[i, j]:.0f}(0,0,0)"
                
                # Create DataFrame for this relationship with detailed ratings
                df_rel = pd.DataFrame(
                    detailed_matrix,
                    index=pd.Index(phys_labels),  # Contact physical objects
                    columns=pd.Index(phys_labels)  # Substrate physical objects
                )
                
                # Save this relationship matrix
                safe_contact_name = contact_name.replace(" ", "_").replace("/", "_")
                safe_substrate_name = substrate_name.replace(" ", "_").replace("/", "_")
                output_filename = f"interaction_matrix_3d_rel{rel_idx}_{safe_contact_name}_to_{safe_substrate_name}.csv"
                output_path_rel = os.path.join(self.data_dir, output_filename)
                df_rel.to_csv(output_path_rel)
                
                non_zero_count = np.count_nonzero(rel_matrix)
                print(f"3D Interaction matrix for relationship {rel_idx} ({contact_name} -> {substrate_name}) exported to: {output_filename}")
                print(f"  Matrix shape: {rel_matrix.shape}, Non-zero entries: {non_zero_count}")
                print(f"  Format: combined_rating(harmony,expressivity,realism)")
            
            # Also export a summary of all relationships
            summary_data = []
            for rel_idx, (contact_idx, substrate_idx) in enumerate(self.virtual_relationship_pairs):
                contact_name = self.virtual_objects[contact_idx].name
                substrate_name = self.virtual_objects[substrate_idx].name
                rel_matrix = self.interaction_matrix_3d[rel_idx, :, :]
                
                summary_data.append({
                    'relationship_index': rel_idx,
                    'contact_virtual_object': contact_name,
                    'substrate_virtual_object': substrate_name,
                    'matrix_shape': f"{rel_matrix.shape[0]}x{rel_matrix.shape[1]}",
                    'non_zero_entries': int(np.count_nonzero(rel_matrix)),
                    'max_rating': float(np.max(rel_matrix)),
                    'mean_rating': float(np.mean(rel_matrix[rel_matrix > 0])) if np.any(rel_matrix > 0) else 0.0
                })
            
            # Save summary
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.data_dir, "interaction_matrix_3d_summary.csv")
            summary_df.to_csv(summary_path, index=False, float_format='%.4f')
            print(f"3D Interaction matrix summary exported to: interaction_matrix_3d_summary.csv")
        
        if not hasattr(self, 'interaction_matrix') and not hasattr(self, 'interaction_matrix_3d'):
            print("Warning: No interaction matrices available for export")

def main(start_pin_idx: int = 0, start_rot_idx: int = 0, auto_resume: bool = True):
    """Main function to run the optimization
    
    Args:
        start_pin_idx: Starting pin index (0-based, 1-indexed in display). 
                      Example: Use 29 to start at pin no.30
        start_rot_idx: Starting rotation index (0-based, 1-indexed in display).
                      Example: Use 0 to start at rotation no.1
        auto_resume: If True, automatically check for and offer to resume from saved progress
    """
    print("ProXeek Global Optimization")
    print("="*40)
    
    # Initialize optimizer
    optimizer = ProXeekOptimizer()
    
    # Load data
    if not optimizer.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Export realism matrix to CSV
    optimizer.export_realism_matrix_to_csv()
    
    # Export interaction matrices to CSV
    optimizer.export_interaction_matrices_to_csv()
    
    # Set loss function weights (can be adjusted)
    optimizer.w_realism = 1
    optimizer.w_interaction = 1
    optimizer.w_spatial = 1
    
    # Enable/disable exclusivity constraint
    optimizer.enable_exclusivity = True
    
    # Control priority weighting (can be adjusted)
    optimizer.set_priority_weighting(True)  # Set to False to disable priority weighting
    
    # ------------------------------------------------------------
    # INPUT: Banned physical objects (image_id, object_id) pairs
    banned_pairs_input: List[Tuple[int, int]] = [
        (1, 5),
        (2, 2),
        # (3, 6),
        # Example: (image_id, object_id)
        # (0, 5), # black ballpoint pen
        # (0, 1), # Apple iMac computer monitor with silver bezel and black screen
        # (0, 2), # small white tabletop fan with round grill
        # (0, 4), # white USB charging cable
        # (0, 8), # green retractable tape measure
        # (0, 3), # black mesh office chair
        # (1, 6), # orange ping pong ball with printed logo
        # (1, 3), # pair of scissors with black blades and orange handles
        # (2, 3), # AA battery
        # (2, 5), # red Shin Ramyun instant noodle packet
        # (3, 1), # black-handled collapsible selfie stick / monopod
        # (3, 4), # cylindrical silver 500g calibration weight marked 'M 500g'
        # (3, 3), # yellow rectangular kitchen sponge with green scrub side
        # (4, 1), # black faux leather two-seater sofa
        # (4, 4), # black and white patterned umbrella with curved handle
    ]
    optimizer.set_banned_physical_objects(banned_pairs_input)
    
    print(f"\nOptimization Parameters:")
    print(f"  Realism weight: {optimizer.w_realism}")
    print(f"  Interaction weight: {optimizer.w_interaction}")
    print(f"  Spatial weight: {optimizer.w_spatial}")
    print(f"  Priority weighting: {optimizer.enable_priority_weighting}")
    print(f"  Exclusivity constraint: {optimizer.enable_exclusivity}")
    
    # Display starting point if custom
    if start_pin_idx > 0 or start_rot_idx > 0:
        print(f"\nCustom Starting Point:")
        print(f"  Pin Index: {start_pin_idx} (Pin no.{start_pin_idx + 1})")
        print(f"  Rotation Index: {start_rot_idx} (Rotation no.{start_rot_idx + 1})")
    
    # Run optimization
    best_assignment = optimizer.optimize(start_pin_idx, start_rot_idx, auto_resume)
    
    if best_assignment:
        # Print results
        optimizer.print_assignment_details(best_assignment)
        
        # Print detailed loss calculation process
        optimizer.print_detailed_loss_calculation(best_assignment)
        
        # Save results
        optimizer.save_results(best_assignment)
        
        # Save Quest-compatible results
        optimizer.save_results_for_quest(best_assignment)
        
        # Export pin & rotate visualization if spatial constraint was used
        optimizer.export_pin_rotate_visualization(best_assignment)
    else:
        print("Optimization failed - no valid assignment found.")

if __name__ == "__main__":
    # ============================================================================
    # CUSTOMIZE STARTING POINT HERE (if needed)
    # ============================================================================
    # To start from a specific configuration, set the values below:
    # - start_pin: Pin number (1-indexed). Example: 30 for pin no.30 → use index 29
    # - start_rot: Rotation number (1-indexed). Example: 1 for rotation no.1 → use index 0
    # 
    # Note: Indices are 0-based, so subtract 1 from the display number
    # Examples:
    #   - To start at pin no.30, rotation no.1: main(start_pin_idx=29, start_rot_idx=0)
    #   - To start at pin no.1, rotation no.5: main(start_pin_idx=0, start_rot_idx=4)
    #   - To disable auto-resume: main(auto_resume=False)
    # ============================================================================
    
    # Parse command-line arguments if provided
    import argparse
    import inspect
    
    # Get default values from main() function signature
    sig = inspect.signature(main)
    default_start_pin = sig.parameters['start_pin_idx'].default
    default_start_rot = sig.parameters['start_rot_idx'].default
    default_auto_resume = sig.parameters['auto_resume'].default
    
    parser = argparse.ArgumentParser(description='ProXeek Global Optimization with Resume Support')
    parser.add_argument('--start-pin', type=int, default=default_start_pin, 
                       help=f'Starting pin index (0-based). Example: 29 for pin no.30 (default: {default_start_pin})')
    parser.add_argument('--start-rot', type=int, default=default_start_rot,
                       help=f'Starting rotation index (0-based). Example: 0 for rotation no.1 (default: {default_start_rot})')
    parser.add_argument('--no-resume', action='store_true',
                       help=f'Disable automatic resume from saved progress (default: {"enabled" if default_auto_resume else "disabled"})')
    
    args = parser.parse_args()
    
    # Run main with specified or default parameters
    main(start_pin_idx=args.start_pin, 
         start_rot_idx=args.start_rot, 
         auto_resume=not args.no_resume) 