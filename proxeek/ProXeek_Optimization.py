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
import numpy as np
import pandas as pd
import itertools
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

# =============================================================================
# CONFIGURATION PARAMETERS - Easy to find and modify
# =============================================================================

# Top-K Filtering Configuration
TOP_K_CONTACT_OBJECTS = 5           # Number of top contact objects to consider (with tie handling)

# =============================================================================

@dataclass
class VirtualObject:
    """Represents a virtual object with its properties"""
    name: str
    index: int  # Index in the virtual objects list
    engagement_level: float  # 0.0-1.0 scale (normalized priority weight)
    involvement_type: str  # grasp, contact, substrate
    position: Optional[np.ndarray] = None  # 3D position in virtual space

@dataclass
class PhysicalObject:
    """Represents a physical object with its properties"""
    name: str
    object_id: int
    image_id: int
    index: int  # Index in the physical objects list
    position: Optional[np.ndarray] = None  # 3D position in world space

@dataclass
class Assignment:
    """Represents an assignment of virtual objects to physical objects"""
    assignment_matrix: np.ndarray  # Binary matrix X[i,j] where i=virtual, j=physical
    virtual_to_physical: Dict[int, int]  # Maps virtual object index to physical object index
    total_loss: float
    loss_components: Dict[str, float]

class ProXeekOptimizer:
    """Global optimization for haptic proxy assignment"""
    
    def __init__(self, data_dir: str = r"C:\Users\aaron\Documents\GitHub\YOLO-World-Seg\proxeek\output"):
        self.data_dir = data_dir
        self.virtual_objects: List[VirtualObject] = []
        self.physical_objects: List[PhysicalObject] = []
        self.realism_matrix: Optional[np.ndarray] = None  # realism_rating[i,j]

        self.interaction_matrix: Optional[np.ndarray] = None  # interaction_rating[j,k] for physical objects
        self.interaction_exists: Optional[np.ndarray] = None  # interaction_exists[i,k] for virtual objects
        
        # Loss function weights
        self.w_realism = 1.0
        self.w_priority = 1.0
        self.w_interaction = 1.0
        # NEW: spatial weight
        self.w_spatial = 1.0
        
        # Matrices for spatial calculations (initialized later)
        self.spatial_group_matrix: Optional[np.ndarray] = None
        self.virtual_distance_matrix: Optional[np.ndarray] = None
        self.virtual_angle_matrix: Optional[np.ndarray] = None
        self.physical_distance_matrix: Optional[np.ndarray] = None
        self.physical_angle_matrix: Optional[np.ndarray] = None
        
        # Constraints
        self.enable_exclusivity = True  # Each physical object used at most once
        
    def load_data(self) -> bool:
        """Load all required data files"""
        try:
            # Load haptic annotation data - find the most recent haptic_annotation file
            haptic_dir = r"C:\Users\aaron\Documents\GitHub\ProXeek\ProXeek\Assets\StreamingAssets\Export"
            haptic_files = [f for f in os.listdir(haptic_dir) if f.startswith("haptic_annotation") and f.endswith(".json")]
            
            if not haptic_files:
                raise FileNotFoundError(f"No haptic annotation files found in {haptic_dir}")
            
            # Use the most recent file (sorted alphabetically, which works for timestamp format)
            haptic_file = os.path.join(haptic_dir, sorted(haptic_files)[-1])
            print(f"Loading haptic annotation file: {haptic_file}")
            
            with open(haptic_file, 'r') as f:
                haptic_data = json.load(f)
            
            # Load physical object database
            physical_file = os.path.join(self.data_dir, "physical_object_database.json")
            with open(physical_file, 'r') as f:
                physical_data = json.load(f)
            
            # Load proxy matching results (for realism ratings)
            proxy_file = os.path.join(self.data_dir, "proxy_matching_results.json")
            with open(proxy_file, 'r') as f:
                proxy_data = json.load(f)
            
            # Load relationship rating results (for interaction ratings)
            relationship_file = os.path.join(self.data_dir, "relationship_rating_results.json")
            with open(relationship_file, 'r') as f:
                relationship_data = json.load(f)
            
            # NEW: Load virtual object database (for positions)
            virtual_file = os.path.join(self.data_dir, "virtual_object_database.json")
            with open(virtual_file, 'r') as f:
                virtual_data = json.load(f)
            
            # Process the loaded data
            self._process_virtual_objects(haptic_data)
            self._process_physical_objects(physical_data)
            # Assign positions to virtual objects
            self._assign_virtual_positions(virtual_data)
            # Build auxiliary matrices
            self._build_realism_matrix(proxy_data)
            self._build_interaction_matrices(haptic_data, relationship_data)
            self._build_distance_matrices()  # for spatial loss
            
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
    
    def _process_virtual_objects(self, haptic_data: Dict) -> None:
        """Process virtual objects from haptic annotation data"""
        node_annotations = haptic_data.get("nodeAnnotations", [])
        high_engagement = haptic_data.get("highEngagementOrder", [])
        medium_engagement = haptic_data.get("mediumEngagementOrder", [])
        low_engagement = haptic_data.get("lowEngagementOrder", [])
        
        # Create complete priority order for granular engagement levels
        complete_priority_order = high_engagement + medium_engagement + low_engagement
        
        # Only include grasp and contact objects (not substrate-only objects)
        filtered_objects = [obj for obj in node_annotations 
                          if obj.get("involvementType") in ["grasp", "contact", "substrate"]]
        
        for i, obj in enumerate(filtered_objects):
            name = obj.get("objectName", "")
            involvement_type = obj.get("involvementType", "")
            
            # Determine granular engagement level (normalized to 0-1 scale)
            engagement_level = 0.0  # default for unranked objects
            if name in complete_priority_order:
                # Higher priority gets higher engagement level
                priority_rank = complete_priority_order.index(name)
                # Scale to 0-1 range instead of 1-N range for better scalability
                engagement_level = (len(complete_priority_order) - priority_rank) / len(complete_priority_order)
            
            virtual_obj = VirtualObject(
                name=name,
                index=i,
                engagement_level=engagement_level,
                involvement_type=involvement_type
            )
            self.virtual_objects.append(virtual_obj)
        
        # Print priority assignments for debugging
        print(f"Priority weights assigned:")
        for virtual_obj in self.virtual_objects:
            print(f"  {virtual_obj.name}: {virtual_obj.engagement_level:.1f}")
    
    def _process_physical_objects(self, physical_data: Dict) -> None:
        """Process physical objects from database"""
        index = 0
        for image_id_str, objects in physical_data.items():
            for obj in objects:
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
        for entry in virtual_data:
            if "objectName" in entry and "globalPosition" in entry:
                gp = entry["globalPosition"]
                name_to_pos[entry["objectName"]] = np.array([
                    float(gp.get("x", 0.0)),
                    float(gp.get("y", 0.0)),
                    float(gp.get("z", 0.0))
                ])
        for v_obj in self.virtual_objects:
            if v_obj.name in name_to_pos:
                v_obj.position = name_to_pos[v_obj.name]
    
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
        ratings_processed = 0
        for rel_result in relationship_data:
            virtual_contact = rel_result.get("virtualContactObject", "")
            virtual_substrate = rel_result.get("virtualSubstrateObject", "")
            contact_obj_id = rel_result.get("contactObject_id", -1)
            contact_img_id = rel_result.get("contactImage_id", -1)
            substrate_obj_id = rel_result.get("substrateObject_id", -1)
            substrate_img_id = rel_result.get("substrateImage_id", -1)
            
            # Find the relationship index
            virtual_pair_key = (virtual_contact, virtual_substrate)
            if virtual_pair_key not in virtual_relationship_name_to_index:
                continue
                
            rel_idx = virtual_relationship_name_to_index[virtual_pair_key]
            
            # Calculate combined rating across the three dimensions
            harmony_rating = rel_result.get("harmony_rating", 0)
            expressivity_rating = rel_result.get("expressivity_rating", 0)
            realism_rating = rel_result.get("realism_rating", 0)
            
            combined_rating = (harmony_rating + expressivity_rating + realism_rating)
            
            # Map to physical object indices
            contact_key = (contact_obj_id, contact_img_id)
            substrate_key = (substrate_obj_id, substrate_img_id)
            
            if (contact_key in physical_id_to_index and 
                substrate_key in physical_id_to_index):
                contact_phys_idx = physical_id_to_index[contact_key]
                substrate_phys_idx = physical_id_to_index[substrate_key]
                self.interaction_matrix_3d[rel_idx, contact_phys_idx, substrate_phys_idx] = combined_rating
                ratings_processed += 1
        
        print(f"Processed {ratings_processed} interaction ratings into 3D matrix")
    
    def _build_distance_matrices(self) -> None:
        """Pre-compute distance and angle matrices for spatial loss"""
        # Virtual objects
        n_virtual = len(self.virtual_objects)
        self.virtual_distance_matrix = np.zeros((n_virtual, n_virtual))
        self.virtual_angle_matrix = np.zeros((n_virtual, n_virtual))
        for i in range(n_virtual):
            pos_i = self.virtual_objects[i].position
            if pos_i is None:
                continue
            for k in range(n_virtual):
                pos_k = self.virtual_objects[k].position
                if pos_k is None:
                    continue
                diff = pos_k - pos_i
                self.virtual_distance_matrix[i, k] = float(np.linalg.norm(diff))
                self.virtual_angle_matrix[i, k] = float(np.arctan2(diff[2], diff[0]))
        
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
    
    def calculate_realism_loss(self, assignment_matrix: np.ndarray) -> float:
        """Calculate L_realism = -∑ᵢ∑ⱼ (realism_rating[i,j] × X[i,j])"""
        if self.realism_matrix is None:
            return 0.0
        sum_value = np.sum(self.realism_matrix * assignment_matrix.astype(float))
        return -float(sum_value)
    
    def calculate_priority_loss(self, assignment_matrix: np.ndarray) -> float:
        """Calculate L_priority = -∑ᵢ (priority[i] × max_realism_for_object[i] × assigned_realism[i])"""
        if self.realism_matrix is None:
            return 0.0
            
        loss = 0.0
        n_virtual = len(self.virtual_objects)
        
        for i in range(n_virtual):
            # Get max realism rating for this virtual object
            max_realism = np.max(self.realism_matrix[i, :])
            
            # Get assigned realism rating
            assigned_realism = np.sum(self.realism_matrix[i, :] * assignment_matrix[i, :])
            
            # Use engagement_level from VirtualObject directly instead of priority_weights array
            priority_weight = self.virtual_objects[i].engagement_level
            
            # Calculate priority loss component
            priority_component = priority_weight * max_realism * assigned_realism
            loss -= priority_component
        
        return loss
    
    def calculate_interaction_loss(self, assignment_matrix: np.ndarray) -> float:
        """Calculate L_interaction = -∑ᵢ∑ₖ (interaction_exists[i,k] × interaction_rating[proxy_assigned[i], proxy_assigned[k]])"""
        if self.interaction_exists is None:
            return 0.0
            
        # Use 3D matrix for accurate interaction calculation
        if hasattr(self, 'interaction_matrix_3d') and hasattr(self, 'virtual_relationship_pairs'):
            return self._calculate_interaction_loss_3d(assignment_matrix)
        else:
            return 0.0
    
    def _calculate_interaction_loss_3d(self, assignment_matrix: np.ndarray) -> float:
        """Calculate interaction loss using 3D interaction matrix"""
        if not hasattr(self, 'interaction_matrix_3d') or not hasattr(self, 'virtual_relationship_pairs'):
            return 0.0
        if self.interaction_matrix_3d is None or self.virtual_relationship_pairs is None or self.interaction_exists is None:
            return 0.0
            
        loss = 0.0
        n_virtual = len(self.virtual_objects)
        
        # Iterate through each virtual relationship
        for rel_idx, (contact_virtual_idx, substrate_virtual_idx) in enumerate(self.virtual_relationship_pairs):
            if self.interaction_exists[contact_virtual_idx, substrate_virtual_idx] > 0:
                # Find assigned physical objects for this virtual relationship
                proxy_contact = np.argmax(assignment_matrix[contact_virtual_idx, :])
                proxy_substrate = np.argmax(assignment_matrix[substrate_virtual_idx, :])
                
                # Get interaction rating from 3D matrix for this specific relationship
                interaction_rating = self.interaction_matrix_3d[rel_idx, proxy_contact, proxy_substrate]
                loss -= interaction_rating
        
        return loss
    
    def calculate_spatial_loss(self, assignment_matrix: np.ndarray) -> float:
        """Calculate L_spatial based only on distance distortions (no direction)"""
        if (self.spatial_group_matrix is None or
            self.virtual_distance_matrix is None or self.physical_distance_matrix is None):
            return 0.0
        # Cast to local vars for clarity
        virtual_dist_mat = self.virtual_distance_matrix
        physical_dist_mat = self.physical_distance_matrix
        n_virtual = len(self.virtual_objects)
        assigned_physical = np.argmax(assignment_matrix, axis=1)
        loss = 0.0
        for i in range(n_virtual):
            for k in range(n_virtual):
                if i == k or self.spatial_group_matrix[i, k] == 0:
                    continue
                v_dist = virtual_dist_mat[i, k]
                p_dist = physical_dist_mat[assigned_physical[i], assigned_physical[k]]
                loss += (v_dist - p_dist) ** 2
        return float(loss)
    
    def calculate_total_loss(self, assignment_matrix: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate the total multi-objective loss function"""
        l_realism = self.calculate_realism_loss(assignment_matrix)
        l_priority = self.calculate_priority_loss(assignment_matrix)
        l_interaction = self.calculate_interaction_loss(assignment_matrix)
        l_spatial = self.calculate_spatial_loss(assignment_matrix)
        
        total_loss = (self.w_realism * l_realism + 
                     self.w_priority * l_priority + 
                     self.w_interaction * l_interaction +
                     self.w_spatial * l_spatial)
        
        loss_components = {
            "L_realism": l_realism,
            "L_priority": l_priority,
            "L_interaction": l_interaction,
            "L_spatial": l_spatial,
            "total": total_loss
        }
        
        return total_loss, loss_components
    
    def is_valid_assignment(self, assignment_matrix: np.ndarray) -> bool:
        """Check if assignment satisfies hard constraints"""
        # Hard constraint 1: Each virtual object gets exactly one proxy
        row_sums = np.sum(assignment_matrix, axis=1)
        if not np.allclose(row_sums, 1.0):
            return False
        
        # Hard constraint 2: Exclusivity (if enabled)
        if self.enable_exclusivity:
            col_sums = np.sum(assignment_matrix, axis=0)
            if np.any(col_sums > 1.0):
                return False
        
        return True
    
    def filter_by_top_k_realism(self, k: Optional[int] = None) -> List[np.ndarray]:
        """Generate assignments using only top-K realism scores for grasp and contact virtual objects"""
        n_virtual = len(self.virtual_objects)
        n_physical = len(self.physical_objects)
        
        # Set default K to fixed value
        if k is None:
            k = TOP_K_CONTACT_OBJECTS
            print(f"Setting K = {k}")
        
        # Ensure K doesn't exceed available physical objects
        k = min(k, n_physical)
        
        print(f"Applying Top-K filtering with K = {k}")
        
        # Check if realism matrix is available
        if self.realism_matrix is None:
            print("Warning: Realism matrix not available, falling back to exhaustive generation")
            # Fall back to exhaustive generation
            valid_assignments = []
            if self.enable_exclusivity:
                for perm in itertools.permutations(range(n_physical), n_virtual):
                    assignment_matrix = np.zeros((n_virtual, n_physical))
                    for i, j in enumerate(perm):
                        assignment_matrix[i, j] = 1.0
                    valid_assignments.append(assignment_matrix)
            else:
                for assignment_tuple in itertools.product(range(n_physical), repeat=n_virtual):
                    assignment_matrix = np.zeros((n_virtual, n_physical))
                    for i, j in enumerate(assignment_tuple):
                        assignment_matrix[i, j] = 1.0
                    valid_assignments.append(assignment_matrix)
            return valid_assignments
        
        # Get top-K physical objects for each virtual object
        top_k_assignments = {}
        for v_idx in range(n_virtual):
            virtual_obj = self.virtual_objects[v_idx]
            
            # Only apply Top-K filtering to grasp and contact objects
            if virtual_obj.involvement_type in ["grasp", "contact"]:
                realism_scores = self.realism_matrix[v_idx, :]
                
                # Handle ties: include all objects with the same score as the k-th object
                if len(realism_scores) <= k:
                    # If we have k or fewer objects, use all of them
                    top_k_indices = list(range(len(realism_scores)))
                else:
                    # Get sorted indices by realism score (highest first)
                    sorted_indices = np.argsort(realism_scores)[::-1]
                    
                    # Find the k-th highest score
                    kth_score = realism_scores[sorted_indices[k-1]]
                    
                    # Include all objects with scores >= k-th score
                    top_k_indices = []
                    for idx in sorted_indices:
                        if realism_scores[idx] >= kth_score:
                            top_k_indices.append(idx)
                        else:
                            break  # Since list is sorted, we can break early
                
                top_k_assignments[v_idx] = top_k_indices
                
                print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): Top-{k} realism scores (with ties): {realism_scores[top_k_indices]}")
                print(f"    Selected {len(top_k_indices)} objects (including ties)")
            else:
                # For substrate objects, use all physical objects since realism ratings are zero
                top_k_assignments[v_idx] = list(range(n_physical))
                print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): Using all physical objects (substrate)")
        
        # Generate assignments using only top-K combinations
        valid_assignments = []
        assignment_count = 0
        
        print(f"Generating assignments from filtered combinations...")
        
        for assignment_tuple in itertools.product(*[top_k_assignments[i] for i in range(n_virtual)]):
            # Check exclusivity constraint if enabled
            if self.enable_exclusivity and len(set(assignment_tuple)) != len(assignment_tuple):
                continue
                
            assignment_matrix = np.zeros((n_virtual, n_physical))
            for i, j in enumerate(assignment_tuple):
                assignment_matrix[i, j] = 1.0
            valid_assignments.append(assignment_matrix)
            assignment_count += 1
            
            # Progress indicator for large searches
            if assignment_count % 10000 == 0:
                print(f"    Generated {assignment_count} assignments...")
        
        print(f"Top-K filtering generated {len(valid_assignments)} assignments")
        return valid_assignments
    
    def generate_all_assignments(self) -> List[np.ndarray]:
        """Generate all valid assignment permutations with optional Top-K filtering"""
        n_virtual = len(self.virtual_objects)
        n_physical = len(self.physical_objects)
        
        if n_physical < n_virtual:
            print(f"Error: Not enough physical objects ({n_physical}) for virtual objects ({n_virtual})")
            return []
        
        # Calculate theoretical assignment count (for informational purposes)
        if self.enable_exclusivity:
            theoretical_count = math.factorial(n_physical) // math.factorial(n_physical - n_virtual)
        else:
            theoretical_count = n_physical ** n_virtual
        
        print(f"Theoretical assignment count: {theoretical_count}")
        
        # Always use Top-K filtering for consistent behavior
        print("Using Top-K filtered assignment generation")
        k = TOP_K_CONTACT_OBJECTS
        return self.filter_by_top_k_realism(k=k)
    
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
        
        print("="*60)

    def optimize(self) -> Optional[Assignment]:
        """Find the optimal assignment with minimum loss"""
        print("Starting global optimization...")
        
        # Print debug information about matrices
        self.print_debug_matrices()
        
        # Generate all possible assignments
        all_assignments = self.generate_all_assignments()
        if not all_assignments:
            return None
        
        best_assignment = None
        best_loss = float('inf')
        best_components = None
        
        print(f"Evaluating {len(all_assignments)} assignments...")
        start_time = time.time()
        
        for i, assignment_matrix in enumerate(all_assignments):
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i}/{len(all_assignments)} assignments ({elapsed:.1f}s)")
            
            # Verify assignment is valid (should be by construction, but double-check)
            if not self.is_valid_assignment(assignment_matrix):
                continue
            
            # Calculate loss
            total_loss, loss_components = self.calculate_total_loss(assignment_matrix)
            
            # Update best assignment if this is better
            if total_loss < best_loss:
                best_loss = total_loss
                best_components = loss_components
                
                # Create virtual-to-physical mapping
                virtual_to_physical = {}
                for virtual_idx in range(len(self.virtual_objects)):
                    physical_idx = np.argmax(assignment_matrix[virtual_idx, :])
                    virtual_to_physical[virtual_idx] = physical_idx
                
                best_assignment = Assignment(
                    assignment_matrix=assignment_matrix.copy(),
                    virtual_to_physical=virtual_to_physical,
                    total_loss=total_loss,
                    loss_components=loss_components
                )
        
        elapsed = time.time() - start_time
        print(f"Optimization completed in {elapsed:.2f}s")
        print(f"Best total loss: {best_loss:.4f}")
        print(f"Loss components: {best_components}")
        
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
                
            # Use engagement_level from VirtualObject directly
            priority = virtual_obj.engagement_level
            
            print(f"{virtual_obj.name} -> {physical_obj.name}")
            print(f"  Priority: {priority:.1f}")
            print(f"  Realism Score: {realism_score:.3f}")
            print(f"  Physical ID: {physical_obj.object_id}, Image: {physical_obj.image_id}")
            print()
    
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
                    "w_priority": self.w_priority,
                    "w_interaction": self.w_interaction,
                    "w_spatial": self.w_spatial
                }
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
                
            # Use engagement_level from VirtualObject directly
            priority_weight = float(virtual_obj.engagement_level)
            
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
                "assignment_matrix_row": assignment.assignment_matrix[virtual_idx, :].tolist()
            }
            results["assignments"].append(assignment_info)
        
        # Save to file
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")

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
                relationship_file = os.path.join(self.data_dir, "relationship_rating_results.json")
                if os.path.exists(relationship_file):
                    with open(relationship_file, 'r') as f:
                        relationship_data = json.load(f)
                    
                    # Create mapping from (contact_id, contact_img, substrate_id, substrate_img) to ratings
                    rating_map = {}
                    for rel_result in relationship_data:
                        if (rel_result.get("virtualContactObject") == contact_name and 
                            rel_result.get("virtualSubstrateObject") == substrate_name):
                            
                            contact_obj_id = rel_result.get("contactObject_id", -1)
                            contact_img_id = rel_result.get("contactImage_id", -1)
                            substrate_obj_id = rel_result.get("substrateObject_id", -1)
                            substrate_img_id = rel_result.get("substrateImage_id", -1)
                            
                            # Get individual dimension ratings
                            harmony_rating = rel_result.get("harmony_rating", 0)
                            expressivity_rating = rel_result.get("expressivity_rating", 0)
                            realism_rating = rel_result.get("realism_rating", 0)
                            combined_rating = rel_result.get("combined_rating", harmony_rating + expressivity_rating + realism_rating)
                            
                            # Create key for mapping
                            rating_key = (contact_obj_id, contact_img_id, substrate_obj_id, substrate_img_id)
                            rating_map[rating_key] = {
                                'harmony': harmony_rating,
                                'expressivity': expressivity_rating,
                                'realism': realism_rating,
                                'combined': combined_rating
                            }
                    
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

def main():
    """Main function to run the optimization"""
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
    optimizer.w_realism = 1.0
    optimizer.w_priority = 0.5
    optimizer.w_interaction = 0.5
    optimizer.w_spatial = 0.5
    
    # Enable/disable exclusivity constraint
    optimizer.enable_exclusivity = True
    
    print(f"\nOptimization Parameters:")
    print(f"  Realism weight: {optimizer.w_realism}")
    print(f"  Priority weight: {optimizer.w_priority}")
    print(f"  Interaction weight: {optimizer.w_interaction}")
    print(f"  Spatial weight: {optimizer.w_spatial}")
    print(f"  Exclusivity constraint: {optimizer.enable_exclusivity}")
    
    # Run optimization
    best_assignment = optimizer.optimize()
    
    if best_assignment:
        # Print results
        optimizer.print_assignment_details(best_assignment)
        
        # Save results
        optimizer.save_results(best_assignment)
    else:
        print("Optimization failed - no valid assignment found.")

if __name__ == "__main__":
    main() 