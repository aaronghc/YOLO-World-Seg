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
import random
from typing import List, Dict, Any, Tuple, Optional, Set
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
    """Global optimization for haptic proxy assignment
    
    The optimizer now uses updated loss functions:
    - L_realism = (1/N_grasp_contact) × -∑ᵢ∑ⱼ (2 × priority_weight[i] × realism_rating[i,j] × X[i,j]) for grasp and contact objects only
    - L_interaction = (1/N_relationships) × -∑ᵢ∑ₖ (interaction_exists[i,k] × interaction_rating[proxy_assigned[i], proxy_assigned[k]] × combined_priority_weight[i,k])
    - L_spatial = (1/N_spatial) × Σᵢₖ [(virtual_distance[i,k] - physical_distance[proxy_i,proxy_k])² × combined_priority_weight[i,k]]
    
    Where:
    - priority_weight[i] uses a three-tier system: High=5.0, Medium=3.0, Low=1.0
    - combined_priority_weight[i,k] = priority_weight[i] + priority_weight[k] for both interaction and spatial losses
    - N_grasp_contact = count of grasp and contact objects (normalization factor for L_realism)
    - N_relationships = count of interaction relationships (normalization factor for L_interaction)  
    - N_spatial = count of spatial relationships (normalization factor for L_spatial)
    """
    
    def __init__(self, data_dir: str = r"C:\Users\aaron\Documents\GitHub\YOLO-World-Seg\proxeek\output"):
        self.data_dir = data_dir
        self.virtual_objects: List[VirtualObject] = []
        self.physical_objects: List[PhysicalObject] = []
        self.realism_matrix: Optional[np.ndarray] = None  # realism_rating[i,j]

        self.interaction_matrix: Optional[np.ndarray] = None  # interaction_rating[j,k] for physical objects
        self.interaction_exists: Optional[np.ndarray] = None  # interaction_exists[i,k] for virtual objects
        
        # Loss function weights
        self.w_realism = 1.0
        self.w_interaction = 1.0
        # NEW: spatial weight
        self.w_spatial = 1.0
        
        # Priority control
        self.enable_priority_weighting = True  # Control whether to apply priority weights
        
        # Matrices for spatial calculations (initialized later)
        self.spatial_group_matrix: Optional[np.ndarray] = None
        self.virtual_distance_matrix: Optional[np.ndarray] = None
        self.virtual_angle_matrix: Optional[np.ndarray] = None
        self.physical_distance_matrix: Optional[np.ndarray] = None
        self.physical_angle_matrix: Optional[np.ndarray] = None
        
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
                physical_data = json.load(f)
            
            # Debug: Show sample physical objects
            sample_objects = []
            for image_id, objects in physical_data.items():
                for obj in objects[:2]:  # First 2 objects
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
            # Build auxiliary matrices
            self._build_realism_matrix(proxy_data)
            self._build_interaction_matrices(haptic_data, relationship_data)
            self._build_distance_matrices()  # for spatial loss
            
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
        high_engagement = haptic_data.get("highEngagementOrder", [])
        medium_engagement = haptic_data.get("mediumEngagementOrder", [])
        low_engagement = haptic_data.get("lowEngagementOrder", [])
        
        # Create complete priority order for granular engagement levels
        complete_priority_order = high_engagement + medium_engagement + low_engagement
        
        print(f"DEBUG: Processing {len(node_annotations)} node annotations")
        print(f"DEBUG: Available involvement types: {list(set(obj.get('involvementType', '') for obj in node_annotations))}")
        
        # Include ALL object types (grasp, contact, substrate)
        filtered_objects = [obj for obj in node_annotations 
                          if obj.get("involvementType") in ["grasp", "contact", "substrate"]]
        
        print(f"DEBUG: Filtered to {len(filtered_objects)} objects")
        for obj in filtered_objects:
            print(f"  DEBUG: {obj.get('objectName', 'Unknown')} - Type: {obj.get('involvementType', 'Unknown')}")
        
        # Create separate priority orders for different object types
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
                    break
        
        print(f"DEBUG: Priority order for grasp/contact objects: {grasp_contact_priority_order}")
        print(f"DEBUG: Priority order for substrate objects: {substrate_priority_order}")
        
        for i, obj in enumerate(filtered_objects):
            name = obj.get("objectName", "")
            involvement_type = obj.get("involvementType", "")
            
            # Calculate engagement level based on three-tier priority system
            engagement_level = 1.0  # default for unranked objects
            
            # Determine engagement level based on which priority list the object is in
            if name in high_engagement:
                engagement_level = 5.0  # High priority
            elif name in medium_engagement:
                engagement_level = 3.0  # Medium priority
            elif name in low_engagement:
                engagement_level = 1.0  # Low priority
            else:
                engagement_level = 1.0  # Default for objects not in any engagement list
            
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
            print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): {virtual_obj.engagement_level:.3f}")
        
        print(f"DEBUG: Created {len(self.virtual_objects)} virtual objects:")
        for i, obj in enumerate(self.virtual_objects):
            print(f"  DEBUG: [{i}] {obj.name} - Type: {obj.involvement_type}")
    
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
            
            print(f"Processing spatial group: {group_title} with {len(object_names)} objects")
            
            # Mark all objects in the group as spatially related to each other
            for i, obj_name_i in enumerate(object_names):
                if obj_name_i not in virtual_name_to_index:
                    print(f"  Warning: Virtual object '{obj_name_i}' not found in virtual objects list")
                    continue
                idx_i = virtual_name_to_index[obj_name_i]
                
                for j, obj_name_j in enumerate(object_names):
                    if obj_name_j not in virtual_name_to_index:
                        print(f"  Warning: Virtual object '{obj_name_j}' not found in virtual objects list")
                        continue
                    idx_j = virtual_name_to_index[obj_name_j]
                    
                    # Mark spatial relationship (symmetric)
                    if idx_i != idx_j:
                        self.spatial_group_matrix[idx_i, idx_j] = 1.0
                        self.spatial_group_matrix[idx_j, idx_i] = 1.0
                        spatial_relationships_found += 1
                        print(f"    Marked spatial relationship: {obj_name_i} <-> {obj_name_j}")
            
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
                contact_priority_weight = self.virtual_objects[contact_virtual_idx].engagement_level
                substrate_priority_weight = self.virtual_objects[substrate_virtual_idx].engagement_level
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
        """Calculate L_spatial based on distance distortions weighted by combined priority weights"""
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
    
    def calculate_total_loss(self, assignment_matrix: np.ndarray, verbose: bool = False) -> Tuple[float, Dict[str, float]]:
        """Calculate the total multi-objective loss function"""
        l_realism = self.calculate_realism_loss(assignment_matrix)
        l_interaction = self.calculate_interaction_loss(assignment_matrix, verbose)
        l_spatial = self.calculate_spatial_loss(assignment_matrix)
        
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

                # Exclude banned indices first
                allowed_indices = [idx for idx in range(len(realism_scores)) if idx not in self.banned_physical_indices]

                # Handle ties: include all objects with the same score as the k-th object among allowed only
                if len(allowed_indices) <= k:
                    # If we have k or fewer allowed objects, use all of them
                    top_k_indices = allowed_indices
                else:
                    # Get allowed indices sorted by realism score (highest first)
                    sorted_allowed = sorted(allowed_indices, key=lambda idx: realism_scores[idx], reverse=True)

                    # Find the k-th highest score among allowed
                    kth_score = realism_scores[sorted_allowed[k-1]]

                    # Include all allowed objects with scores >= k-th score (tie handling)
                    top_k_indices = []
                    for idx in sorted_allowed:
                        if realism_scores[idx] >= kth_score:
                            top_k_indices.append(idx)
                        else:
                            break  # Since list is sorted, we can break early
                
                top_k_assignments[v_idx] = top_k_indices
                
                print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): Top-{k} realism scores (with ties): {realism_scores[top_k_indices]}")
                print(f"    Selected {len(top_k_indices)} objects (including ties):")
                for idx in top_k_indices:
                    physical_obj_name = self.physical_objects[idx].name
                    score = realism_scores[idx]
                    print(f"      - {physical_obj_name}: {score:.3f}")
            else:
                # For substrate objects, use all physical objects since realism ratings are zero
                substrate_indices = [idx for idx in range(n_physical) if idx not in self.banned_physical_indices]
                
                # Add randomization for substrate objects when interaction weight is 0
                if self.w_interaction == 0:
                    random.shuffle(substrate_indices)
                    print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): Using all physical objects (substrate) - RANDOMIZED ORDER due to w_interaction=0:")
                else:
                    print(f"  {virtual_obj.name} ({virtual_obj.involvement_type}): Using all physical objects (substrate):")
                
                top_k_assignments[v_idx] = substrate_indices
                for idx in substrate_indices:
                    physical_obj_name = self.physical_objects[idx].name
                    print(f"      - {physical_obj_name}")
        
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
        
        # Print spatial matrices information
        print(f"\nSpatial Matrices:")
        print("-" * 40)
        
        # Spatial group matrix
        if self.spatial_group_matrix is not None:
            spatial_relationships = np.count_nonzero(self.spatial_group_matrix)
            print(f"Spatial Group Matrix: {spatial_relationships} non-zero entries")
            
            if spatial_relationships > 0:
                print("Spatial relationships found:")
                for i in range(self.spatial_group_matrix.shape[0]):
                    for j in range(self.spatial_group_matrix.shape[1]):
                        if self.spatial_group_matrix[i, j] > 0 and i != j:
                            obj_i_name = self.virtual_objects[i].name
                            obj_j_name = self.virtual_objects[j].name
                            print(f"  {obj_i_name} <-> {obj_j_name}")
            else:
                print("No spatial relationships found")
        else:
            print("Spatial Group Matrix: Not initialized")
        
        # Virtual distance matrix
        if self.virtual_distance_matrix is not None:
            non_zero_distances = np.count_nonzero(self.virtual_distance_matrix)
            print(f"Virtual Distance Matrix: {non_zero_distances} non-zero entries")
            
            # Show ALL distances (not just samples)
            if non_zero_distances > 0:
                print("Complete Virtual Distance Matrix:")
                print("  Format: Object_A -> Object_B: distance")
                
                # First, show which objects have positions
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
                
                print("  All distance entries:")
                for i in range(self.virtual_distance_matrix.shape[0]):
                    for j in range(self.virtual_distance_matrix.shape[1]):
                        if self.virtual_distance_matrix[i, j] > 0 and i != j:
                            obj_i_name = self.virtual_objects[i].name
                            obj_j_name = self.virtual_objects[j].name
                            distance = self.virtual_distance_matrix[i, j]
                            print(f"    {obj_i_name} -> {obj_j_name}: {distance:.3f}m")
                
                # Also show the matrix structure
                print(f"  Matrix shape: {self.virtual_distance_matrix.shape}")
                print(f"  Expected entries for {len(self.virtual_objects)} objects: {len(self.virtual_objects)} × {len(self.virtual_objects)} - {len(self.virtual_objects)} = {(len(self.virtual_objects) * len(self.virtual_objects)) - len(self.virtual_objects)}")
            else:
                print("  No distances calculated - all objects may be missing positions")
        else:
            print("Virtual Distance Matrix: Not initialized")
        
        # Physical distance matrix
        if self.physical_distance_matrix is not None:
            non_zero_distances = np.count_nonzero(self.physical_distance_matrix)
            print(f"Physical Distance Matrix: {non_zero_distances} non-zero entries")
        else:
            print("Physical Distance Matrix: Not initialized")
        
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
        
        best_assignments = []  # Store all assignments with minimum loss
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
        virtual_to_physical = {}
        for virtual_idx in range(len(self.virtual_objects)):
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
        
        # Recalculate all losses with verbose output
        print("\n1. L_REALISM CALCULATION:")
        print("-" * 40)
        l_realism = self.calculate_realism_loss(assignment.assignment_matrix)
        print(f"Final L_realism: {l_realism:.4f}")
        
        print("\n2. L_INTERACTION CALCULATION:")
        print("-" * 40)
        l_interaction = self.calculate_interaction_loss(assignment.assignment_matrix, verbose=True)
        
        print("\n3. L_SPATIAL CALCULATION:")
        print("-" * 40)
        l_spatial = self.calculate_spatial_loss(assignment.assignment_matrix)
        print(f"Final L_spatial: {l_spatial:.4f}")
        
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
    optimizer.w_realism = 1
    optimizer.w_interaction = 1
    optimizer.w_spatial = 1
    
    # Enable/disable exclusivity constraint
    optimizer.enable_exclusivity = True
    
    # Control priority weighting (can be adjusted)
    optimizer.set_priority_weighting(False)  # Set to False to disable priority weighting
    
    # ------------------------------------------------------------
    # INPUT: Banned physical objects (image_id, object_id) pairs
    banned_pairs_input: List[Tuple[int, int]] = [
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
    
    # Run optimization
    best_assignment = optimizer.optimize()
    
    if best_assignment:
        # Print results
        optimizer.print_assignment_details(best_assignment)
        
        # Print detailed loss calculation process
        optimizer.print_detailed_loss_calculation(best_assignment)
        
        # Save results
        optimizer.save_results(best_assignment)
    else:
        print("Optimization failed - no valid assignment found.")

if __name__ == "__main__":
    main() 