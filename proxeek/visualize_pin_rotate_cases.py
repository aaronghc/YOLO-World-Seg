"""
Visualize Pin & Rotate Cases: Show virtual environment and play area overlapping at different configurations
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from typing import Dict, List, Tuple

def load_data():
    """Load play area and virtual environment data"""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    # Load physical object database (for play area)
    physical_db_path = os.path.join(output_dir, 'physical_object_database.json')
    with open(physical_db_path, 'r') as f:
        physical_data = json.load(f)
    
    # Load haptic annotation (for virtual environment)
    haptic_files = [f for f in os.listdir(output_dir) if f.startswith("haptic_annotation") and f.endswith(".json")]
    haptic_file = os.path.join(output_dir, sorted(haptic_files)[-1])
    with open(haptic_file, 'r') as f:
        haptic_data = json.load(f)
    
    return physical_data, haptic_data

def extract_play_area(physical_data):
    """Extract play area boundary points"""
    play_area = physical_data.get('playArea')
    if not play_area:
        return None
    
    boundary_points = play_area.get('boundaryPoints', [])
    points_2d = np.array([[pt['x'], pt['z']] for pt in boundary_points])
    
    center = np.array([play_area['center']['x'], play_area['center']['z']])
    
    return {
        'points': points_2d,
        'center': center,
        'width': play_area['width'],
        'depth': play_area['depth']
    }

def extract_virtual_env(haptic_data):
    """Extract virtual environment bounds"""
    virt_env_bounds = haptic_data.get('virtualEnvironmentBounds')
    if not virt_env_bounds:
        return None
    
    footprint_2d = virt_env_bounds.get('footprint2D', {})
    
    min_x = footprint_2d.get('minX', 0)
    max_x = footprint_2d.get('maxX', 0)
    min_z = footprint_2d.get('minZ', 0)
    max_z = footprint_2d.get('maxZ', 0)
    
    # Create corners
    corners = np.array([
        [min_x, min_z],
        [max_x, min_z],
        [max_x, max_z],
        [min_x, max_z]
    ])
    
    center_x = footprint_2d.get('centerX', 0)
    center_z = footprint_2d.get('centerZ', 0)
    center = np.array([center_x, center_z])
    
    virt_env_center = haptic_data.get('virtualEnvironmentCenter', {})
    rotation_center = np.array([
        virt_env_center.get('x', 0),
        virt_env_center.get('z', 0)
    ])
    
    return {
        'corners': corners,
        'center': center,
        'rotation_center': rotation_center,
        'width': footprint_2d.get('width', 0),
        'depth': footprint_2d.get('depth', 0)
    }

def extract_virtual_objects(haptic_data):
    """Extract virtual object positions, types, and bounds"""
    node_annotations = haptic_data.get('nodeAnnotations', [])
    
    interactables = []
    pedestals = []
    surroundings = []
    
    for obj in node_annotations:
        pos = obj.get('globalPosition', {})
        pos_2d = np.array([pos.get('x', 0), pos.get('z', 0)])
        
        name = obj.get('objectName', 'Unknown')
        involvement = obj.get('involvementType', '')
        
        # Extract 2D bounds if available
        # NOTE: bounds_2d from Unity export contains pre-computed oriented corners
        # DO NOT reconstruct from min/max as that creates axis-aligned boxes
        bounds_2d = obj.get('bounds2D', {})
        bounds_corners = None
        if bounds_2d and 'corners' in bounds_2d:
            corners_data = bounds_2d['corners']
            if corners_data and len(corners_data) >= 3:
                bounds_corners = np.array([[c.get('x', 0), c.get('z', 0)] for c in corners_data])
        
        obj_data = {
            'name': name, 
            'pos': pos_2d,
            'bounds': bounds_corners
        }
        
        if involvement in ['grasp', 'contact']:
            interactables.append(obj_data)
        elif involvement == 'pedestal':
            pedestals.append(obj_data)
        elif involvement == 'surroundings':
            surroundings.append(obj_data)
    
    return interactables, pedestals, surroundings

def extract_physical_objects(physical_data):
    """Extract physical object positions if available, or mark as unpositioned"""
    data = physical_data.get('data', physical_data)
    
    physical_objs = []
    unpositioned_objs = []
    
    for image_id, objects in data.items():
        if image_id in ['action', 'timestamp', 'total_objects', 'playArea']:
            continue  # Skip metadata
        if not isinstance(objects, list):
            continue
        
        for obj in objects:
            name = obj.get('object', 'Unknown')
            world_pos = obj.get('worldposition')
            if world_pos:
                pos_2d = np.array([world_pos['x'], world_pos['z']])
                physical_objs.append({'name': name, 'pos': pos_2d, 'positioned': True})
            else:
                # Object exists but doesn't have world position yet
                unpositioned_objs.append({'name': name, 'positioned': False})
    
    return physical_objs, unpositioned_objs

def rotate_point_2d(point, angle, center):
    """Rotate a 2D point around a center"""
    p = point - center
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = np.array([
        p[0] * cos_a - p[1] * sin_a,
        p[0] * sin_a + p[1] * cos_a
    ])
    return rotated + center

def rotate_polygon(polygon, angle, center):
    """Rotate a polygon around a center"""
    return np.array([rotate_point_2d(pt, angle, center) for pt in polygon])

def generate_pin_grid(play_area, spacing=0.291):
    """Generate pin points on play area"""
    center = play_area['center']
    width = play_area['width']
    depth = play_area['depth']
    
    x_min = center[0] - width / 2
    x_max = center[0] + width / 2
    z_min = center[1] - depth / 2
    z_max = center[1] + depth / 2
    
    x_points = np.arange(center[0], x_max + spacing/2, spacing)
    x_points = np.concatenate([np.arange(center[0] - spacing, x_min - spacing/2, -spacing)[::-1], x_points])
    
    z_points = np.arange(center[1], z_max + spacing/2, spacing)
    z_points = np.concatenate([np.arange(center[1] - spacing, z_min - spacing/2, -spacing)[::-1], z_points])
    
    xx, zz = np.meshgrid(x_points, z_points)
    pin_points = np.stack([xx.flatten(), zz.flatten()], axis=1)
    
    # Filter to only points inside play area
    boundary = play_area['points']
    valid_pins = []
    for pin in pin_points:
        if point_in_polygon(pin, boundary):
            valid_pins.append(pin)
    
    return np.array(valid_pins) if valid_pins else pin_points

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
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

def plot_configuration(ax, play_area, virt_env, interactables, pedestals, surroundings, 
                       physical_objs, unpositioned_objs, pin_point, rotation_angle, title):
    """Plot a single pin & rotate configuration"""
    
    # Plot play area
    play_poly = Polygon(play_area['points'], fill=True, alpha=0.2, 
                       facecolor='lightblue', edgecolor='blue', linewidth=2, label='Play Area')
    ax.add_patch(play_poly)
    
    # Plot play area center (smaller)
    ax.plot(play_area['center'][0], play_area['center'][1], 
           'b^', markersize=6, label='Play Area Center')
    
    # Transform virtual environment
    virt_center = virt_env['rotation_center']
    translation = pin_point - virt_center
    
    # Rotate and translate virtual env corners
    rotated_corners = rotate_polygon(virt_env['corners'], rotation_angle, virt_center)
    translated_corners = rotated_corners + translation
    
    # Plot virtual environment
    virt_poly = Polygon(translated_corners, fill=True, alpha=0.2,
                       facecolor='lightgreen', edgecolor='green', linewidth=2, 
                       linestyle='--', label='Virtual Environment')
    ax.add_patch(virt_poly)
    
    # Plot pin point (where virtual center is placed) - smaller
    ax.plot(pin_point[0], pin_point[1], 'r*', markersize=10, 
           label=f'Pin Point (Virtual Center)')
    
    # Transform and plot virtual objects
    for obj in interactables:
        rotated_pos = rotate_point_2d(obj['pos'], rotation_angle, virt_center)
        transformed_pos = rotated_pos + translation
        
        # Draw bounds if available
        if obj['bounds'] is not None:
            rotated_bounds = rotate_polygon(obj['bounds'], rotation_angle, virt_center)
            transformed_bounds = rotated_bounds + translation
            bounds_poly = Polygon(transformed_bounds, fill=False, 
                                edgecolor='darkgreen', linewidth=1.5, linestyle='-', alpha=0.7)
            ax.add_patch(bounds_poly)
        
        # Draw position marker (smaller)
        ax.plot(transformed_pos[0], transformed_pos[1], 'go', markersize=5, 
               markeredgecolor='darkgreen', markeredgewidth=1.5)
        ax.text(transformed_pos[0], transformed_pos[1] + 0.15, obj['name'], 
               fontsize=7, ha='center', color='darkgreen', fontweight='bold')
    
    for obj in pedestals:
        rotated_pos = rotate_point_2d(obj['pos'], rotation_angle, virt_center)
        transformed_pos = rotated_pos + translation
        
        # Draw bounds if available (pedestals should have bounds)
        if obj['bounds'] is not None:
            rotated_bounds = rotate_polygon(obj['bounds'], rotation_angle, virt_center)
            transformed_bounds = rotated_bounds + translation
            bounds_poly = Polygon(transformed_bounds, fill=True, alpha=0.15,
                                facecolor='orange', edgecolor='orange', 
                                linewidth=2, linestyle='-')
            ax.add_patch(bounds_poly)
        
        # Draw position marker (smaller)
        ax.plot(transformed_pos[0], transformed_pos[1], 'ys', markersize=6,
               markeredgecolor='darkorange', markeredgewidth=2)
        ax.text(transformed_pos[0], transformed_pos[1] + 0.15, obj['name'], 
               fontsize=7, ha='center', color='darkorange', fontweight='bold')
    
    for obj in surroundings:
        rotated_pos = rotate_point_2d(obj['pos'], rotation_angle, virt_center)
        transformed_pos = rotated_pos + translation
        
        # Draw bounds if available
        if obj['bounds'] is not None:
            rotated_bounds = rotate_polygon(obj['bounds'], rotation_angle, virt_center)
            transformed_bounds = rotated_bounds + translation
            bounds_poly = Polygon(transformed_bounds, fill=True, alpha=0.1,
                                facecolor='purple', edgecolor='purple', 
                                linewidth=1.5, linestyle='--')
            ax.add_patch(bounds_poly)
        
        # Draw position marker (smaller)
        ax.plot(transformed_pos[0], transformed_pos[1], 'md', markersize=4,
               markeredgecolor='purple', markeredgewidth=1)
    
    # Plot physical objects with labels (smaller markers)
    if physical_objs:
        for idx, obj in enumerate(physical_objs):
            ax.plot(obj['pos'][0], obj['pos'][1], 'rx', markersize=5, 
                   markeredgewidth=2, alpha=0.8)
            # Add small label for first few physical objects to avoid clutter
            if idx < 3:  # Only label first 3 to avoid clutter
                ax.text(obj['pos'][0] + 0.05, obj['pos'][1] + 0.05, 
                       obj['name'][:20], fontsize=6, ha='left', 
                       color='red', alpha=0.7)
    
    # Add text box showing unpositioned physical objects
    if unpositioned_objs:
        obj_list = ', '.join([obj['name'][:15] for obj in unpositioned_objs[:5]])
        if len(unpositioned_objs) > 5:
            obj_list += f'... (+{len(unpositioned_objs)-5} more)'
        info_text = f'Physical objects (no position):\n{obj_list}'
        ax.text(0.02, 0.02, info_text, 
               transform=ax.transAxes, fontsize=6, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, edgecolor='red', linewidth=0.5))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    
    # Add rotation angle text
    angle_deg = np.rad2deg(rotation_angle)
    ax.text(0.02, 0.98, f'Rotation: {angle_deg:.0f}°', 
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def visualize_cases():
    """Generate and save visualization of different pin & rotate cases"""
    print("Loading data...")
    physical_data, haptic_data = load_data()
    
    play_area = extract_play_area(physical_data)
    virt_env = extract_virtual_env(haptic_data)
    interactables, pedestals, surroundings = extract_virtual_objects(haptic_data)
    physical_objs, unpositioned_objs = extract_physical_objects(physical_data)
    
    if not play_area:
        print("❌ No play area data found!")
        return
    
    if not virt_env:
        print("❌ No virtual environment data found!")
        return
    
    print(f"✓ Play area: {play_area['width']:.2f}m × {play_area['depth']:.2f}m")
    print(f"✓ Virtual env: {virt_env['width']:.2f}m × {virt_env['depth']:.2f}m")
    print(f"✓ Interactables: {len(interactables)}, Pedestals: {len(pedestals)}, Surroundings: {len(surroundings)}")
    print(f"✓ Physical objects: {len(physical_objs)} positioned, {len(unpositioned_objs)} unpositioned")
    
    # Debug: Check which objects have bounds
    print("\nBounds information:")
    for obj in interactables:
        bounds_status = "✓ Has bounds" if obj['bounds'] is not None else "✗ No bounds"
        print(f"  Interactable '{obj['name']}': {bounds_status}")
    for obj in pedestals:
        bounds_status = "✓ Has bounds" if obj['bounds'] is not None else "✗ No bounds"
        if obj['bounds'] is not None:
            bounds_size = np.max(obj['bounds'], axis=0) - np.min(obj['bounds'], axis=0)
            print(f"  Pedestal '{obj['name']}': {bounds_status} (size: {bounds_size[0]:.2f}m × {bounds_size[1]:.2f}m)")
        else:
            print(f"  Pedestal '{obj['name']}': {bounds_status}")
    for obj in surroundings:
        bounds_status = "✓ Has bounds" if obj['bounds'] is not None else "✗ No bounds"
        print(f"  Surroundings '{obj['name']}': {bounds_status}")
    
    # Generate pin grid
    pin_points = generate_pin_grid(play_area, spacing=0.291)
    print(f"✓ Generated {len(pin_points)} pin points")
    
    # Select interesting configurations
    # 1. Center pin, 0° rotation
    # 2. Center pin, 90° rotation
    # 3. Corner pin, 0° rotation
    # 4. Corner pin, 45° rotation
    # 5. Edge pin, 180° rotation
    # 6. Random pin, 270° rotation
    
    center_pin = play_area['center']
    
    # Find corner and edge pins
    if len(pin_points) > 1:
        distances = np.linalg.norm(pin_points - center_pin, axis=1)
        farthest_idx = np.argmax(distances)
        corner_pin = pin_points[farthest_idx]
        
        mid_dist_idx = np.argsort(distances)[len(distances)//2]
        edge_pin = pin_points[mid_dist_idx]
        
        random_idx = len(pin_points) // 4
        random_pin = pin_points[random_idx]
    else:
        corner_pin = edge_pin = random_pin = center_pin
    
    configurations = [
        (center_pin, 0, "Center Pin, 0° Rotation"),
        (center_pin, np.deg2rad(90), "Center Pin, 90° Rotation"),
        (corner_pin, 0, "Corner Pin, 0° Rotation"),
        (corner_pin, np.deg2rad(45), "Corner Pin, 45° Rotation"),
        (edge_pin, np.deg2rad(180), "Edge Pin, 180° Rotation"),
        (random_pin, np.deg2rad(270), "Random Pin, 270° Rotation"),
    ]
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (pin_point, rotation_angle, title) in enumerate(configurations):
        plot_configuration(
            axes[idx], play_area, virt_env, interactables, pedestals, 
            surroundings, physical_objs, unpositioned_objs, pin_point, rotation_angle, title
        )
    
    # Add legend to last subplot
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Add custom legend entries for virtual objects
    from matplotlib.lines import Line2D
    custom_handles = handles + [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
               markersize=8, markeredgecolor='darkgreen', markeredgewidth=1.5, label='Interactables'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='y', 
               markersize=10, markeredgecolor='orange', markeredgewidth=1.5, label='Pedestals'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='m', 
               markersize=6, markeredgecolor='purple', markeredgewidth=1, label='Surroundings'),
        Line2D([0], [0], marker='x', color='r', markersize=6, 
               markeredgewidth=2, alpha=0.7, linestyle='None', label='Physical Objects')
    ]
    
    fig.legend(custom_handles, [h.get_label() for h in custom_handles], 
              loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.suptitle('Pin & Rotate Configuration Cases\nVirtual Environment (green) vs Play Area (blue)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    output_path = os.path.join(output_dir, 'pin_rotate_cases.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Also create a detailed view of one interesting case
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    plot_configuration(
        ax2, play_area, virt_env, interactables, pedestals, 
        surroundings, physical_objs, unpositioned_objs, corner_pin, np.deg2rad(45),
        "Detailed View: Corner Pin, 45° Rotation"
    )
    
    # Add all pin points as small dots
    ax2.scatter(pin_points[:, 0], pin_points[:, 1], c='gray', s=10, alpha=0.3, label='Available Pin Points')
    
    ax2.legend(loc='upper right', fontsize=8)
    
    output_path2 = os.path.join(output_dir, 'pin_rotate_detailed.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Detailed view saved to: {output_path2}")
    
    plt.show()

if __name__ == '__main__':
    visualize_cases()

