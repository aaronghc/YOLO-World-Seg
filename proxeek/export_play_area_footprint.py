#!/usr/bin/env python3
"""
Play Area Footprint Exporter

This module generates a 2D top-down visualization of the physical play area,
showing the boundary, recognized physical objects, and spacing grid.
"""

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional

# Visualization constants
IMAGE_SIZE = 2048
PADDING = 100
GRID_SPACING_METERS = 0.5  # 0.5m grid spacing
BACKGROUND_COLOR = (255, 255, 255, 255)  # White
BOUNDARY_COLOR = (0, 128, 0, 255)  # Green for boundary
GRID_COLOR = (200, 200, 200, 128)  # Light gray for grid
OBJECT_COLOR = (0, 100, 200, 255)  # Blue for physical objects
OBJECT_DOT_RADIUS = 6


def load_physical_object_database(file_path: str) -> Dict:
    """Load physical object database JSON"""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_play_area_and_objects(data: Dict) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Extract play area boundary and physical objects from database
    
    Returns:
        Tuple of (play_area_data, list_of_physical_objects)
    """
    play_area = data.get('playArea', None)
    
    # Extract physical objects from all images
    physical_objects = []
    if 'data' in data and isinstance(data['data'], dict):
        for image_id, objects in data['data'].items():
            physical_objects.extend(objects)
    
    return play_area, physical_objects


def calculate_bounds(play_area: Optional[Dict], objects: List[Dict]) -> Tuple[float, float, float, float]:
    """
    Calculate overall bounds for visualization
    
    Returns:
        (min_x, max_x, min_z, max_z)
    """
    min_x, max_x = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    
    # Include play area boundary if available
    if play_area and 'boundaryPoints' in play_area:
        for point in play_area['boundaryPoints']:
            min_x = min(min_x, point['x'])
            max_x = max(max_x, point['x'])
            min_z = min(min_z, point['z'])
            max_z = max(max_z, point['z'])
    
    # Include physical objects
    for obj in objects:
        if 'worldposition' in obj:
            pos = obj['worldposition']
            min_x = min(min_x, pos['x'])
            max_x = max(max_x, pos['x'])
            min_z = min(min_z, pos['z'])
            max_z = max(max_z, pos['z'])
    
    # Add padding
    pad = 0.5  # 0.5m padding
    min_x -= pad
    max_x += pad
    min_z -= pad
    max_z += pad
    
    # Ensure minimum size
    width = max_x - min_x
    depth = max_z - min_z
    if width < 2.0:
        center_x = (min_x + max_x) * 0.5
        min_x = center_x - 1.0
        max_x = center_x + 1.0
    if depth < 2.0:
        center_z = (min_z + max_z) * 0.5
        min_z = center_z - 1.0
        max_z = center_z + 1.0
    
    return min_x, max_x, min_z, max_z


def world_to_screen(x: float, z: float, bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """
    Convert world coordinates (X, Z) to screen coordinates
    
    Args:
        x, z: World coordinates
        bounds: (min_x, max_x, min_z, max_z)
    
    Returns:
        (screen_x, screen_y) in pixels
    """
    min_x, max_x, min_z, max_z = bounds
    
    world_width = max_x - min_x
    world_depth = max_z - min_z
    max_dimension = max(world_width, world_depth)
    scale = (IMAGE_SIZE - 2 * PADDING) / max_dimension
    
    # Offset from min corner
    offset_x = x - min_x
    offset_z = z - min_z
    
    # Convert to screen space (flip Z for top-down view)
    screen_x = PADDING + offset_x * scale
    screen_y = PADDING + (world_depth - offset_z) * scale
    
    return int(screen_x), int(screen_y)


def draw_grid(draw: ImageDraw, bounds: Tuple[float, float, float, float], grid_spacing: float):
    """Draw grid lines at regular intervals"""
    min_x, max_x, min_z, max_z = bounds
    
    # Vertical grid lines (constant X)
    x = np.ceil(min_x / grid_spacing) * grid_spacing
    while x <= max_x:
        screen_x1, screen_y1 = world_to_screen(x, min_z, bounds)
        screen_x2, screen_y2 = world_to_screen(x, max_z, bounds)
        draw.line([(screen_x1, screen_y1), (screen_x2, screen_y2)], fill=GRID_COLOR, width=1)
        x += grid_spacing
    
    # Horizontal grid lines (constant Z)
    z = np.ceil(min_z / grid_spacing) * grid_spacing
    while z <= max_z:
        screen_x1, screen_y1 = world_to_screen(min_x, z, bounds)
        screen_x2, screen_y2 = world_to_screen(max_x, z, bounds)
        draw.line([(screen_x1, screen_y1), (screen_x2, screen_y2)], fill=GRID_COLOR, width=1)
        z += grid_spacing


def draw_play_area_boundary(draw: ImageDraw, play_area: Dict, bounds: Tuple[float, float, float, float]):
    """Draw play area boundary as a polygon"""
    if 'boundaryPoints' not in play_area:
        return
    
    boundary_points = play_area['boundaryPoints']
    if len(boundary_points) < 3:
        return
    
    # Convert to screen coordinates
    screen_points = []
    for point in boundary_points:
        screen_x, screen_y = world_to_screen(point['x'], point['z'], bounds)
        screen_points.append((screen_x, screen_y))
    
    # Draw filled polygon (semi-transparent)
    fill_color = (0, 128, 0, 30)  # Light green fill
    draw.polygon(screen_points, fill=fill_color, outline=BOUNDARY_COLOR)
    
    # Draw thicker outline
    for i in range(len(screen_points)):
        start = screen_points[i]
        end = screen_points[(i + 1) % len(screen_points)]
        draw.line([start, end], fill=BOUNDARY_COLOR, width=3)


def draw_physical_objects(draw: ImageDraw, objects: List[Dict], bounds: Tuple[float, float, float, float]):
    """Draw physical objects as dots"""
    for obj in objects:
        if 'worldposition' not in obj:
            continue
        
        pos = obj['worldposition']
        screen_x, screen_y = world_to_screen(pos['x'], pos['z'], bounds)
        
        # Draw dot
        bbox = [
            screen_x - OBJECT_DOT_RADIUS,
            screen_y - OBJECT_DOT_RADIUS,
            screen_x + OBJECT_DOT_RADIUS,
            screen_y + OBJECT_DOT_RADIUS
        ]
        draw.ellipse(bbox, fill=OBJECT_COLOR, outline=(0, 50, 150, 255), width=2)


def add_legend(draw: ImageDraw, play_area: Optional[Dict]):
    """Add legend explaining the visualization"""
    legend_x = 20
    legend_y = 20
    line_height = 25
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Title
    draw.text((legend_x, legend_y), "Play Area Footprint", fill=(0, 0, 0, 255), font=font)
    legend_y += line_height * 1.5
    
    # Grid spacing
    draw.text((legend_x, legend_y), f"Grid: {GRID_SPACING_METERS}m spacing", fill=(100, 100, 100, 255), font=font)
    legend_y += line_height
    
    # Play area info
    if play_area:
        width = play_area.get('width', 0)
        depth = play_area.get('depth', 0)
        area = play_area.get('area', 0)
        draw.text((legend_x, legend_y), f"Play Area: {width:.2f}m × {depth:.2f}m ({area:.2f}m²)", 
                  fill=(0, 128, 0, 255), font=font)
        legend_y += line_height
    
    # Color legend
    draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=BOUNDARY_COLOR)
    draw.text((legend_x + 20, legend_y), "Play area boundary", fill=(0, 0, 0, 255), font=font)
    legend_y += line_height
    
    draw.ellipse([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=OBJECT_COLOR)
    draw.text((legend_x + 20, legend_y), "Physical objects", fill=(0, 0, 0, 255), font=font)


def export_play_area_footprint(database_file: str, output_file: str):
    """
    Export play area footprint visualization
    
    Args:
        database_file: Path to physical_object_database.json
        output_file: Path to output PNG file
    """
    print(f"Loading physical object database: {database_file}")
    data = load_physical_object_database(database_file)
    
    print("Extracting play area and objects...")
    play_area, objects = extract_play_area_and_objects(data)
    
    if play_area is None:
        print("⚠️ Warning: No play area data found in database")
    else:
        print(f"✓ Play area found: {len(play_area.get('boundaryPoints', []))} boundary points")
    
    print(f"✓ Found {len(objects)} physical objects")
    
    # Calculate bounds
    bounds = calculate_bounds(play_area, objects)
    min_x, max_x, min_z, max_z = bounds
    print(f"World bounds: X=[{min_x:.2f}, {max_x:.2f}], Z=[{min_z:.2f}, {max_z:.2f}]")
    
    # Create image
    image = Image.new('RGBA', (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image, 'RGBA')
    
    # Draw grid
    print("Drawing grid...")
    draw_grid(draw, bounds, GRID_SPACING_METERS)
    
    # Draw play area boundary
    if play_area:
        print("Drawing play area boundary...")
        draw_play_area_boundary(draw, play_area, bounds)
    
    # Draw physical objects
    print("Drawing physical objects...")
    draw_physical_objects(draw, objects, bounds)
    
    # Add legend
    add_legend(draw, play_area)
    
    # Save image
    image.save(output_file, 'PNG')
    print(f"✓ Play area footprint exported to: {output_file}")
    
    return output_file


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export play area footprint visualization')
    parser.add_argument('database_file', help='Path to physical_object_database.json')
    parser.add_argument('--output', '-o', help='Output PNG file path', 
                        default='play_area_footprint.png')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.database_file):
        print(f"Error: Database file not found: {args.database_file}")
        return 1
    
    try:
        export_play_area_footprint(args.database_file, args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

