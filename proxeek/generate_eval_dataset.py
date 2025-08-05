import csv
import json
import os
import random
import base64
from collections import defaultdict
from PIL import Image
import io

def get_property_system_prompt(property_type):
    """Get the system prompt for a specific property type with detailed rubrics"""
    
    # Property-specific rubrics from ProXeek system
    rubrics = {
        "Inertia": """
Inertia:
- 1-Strong Disagree
  - The weight difference is immediately and jarringly noticeable upon first contact
  - Center of mass feels completely misaligned (e.g., top-heavy physical object for a bottom-heavy virtual object)
  - Movement resistance feels entirely wrong (e.g., extremely light physical plastic bottle for a heavy virtual sledgehammer)
- 7-Strong Agree
  - Weight feels natural as expected throughout the entire interaction
  - Center of mass location allows intuitive and stable manipulation
  - Movement resistance and momentum feel completely consistent with the virtual object
""",
        "Interactivity": """
Interactivity:
- 1-Strong Disagree
  - Required interactive elements are completely absent or non-functional
  - User cannot perform the intended actions at all
- 7-Strong Agree
  - All interactive elements are present and function intuitively as expected
  - Degrees of freedom match exactly (rotation axes, sliding directions, button positions)
""",
        "Outline": """
Outline:
- 1-Strong Disagree
  - Size mismatch is immediately apparent and disrupts grip formation
  - Basic shape category is entirely different (e.g., spherical physical object for a virtual tetrahedron)
  - Key affordances or contact points are absent
- 7-Strong Agree
  - Size and proportions feel completely natural in the hand
  - Shape affords all expected grips and manipulation patterns
""",
        "Texture": """
Texture:
- 1-Strong Disagree
  - Surface finishing is shockingly different from expectations (e.g., extremely rough physical surface for virtual polished glass)
  - Tactile landmarks are missing or misplaced
- 7-Strong Agree
  - Surface texture feels exactly as anticipated
  - Texture transitions occur at expected locations
""",
        "Hardness": """
Hardness:
- 1-Strong Disagree
  - Compliance is completely wrong, it affects basic interaction (e.g., soft foam for a virtual metal tool)
  - Deformation behavior is shocking and breaks immersion
- 7-Strong Agree
  - Material hardness feels precisely as expected
  - Deformation behavior matches material expectations perfectly
""",
        "Temperature": """
Temperature:
- 1-Strong Disagree
  - Temperature sensation is shockingly wrong or opposite to expectations (e.g., warm/hot physical object for virtual ice cube)
  - Thermal conductivity creates wrong sensations (e.g., insulating material for a virtual metal object)
- 7-Strong Agree
  - Initial temperature matches the expected thermal sensation
  - Heat flow during contact feels natural for the material type
"""
    }
    
    # Get the specific rubric for this property type
    property_rubric = rubrics.get(property_type, "")
    
    return f"""You are an expert in haptic design who specializes in evaluating how well physical objects can serve as haptic proxies for virtual objects in VR.

Your task is to evaluate how well each physical object can replicate the {property_type} aspect of the virtual object's interaction, considering the Virtual Object Interaction Activity, the Physical Object Utilization Method, and the visual characteristics of both virtual and physical objects shown in the images.

Rate each physical object on a 7-point Likert scale based on how well its {property_type} characteristics support the intended proxy interaction:
1 - Strongly Disagree 
2 - Disagree
3 - Somewhat Disagree
4 - Neutral
5 - Somewhat Agree
6 - Agree
7 - Strongly Agree

Use the following rubric to guide your evaluation:
{property_rubric}

Format your response as: rating: x
"""

def encode_image_to_base64(image_path):
    """Convert an image file to base64 data URL, ensuring RGB/RGBA format"""
    try:
        # Open image with PIL
        with Image.open(image_path) as img:
            # Check and convert color mode if needed
            original_mode = img.mode
            print(f"🖼️  Processing {os.path.basename(image_path)}: {original_mode} mode", end="")
            
            # Convert to RGB or RGBA as needed
            if img.mode not in ['RGB', 'RGBA']:
                if img.mode == 'P' and 'transparency' in img.info:
                    # Palette mode with transparency -> RGBA
                    img = img.convert('RGBA')
                    print(f" → RGBA")
                elif img.mode in ['P', 'L', 'LA']:
                    # Palette or grayscale -> RGB
                    img = img.convert('RGB')
                    print(f" → RGB")
                else:
                    # Other modes -> RGB
                    img = img.convert('RGB')
                    print(f" → RGB")
            else:
                print(f" ✅")
            
            # Save to BytesIO buffer
            buffer = io.BytesIO()
            
            # Determine output format
            ext = os.path.splitext(image_path)[1].lower()
            if ext == '.png' or img.mode == 'RGBA':
                img.save(buffer, format='PNG')
                mime_type = 'image/png'
            elif ext in ['.jpg', '.jpeg']:
                # JPEG doesn't support transparency, ensure RGB
                if img.mode == 'RGBA':
                    # Create white background for transparency
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1])  # Use alpha as mask
                    img = rgb_img
                img.save(buffer, format='JPEG', quality=90)
                mime_type = 'image/jpeg'
            else:
                # Default to PNG for other formats
                img.save(buffer, format='PNG')
                mime_type = 'image/png'
            
            # Encode to base64
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:{mime_type};base64,{image_data}"
            
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return None

def find_object_image(object_name, image_folders):
    """Find the image file for a given object name"""
    # Try various matching strategies
    search_names = [
        object_name,
        object_name.replace(' ', ''),
        object_name.replace(' ', '_'),
        object_name.replace(' ', '-'),
    ]
    
    for folder in image_folders:
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            # Remove extension for comparison
            file_base = os.path.splitext(filename)[0]
            
            # Try exact matches and partial matches
            for search_name in search_names:
                if (search_name.lower() == file_base.lower() or 
                    search_name.lower() in file_base.lower() or
                    file_base.lower() in search_name.lower()):
                    return os.path.join(folder, filename)
    
    return None

def create_eval_user_message(row, virtual_image_url, physical_image_url, property_type):
    """Create user message with images and text for evaluation"""
    content = [
        {
            "type": "text",
            "text": f"""Virtual Object: {row.get('Virtual Object', 'N/A')}
Physical Object: {row.get('Physical Object', 'N/A')}

Virtual Object Interaction Activity: {row.get('Virtual Object Interaction Activity', 'N/A')}
Physical Object Utilization Method: {row.get('Physical Object Utilization Method', 'N/A')}

Virtual Object Bounding Box: {row.get('Virtual Object Bounding Box', 'N/A')}

Please rate how well this physical object can deliver the expected {property_type} haptic feedback for the virtual object, based on the images and information provided."""
        },
        {
            "type": "text", 
            "text": "Virtual Object Image:"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": virtual_image_url,
                "detail": "high"
            }
        },
        {
            "type": "text",
            "text": "Physical Object Image:"
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": physical_image_url,
                "detail": "high"
            }
        }
    ]
    
    return content

def generate_eval_dataset(items_per_virtual_object=1):
    """Generate evaluation dataset from Results_feed_testing.csv
    
    Args:
        items_per_virtual_object (int): Number of items to select per virtual object. Default: 3
    """
    properties = ['Inertia', 'Interactivity', 'Outline', 'Texture', 'Hardness', 'Temperature']
    eval_dataset = []
    
    # Define image folder paths
    script_dir = os.path.dirname(__file__)
    virtual_folder = os.path.join(script_dir, 'Virtual Objects')
    physical_folder = os.path.join(script_dir, 'Physical Objects')
    
    print(f"🔍 Looking for images in:")
    print(f"   Virtual: {virtual_folder}")
    print(f"   Physical: {physical_folder}")
    
    # Load CSV data
    csv_path = os.path.join(script_dir, 'output', 'Results_feed_testing.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Testing CSV file not found at {csv_path}")
        return []
    
    # Read all rows and group by virtual object
    virtual_object_groups = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Debug: print column names
        print(f"📊 CSV columns: {list(reader.fieldnames)}")
        
        for row in reader:
            virtual_object = row.get('Virtual Object', '').strip()
            if virtual_object:
                virtual_object_groups[virtual_object].append(row)
    
    print(f"📊 Found {len(virtual_object_groups)} unique virtual objects")
    
    # Select multiple rows per virtual object (randomly)
    random.seed(99)  # For reproducibility
    selected_rows = []
    
    for virtual_object, rows in virtual_object_groups.items():
        # Randomly select up to items_per_virtual_object rows for this virtual object
        available_rows = len(rows)
        num_to_select = min(items_per_virtual_object, available_rows)
        
        if num_to_select > 0:
            selected_for_this_object = random.sample(rows, num_to_select)
            selected_rows.extend(selected_for_this_object)
            print(f"🎲 {virtual_object}: selected {num_to_select}/{available_rows} items")
    
    print(f"📊 Total selected rows: {len(selected_rows)} ({items_per_virtual_object} per virtual object)")
    
    skipped_missing_images = 0
    processed_rows = 0
    
    # Process selected rows
    for row_num, row in enumerate(selected_rows, 1):
        processed_rows += 1
        
        # Extract virtual and physical object names
        virtual_object = row.get('Virtual Object', '').strip()
        physical_object = row.get('Physical Object', '').strip()
        
        if not virtual_object or not physical_object:
            print(f"⚠️  Row {row_num}: Missing object names")
            continue
        
        # Find corresponding images
        virtual_image_path = find_object_image(virtual_object, [virtual_folder])
        physical_image_path = find_object_image(physical_object, [physical_folder])
        
        if not virtual_image_path:
            print(f"⚠️  Row {row_num}: Virtual object image not found: {virtual_object}")
            skipped_missing_images += 1
            continue
            
        if not physical_image_path:
            print(f"⚠️  Row {row_num}: Physical object image not found: {physical_object}")
            skipped_missing_images += 1
            continue
        
        # Convert images to base64
        virtual_image_url = encode_image_to_base64(virtual_image_path)
        physical_image_url = encode_image_to_base64(physical_image_path)
        
        if not virtual_image_url or not physical_image_url:
            print(f"❌ Row {row_num}: Failed to encode images")
            skipped_missing_images += 1
            continue
        
        print(f"✅ Row {row_num}: Found images for {virtual_object} + {physical_object}")
        
        # Check which properties have valid ratings
        valid_properties = []
        for prop in properties:
            if prop in row and row[prop].strip() and row[prop].strip().replace('.', '').isdigit():
                rating_val = float(row[prop].strip())
                if 1 <= rating_val <= 7:
                    valid_properties.append(prop)
        
        if not valid_properties:
            print(f"⚠️  Row {row_num}: No valid property ratings")
            continue
        
        # Create evaluation entries for each valid property
        for property_type in valid_properties:
            rating_str = row[property_type].strip()
            ground_truth_rating = int(float(rating_str))  # Convert to int (handles "3.0" -> 3)
            
            # Create user message with images
            user_content = create_eval_user_message(
                row, virtual_image_url, physical_image_url, property_type
            )
            
            # Create evaluation entry (format for OpenAI Evals API)
            eval_entry = {
                "item": {
                    "virtual_object": virtual_object,
                    "physical_object": physical_object,
                    "property_type": property_type,
                    "ground_truth_rating": ground_truth_rating,
                    "user_message": user_content,
                    "system_prompt": get_property_system_prompt(property_type),
                    "interaction_activity": row.get('Virtual Object Interaction Activity', 'N/A'),
                    "utilization_method": row.get('Physical Object Utilization Method', 'N/A'),
                    "bounding_box": row.get('Virtual Object Bounding Box', 'N/A')
                }
            }
            
            eval_dataset.append(eval_entry)
    
    print(f"\n📊 Evaluation Dataset Generation Summary:")
    print(f"   Processed rows: {processed_rows}")
    print(f"   Skipped (missing images): {skipped_missing_images}") 
    print(f"   Generated evaluation items: {len(eval_dataset)}")
    
    return eval_dataset

def save_eval_dataset_as_jsonl(dataset, filename):
    """Save evaluation dataset as JSONL format for OpenAI Evals API"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"✅ Saved {len(dataset)} evaluation items to {filename}")

def main(items_per_virtual_object=5):
    """Main function for generating evaluation dataset
    
    Args:
        items_per_virtual_object (int): Number of items to select per virtual object. Default: 3
    """
    print("🎯 ProXeek Model Evaluation Dataset Generation")
    print("=" * 60)
    print("📋 Comparing: ft:gpt-4o-2024-08-06:mosra::C04ZOYAf vs gpt-4o-2024-08-06")
    
    # Generate evaluation dataset
    eval_dataset = generate_eval_dataset(items_per_virtual_object=items_per_virtual_object)
    
    if not eval_dataset:
        print("❌ No evaluation dataset generated. Check your CSV file and images.")
        return
    
    # Save dataset
    save_eval_dataset_as_jsonl(eval_dataset, 'eval_dataset.jsonl')
    
    print(f"\n🎉 Evaluation dataset ready!")
    print(f"   Total evaluation items: {len(eval_dataset)}")
    print(f"   Items per virtual object: {items_per_virtual_object}")
    print(f"   Properties tested: Inertia, Interactivity, Outline, Texture, Hardness, Temperature")
    print(f"   Ready for OpenAI Evals API")

if __name__ == "__main__":
    main() 