#!/usr/bin/env python3
"""
Sample VFT Data Format Viewer

This script shows what the vision fine-tuning data looks like
including the base64-encoded images.
"""
import json
import os
from generate_vft_dataset import generate_vft_dataset, split_dataset

def truncate_base64(base64_data, max_length=100):
    """Truncate a base64 string for display purposes"""
    if len(base64_data) > max_length:
        return base64_data[:max_length] + f"... [{len(base64_data)} total chars]"
    return base64_data

def display_content_item(item, indent="      "):
    """Display a content item with proper formatting"""
    if item.get('type') == 'text':
        print(f"{indent}📝 Text: {item.get('text', '')[:100]}...")
    elif item.get('type') == 'image_url':
        image_url = item.get('image_url', {})
        url = image_url.get('url', '')
        detail = image_url.get('detail', 'auto')
        
        if url.startswith('data:'):
            # Extract format and truncate base64
            parts = url.split(',')
            if len(parts) == 2:
                mime_info = parts[0]  # e.g., "data:image/png;base64"
                base64_data = parts[1]
                print(f"{indent}🖼️  Image: {mime_info}")
                print(f"{indent}    Detail: {detail}")
                print(f"{indent}    Data: {truncate_base64(base64_data, 80)}")
            else:
                print(f"{indent}🖼️  Image: {truncate_base64(url, 100)}")
        else:
            print(f"{indent}🖼️  Image URL: {url}")
            print(f"{indent}    Detail: {detail}")

def display_sample_entry(entry, sample_num):
    """Display a sample entry in a readable format"""
    print(f"\n📝 Sample {sample_num}: {entry.get('virtual_object', 'N/A')} + {entry.get('physical_object', 'N/A')}")
    print(f"   Property: {entry.get('property_type', 'N/A')}")
    print(f"   Rating: {entry.get('rating', 'N/A')}")
    
    messages = entry.get('messages', [])
    
    for i, message in enumerate(messages):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        if role == 'system':
            print(f"   🔧 System: {content[:120]}...")
        elif role == 'user':
            print(f"   👤 User Content:")
            if isinstance(content, list):
                for j, item in enumerate(content):
                    print(f"      [{j+1}]", end=" ")
                    display_content_item(item)
            else:
                print(f"      📝 {content[:100]}...")
        elif role == 'assistant':
            print(f"   🤖 Assistant: {content}")

def count_images_in_dataset(dataset):
    """Count total number of images in the dataset"""
    total_images = 0
    total_text_items = 0
    
    for entry in dataset:
        messages = entry.get('messages', [])
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'image_url':
                        total_images += 1
                    elif item.get('type') == 'text':
                        total_text_items += 1
    
    return total_images, total_text_items

def main():
    print("🎯 ProXeek Vision Fine-Tuning Data Format Sample")
    print("=" * 60)
    
    # Generate a small sample dataset for preview
    print("📊 Generating sample data with images...")
    vft_dataset = generate_vft_dataset(max_rows=20)  # Small sample for quick preview
    
    if not vft_dataset:
        print("❌ No dataset generated. Check your CSV file and images.")
        return
    
    train_vft, test_vft = split_dataset(vft_dataset, test_per_property=2)
    
    print(f"✅ Generated {len(train_vft)} training examples")
    print(f"✅ Generated {len(test_vft)} testing examples")
    
    # Count images
    train_images, train_text = count_images_in_dataset(train_vft)
    test_images, test_text = count_images_in_dataset(test_vft)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training: {train_images} images, {train_text} text items")
    print(f"   Testing: {test_images} images, {test_text} text items")
    
    # Show first few training examples
    print(f"\n🔍 Sample Training Examples:")
    print("-" * 50)
    
    for i, entry in enumerate(train_vft[:3]):  # Show first 3 examples
        display_sample_entry(entry, i + 1)
    
    # Show the JSON structure of one complete example
    print(f"\n📋 Complete JSON Structure (Sample 1):")
    print("-" * 50)
    
    if train_vft:
        sample = train_vft[0]
        # Create a clean version for display (truncate base64 data)
        display_sample = json.loads(json.dumps(sample))
        
        # Truncate base64 data for readability
        messages = display_sample.get('messages', [])
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'image_url' and item.get('image_url', {}).get('url', '').startswith('data:'):
                        url = item['image_url']['url']
                        parts = url.split(',')
                        if len(parts) == 2:
                            item['image_url']['url'] = f"{parts[0]},{truncate_base64(parts[1], 50)}"
        
        # Remove extra fields for cleaner display
        clean_sample = {"messages": display_sample["messages"]}
        print(json.dumps(clean_sample, indent=2, ensure_ascii=False)[:1500] + "...")
    
    print(f"\n🆚 Key Differences from Text-Only Fine-tuning:")
    print("   Vision Fine-tuning:")
    print("   ✅ Includes actual images as base64 data URLs")
    print("   ✅ Model can see visual characteristics of objects")
    print("   ✅ Multi-modal input (text + images)")
    print("   ✅ Better understanding of spatial relationships")
    print("   ✅ Direct visual property assessment")
    
    print("\n   Text-Only Fine-tuning:")
    print("   ❌ Relies on text descriptions only")
    print("   ❌ No visual understanding")
    print("   ❌ May miss visual cues important for haptics")
    
    print(f"\n🎯 Expected VFT Results:")
    print("   ✅ More accurate haptic property ratings")
    print("   ✅ Better understanding of object shapes/textures")
    print("   ✅ Visual-haptic correlation learning")
    print("   ✅ Improved spatial reasoning for VR proxies")
    
    # Show image matching status
    print(f"\n🔗 Image Matching Status:")
    properties = ['Inertia', 'Interactivity', 'Outline', 'Texture', 'Hardness', 'Temperature']
    for prop in properties:
        count = len([entry for entry in train_vft if entry.get('property_type') == prop])
        print(f"   {prop}: {count} training examples with images")

if __name__ == "__main__":
    main() 