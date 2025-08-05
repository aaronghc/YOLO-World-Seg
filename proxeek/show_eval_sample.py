import json
import os
from generate_eval_dataset import generate_eval_dataset

def show_eval_sample(max_samples=3):
    """Show sample evaluation dataset entries"""
    print("🎯 ProXeek Evaluation Dataset Sample Preview")
    print("=" * 60)
    
    # Generate a small sample dataset
    print("📊 Generating sample evaluation dataset...")
    eval_dataset = generate_eval_dataset(items_per_virtual_object=1)  # Small sample for preview
    
    if not eval_dataset:
        print("❌ No evaluation dataset generated")
        return
    
    print(f"\n✅ Generated {len(eval_dataset)} evaluation items")
    print(f"📋 Showing first {min(max_samples, len(eval_dataset))} samples:\n")
    
    for i, entry in enumerate(eval_dataset[:max_samples]):
        item = entry['item']
        print(f"🔍 Sample {i+1}:")
        print(f"   Virtual Object: {item['virtual_object']}")
        print(f"   Physical Object: {item['physical_object']}")
        print(f"   Property: {item['property_type']}")
        print(f"   Ground Truth Rating: {item['ground_truth_rating']}")
        print(f"   Interaction Activity: {item['interaction_activity']}")
        print(f"   Utilization Method: {item['utilization_method']}")
        print(f"   Bounding Box: {item['bounding_box']}")
        
        # Show system prompt (truncated)
        system_prompt = item['system_prompt']
        if len(system_prompt) > 200:
            system_prompt = system_prompt[:200] + "..."
        print(f"   System Prompt: {system_prompt}")
        
        # Show user message structure
        user_msg = item['user_message']
        print(f"   User Message Structure:")
        for j, content in enumerate(user_msg):
            if content['type'] == 'text':
                text_preview = content['text'][:100] + "..." if len(content['text']) > 100 else content['text']
                print(f"     [{j}] Text: {text_preview}")
            elif content['type'] == 'image_url':
                image_url = content['image_url']['url']
                # Show just the format, not the full base64 string
                if image_url.startswith('data:'):
                    mime_type = image_url.split(';')[0].split(':')[1]
                    print(f"     [{j}] Image: {mime_type} (base64 encoded)")
                else:
                    print(f"     [{j}] Image: {image_url}")
        
        print()  # Empty line between samples
    
    # Show dataset statistics
    print("📈 Dataset Statistics:")
    
    # Group by property type
    property_counts = {}
    virtual_objects = set()
    physical_objects = set()
    
    for entry in eval_dataset:
        item = entry['item']
        prop = item['property_type']
        property_counts[prop] = property_counts.get(prop, 0) + 1
        virtual_objects.add(item['virtual_object'])
        physical_objects.add(item['physical_object'])
    
    print(f"   Total Items: {len(eval_dataset)}")
    print(f"   Virtual Objects: {len(virtual_objects)}")
    print(f"   Physical Objects: {len(physical_objects)}")
    print(f"   Property Distribution:")
    for prop, count in sorted(property_counts.items()):
        print(f"     {prop}: {count}")
    
    print(f"\n💾 To save full dataset, run: python generate_eval_dataset.py")
    print(f"🚀 To start evaluation, run: python create_model_eval.py")

def main():
    """Main function"""
    show_eval_sample(max_samples=2)

if __name__ == "__main__":
    main() 