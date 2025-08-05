import os
import json
import requests
from dotenv import load_dotenv
import sys

# Import our VFT dataset generation functions
from generate_vft_dataset import generate_vft_dataset, split_dataset, save_dataset_as_jsonl

def load_api_config():
    """Load API configuration from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    api_key = os.getenv('OPENAI_FINETUNE_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_FINETUNE_API_KEY not found in .env file")
        print("   Please add: OPENAI_FINETUNE_API_KEY=your_finetune_key_here")
        sys.exit(1)
    
    return {
        'api_key': api_key,
        'base_url': 'https://api.openai.com/v1',
        'headers': {'Authorization': f'Bearer {api_key}'}
    }

def upload_file(file_path, purpose, config):
    """Upload a file to the API"""
    print(f"📤 Uploading {file_path}...")
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'purpose': purpose}
        
        response = requests.post(
            f"{config['base_url']}/files",
            headers=config['headers'],
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        file_id = result['id']
        print(f"✅ Upload successful! File ID: {file_id}")
        return file_id
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
        return None

def create_vft_job(train_file_id, test_file_id, config):
    """Create a vision fine-tuning job"""
    print("🚀 Creating Vision Fine-tuning job...")
    
    job_data = {
        "training_file": train_file_id,
        "validation_file": test_file_id,
        "model": "gpt-4o-2024-08-06",
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": 2,
                    "batch_size": 1,
                    "learning_rate_multiplier": 1.0
                }
            }
        }
    }
    
    response = requests.post(
        f"{config['base_url']}/fine_tuning/jobs",
        headers={**config['headers'], 'Content-Type': 'application/json'},
        json=job_data
    )
    
    if response.status_code == 200:
        result = response.json()
        job_id = result['id']
        print(f"✅ Vision Fine-tuning job created successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Status: {result.get('status', 'unknown')}")
        return job_id
    else:
        print(f"❌ Job creation failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def main():
    """Main workflow for creating VFT fine-tuning job"""
    print("🎯 ProXeek Property Rating Vision Fine-tuning Setup")
    print("=" * 60)
    
    # Load API configuration
    config = load_api_config()
    print(f"✅ API configuration loaded (base_url: {config['base_url']})")
    
    # Step 1: Generate vision datasets with images
    print("\n📊 Step 1: Generating vision datasets with images...")
    # You can change max_rows to limit dataset size (e.g., max_rows=50 for testing)
    vft_dataset = generate_vft_dataset(max_rows=None)  # None = use all rows
    
    if not vft_dataset:
        print("❌ Failed to generate dataset. Check your CSV file and images.")
        return
    
    print(f"Generated {len(vft_dataset)} total examples with images")
    
    # Split into training and testing
    train_vft, test_vft = split_dataset(vft_dataset, test_per_property=3)
    print(f"Training examples: {len(train_vft)}")
    print(f"Testing examples: {len(test_vft)}")
    
    # Step 2: Save datasets in JSONL format
    print("\n💾 Step 2: Saving vision fine-tuning datasets...")
    train_filename = 'vft_property_rating_train.jsonl'
    test_filename = 'vft_property_rating_test.jsonl'
    
    save_dataset_as_jsonl(train_vft, train_filename)
    save_dataset_as_jsonl(test_vft, test_filename)
    print(f"✅ Datasets saved as {train_filename} and {test_filename}")
    
    # Step 3: Upload files (use full paths)
    print("\n📤 Step 3: Uploading files...")
    script_dir = os.path.dirname(__file__)
    train_file_path = os.path.join(script_dir, train_filename)
    test_file_path = os.path.join(script_dir, test_filename)
    
    train_file_id = upload_file(train_file_path, 'fine-tune', config)
    test_file_id = upload_file(test_file_path, 'fine-tune', config)
    
    if not train_file_id or not test_file_id:
        print("❌ Failed to upload files. Exiting.")
        return
    
    # Step 4: Create vision fine-tuning job
    print("\n🏗️  Step 4: Creating vision fine-tuning job...")
    job_id = create_vft_job(train_file_id, test_file_id, config)
    
    if job_id:
        print("\n🎉 SUCCESS! Vision Fine-tuning job created successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Model: gpt-4o-2024-08-06")
        print(f"   Method: Vision Fine-tuning (Supervised)")
        print(f"   Training examples: {len(train_vft)} (with images)")
        print(f"   Testing examples: {len(test_vft)} (with images)")
        print("\n📝 Next steps:")
        print("   1. Monitor job progress in the fine-tuning dashboard")
        print("   2. Review training metrics and loss curves")
        print("   3. Test the fine-tuned model with actual images when training completes")
        print("   4. The model will be able to see both virtual and physical objects!")
        
        # Save job info for reference
        job_info = {
            'job_id': job_id,
            'model': 'gpt-4o-2024-08-06',
            'method': 'vision_fine_tuning',
            'training_file_id': train_file_id,
            'validation_file_id': test_file_id,
            'training_examples': len(train_vft),
            'testing_examples': len(test_vft),
            'includes_images': True
        }
        
        job_info_path = os.path.join(os.path.dirname(__file__), 'vft_job_info.json')
        with open(job_info_path, 'w') as f:
            json.dump(job_info, f, indent=2)
        print(f"   5. Job details saved to vft_job_info.json")
        
    else:
        print("\n❌ Failed to create fine-tuning job")

if __name__ == "__main__":
    main() 