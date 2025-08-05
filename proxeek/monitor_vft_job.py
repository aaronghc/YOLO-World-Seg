import os
import json
import requests
import time
from dotenv import load_dotenv
import sys
import re

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

def get_job_status(job_id, config):
    """Get the current status of a fine-tuning job"""
    response = requests.get(
        f"{config['base_url']}/fine_tuning/jobs/{job_id}",
        headers=config['headers']
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get job status: {response.status_code} - {response.text}")
        return None

def get_job_events(job_id, config, limit=10):
    """Get recent events for a fine-tuning job"""
    response = requests.get(
        f"{config['base_url']}/fine_tuning/jobs/{job_id}/events",
        headers=config['headers'],
        params={'limit': limit}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to get job events: {response.status_code} - {response.text}")
        return None

def display_job_status(job_info):
    """Display formatted job status information for VFT"""
    status = job_info.get('status', 'unknown')
    method = job_info.get('method', {})
    method_type = method.get('type', 'unknown')
    
    print(f"🖼️  Job Status: {status.upper()}")
    print(f"   Model: {job_info.get('model', 'N/A')}")
    print(f"   Method: Vision Fine-tuning ({method_type.title()})")
    print(f"   Created: {job_info.get('created_at', 'N/A')}")
    
    if 'training_file' in job_info:
        print(f"   Training File: {job_info['training_file']}")
    if 'validation_file' in job_info:
        print(f"   Validation File: {job_info['validation_file']}")
    
    # Show hyperparameters if available
    if method_type == 'supervised' and 'supervised' in method:
        hyper = method['supervised'].get('hyperparameters', {})
        if hyper:
            print(f"   Hyperparameters:")
            for key, value in hyper.items():
                print(f"     - {key}: {value}")
    
    # Show progress info
    if job_info.get('trained_tokens'):
        print(f"   Trained Tokens: {job_info['trained_tokens']:,}")
    
    # Show epoch progress if available
    if status in ['running', 'validating']:
        # Try to extract epoch info from various sources
        current_epoch = None
        total_epochs = None
        
        # Check hyperparameters for total epochs
        if method_type == 'supervised' and 'supervised' in method:
            hyper = method['supervised'].get('hyperparameters', {})
            total_epochs = hyper.get('n_epochs', 3)  # Default to 3 if not specified
        
        # Check if there's progress info in the job object
        if 'training_file' in job_info and 'validation_file' in job_info:
            # For vision fine-tuning, we can estimate progress
            print(f"   📊 Training Progress:")
            if total_epochs:
                print(f"      Total Epochs: {total_epochs}")
            print(f"      Status: {status.title()}")
            if status == 'validating':
                print(f"      🔄 Currently validating model performance...")
            elif status == 'running':
                print(f"      🚀 Training in progress...")
    
    # Show fine-tuned model if completed
    if job_info.get('fine_tuned_model'):
        print(f"   ✅ Fine-tuned Model: {job_info['fine_tuned_model']}")
        print(f"   🎯 This model can now see and analyze images!")
    
    # Show error if failed
    if job_info.get('error'):
        error = job_info['error']
        if isinstance(error, dict) and error.get('message'):
            print(f"   ❌ Error: {error['message']}")
        else:
            print(f"   ❌ Error: {error}")

def display_recent_events(events):
    """Display recent job events with epoch tracking"""
    if not events or 'data' not in events:
        return
    
    print(f"\n📝 Recent Events:")
    
    # Look for training progress in events
    latest_epoch_info = None
    
    for event in events['data'][:10]:  # Check more events for epoch info
        timestamp = event.get('created_at', 'N/A')
        level = event.get('level', 'info')
        message = event.get('message', 'No message')
        
        # Try to extract epoch information from message
        epoch_match = re.search(r'[Ee]poch (\d+)(?:/(\d+))?', message)
        step_match = re.search(r'[Ss]tep (\d+)(?:/(\d+))?', message)
        
        if epoch_match and not latest_epoch_info:
            current_epoch = epoch_match.group(1)
            total_epochs = epoch_match.group(2) if epoch_match.group(2) else None
            latest_epoch_info = (current_epoch, total_epochs)
        
        # Color code by level
        if level == 'error':
            prefix = "❌"
        elif level == 'warn':
            prefix = "⚠️ "
        else:
            prefix = "ℹ️ "
    
    # Display epoch progress if found
    if latest_epoch_info:
        current_epoch, total_epochs = latest_epoch_info
        if total_epochs:
            print(f"   🎯 Current Progress: Epoch {current_epoch}/{total_epochs}")
        else:
            print(f"   🎯 Current Progress: Epoch {current_epoch}")
        print()
    
    # Display recent events (limit to 5 for readability)
    for event in events['data'][:5]:
        timestamp = event.get('created_at', 'N/A')
        level = event.get('level', 'info')
        message = event.get('message', 'No message')
        
        # Color code by level
        if level == 'error':
            prefix = "❌"
        elif level == 'warn':
            prefix = "⚠️ "
        else:
            prefix = "ℹ️ "
        
        print(f"   {prefix} {message}")

def monitor_job(job_id, config, watch=False):
    """Monitor a vision fine-tuning job"""
    print(f"🔍 Monitoring Vision Fine-tuning Job: {job_id}")
    print("=" * 60)
    
    while True:
        # Get job status
        job_info = get_job_status(job_id, config)
        if not job_info:
            break
        
        # Display status
        display_job_status(job_info)
        
        # Get and display recent events
        events = get_job_events(job_id, config)
        display_recent_events(events)
        
        status = job_info.get('status', 'unknown')
        
        # Check if job is complete
        if status in ['succeeded', 'failed', 'cancelled']:
            print(f"\n🏁 Job {status.upper()}!")
            if status == 'succeeded':
                model_id = job_info.get('fine_tuned_model', 'N/A')
                print(f"✅ Your vision fine-tuned model is ready: {model_id}")
                print("\n🎯 How to use your vision model:")
                print(f"   Model ID: {model_id}")
                print("   This model can now:")
                print("   • See and analyze both virtual and physical object images")
                print("   • Rate haptic properties based on visual characteristics")
                print("   • Understand spatial relationships and object properties")
                print("   • Use in Chat Completions API with image inputs")
            break
        
        if not watch:
            break
        
        # Wait before next check
        print(f"\n⏳ Waiting 30 seconds before next check... (Ctrl+C to stop)")
        try:
            time.sleep(30)
            print("\n" + "=" * 60)
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            break

def main():
    """Main function"""
    if len(sys.argv) < 2:
        # Try to load job ID from saved file
        job_info_path = os.path.join(os.path.dirname(__file__), 'vft_job_info.json')
        if os.path.exists(job_info_path):
            with open(job_info_path, 'r') as f:
                job_info = json.load(f)
                job_id = job_info.get('job_id')
                if job_id:
                    print(f"📁 Using job ID from vft_job_info.json: {job_id}")
                else:
                    print("❌ No job ID found in vft_job_info.json")
                    sys.exit(1)
        else:
            print("Usage: python monitor_vft_job.py <job_id> [--watch]")
            print("   or: python monitor_vft_job.py [--watch]  (uses job ID from vft_job_info.json)")
            sys.exit(1)
    else:
        job_id = sys.argv[1]
    
    watch = '--watch' in sys.argv
    
    # Load API configuration
    config = load_api_config()
    
    # Monitor the job
    monitor_job(job_id, config, watch)

if __name__ == "__main__":
    main() 