#!/usr/bin/env python3
"""
Send Optimization Results to Quest

This standalone script sends the optimization_results_for_quest.json file
to the Quest device via the local server endpoint.

Usage:
    python send_results_to_quest.py
    
    Or optionally specify a custom file:
    python send_results_to_quest.py path/to/custom_results.json
"""

import os
import sys
import json
import requests
from typing import Dict, Optional

# Configuration
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEFAULT_RESULTS_FILE = "optimization_results_for_quest.json"
QUEST_SERVER_URL = "http://localhost:5000/send_to_quest"
REQUEST_TIMEOUT = 10  # seconds


def load_optimization_results(file_path: str) -> Optional[Dict]:
    """Load optimization results from JSON file
    
    Args:
        file_path: Path to the optimization results JSON file
        
    Returns:
        Dictionary containing optimization results, or None if loading fails
    """
    try:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded optimization results from: {file_path}")
        return data
    
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def validate_payload(payload: Dict) -> bool:
    """Validate that the payload has the required Quest format
    
    Args:
        payload: The optimization results payload
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["action", "data", "timestamp", "total_assignments"]
    
    for field in required_fields:
        if field not in payload:
            print(f"❌ Error: Missing required field '{field}' in payload")
            return False
    
    if payload["action"] != "optimization_results":
        print(f"⚠ Warning: Expected action='optimization_results', got '{payload['action']}'")
    
    return True


def send_to_quest(payload: Dict) -> bool:
    """Send optimization results to Quest device
    
    Args:
        payload: The optimization results payload in Quest format
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print("\n" + "="*60)
        print("Sending Optimization Results to Quest")
        print("="*60)
        
        # Display payload summary
        total_assignments = payload.get("total_assignments", 0)
        pin_point = payload.get("pinPoint")
        rotation_angle = payload.get("rotationAngle")
        timestamp = payload.get("timestamp")
        
        print(f"\n📦 Payload Summary:")
        print(f"   • Total assignments: {total_assignments}")
        print(f"   • Timestamp: {timestamp}")
        
        if pin_point is not None:
            print(f"   • Pin point: [{pin_point[0]:.3f}, {pin_point[1]:.3f}, {pin_point[2]:.3f}]")
        else:
            print(f"   • Pin point: None (no spatial transformation)")
        
        if rotation_angle is not None:
            print(f"   • Rotation angle: {rotation_angle:.1f}°")
        else:
            print(f"   • Rotation angle: None")
        
        # Display assignments
        print(f"\n📋 Assignments:")
        assignments = payload.get("data", [])
        for i, assignment in enumerate(assignments, 1):
            virtual_name = assignment.get("virtualObjectName", "Unknown")
            proxy_name = assignment.get("proxyObjectName", "Unknown")
            print(f"   {i}. {virtual_name} → {proxy_name}")
        
        # Send to Quest
        print(f"\n🔄 Sending to Quest server: {QUEST_SERVER_URL}")
        print(f"   Timeout: {REQUEST_TIMEOUT} seconds")
        
        response = requests.post(
            QUEST_SERVER_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        
        # Check response
        if response.status_code == 200:
            print(f"\n✅ Successfully sent optimization results to Quest!")
            print(f"   HTTP Status: {response.status_code}")
            
            # Try to parse response message if available
            try:
                response_data = response.json()
                if "message" in response_data:
                    print(f"   Server response: {response_data['message']}")
            except:
                pass  # Response might not be JSON
            
            return True
        else:
            print(f"\n❌ Failed to send results to Quest")
            print(f"   HTTP Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print(f"\n❌ Error: Request timed out after {REQUEST_TIMEOUT} seconds")
        print(f"   Make sure the Quest server is running on {QUEST_SERVER_URL}")
        return False
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to Quest server")
        print(f"   Make sure the server is running on {QUEST_SERVER_URL}")
        print(f"   Check that:")
        print(f"   1. The Quest device is powered on")
        print(f"   2. The local server is running (usually on port 5000)")
        print(f"   3. Your network connection is active")
        return False
    
    except Exception as e:
        print(f"\n❌ Error sending to Quest: {e}")
        return False


def main():
    """Main function"""
    print("\n" + "="*60)
    print("ProXeek - Send Optimization Results to Quest")
    print("="*60 + "\n")
    
    # Determine file path
    if len(sys.argv) > 1:
        # User provided custom file path
        file_path = sys.argv[1]
        print(f"Using custom results file: {file_path}")
    else:
        # Use default file path
        file_path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_FILE)
        print(f"Using default results file: {file_path}")
    
    # Load optimization results
    payload = load_optimization_results(file_path)
    if payload is None:
        print("\n❌ Failed to load optimization results. Exiting.")
        sys.exit(1)
    
    # Validate payload format
    if not validate_payload(payload):
        print("\n❌ Invalid payload format. Exiting.")
        sys.exit(1)
    
    # Send to Quest
    success = send_to_quest(payload)
    
    # Exit with appropriate code
    if success:
        print("\n" + "="*60)
        print("✅ Operation completed successfully!")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ Operation failed. Please check the errors above.")
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

