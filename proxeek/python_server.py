from flask import Flask, request, jsonify
import subprocess
import os
import sys
import json
import traceback
import threading
import queue
import requests

app = Flask(__name__)

# Increase maximum content length to handle multiple images
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Allow up to 100MB payloads

# Store Quest connection info (will be set when Quest connects)
quest_callback_url = None

# Path to your Unity project's Python scripts
SCRIPTS_PATH = r"C:\Users\aaron\Documents\GitHub\YOLO-World-Seg\proxeek"


@app.route('/run_python', methods=['POST'])
def run_python():
    try:
        print("Received request of size:", len(request.data))
        data = request.json

        if not data:
            print("No JSON data received")
            return jsonify({'status': 'error', 'output': 'No JSON data received'}), 400

        print("Parsed JSON data keys:", data.keys())

        if 'action' not in data or data['action'] != 'run_script':
            print("Invalid request format - missing 'action' or not 'run_script'")
            return jsonify({'status': 'error', 'output': 'Invalid request format'}), 400

        # Prompt user to select which script to run
        print("\n" + "="*60)
        print("SELECT SCRIPT TO RUN:")
        print("  1 - ProXeek.py (Full pipeline or skip mode)")
        print("  2 - ProXeek_ObjectRecognition.py (Object recognition only)")
        print("="*60)
        
        while True:
            user_choice = input("Enter your choice (1 or 2): ").strip()
            if user_choice == '1':
                script_name = 'ProXeek.py'
                print(f"Selected: ProXeek.py")
                break
            elif user_choice == '2':
                script_name = 'ProXeek_ObjectRecognition.py'
                print(f"Selected: ProXeek_ObjectRecognition.py")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        params = data.get('params', {})
        
        # If ProXeek.py was selected, ask if user wants to skip object recognition
        if script_name == 'ProXeek.py':
            print("\n" + "="*60)
            print("OBJECT RECOGNITION MODE:")
            print("  1 - Run full pipeline (object recognition + YOLO + segmentation)")
            print("  2 - Skip object recognition (use existing physical_object_database.json)")
            print("="*60)
            
            while True:
                skip_choice = input("Enter your choice (1 or 2): ").strip()
                if skip_choice == '1':
                    params['skipObjectRecognition'] = False
                    print("Selected: Run full pipeline")
                    break
                elif skip_choice == '2':
                    params['skipObjectRecognition'] = True
                    print("Selected: Skip object recognition (using existing database)")
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")

        print(f"Script name: {script_name}")
        if 'environmentImageBase64List' in params:
            print(f"Received {len(params['environmentImageBase64List'])} environment images")
        if 'virtualObjectSnapshots' in params:
            print(f"Received {len(params['virtualObjectSnapshots'])} virtual object snapshots")
        if 'arrangementSnapshots' in params:
            print(f"Received {len(params['arrangementSnapshots'])} arrangement snapshots")

        # Full path to the script
        script_path = os.path.join(SCRIPTS_PATH, script_name)

        if not os.path.exists(script_path):
            print(f"Script not found: {script_path}")
            return jsonify({'status': 'error', 'output': f'Script not found: {script_name}'}), 404

        # Create a temporary JSON file with parameters
        params_path = os.path.join(SCRIPTS_PATH, 'temp_params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f)

        print(f"Running script: {sys.executable} {script_path} {params_path}")

        # Create a list to collect output
        output_lines = []

        def stream_output(pipe, prefix):
            """Read from pipe and print + store output"""
            for line in iter(pipe.readline, ''):
                if line:
                    formatted_line = f"[{prefix}] {line.rstrip()}"
                    print(formatted_line, flush=True)  # Print to Flask console
                    output_lines.append(line)
            pipe.close()

        # Run the Python script with real-time output streaming
        process = subprocess.Popen(
            [sys.executable, script_path, params_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Create threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=stream_output, 
            args=(process.stdout, "ProXeek")
        )
        stderr_thread = threading.Thread(
            target=stream_output, 
            args=(process.stderr, "ProXeek-ERROR")
        )

        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to complete
        return_code = process.wait(timeout=5000)
        
        # Wait for output threads to finish
        stdout_thread.join()
        stderr_thread.join()

        # Clean up the temporary file
        if os.path.exists(params_path):
            os.remove(params_path)

        # Join all output lines
        full_output = ''.join(output_lines)

        # Check for errors
        if return_code != 0:
            print(f"Script execution failed with code {return_code}")
            return jsonify({
                'status': 'error',
                'output': f"Error executing script (code {return_code}):\n{full_output}"
            }), 500

        # Return the output
        print(f"Script executed successfully, output length: {len(full_output)}")
        return jsonify({
            'status': 'success',
            'output': full_output
        })

    except Exception as e:
        print(f"Server error: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'output': f"Server error: {str(e)}"}), 500


@app.route('/register_quest', methods=['POST'])
def register_quest():
    """Register Quest's callback URL for receiving bounding box data"""
    global quest_callback_url
    try:
        data = request.json
        if not data or 'callback_url' not in data:
            return jsonify({'status': 'error', 'message': 'callback_url required'}), 400
        
        quest_callback_url = data['callback_url']
        print(f"Quest registered with callback URL: {quest_callback_url}")
        
        return jsonify({'status': 'success', 'message': 'Quest registered successfully'})
    except Exception as e:
        print(f"Error registering Quest: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Registration failed: {str(e)}"}), 500


@app.route('/send_to_quest', methods=['POST'])
def send_to_quest():
    """Send bounding box data to Quest"""
    global quest_callback_url
    try:
        if not quest_callback_url:
            print("Quest callback URL not registered - cannot send data")
            return jsonify({'status': 'error', 'message': 'Quest not registered'}), 400
        
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        print(f"Sending bounding box data to Quest at: {quest_callback_url}")
        print(f"Data contains {data.get('total_objects', 0)} objects")
        print(f"Data contains data: {data}")
        
        # Forward the data to Quest
        response = requests.post(
            quest_callback_url,
            json=data,
            timeout=15  # 15 second timeout
        )
        
        if response.status_code == 200:
            print("Successfully sent bounding box data to Quest")
            return jsonify({'status': 'success', 'message': 'Data sent to Quest successfully'})
        else:
            print(f"Failed to send data to Quest: HTTP {response.status_code}")
            return jsonify({'status': 'error', 'message': f'Quest responded with {response.status_code}'}), 502
            
    except requests.exceptions.Timeout:
        print("Timeout sending data to Quest")
        return jsonify({'status': 'error', 'message': 'Timeout sending to Quest'}), 504
    except requests.exceptions.ConnectionError:
        print("Connection error sending data to Quest")
        return jsonify({'status': 'error', 'message': 'Cannot connect to Quest'}), 502
    except Exception as e:
        print(f"Error sending data to Quest: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Send failed: {str(e)}"}), 500


# ============================================================================
#  NEW ENDPOINT: receive_updated_bounding_boxes
#  Quest sends back bounding box data where world positions have been added.
#  This endpoint stores the data as the final physical_object_database.json.
# ============================================================================


@app.route('/receive_updated_bounding_boxes', methods=['POST'])
def receive_updated_bounding_boxes():
    """Receive updated bounding box data (with world positions) from Quest and save it as the final physical_object_database.json"""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON body received'}), 400

        # Ensure output directory exists
        output_dir = os.path.join(SCRIPTS_PATH, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Save the FULL data (including playArea if present) to physical_object_database.json
        output_path = os.path.join(output_dir, 'physical_object_database.json')
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        obj_count = 0
        try:
            # Count objects in the 'data' field
            obj_data = data.get('data', {})
            obj_count = sum(len(v) for v in obj_data.values())
        except Exception:
            pass

        print(f"Received updated bounding box data. Saved {obj_count} objects to {output_path}")
        
        # Check if play area data is present (just for logging)
        play_area = data.get('playArea', None)
        if play_area:
            print(f"Play area data found: {len(play_area.get('boundaryPoints', []))} boundary points")
            print(f"  Width: {play_area.get('width', 0):.2f}m, Depth: {play_area.get('depth', 0):.2f}m")
            
            # Generate play area footprint visualization
            try:
                from export_play_area_footprint import export_play_area_footprint
                
                footprint_path = os.path.join(output_dir, 'play_area_footprint.png')
                export_play_area_footprint(output_path, footprint_path)
                print(f"✓ Play area footprint exported to: {footprint_path}")
            except Exception as footprint_error:
                print(f"Warning: Could not export play area footprint: {footprint_error}")
        else:
            print("No play area data in received payload (boundary not configured on Quest)")

        return jsonify({'status': 'success', 'message': 'Updated physical object database saved'}), 200

    except Exception as e:
        print(f"Error handling updated bounding box data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print(f"Starting Python server on port 5000")
    print(f"Scripts path: {SCRIPTS_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)