import os
import sys
import json
import base64
from io import BytesIO
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image
import asyncio
from typing import List, Dict, Any, Optional, Union
from pydantic import SecretStr
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import uuid
import difflib
import requests

# Add YOLO-World + EfficientSAM imports
import torch
import cv2
import numpy as np
from inference.models import YOLOWorld
from utils.efficient_sam import load, inference_with_boxes
import supervision as sv
from supervision.detection.core import Detections
import tempfile

# =============================================================================
# CONFIGURATION PARAMETERS - Easy to find and modify
# =============================================================================

# Substrate Utilization Process - Overlapping Batch Configuration
SUBSTRATE_BATCH_SIZE = 50        # Number of tasks per batch
SUBSTRATE_BATCH_INTERVAL = 1.5  # Seconds between starting new batches

# Property Rating Process - Overlapping Batch Configuration
PROPERTY_RATING_BATCH_SIZE = 50  # Number of tasks per batch
PROPERTY_RATING_BATCH_INTERVAL = 2  # Seconds between starting new batches

# Relationship Rating Process - Overlapping Batch Configuration
RELATIONSHIP_RATING_BATCH_SIZE = 50  # Number of tasks per batch
RELATIONSHIP_RATING_BATCH_INTERVAL = 1  # Seconds between starting new batches

# Process Activation Switches
ENABLE_PROXY_MATCHING = True        # Set to True to enable proxy matching and dependent processes
ENABLE_RELATIONSHIP_RATING = True   # Set to True to enable relationship rating

# =============================================================================

# Set up logging to help debug
def log(message):
    print(f"LOG: {message}")
    sys.stdout.flush()

# Helper function to extract text content from LLM response
def extract_response_text(response_content) -> str:
    """Extract text content from LLM response, handling different response formats"""
    if isinstance(response_content, list):
        # If it's a list, join the elements or take the first string element
        text_content = ""
        for item in response_content:
            if isinstance(item, str):
                text_content += item
            elif isinstance(item, dict) and 'text' in item:
                text_content += item['text']
        return text_content
    elif isinstance(response_content, str):
        return response_content
    else:
        return str(response_content)

# Note: LangChain ChatOpenAI automatically handles LangSmith tracing when environment variables are set
# No need for manual run IDs or langsmith_extra parameters


log("Script started")

# Helper function to normalize object names for matching
def normalize_name(name):
    # Convert to lowercase, replace spaces with underscores, remove special characters
    return name.lower().replace(" ", "_").replace("-", "_")

# Helper function to retry async operations with exponential backoff and timeout
async def retry_with_backoff(async_func, max_retries=3, base_delay=1.0, backoff_factor=2.0, timeout_seconds=300):
    """Retry an async function with exponential backoff and timeout"""
    for attempt in range(max_retries):
        try:
            # Add timeout to each attempt
            result = await asyncio.wait_for(async_func(), timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError as e:
            if attempt == max_retries - 1:
                log(f"Final retry attempt timed out after {timeout_seconds} seconds")
                raise e
            
            delay = base_delay * (backoff_factor ** attempt)
            log(f"Attempt {attempt + 1} timed out after {timeout_seconds} seconds. Retrying in {delay:.1f} seconds...")
            await asyncio.sleep(delay)
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed, re-raise the exception
                log(f"Final retry attempt failed: {e}")
                raise e
            
            delay = base_delay * (backoff_factor ** attempt)
            log(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f} seconds...")
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    raise Exception("Max retries exceeded")

# Function to initialize YOLO-World model
def initialize_models():
    """Initialize YOLO-World and EfficientSAM models"""
    global yolo_world_model, efficient_sam_model
    
    if yolo_world_model is not None and efficient_sam_model is not None:
        return yolo_world_model, efficient_sam_model
    
    try:
        log("Initializing YOLO-World and EfficientSAM models...")
        
        # Initialize YOLO-World model
        yolo_world_model = YOLOWorld(model_id="yolo_world/l")
        
        # Move YOLO-World model to GPU if available
        if torch.cuda.is_available():
            try:
                # Check if YOLO-World model has a .to() method
                if hasattr(yolo_world_model, 'to'):
                    yolo_world_model = yolo_world_model.to(DEVICE)
                    log(f"YOLO-World model moved to {DEVICE}")
                elif hasattr(yolo_world_model, 'model') and hasattr(yolo_world_model.model, 'to'):
                    yolo_world_model.model = yolo_world_model.model.to(DEVICE)
                    log(f"YOLO-World underlying model moved to {DEVICE}")
                else:
                    log("Warning: Could not move YOLO-World model to GPU - no .to() method found")
            except Exception as e:
                log(f"Warning: Failed to move YOLO-World model to GPU: {e}")
        
        # Initialize EfficientSAM model
        efficient_sam_model = load(device=DEVICE)
        
        log(f"YOLO-World and EfficientSAM models initialized successfully on device: {DEVICE}")
        return yolo_world_model, efficient_sam_model
        
    except Exception as e:
        log(f"Error initializing models: {e}")
        return None, None

# Function to run YOLO-World + EfficientSAM detection on an image
def run_yolo_segmentation(image_base64: str, object_names: List[str], image_id: int) -> List[Dict]:
    """Run YOLO-World + EfficientSAM segmentation on an image with specified object names"""
    try:
        # Initialize models if needed
        yolo_model, sam_model = initialize_models()
        if yolo_model is None or sam_model is None:
            log(f"Models not available for image {image_id}")
            return []
        
        # Convert base64 to numpy array
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            log(f"Failed to decode image {image_id}")
            return []
        
        # Convert BGR to RGB for YOLO-World
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set classes and run YOLO-World detection
        yolo_model.set_classes(object_names)
        results = yolo_model.infer(image_rgb, confidence=CONFIDENCE_THRESHOLD)
        detections = Detections.from_inference(results)
        
        # Apply NMS
        detections = detections.with_nms(
            class_agnostic=False,
            threshold=IOU_THRESHOLD
        )
        
        # Run EfficientSAM segmentation on detected bounding boxes
        if len(detections.xyxy) > 0:
            detections.mask = inference_with_boxes(
                image=image_rgb,
                xyxy=detections.xyxy,
                model=sam_model,
                device=DEVICE
            )
        
        # Convert detections to our format
        segmentations = []
        if len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                class_id = detections.class_id[i] if detections.class_id is not None else 0
                confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                mask = detections.mask[i] if detections.mask is not None else None
                
                # Get the detected object name
                detected_object_name = object_names[class_id] if class_id < len(object_names) else "unknown"
                
                # Calculate robust mask center (always inside the mask)
                if mask is not None and mask.any():
                    # --- Robust center: point farthest from edges (distance transform) ---
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

                    # Zero out a safety border to avoid selecting points on the image edge
                    h, w = dist_map.shape
                    edge_buffer = max(15, int(0.01 * min(h, w)))  # 1% of smaller dimension (>=3px)
                    dist_map[:edge_buffer, :] = 0
                    dist_map[-edge_buffer:, :] = 0
                    dist_map[:, :edge_buffer] = 0
                    dist_map[:, -edge_buffer:] = 0

                    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist_map)

                    if maxVal > 0:
                        # Use the safest interior point
                        mask_center_x = float(maxLoc[0])
                        mask_center_y = float(maxLoc[1])
                    else:
                        # Fall back to centroid if no interior point away from border found
                        mask_coords = np.where(mask)
                        if len(mask_coords[0]) > 0:
                            mask_center_y = float(np.mean(mask_coords[0]))
                            mask_center_x = float(np.mean(mask_coords[1]))
                        else:
                            mask_center_x = float((bbox[0] + bbox[2]) / 2)
                            mask_center_y = float((bbox[1] + bbox[3]) / 2)

                    # Extract mask contours for visualization
                    mask_contours = []
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        contour_points = contour.reshape(-1, 2).tolist()
                        mask_contours.append(contour_points)
                else:
                    # Fallback to bbox center if mask is None or empty
                    mask_center_x = float((bbox[0] + bbox[2]) / 2)
                    mask_center_y = float((bbox[1] + bbox[3]) / 2)
                    mask_contours = []
                
                segmentation = {
                    "bbox": {
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]), 
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3]),
                        "width": float(bbox[2] - bbox[0]),
                        "height": float(bbox[3] - bbox[1]),
                        "center_x": float((bbox[0] + bbox[2]) / 2),
                        "center_y": float((bbox[1] + bbox[3]) / 2)
                    },
                    "mask_center": {
                        "x": mask_center_x,
                        "y": mask_center_y
                    },
                    "mask_contours": mask_contours,  # Add mask contours for visualization
                    "detected_object_name": detected_object_name,
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "image_id": image_id,
                    "has_mask": mask is not None
                }
                segmentations.append(segmentation)
        
        log(f"YOLO-World + EfficientSAM segmentation completed for image {image_id}: found {len(segmentations)} objects")
        return segmentations
        
    except Exception as e:
        log(f"Error in YOLO segmentation for image {image_id}: {e}")
        return []

# Check if we're running from Unity or from the server
if len(sys.argv) > 1:
    # Running from server with parameters file
    log(f"Running with parameters file: {sys.argv[1]}")
    params_file = sys.argv[1]

    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        log(f"Loaded parameters")

        # Extract data from parameters
        haptic_annotation_json = params.get('hapticAnnotationJson', '')
        environment_image_base64_list = params.get('environmentImageBase64List', [])
        virtual_object_snapshots = params.get('virtualObjectSnapshots', [])
        arrangement_snapshots = params.get('arrangementSnapshots', [])

        log(f"Found {len(environment_image_base64_list)} environment images")
        log(f"Found {len(virtual_object_snapshots)} virtual object snapshots")
        log(f"Found {len(arrangement_snapshots)} arrangement snapshots")
        log(f"Haptic annotation JSON present: {'Yes' if haptic_annotation_json else 'No'}")

    except Exception as e:
        log(f"Error reading parameters file: {e}")
        haptic_annotation_json = ''
        environment_image_base64_list = []
        virtual_object_snapshots = []
        arrangement_snapshots = []
else:
    # Default when running from Unity Editor
    log("No parameters file provided, using defaults")
    haptic_annotation_json = ''
    environment_image_base64_list = []
    virtual_object_snapshots = []
    arrangement_snapshots = []

# Get the project path
script_dir = os.path.dirname(os.path.abspath(__file__))
log(f"Script directory: {script_dir}")

# Add the script directory to sys.path
if script_dir not in sys.path:
    sys.path.append(script_dir)
    log(f"Added {script_dir} to sys.path")

# Load environment variables
try:
    load_dotenv(os.path.join(script_dir, '.env'))
    log("Loaded .env file")
except Exception as e:
    log(f"Error loading .env file: {e}")

# Get API keys
api_key = os.environ.get("OPENAI_API_KEY")
langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")

log(f"API key found: {'Yes' if api_key else 'No'}")
log(f"Langchain API key found: {'Yes' if langchain_api_key else 'No'}")

# If keys not found in environment, try to read directly from .env file
if not api_key or not langchain_api_key:
    try:
        log("Trying to read API keys directly from .env file")
        with open(os.path.join(script_dir, '.env'), 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.strip().split('=', 1)[1].strip('"\'')
                    log("Found OPENAI_API_KEY in .env file")
                elif line.startswith('LANGCHAIN_API_KEY='):
                    langchain_api_key = line.strip().split('=', 1)[1].strip('"\'')
                    log("Found LANGCHAIN_API_KEY in .env file")
    except Exception as e:
        log(f"Error reading .env file directly: {e}")

# Set up LangSmith tracing (using LANGSMITH_* variables as per best practices)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "ProXeek-Haptic-System"  # Set project name for better organization

if langchain_api_key:
    os.environ["LANGSMITH_API_KEY"] = langchain_api_key
    # Also set LANGCHAIN_* for backward compatibility
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    log("LangSmith tracing enabled with API key and project: ProXeek-Haptic-System")
    
    # Verify tracing setup
    try:
        from langsmith import Client
        ls_client = Client()
        log("LangSmith client initialized successfully")
    except Exception as e:
        log(f"Warning: LangSmith client initialization failed: {e}")
else:
    log("Warning: LangSmith API key not found - tracing may not work properly")

# o4-mini-2025-04-16
# Initialize the physical object recognition LLM
# Note: LangChain ChatOpenAI has built-in LangSmith integration - no need for wrap_openai()
physical_object_recognition_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    temperature=0.1,
    base_url="https://api.nuwaapi.com/v1",
    api_key=SecretStr(api_key) if api_key else None
)

# Initialize the virtual object processing LLM
virtual_object_processor_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    temperature=0.1,
    base_url="https://api.nuwaapi.com/v1",
    api_key=SecretStr(api_key) if api_key else None
)

# Initialize the proxy matching LLM
proxy_matching_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    temperature=0.1,
    base_url="https://api.nuwaapi.com/v1",
    api_key=SecretStr(api_key) if api_key else None
)

# Initialize the property rating LLM
property_rating_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    temperature=0.1,
    base_url="https://api.nuwaapi.com/v1",
    api_key=SecretStr(api_key) if api_key else None
)

# Initialize the relationship rating LLM
relationship_rating_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    temperature=0.1,
    base_url="https://api.nuwaapi.com/v1",
    api_key=SecretStr(api_key) if api_key else None
)
log("Initialized relationship_rating_llm for LangSmith tracing")

# Initialize the substrate utilization LLM
substrate_utilization_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    temperature=0.1,
    base_url="https://api.nuwaapi.com/v1",
    api_key=SecretStr(api_key) if api_key else None
)



# YOLO-World + EfficientSAM configuration  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.01
IOU_THRESHOLD = 0.5

# Log CUDA availability and device info
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"CUDA device count: {torch.cuda.device_count()}")
    log(f"Current CUDA device: {torch.cuda.current_device()}")
    log(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    log(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    log("CUDA not available - running on CPU")
log(f"Selected device: {DEVICE}")

# Initialize models (will be done lazily when needed)
yolo_world_model = None
efficient_sam_model = None

# System prompt for object recognition
object_recognition_system_prompt = """
You are an expert computer vision system that identifies objects in images.

For each image, create a detailed list of all recognizable objects with the following information:

1. Its name with some details (e.g., "white cuboid airpods case")
2. Its position in the image (e.g., "bottom left of the image")

FORMAT YOUR RESPONSE AS A JSON ARRAY with the following structure:

```json
[
  {
    "object_id": 1,
    "object": "object name with some details",
    "position": "position in image"
  },
  {
    "object_id": 2,
    ...
  }
]
```

Be comprehensive and include all clearly visible objects.
"""

# System prompt for virtual object processing
virtual_object_processor_system_prompt = """
You are an expert in user interaction analysis who specializes in understanding how users physically interact with virtual objects in VR.

Your task is to analyze virtual object metadata and its snapshot, and deduce how users would likely interact with these virtual objects based on the annotations. This will help identify appropriate physical proxies from the real environment.

For each virtual object, consider the following information:
- objectName: The target virtual object in the VR scene
- involvementType: grasp: objects that users hold and manipulate. contact: objects that users nudge/hit/kick with their body parts. substrate: objects that users interact with indirectly using other objects.
- usage: Overall usage of this virtual object in the VR scene
- inertia: Highly expected haptic feedback, if any, regarding the target virtual object's mass, weight distribution, and resistance to movement.
- interactivity: Highly expected haptic feedback, if any, regarding how the virtual object responds to user actions.
- outline: Highly expected haptic feedback, if any, regarding the target virtual object's shape and size.
- texture: Highly expected haptic feedback, if any, regarding the target virtual object's surface feel and tactile patterns.
- hardness: Highly expected haptic feedback, if any, regarding the target virtual object's rigidity, compliance, and deformation.
- temperature: Highly expected haptic feedback, if any, regarding the target virtual object's thermal properties and heat transfer.
- dimensions_meters: Physical dimensions of the object

Consider these deduction strategies based on involvement type:

For grasp objects:
- Identify the grip type (power grip, precision grip, pinch, etc.)
- Consider manipulation patterns (swing, rotate, squeeze, trigger, etc.)
- Analyze tool-like behaviors (pointing, striking, cutting motions, etc.)

For contact objects:
- Identify contact method (press, push, strike, slide, etc.)
- Determine body part involved (finger, palm, fist, foot, etc.)
- Consider interaction duration (momentary tap vs. sustained contact)

Create a comprehensive interaction deduction that:
1. Identifies the specific way users would interact with the object
2. Describes the physical movements and contact patterns
3. Explains how the object would respond to user interaction

FORMAT YOUR RESPONSE AS A JSON ARRAY where each object has:
- "objectName": Name of the virtual object
- "interactionDeduction": Your comprehensive interaction deduction

The JSON should look like:
```json
[
  {
    "objectName": "Example Object",
    "interactionDeduction": "Detailed description of how users would interact with this object..."
  },
  ...
]
```
"""

# System prompt for proxy matching
proxy_matching_system_prompt = """
You are an expert in haptic design who specializes in finding physical proxies for virtual objects in VR.

Your task is to analyze ONE virtual object and evaluate ALL physical objects from the environment as potential haptic proxies.

First, carefully consider the deduced interaction for the virtual object—how users are expected to interact with it

For each physical object, propose a specific method to utilize it as a haptic proxy that best replicates the deduced interaction of the virtual object.

Focus on matching the most important haptic properties of the virtual object (those with higher importance values), but always ensure your proxy method enables the user to perform the same type of interaction as described in the interaction deduction.

Make sure to include the object_id and image_id for each physical object exactly as they appear in the detected objects list.

CRITICAL REQUIREMENT: You MUST evaluate and generate a utilization method for EVERY SINGLE physical object shown in list (Detected Objects in this Snapshot). Do not skip any objects. Even if an object seems unsuitable, you must still propose a utilization method and explain how it could potentially be used as a proxy.

IMPORTANT: Image IDs begin at 0 (not 1). The first image has image_id=0, the second has image_id=1, etc.
"""

# Function to generate property-specific system prompt
def get_property_rating_system_prompt(property_name):
    property_type = property_name.replace("Value", "")
    
    # Base prompt
    base_prompt = f"""
You are an expert in haptic design who specializes in evaluating how well physical objects can deliver the expected haptic feedback for virtual objects in VR.

Your task is to evaluate how well each physical object can provide the expected {property_type} haptic feedback described for the virtual object. Focus on the specific haptic description provided rather than making general assumptions about the property type.

Rate each physical object on a 7-point Likert scale based on how well it can deliver the described {property_type} experience:
1 - Strongly Disagree 
2 - Disagree
3 - Somewhat Disagree
4 - Neutral
5 - Somewhat Agree
6 - Agree
7 - Strongly Agree

Use the following rubric to guide your evaluation:
"""

    # Property-specific rubrics
    rubrics = {
        "inertia": """
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
        "interactivity": """
Interactivity:
- 1-Strong Disagree
  - Required interactive elements are completely absent or non-functional
  - User cannot perform the intended actions at all
- 7-Strong Agree
  - All interactive elements are present and function intuitively as expected
  - Degrees of freedom match exactly (rotation axes, sliding directions, button positions)
""",
        "outline": """
Outline:
- 1-Strong Disagree
  - Size mismatch is immediately apparent and disrupts grip formation
  - Basic shape category is entirely different (e.g., spherical physical object for a virtual tetrahedron)
  - Key affordances or contact points are absent
- 7-Strong Agree
  - Size and proportions feel completely natural in the hand
  - Shape affords all expected grips and manipulation patterns
""",
        "texture": """
Texture:
- 1-Strong Disagree
  - Surface finishing is shockingly different from expectations (e.g., extremely rough physical surface for virtual polished glass)
  - Tactile landmarks are missing or misplaced
- 7-Strong Agree
  - Surface texture feels exactly as anticipated
  - Texture transitions occur at expected locations
""",
        "hardness": """
Hardness:
- 1-Strong Disagree
  - Compliance is completely wrong, it affects basic interaction (e.g., soft foam for a virtual metal tool)
  - Deformation behavior is shocking and breaks immersion
- 7-Strong Agree
  - Material hardness feels precisely as expected
  - Deformation behavior matches material expectations perfectly
""",
        "temperature": """
Temperature:
- 1-Strong Disagree
  - Temperature sensation is shockingly wrong or opposite to expectations (e.g., warm/hot physical object for virtual ice cube)
  - Thermal conductivity creates wrong sensations (e.g., insulating material for a virtual metal object)
- 7-Strong Agree
  - Initial temperature matches the expected thermal sensation
  - Heat flow during contact feels natural for the material type
"""
    }
    
    # Output format
    output_format = """
FORMAT YOUR RESPONSE AS A JSON ARRAY with the following structure:
```json
[
  {
    "virtualObject": "name of the virtual object",
    "property": "name of the property being evaluated",
    "physicalObject": "name of the physical object",
    "object_id": 1,
    "image_id": 0,
    "rating": 5,
    "explanation": "Brief explanation of why this rating was given"
  },
  ...
]
```

Make sure to include ALL physical objects in your evaluation, even those with low ratings.
"""
    
    # Construct the complete prompt with only the relevant property rubric
    full_prompt = base_prompt + rubrics.get(property_type.lower(), "") + output_format
    return full_prompt

# Function to generate relationship dimension-specific system prompt
def get_relationship_rating_system_prompt(dimension_name):
    dimension_type = dimension_name.lower()
    
    # Base prompt
    base_prompt = f"""
You are an expert in haptic design who specializes in evaluating how well pairs of physical objects can simulate the expected haptic feedback when two virtual objects interact with each other in VR.

You will be provided with pre-generated substrate utilization methods for each contact-substrate pair. Your task is to critically evaluate how well each pair can deliver the expected haptic feedback, considering both the contact object's utilization method and the provided substrate utilization method.

Rate each physical object pair on a 7-point Likert scale for {dimension_name}:
1 - Strongly Disagree 
2 - Disagree
3 - Somewhat Disagree
4 - Neutral
5 - Somewhat Agree
6 - Agree
7 - Strongly Agree

Focus specifically on the {dimension_name} dimension:
"""

    # Dimension-specific rubrics
    rubrics = {
        "harmony": """
**Harmony Dimension**: "I felt the haptic feedback was well coordinated with visual feedback"

Focus on the synchronization of contact-substrate contact:

Score 1 - Strongly Disagree:
- Physical contact happens noticeably before visual contact or visual contact occurs with no physical sensation
- Force direction contradicts visual motion
- Visual substrate responses don't match felt impact intensity

Score 7 - Strongly Agree:
- Physical and visual contact perfectly synchronized
- Every visual contact event has corresponding haptic feedback
- Force vectors align naturally with visual physics
- Substrate visual behavior matches haptic intensity
""",
        "expressivity": """
**Expressivity Dimension**: "I felt the contact object effectively conveyed substrate properties and interaction variations through my hand"

Focus on how well substrate properties are conveyed through the contact object:

Score 1 - Strongly Disagree:
- Single uniform feedback regardless of interaction parameters
- No variation with impact speed, angle, or force
- Contact object fails to transmit any substrate characteristics

Score 7 - Strongly Agree:
- Rich feedback variations convey substrate properties clearly
- Natural variations in speed, angle, and force produce different haptic responses
- Contact object effectively transmits substrate material properties and surface characteristics
""",
        "realism": """
**Realism Dimension**: "I felt using this physical contact object on this physical substrate closely simulated the intended haptic feedback"

Focus on how well the overall contact-substrate interaction matches the expected haptic experience:

Score 1 - Strongly Disagree:
- The interaction feels fundamentally wrong (e.g., soft bouncy feedback when hammering should feel solid)
- Missing essential haptic elements that define the expected haptic feedback
- Overall experience contradicts the virtual interaction expectations

Score 7 - Strongly Agree:
- All characteristic sensations of the real interaction are present (impacts, resistance, texture transmission, etc.)
- The physical pairing naturally affords the same manipulation techniques as the virtual scenario
- The combined utilization methods deliver the essential haptic elements described in the expected feedback
"""
    }
    
    # Output format
    output_format = f"""
FORMAT YOUR RESPONSE AS A JSON ARRAY with the following structure:
```json
[
  {{
    "virtualContactObject": "FULL name of virtual contact object",
    "virtualSubstrateObject": "FULL name of virtual substrate object", 
    "physicalContactObject": "FULL name of physical contact object",
    "physicalSubstrateObject": "FULL name of physical substrate object",
    "contactObject_id": 1,
    "contactImage_id": 0,
    "substrateObject_id": 2,
    "substrateImage_id": 1,
    "contactUtilizationMethod": "utilization method for the contact object",
    "substrateUtilizationMethod": "your planned utilization method for the substrate object",
    "{dimension_type}_rating": 5,
    "{dimension_type}_explanation": "Brief explanation for {dimension_type} rating considering both utilization methods"
  }},
  ...
]
```

Include EVERY pair in your evaluation.
"""
    
    # Construct the complete prompt with only the relevant dimension rubric
    full_prompt = base_prompt + rubrics.get(dimension_type, "") + output_format
    return full_prompt



# Function to generate substrate-type-specific system prompt
def get_substrate_utilization_system_prompt(substrate_type):
    # Base prompt components
    base_intro = """You are an expert in haptic design who specializes in determining how physical objects can be utilized as substrates to simulate virtual substrate objects in VR.

Your task is to analyze ONE specific virtual contact-substrate relationship and determine how each physical object in the environment could be utilized as a substrate to work with a specific physical contact object."""
    
    # Type-specific context and instructions
    if substrate_type == "pure":
        context_section = """
**Context**: The virtual substrate object is a PURE SUBSTRATE (involvementType = "substrate") that users do not directly interact with, but rather interact with indirectly through other objects.

You will be given:
1. A virtual contact object and its corresponding physical contact object with its utilization method
2. A virtual substrate object (pure substrate) and the expected haptic feedback for their interaction
3. All physical objects in the environment as potential substrate candidates

For each physical substrate candidate, determine:
- How it should be positioned, oriented, or prepared to serve as a substrate
- What specific properties or features should be utilized
- How it would interact with the given physical contact object's utilization method
- What modifications or setup might be needed"""
    else:  # dual-role
        context_section = """
**Important Context**: The virtual substrate object is a DUAL-ROLE object that:
1. Serves as a contact object that users directly interact with (already has proxy matching results)
2. Also serves as a substrate object in relationships with other contact objects

You will be given:
1. A virtual contact object and its corresponding physical contact object with its utilization method
2. A virtual substrate object (dual-role) that ALSO has existing utilization methods from proxy matching
3. All physical objects in the environment as potential substrate candidates
4. Existing utilization methods for physical objects when used as proxies for the dual-role virtual substrate object

For each physical substrate candidate, consider:
- Its existing utilization method as a proxy for the virtual substrate object (if any)
- How this existing method can be adapted or enhanced for substrate use
- How it should be positioned/oriented to work with the contact object's utilization method
- How to combine its contact-proxy role with its substrate role
- What additional setup or modifications might be needed

**Key Consideration**: When a physical object already has a utilization method for the virtual substrate object from proxy matching, build upon that existing method rather than completely replacing it. The substrate utilization should complement and work with the existing contact utilization."""
    
    # Common evaluation guidelines
    guidelines_section = """
Use the following evaluation guidelines when planning substrate utilization methods:

**Harmony Considerations:**
- Ensure the substrate setup allows for synchronized physical and visual contact
- Consider how the substrate positioning affects force direction alignment
- Plan for substrate responses that match expected visual behavior

**Expressivity Considerations:**
- Think about how the substrate can provide varied feedback based on interaction parameters
- Consider how different contact speeds, angles, or forces would affect the substrate response
- Plan for substrate properties that can be effectively conveyed through the contact object

**Realism Considerations:**
- Focus on substrate utilization that delivers the essential haptic sensations described in the expected feedback
- Ensure the substrate setup naturally affords the intended manipulation techniques
- Consider how well the substrate can simulate the virtual substrate's key properties"""
    
    # Common output format
    output_format = """
FORMAT YOUR RESPONSE AS A JSON ARRAY with the following structure:
```json
[
  {
    "virtualContactObject": "name of virtual contact object",
    "virtualSubstrateObject": "name of virtual substrate object",
    "physicalContactObject": "name of physical contact object",
    "contactObject_id": 1,
    "contactImage_id": 0,
    "contactUtilizationMethod": "utilization method for the contact object",
    "expectedHapticFeedback": "expected haptic feedback for the interaction",
    "physicalSubstrateObject": "name of physical substrate object",
    "substrateObject_id": 2,
    "substrateImage_id": 1,
    "substrateUtilizationMethod": "detailed method to utilize this object as a substrate for the given contact object"
  },
  ...
]
```

Include ALL physical objects as potential substrate candidates in your evaluation."""
    
    # Combine all sections
    full_prompt = base_intro + context_section + guidelines_section + output_format
    return full_prompt

# Function to process a single image and recognize objects
@traceable(run_type="llm", metadata={"process": "physical_object_extraction"})
async def process_single_image(image_base64: str, image_id: int) -> Dict[str, Any]:
    try:
        # Create the human message with image content
        human_message_content = [
            {"type": "text", "text": f"Identify all objects in this image (image ID: {image_id})."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}}
        ]
        
        # Create the messages
        messages = [
            SystemMessage(content=object_recognition_system_prompt),
            HumanMessage(content=human_message_content)
        ]
        
        # Get response from the model
        log(f"Sending image {image_id} to object recognition model")
        # LangChain ChatOpenAI has built-in LangSmith tracing - no extra config needed
        response = await physical_object_recognition_llm.ainvoke(messages)
        log(f"Received response for image {image_id}")
        
        # Extract JSON from response
        response_text = extract_response_text(response.content)
        
        # Find JSON content between ```json and ```
        json_start = response_text.find("```json")
        if json_start != -1:
            json_start += 7  # Length of ```json
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                json_content = response_text[json_start:json_end].strip()
            else:
                json_content = response_text[json_start:].strip()
        else:
            # Try to find any JSON array in the response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                json_content = response_text[json_start:json_end].strip()
            else:
                json_content = response_text
        
        try:
            # Parse the JSON response
            objects = json.loads(json_content)
            
            # Add image_id to each object
            for obj in objects:
                obj["image_id"] = image_id
                
            return {"image_id": image_id, "objects": objects, "status": "success"}
            
        except json.JSONDecodeError as e:
            log(f"Error parsing JSON for image {image_id}: {e}")
            log(f"Raw content: {json_content}")
            return {"image_id": image_id, "objects": [], "status": "error", "error": str(e)}
            
    except Exception as e:
        log(f"Error processing image {image_id}: {e}")
        return {"image_id": image_id, "objects": [], "status": "error", "error": str(e)}

# Process multiple images concurrently
@traceable(run_type="chain", metadata={"process": "physical_object_extraction_batch"})
async def process_multiple_images(environment_images: List[str]) -> Dict[int, List[Dict]]:
    tasks = []
    for i, image_base64 in enumerate(environment_images):
        tasks.append(process_single_image(image_base64, i))
    
    results = await asyncio.gather(*tasks)
    
    # Organize results into a database
    object_database = {}
    for result in results:
        image_id = result["image_id"]
        if result["status"] == "success":
            object_database[image_id] = result["objects"]
        else:
            log(f"Processing failed for image {image_id}: {result.get('error', 'Unknown error')}")
            object_database[image_id] = []
            
    return object_database

# Function to enhance physical object database with YOLO-World bounding boxes
@traceable(run_type="chain", metadata={"process": "yolo_segmentation_enhancement"})
async def enhance_with_yolo_segmentation(object_database: Dict[int, List[Dict]], environment_images: List[str]) -> Dict[int, List[Dict]]:
    """Enhance the physical object database with YOLO-World + EfficientSAM segmentation"""
    log("Starting YOLO-World + EfficientSAM segmentation enhancement...")
    
    enhanced_database = {}
    
    for image_id, objects in object_database.items():
        log(f"Processing YOLO segmentation for image {image_id} with {len(objects)} objects")
        
        if len(objects) == 0:
            enhanced_database[image_id] = objects
            continue
        
        # Extract object names with positions for YOLO detection
        # This sends the full object description including position to YOLO-World
        # Format: "white handheld scanner (bottom right of the image)"
        object_names = []
        for obj in objects:
            # Include both object name and position in the format "object (position)"
            object_name = obj.get("object", "").strip()
            if object_name:
                object_names.append(object_name)
        
        # Remove duplicates while preserving order
        unique_names = []
        seen = set()
        for name in object_names:
            if name not in seen:
                unique_names.append(name)
                seen.add(name)
        
        log(f"Running YOLO segmentation on image {image_id} for objects with positions: {unique_names}")
        
        # Run YOLO segmentation
        try:
            image_base64 = environment_images[image_id]
            yolo_segmentations = run_yolo_segmentation(image_base64, unique_names, image_id)
            
            # Match YOLO segmentations to LLM-identified objects
            enhanced_objects = []
            for obj in objects:
                enhanced_obj = obj.copy()
                
                # Try to find matching YOLO segmentation
                obj_name = obj.get("object", "").strip().lower()
                best_match = None
                best_confidence = 0.0
                
                for segmentation in yolo_segmentations:
                    detected_name = segmentation.get("detected_object_name", "").lower()
                    confidence = segmentation.get("confidence", 0.0)
                    
                    # Extract just the object name part (before parentheses) for matching
                    obj_name_part = obj_name.split("(")[0].strip()
                    detected_name_part = detected_name.split("(")[0].strip()
                    
                    # Simple name matching (can be improved with fuzzy matching if needed)
                    if detected_name_part in obj_name_part or obj_name_part in detected_name_part:
                        if confidence > best_confidence:
                            best_match = segmentation
                            best_confidence = confidence
                
                # Add segmentation information if found
                if best_match:
                    enhanced_obj["yolo_segmentation"] = {
                        "bbox": best_match["bbox"],
                        "mask_center": best_match["mask_center"],
                        "confidence": best_match["confidence"],
                        "detected_name": best_match["detected_object_name"],
                        "has_mask": best_match["has_mask"],
                        "mask_contours": best_match.get("mask_contours", [])
                    }
                    # Keep original 'position'; do not overwrite with mask center coordinates
                    log(f"Matched '{obj.get('object', '')}' with YOLO segmentation '{best_match['detected_object_name']}' (confidence: {best_confidence:.3f})")
                else:
                    enhanced_obj["yolo_segmentation"] = None
                    log(f"No YOLO match found for '{obj.get('object', '')}'")
                
                enhanced_objects.append(enhanced_obj)
            
            # Also add any unmatched YOLO segmentations as additional objects
            matched_segmentations = set()
            for obj in enhanced_objects:
                if obj.get("yolo_segmentation"):
                    matched_segmentations.add(obj["yolo_segmentation"]["detected_name"])
            
            for segmentation in yolo_segmentations:
                detected_name = segmentation["detected_object_name"]
                if detected_name not in matched_segmentations:
                    # Create a new object entry for unmatched YOLO segmentation
                    additional_obj = {
                        "object_id": len(enhanced_objects) + 1,
                        "object": f"{detected_name} (YOLO-only segmentation)",
                        "position": f"detected at mask center ({segmentation['mask_center']['x']:.0f}, {segmentation['mask_center']['y']:.0f})",
                        "image_id": image_id,
                        "yolo_segmentation": {
                            "bbox": segmentation["bbox"],
                            "mask_center": segmentation["mask_center"],
                            "confidence": segmentation["confidence"],
                            "detected_name": detected_name,
                            "has_mask": segmentation["has_mask"],
                            "mask_contours": segmentation.get("mask_contours", [])
                        }
                    }
                    enhanced_objects.append(additional_obj)
                    log(f"Added YOLO-only segmentation: '{detected_name}' (confidence: {segmentation['confidence']:.3f})")
            
            # ------------------------------------------------------------------
            # SECOND PASS: ultra-low-threshold YOLO detection for still-unmatched
            # objects (e.g., "blue plastic food storage container with a clear
            # lid").  Runs once per remaining object using the format
            #   "object name (position)"  and confidence 0.0001.
            # ------------------------------------------------------------------
            for obj in enhanced_objects:
                if obj.get("yolo_segmentation") is None:
                    obj_name = obj.get("object", "").strip()
                    obj_pos = obj.get("position", "").strip()
                    if not obj_name:
                        continue  # skip nameless entries

                    query_name = f"{obj_name} ({obj_pos})" if obj_pos else obj_name

                    # Temporarily override global confidence threshold
                    global CONFIDENCE_THRESHOLD
                    _orig_conf = CONFIDENCE_THRESHOLD
                    CONFIDENCE_THRESHOLD = 0.0001
                    try:
                        low_segs = run_yolo_segmentation(image_base64, [query_name], image_id)
                        if low_segs:
                            seg = low_segs[0]
                            obj["yolo_segmentation"] = {
                                "bbox": seg["bbox"],
                                "mask_center": seg["mask_center"],
                                "confidence": seg["confidence"],
                                "detected_name": seg["detected_object_name"],
                                "has_mask": seg["has_mask"],
                                "mask_contours": seg.get("mask_contours", []),
                            }
                            log(
                                f"Second-pass YOLO matched '{obj_name}' in image {image_id} "
                                f"(conf {seg['confidence']:.4f})"
                            )
                    except Exception as e:
                        log(
                            f"Second-pass YOLO error for '{obj_name}' in image {image_id}: {e}"
                        )
                    finally:
                        CONFIDENCE_THRESHOLD = _orig_conf
            
            enhanced_database[image_id] = enhanced_objects
            
        except Exception as e:
            log(f"Error in YOLO segmentation for image {image_id}: {e}")
            # Keep original objects without YOLO enhancement
            enhanced_database[image_id] = objects
    
    total_enhanced = sum(len(objects) for objects in enhanced_database.values())
    log(f"YOLO-World + EfficientSAM segmentation enhancement completed. Enhanced {total_enhanced} objects across {len(enhanced_database)} images")
    
    return enhanced_database

# Function to export annotated images with bounding boxes
@traceable(run_type="chain", metadata={"process": "image_annotation_export"})
async def export_annotated_images(enhanced_database: Dict[int, List[Dict]], environment_images: List[str], output_dir: str) -> List[str]:
    """Export environment images with segmentation annotations drawn on them"""
    log("Exporting annotated images with segmentation annotations...")
    
    annotated_image_paths = []
    
    # Ensure output directory exists
    annotated_dir = os.path.join(output_dir, "annotated_images")
    os.makedirs(annotated_dir, exist_ok=True)
    
    for image_id, objects in enhanced_database.items():
        try:
            # Convert base64 to image
            image_data = base64.b64decode(environment_images[image_id])
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                log(f"Failed to decode image {image_id}")
                continue
            
            # Draw bounding boxes, mask centers, labels, and mask overlays
            for obj in objects:
                yolo_segmentation = obj.get("yolo_segmentation")
                if yolo_segmentation and yolo_segmentation.get("bbox"):
                    bbox = yolo_segmentation["bbox"]
                    confidence = yolo_segmentation["confidence"]
                    detected_name = yolo_segmentation["detected_name"]
                    mask_center = yolo_segmentation.get("mask_center")
                    mask_contours = yolo_segmentation.get("mask_contours", [])
                    
                    # Extract coordinates
                    x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                    x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                    
                    # Draw mask overlay if available
                    if mask_contours and len(mask_contours) > 0:
                        # Create a semi-transparent overlay for the mask
                        mask_overlay = image.copy()
                        
                        # Draw filled contours for mask visualization
                        for contour_points in mask_contours:
                            if len(contour_points) > 2:  # Need at least 3 points for a polygon
                                # Convert points to numpy array
                                contour_array = np.array(contour_points, dtype=np.int32)
                                
                                # Draw filled contour with semi-transparent color
                                cv2.fillPoly(mask_overlay, [contour_array], (0, 255, 255))  # Cyan color
                        
                        # Blend the mask overlay with the original image
                        alpha = 0.3  # Transparency factor
                        image = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
                        
                        # Draw contour outlines
                        for contour_points in mask_contours:
                            if len(contour_points) > 2:
                                contour_array = np.array(contour_points, dtype=np.int32)
                                cv2.polylines(image, [contour_array], True, (0, 255, 255), 2)  # Cyan outline
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green color
                    thickness = 2
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw mask center point if available
                    if mask_center:
                        center_x, center_y = int(mask_center["x"]), int(mask_center["y"])
                        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)  # Red circle
                        cv2.circle(image, (center_x, center_y), 8, (0, 0, 255), 2)   # Red outline
                    
                    # Draw label with confidence and mask info
                    mask_info = " (with mask)" if mask_contours and len(mask_contours) > 0 else " (no mask)"
                    label = f"{detected_name}: {confidence:.2f}{mask_info}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    # Draw label background
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save annotated image
            output_path = os.path.join(annotated_dir, f"annotated_image_{image_id}.jpg")
            cv2.imwrite(output_path, image)
            annotated_image_paths.append(output_path)
            
            log(f"Saved annotated image {image_id} to {output_path}")
            
        except Exception as e:
            log(f"Error annotating image {image_id}: {e}")
            continue
    
    log(f"Exported {len(annotated_image_paths)} annotated images to {annotated_dir}")
    return annotated_image_paths

# Function to save object database to JSON file
def save_object_database(object_db: Dict[int, List[Dict]], output_path: str) -> Optional[str]:
    try:
        # --- NEW: merge with existing data to preserve Quest world positions ---
        merged_db: Dict[str, List[Dict]] = {}
        existing_db = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as prev_f:
                    existing_db = json.load(prev_f)
            except Exception as merge_err:
                log(f"Warning: Failed reading existing physical database for merge: {merge_err}")
                existing_db = {}
 
        # Deep-copy incoming db first (str keys for json)
        for img_id, obj_list in object_db.items():
            merged_db[str(img_id)] = []
            for obj in obj_list:
                merged_obj = json.loads(json.dumps(obj))  # simple deep copy
 
                # Try to pull worldposition from existing db if not present
                if "worldposition" not in merged_obj:
                    prev_objs = existing_db.get(str(img_id), [])
                    for prev in prev_objs:
                        if (
                            prev.get("object_id") == merged_obj.get("object_id")
                            and "worldposition" in prev
                        ):
                            merged_obj["worldposition"] = prev["worldposition"]
                            break
 
                # Remove heavy mask contours before save
                yolo_seg = merged_obj.get("yolo_segmentation")
                if yolo_seg and "mask_contours" in yolo_seg:
                    del yolo_seg["mask_contours"]
                merged_db[str(img_id)].append(merged_obj)
 
        # ------------------------------------------------------------------
        # Persist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(merged_db, f, indent=2)
 
        log(f"Object database saved to {output_path} (merged world positions where available)")
        return output_path
    except Exception as e:
        log(f"Error saving object database: {e}")
        return None

# New function to process virtual objects and generate deduced interaction patterns
@traceable(run_type="llm", metadata={"process": "virtual_object_processing"})
async def process_virtual_objects(haptic_annotation_json: str) -> List[Dict]:
    if not haptic_annotation_json:
        log("No haptic annotation data provided")
        return []
    
    try:
        # Parse the haptic annotation JSON
        haptic_data = json.loads(haptic_annotation_json)
        node_annotations = haptic_data.get("nodeAnnotations", [])
        
        if not node_annotations:
            log("No node annotations found in haptic data")
            return []
        
        log(f"Found {len(node_annotations)} virtual objects in haptic annotation data")
        
        # Create a map of normalized object name to snapshot for flexible lookup
        object_snapshot_map = {}
        normalized_name_map = {}  # Maps normalized names back to original names
        
        # First, create a map of normalized names to original snapshots
        for snapshot in virtual_object_snapshots:
            if 'objectName' in snapshot and 'imageBase64' in snapshot:
                original_name = snapshot['objectName']
                normalized_name = normalize_name(original_name)
                object_snapshot_map[normalized_name] = snapshot['imageBase64']
                normalized_name_map[normalized_name] = original_name
                # Also add the original name for direct matches
                object_snapshot_map[original_name] = snapshot['imageBase64']
        
        log(f"Found {len(object_snapshot_map)} virtual object snapshots")
        log(f"Normalized names: {list(normalized_name_map.keys())}")
        
        # Build the human message content with objects and their snapshots
        human_message_content = []
        
        # Add introduction text
        # human_message_content.append({
        #     "type": "text", 
        #     "text": "Please analyze the following virtual objects and create detailed haptic feedback descriptions for each. Focus on the properties with higher importance values (those with higher *Value numbers)."
        # })
        
        # Process each virtual object one by one, but only grasp and contact objects
        for node in node_annotations:
            object_name = node.get("objectName", "Unknown Object")
            involvement_type = node.get("involvementType", "")
            
            # Skip substrate objects - only process grasp and contact objects
            if involvement_type not in ["grasp", "contact"]:
                log(f"Skipping {object_name} with involvementType: {involvement_type}")
                continue
                
            normalized_object_name = normalize_name(object_name)
            
            # Create filtered object data with only the fields mentioned in system prompt
            filtered_node = {
                "objectName": node.get("objectName", ""),
                "involvementType": node.get("involvementType", ""),
                "usage": node.get("usage", ""),
                "inertia": node.get("inertia", ""),
                "interactivity": node.get("interactivity", ""),
                "outline": node.get("outline", ""),
                "texture": node.get("texture", ""),
                "hardness": node.get("hardness", ""),
                "temperature": node.get("temperature", ""),
                "dimensions_meters": node.get("dimensions_meters", {})
            }
            
            # Add object's filtered annotation data as JSON
            object_json = json.dumps(filtered_node, indent=2)
            object_text = f"\n\n## Virtual Object: {object_name}\n```json\n{object_json}\n```"
            
            # Add text content for this object
            human_message_content.append({
                "type": "text",
                "text": object_text
            })
            
            # Try to find snapshot with various name formats
            snapshot_found = False
            
            # First try direct match
            if object_name in object_snapshot_map:
                log(f"Found snapshot for {object_name} (direct match)")
                human_message_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{object_snapshot_map[object_name]}", 
                        "detail": "high"
                    }
                })
                snapshot_found = True
            # Then try normalized match
            elif normalized_object_name in object_snapshot_map:
                log(f"Found snapshot for {object_name} (normalized as {normalized_object_name})")
                human_message_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{object_snapshot_map[normalized_object_name]}", 
                        "detail": "high"
                    }
                })
                snapshot_found = True
            # Finally try a fuzzy match
            else:
                # Try to find partial matches
                potential_matches = [norm_name for norm_name in normalized_name_map.keys() 
                                    if normalized_object_name in norm_name or norm_name in normalized_object_name]
                
                if potential_matches:
                    best_match = potential_matches[0]  # Take the first match
                    log(f"Found snapshot for {object_name} (fuzzy match: {normalized_name_map[best_match]})")
                    human_message_content.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{object_snapshot_map[best_match]}", 
                            "detail": "high"
                        }
                    })
                    snapshot_found = True
            
            if not snapshot_found:
                log(f"No snapshot found for {object_name} (normalized: {normalized_object_name})")
        
        # Add final instruction
        # human_message_content.append({
        #     "type": "text",
        #     "text": "\nDescribe what makes a good physical proxy for each object based on its haptic properties."
        # })
        
        # Create the messages
        messages = [
            SystemMessage(content=virtual_object_processor_system_prompt),
            HumanMessage(content=human_message_content)
        ]
        
        # Get response from the model
        log("Sending virtual object data to LLM for haptic feedback processing")
        # LangChain ChatOpenAI has built-in LangSmith tracing - no extra config needed
        response = await virtual_object_processor_llm.ainvoke(messages)
        log("Received haptic feedback descriptions")
        
        # Extract JSON from response using a more robust approach
        response_text = extract_response_text(response.content)
        
        # First try to find JSON between code blocks
        json_start = response_text.find("```json")
        if json_start != -1:
            json_start += 7  # Length of ```json
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                json_content = response_text[json_start:json_end].strip()
            else:
                json_content = response_text[json_start:].strip()
        else:
            # Try to find JSON array directly
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                json_content = response_text[json_start:json_end].strip()
            else:
                # As a fallback, try to use the entire response
                json_content = response_text
        
        try:
            # Parse the JSON response
            haptic_feedback_data = json.loads(json_content)
            
            # Create a mapping from object name to haptic feedback
            haptic_feedback_map = {item["objectName"]: item["interactionDeduction"] for item in haptic_feedback_data}
            
            # Merge the original node annotations with the haptic feedback descriptions
            # Only include grasp and contact objects
            enhanced_node_annotations = []
            for node in node_annotations:
                involvement_type = node.get("involvementType", "")
                
                # Skip substrate objects - only include grasp and contact objects
                if involvement_type not in ["grasp", "contact"]:
                    continue
                    
                object_name = node["objectName"]
                enhanced_node = node.copy()
                enhanced_node["interactionDeduction"] = haptic_feedback_map.get(object_name, "No interaction deduction available")
                enhanced_node_annotations.append(enhanced_node)
            
            return enhanced_node_annotations
            
        except json.JSONDecodeError as e:
            log(f"Error parsing haptic feedback JSON: {e}")
            log(f"Raw content: {json_content}")
            
            # Return the original node annotations without haptic feedback as a fallback
            # Only include grasp and contact objects
            return [node.copy() for node in node_annotations 
                    if node.get("involvementType", "") in ["grasp", "contact"]]
            
    except Exception as e:
        log(f"Error processing virtual objects: {e}")
        import traceback
        log(traceback.format_exc())
        return []

# Function to match a single virtual object with physical objects
@traceable(run_type="llm", metadata={"process": "proxy_matching"})
async def match_single_virtual_object(virtual_object, environment_images, physical_object_database, object_snapshot_map):
    try:
        virtual_object_name = virtual_object.get("objectName", "Unknown Object")
        log(f"Matching proxies for virtual object: {virtual_object_name}")
        
        # Build the human message content
        human_message_content = []
        
        # 1. Add the virtual object information and haptic feedback
        virtual_object_text = f"""# Virtual Object to Evaluate: {virtual_object_name}

## Haptic Properties
```json
{json.dumps(virtual_object, indent=2)}
```

"""
        human_message_content.append({
            "type": "text", 
            "text": virtual_object_text
        })
        
        # 2. Add virtual object snapshot if available
        normalized_object_name = normalize_name(virtual_object_name)
        snapshot_found = False
        
        if virtual_object_name in object_snapshot_map:
            log(f"Adding snapshot for virtual object: {virtual_object_name}")
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[virtual_object_name]}", 
                    "detail": "high"
                }
            })
            snapshot_found = True
        elif normalized_object_name in object_snapshot_map:
            log(f"Adding snapshot for virtual object: {virtual_object_name} (normalized)")
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[normalized_object_name]}", 
                    "detail": "high"
                }
            })
            snapshot_found = True
            
        # 3. Add introduction to physical environment
        human_message_content.append({
            "type": "text", 
            "text": "\n# Physical Environment\nBelow are snapshots of the physical environment with detected objects that could serve as haptic proxies:"
        })
        
        # Pre-check if we actually have objects in the database to avoid "no objects detected" message
        total_objects = sum(len(objects) for objects in physical_object_database.values())
        log(f"Preparing to display {total_objects} physical objects from database")
        
        # 4. Add environment snapshots with their detected objects
        for i, image_base64 in enumerate(environment_images):
            # Add the environment snapshot
            human_message_content.append({
                "type": "text", 
                "text": f"\n## Environment Snapshot {i+1}\n"
            })
            
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}", 
                    "detail": "high"
                }
            })
            
            # Add the detected objects for this snapshot
            objects_in_snapshot = physical_object_database.get(str(i), [])
            objects_text = "\n### Detected Objects in this Snapshot\n"
            
            if objects_in_snapshot:
                for obj in objects_in_snapshot:
                    objects_text += f"- Object ID {obj['object_id']}: {obj['object']} ({obj['position']})\n"
                    objects_text += f"  Image ID: {i}\n"
                    
                    # Collect all utilization methods for this specific physical object
                    utilization_methods = []
                    
                    # Convert object_id to both string and int for flexible comparison
                    obj_id_int = obj['object_id']
                    if isinstance(obj_id_int, str):
                        try:
                            obj_id_int = int(obj_id_int)
                        except ValueError:
                            pass
                    
                    obj_id_str = str(obj['object_id'])
                    
                    for proxy_result in proxy_matching_results:
                        # Get proxy object_id in both formats for comparison
                        proxy_obj_id = proxy_result.get('object_id')
                        proxy_obj_id_str = str(proxy_obj_id) if proxy_obj_id is not None else None
                        
                        # Get proxy image_id in both formats for comparison
                        proxy_img_id = proxy_result.get('image_id')
                        proxy_img_id_int = proxy_img_id
                        if isinstance(proxy_img_id, str):
                            try:
                                proxy_img_id_int = int(proxy_img_id)
                            except ValueError:
                                pass
                        
                        # More flexible comparison with multiple type checks
                        if ((proxy_obj_id == obj['object_id'] or proxy_obj_id == obj_id_int or proxy_obj_id_str == obj_id_str) and
                            (proxy_img_id == i or proxy_img_id_int == i)):
                            util_method = proxy_result.get("utilizationMethod", "")
                            matched_virtual = proxy_result.get("virtualObject", "Unknown")
                            if util_method:
                                log(f"Found utilization method for {matched_virtual}, obj_id: {proxy_obj_id}, img_id: {proxy_img_id}")
                                utilization_methods.append(f"    • {matched_virtual}: {util_method}")
                            else:
                                log(f"Found proxy result for {matched_virtual} but no utilization method")
                    
                    # Add all utilization methods for this object in an organized way
                    if utilization_methods:
                        objects_text += f"  Available Utilization Methods:\n"
                        for method in utilization_methods:
                            objects_text += f"{method}\n"
                    else:
                        objects_text += f"  No utilization methods available for this object\n"
                        log(f"No utilization methods found for obj_id: {obj['object_id']}, img_id: {i}")
                    
                    objects_text += f"\n"  # Add spacing between objects
            else:
                # Check if we should look for objects in a different format (in case image_id is stored as integer keys)
                objects_in_snapshot = physical_object_database.get(i, [])
                if objects_in_snapshot:
                    for obj in objects_in_snapshot:
                        objects_text += f"- Object ID {obj['object_id']}: {obj['object']} ({obj['position']})\n"
                        # Also display the correct image_id to ensure consistency
                        objects_text += f"  Image ID: {i}\n"
                    else:
                        objects_text += "- No objects detected in this snapshot\n"
                
            human_message_content.append({
                "type": "text", 
                "text": objects_text
            })
        
        # 5. Add final instructions
        human_message_content.append({
            "type": "text", 
            "text": """
# Your Task

1. Evaluate EACH physical object as a potential haptic proxy for the virtual object.
2. For EACH physical object, propose a specific method to utilize it as a haptic proxy.

CRITICAL REQUIREMENTS:
- You MUST include EVERY SINGLE physical object listed above in your response
- Do NOT skip any objects, even if they seem unsuitable
- Generate a utilization method for each object, explaining how it could be used as a proxy
- Count the total number of physical objects shown and ensure your JSON array has the same number of entries

FORMAT YOUR RESPONSE AS A JSON ARRAY with objects having the following structure:

```json
[
  {
    "virtualObject": "name of the virtual object",
    "physicalObject": "name of the physical object",
    "object_id": 1,
    "image_id": 0,
    "proxyLocation": "location of the physical object in the environment",
    "utilizationMethod": "detailed method to use this object as a proxy"
  },
  ...
]
```

IMPORTANT: 
- Make sure to use the EXACT image_id values shown above for each object
- Include ALL physical objects in your evaluation - no exceptions
- Your JSON array length must match the total number of physical objects shown
"""
        })
        
        # Create the messages
        messages = [
            SystemMessage(content=proxy_matching_system_prompt),
            HumanMessage(content=human_message_content)
        ]
        
        # Get response from the model
        log(f"Sending proxy matching request for {virtual_object_name}")
        # LangChain ChatOpenAI has built-in LangSmith tracing - no extra config needed
        response = await proxy_matching_llm.ainvoke(messages)
        log(f"Received method proposals for {virtual_object_name}")
        
        # Extract JSON from response
        response_text = extract_response_text(response.content)
        
        # Try to find JSON array
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_content = response_text[json_start:json_end]
        else:
            # Try to find JSON between code blocks
            json_start = response_text.find("```json")
            if json_start != -1:
                json_start += 7  # Length of ```json
                json_end = response_text.find("```", json_start)
                if json_end != -1:
                    json_content = response_text[json_start:json_end].strip()
                else:
                    json_content = response_text[json_start:].strip()
            else:
                # As a fallback, use the entire response
                json_content = response_text
        
        try:
            # Parse the JSON response
            matching_results = json.loads(json_content)
            if not isinstance(matching_results, list):
                matching_results = []
            
            # Convert any "imageId" keys to "image_id" right away
            matching_results = rename_key_in_json(matching_results, "imageId", "image_id")
            
            # Add the original virtual object info to each result
            for result in matching_results:
                if isinstance(result, dict):
                    result["virtualObjectInfo"] = virtual_object
                    
                    # Make sure we have object_id and image_id in the result
                    if "object_id" not in result:
                        log(f"Missing object_id in result for {result.get('physicalObject', 'unknown object')}")
                        # Try to find the object in the database
                        img_id = result.get("image_id")
                        phys_obj = result.get("physicalObject")
                        if img_id is not None and phys_obj:
                            img_id_str = str(img_id)
                            objects_in_img = physical_object_database.get(img_id_str, [])
                            for obj in objects_in_img:
                                if obj["object"] == phys_obj:
                                    result["object_id"] = obj["object_id"]
                                    log(f"Found object_id {obj['object_id']} for {phys_obj}")
                                    break
                    
                    # Ensure consistent property types
                    if "object_id" in result and isinstance(result["object_id"], str):
                        try:
                            result["object_id"] = int(result["object_id"])
                        except ValueError:
                            pass
                    
                    if "image_id" in result and isinstance(result["image_id"], str):
                        try:
                            result["image_id"] = int(result["image_id"])
                        except ValueError:
                            pass
                    
                    # Double check that image_id matches the database - fix any +1 offset
                    img_id = result.get("image_id")
                    if img_id is not None and isinstance(img_id, int) and img_id > 0:
                        obj_id = result.get("object_id")
                        phys_obj = result.get("physicalObject")
                        
                        # Check if this is incorrectly offset
                        correct_img_id = img_id - 1
                        img_id_str = str(correct_img_id)
                        
                        # Look in the database for a matching object at the offset-fixed image_id
                        found_match = False
                        if img_id_str in physical_object_database:
                            for obj in physical_object_database[img_id_str]:
                                if (obj_id is not None and obj.get("object_id") == obj_id) or obj.get("object") == phys_obj:
                                    # Set the correct image_id
                                    result["image_id"] = correct_img_id
                                    log(f"Fixed image_id offset: was {img_id}, now {correct_img_id}")
                                    found_match = True
                                    break
                        
                        # If no matching object was found with the offset, leave the image_id as is
                        if not found_match:
                            log(f"Could not find matching object for offset correction: {phys_obj} (ID: {obj_id}, Image: {img_id})")
                    
                    # Make sure the physical object properties are from the database
                    img_id = result.get("image_id")
                    obj_id = result.get("object_id")
                    
                    if img_id is not None and obj_id is not None:
                        img_id_str = str(img_id)
                        objects_in_img = physical_object_database.get(img_id_str, [])
                        for obj in objects_in_img:
                            if obj.get("object_id") == obj_id:
                                # Use the database values for consistency
                                result["physicalObject"] = obj["object"]
                                result["proxyLocation"] = obj["position"]
                                break
            
            return matching_results
            
        except json.JSONDecodeError as e:
            log(f"Error parsing proxy matching JSON for {virtual_object_name}: {e}")
            log(f"Raw content: {json_content}")
            
            # Return a basic result with the error
            return [{
                "virtualObject": virtual_object_name,
                "error": f"Failed to parse response: {str(e)}",
                "rawResponse": response_text[:500]  # First 500 chars
            }]
            
    except Exception as e:
        log(f"Error in proxy matching for {virtual_object.get('objectName', 'unknown')}: {e}")
        import traceback
        log(traceback.format_exc())
        
        # Return a basic result with the error
        return [{
            "virtualObject": virtual_object.get("objectName", "unknown"),
            "error": f"Processing error: {str(e)}"
        }]

# Function to run proxy matching for all virtual objects in parallel
@traceable(run_type="chain", metadata={"process": "proxy_matching_batch"})
async def run_proxy_matching(virtual_objects, environment_images, physical_object_database, object_snapshot_map):
    tasks = []
    for virtual_object in virtual_objects:
        task = match_single_virtual_object(
            virtual_object, 
            environment_images, 
            physical_object_database, 
            object_snapshot_map
        )
        tasks.append(task)
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results - flatten the array of arrays
    all_matching_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log(f"Error in proxy matching for object {i}: {result}")
            # Create fallback entry
            fallback_entry = {
                "virtualObject": virtual_objects[i].get("objectName", f"Object {i}"),
                "error": f"Task error: {str(result)}"
            }
            all_matching_results.append(fallback_entry)
        elif isinstance(result, list):
            # Each result is an array of matching results for a single virtual object
            all_matching_results.extend(result)
    
    # Log summary of results
    log(f"Completed proxy matching with {len(all_matching_results)} total matches across {len(virtual_objects)} virtual objects")
    
    return all_matching_results

# Function to rate a single property of a virtual object against all physical objects
@traceable(run_type="llm", metadata={"process": "property_rating"})
async def rate_single_property(virtual_object, property_name, environment_images, physical_object_database, object_snapshot_map, proxy_matching_results, run_index=1):
    # Log information about the proxy matching results
    log(f"Property rating for {property_name} (run {run_index}) with {len(proxy_matching_results)} proxy matching results")
    
    # If no proxy matching results, try loading directly from file
    if len(proxy_matching_results) == 0:
        try:
            output_dir = os.path.join(script_dir, "output")
            proxy_output_path = os.path.join(output_dir, "proxy_matching_results.json")
            if os.path.exists(proxy_output_path):
                log(f"Loading proxy matching results from {proxy_output_path}")
                with open(proxy_output_path, 'r') as f:
                    proxy_matching_results = json.load(f)
                log(f"Loaded {len(proxy_matching_results)} proxy matching results from file")
                if len(proxy_matching_results) > 0:
                    log(f"Sample: {proxy_matching_results[0].get('utilizationMethod', 'N/A')[:50]}...")
            else:
                log(f"Warning: Proxy matching results file not found at {proxy_output_path}")
        except Exception as e:
            log(f"Error loading proxy matching results: {e}")
    try:
        virtual_object_name = virtual_object.get("objectName", "Unknown Object")
        log(f"Rating {property_name} for virtual object: {virtual_object_name} (run {run_index})")
        
        # Get the property description
        property_description = virtual_object.get(property_name.replace("Value", ""), "")
        
        # Get a property-specific system prompt
        property_system_prompt = get_property_rating_system_prompt(property_name)
        
        # Build the human message content
        human_message_content = []
        
        # 1. Add the virtual object property information
        interaction_deduction = virtual_object.get("interactionDeduction", "No interaction deduction available")
        
        property_text = f"""# Property Rating Task

## Virtual Object: {virtual_object_name}
## Property to Evaluate: {property_name.replace("Value", "")}
## Property Description: {property_description}
## Interaction Deduction: {interaction_deduction}

Please rate how well each physical object can deliver the expected {property_name.replace("Value", "")} haptic feedback described above, considering the deduced interaction pattern when used according to the utilization method.
"""
        human_message_content.append({
            "type": "text", 
            "text": property_text
        })
        
        # 2. Add virtual object snapshot if available
        normalized_object_name = normalize_name(virtual_object_name)
        snapshot_found = False
        
        if virtual_object_name in object_snapshot_map:
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[virtual_object_name]}", 
                    "detail": "high"
                }
            })
            snapshot_found = True
        elif normalized_object_name in object_snapshot_map:
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[normalized_object_name]}", 
                    "detail": "high"
                }
            })
            snapshot_found = True
            
        # 3. Add introduction to physical environment
        human_message_content.append({
            "type": "text", 
            "text": "\n# Physical Environment\nBelow are snapshots of the physical environment with detected objects:"
        })
        
        # 4. Add environment snapshots with their detected objects and utilization methods
        for i, image_base64 in enumerate(environment_images):
            # Add the environment snapshot
            human_message_content.append({
                "type": "text", 
                "text": f"\n## Environment Snapshot {i+1}\n"
            })
            
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}", 
                    "detail": "high"
                }
            })
            
            # Add objects from proxy_matching_results for this image
            objects_text = "\n### Objects in this Snapshot\n"
            
            # Group objects by image_id
            image_objects = []
            log(f"Searching through {len(proxy_matching_results)} proxy results for image_id {i}")
            for proxy_result in proxy_matching_results:
                proxy_img_id = proxy_result.get('image_id')
                # Convert to int if it's a string
                if isinstance(proxy_img_id, str):
                    try:
                        proxy_img_id = int(proxy_img_id)
                    except ValueError:
                        pass
                
                # If this object belongs to the current image
                if proxy_img_id == i:
                    log(f"Found object for image_id {i}: {proxy_result.get('physicalObject', 'Unknown')}")
                    has_method = 'utilizationMethod' in proxy_result and proxy_result['utilizationMethod']
                    log(f"Has utilization method: {has_method}")
                    image_objects.append(proxy_result)
            
            # Sort by object_id for consistency
            image_objects.sort(key=lambda x: x.get('object_id', 0))
            
            # Remove duplicates (same object_id)
            unique_objects = []
            seen_object_ids = set()
            for obj in image_objects:
                obj_id = obj.get('object_id')
                if obj_id not in seen_object_ids:
                    seen_object_ids.add(obj_id)
                    unique_objects.append(obj)
            
            if unique_objects:
                for obj in unique_objects:
                    obj_id = obj.get('object_id', 'Unknown')
                    obj_name = obj.get('physicalObject', 'Unknown object')
                    obj_location = obj.get('proxyLocation', 'Unknown position')
                    
                    objects_text += f"- Object ID: {obj_id} - {obj_name} ({obj_location})\n"
                    
                    # Only show utilization methods that match the current virtual object
                    utilization_added = False
                    for proxy_result in proxy_matching_results:
                        # Check if this proxy result matches the current object ID and image ID
                        if (proxy_result.get('object_id') == obj_id and 
                            proxy_result.get('image_id') == i and
                            proxy_result.get('virtualObject') == virtual_object_name):
                            
                            util_method = proxy_result.get("utilizationMethod", "")
                            if util_method:
                                log(f"Found matching utilization method for {virtual_object_name}")
                                objects_text += f"  Utilization Method: {util_method}\n"
                                utilization_added = True
                                break
                    
                    if not utilization_added:
                        objects_text += f"  No utilization method for {virtual_object_name}\n"
                    
                    objects_text += f"  Image ID: {i}\n\n"
            else:
                objects_text += "- No objects found in proxy matching results for this snapshot\n"
            
            human_message_content.append({
                "type": "text", 
                "text": objects_text
            })
        
        # 5. Add final instructions
        human_message_content.append({
            "type": "text", 
            "text": f"""
# Your Task

For each physical object, evaluate the statement: "I felt the haptic feedback closely mimicked the {property_name.replace("Value", "")}" on a 7-point Likert scale:
1 - Strongly Disagree 
2 - Disagree
3 - Somewhat Disagree
4 - Neutral
5 - Somewhat Agree
6 - Agree
7 - Strongly Agree

FORMAT YOUR RESPONSE AS A JSON ARRAY with objects having the following structure:

```json
[
  {{
    "virtualObject": "{virtual_object_name}",
    "property": "{property_name.replace("Value", "")}",
    "physicalObject": "name of the physical object",
    "object_id": 1,
    "image_id": 0,
    "rating": 5,
    "explanation": "Brief explanation of why this rating was given"
  }},
  ...
]
```

IMPORTANT: Include ALL physical objects in your evaluation, even those with low ratings.
"""
        })
        
        # Create the messages
        messages = [
            SystemMessage(content=property_system_prompt),
            HumanMessage(content=human_message_content)
        ]
        
        # Get response from the model
        log(f"Sending property rating request for {property_name} of {virtual_object_name} (run {run_index})")
        # LangChain ChatOpenAI has built-in LangSmith tracing - no extra config needed
        response = await property_rating_llm.ainvoke(messages)
        log(f"Received property ratings for {property_name} of {virtual_object_name} (run {run_index})")
        
        # Extract JSON from response
        response_text = extract_response_text(response.content)
        
        # Try to find JSON array
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_content = response_text[json_start:json_end]
        else:
            # Try to find JSON between code blocks
            json_start = response_text.find("```json")
            if json_start != -1:
                json_start += 7  # Length of ```json
                json_end = response_text.find("```", json_start)
                if json_end != -1:
                    json_content = response_text[json_start:json_end].strip()
                else:
                    json_content = response_text[json_start:].strip()
            else:
                # As a fallback, use the entire response
                json_content = response_text
        
        try:
            # Parse the JSON response
            rating_results = json.loads(json_content)
            
            # Add the property value to each result and rename the rating field based on run_index
            rating_key = f"rating_{run_index}"
            for result in rating_results:
                # Get the property value from the virtual object
                property_value = virtual_object.get(property_name, 0.0)
                result["propertyValue"] = property_value
                
                # Rename the rating field
                if "rating" in result:
                    result[rating_key] = result["rating"]
                    del result["rating"]
                
                # Remove any extra fields not in the required output format
                keys_to_keep = ["virtualObject", "property", "physicalObject", "object_id", "image_id", rating_key, "explanation", "propertyValue"]
                for key in list(result.keys()):
                    if key not in keys_to_keep:
                        del result[key]
                
            return rating_results
            
        except json.JSONDecodeError as e:
            log(f"Error parsing property rating JSON for {property_name} of {virtual_object_name} (run {run_index}): {e}")
            log(f"Raw content: {json_content}")
            
            # Return a basic result with the error
            return [{
                "virtualObject": virtual_object_name,
                "property": property_name.replace("Value", ""),
                "error": f"Failed to parse response: {str(e)}",
                "rawResponse": response_text[:500]  # First 500 chars
            }]
            
    except Exception as e:
        log(f"Error in property rating for {property_name} of {virtual_object.get('objectName', 'unknown')} (run {run_index}): {e}")
        import traceback
        log(traceback.format_exc())
        
        # Return a basic result with the error
        return [{
            "virtualObject": virtual_object.get("objectName", "unknown"),
            "property": property_name.replace("Value", ""),
            "error": f"Processing error: {str(e)}"
        }]

# Function to run property ratings for all virtual objects with overlapping batch approach
@traceable(run_type="chain", metadata={"process": "property_rating_batch"})
async def run_property_ratings(virtual_objects, environment_images, physical_object_database, object_snapshot_map, proxy_matching_results):
    log(f"run_property_ratings received {len(proxy_matching_results)} proxy matching results")
    
    # Check sample proxy result for utilization method
    if len(proxy_matching_results) > 0:
        sample = proxy_matching_results[0]
        log(f"Sample proxy result keys: {list(sample.keys())}")
        if 'utilizationMethod' in sample:
            log(f"Sample utilization method: {sample['utilizationMethod'][:50]}...")
    
    property_names = ["inertiaValue", "interactivityValue", "outlineValue", "textureValue", "hardnessValue", "temperatureValue"]
    
    # Create all property rating tasks (same as before)
    all_tasks = []
    
    for virtual_object in virtual_objects:
        virtual_object_name = virtual_object.get("objectName", "Unknown Object")
        
        # For each property with value > 0, create 3 tasks (one for each run)
        for property_name in property_names:
            property_value = virtual_object.get(property_name, 0.0)
            
            # Only rate properties that are highlighted (value > 0)
            if property_value > 0:
                log(f"Creating 3 property rating tasks for {property_name} of {virtual_object_name} (value: {property_value})")
                
                # Create 3 tasks per property for statistical reliability
                for run_index in range(1, 4):  # Run 1, 2, 3
                    task = rate_single_property(
                        virtual_object,
                        property_name,
                        environment_images,
                        physical_object_database,
                        object_snapshot_map,
                        proxy_matching_results,
                        run_index
                    )
                    all_tasks.append(task)
    
    # Run tasks with overlapping batches for maximum throughput (same approach as substrate utilization)
    log(f"Running {len(all_tasks)} property rating tasks with overlapping batches (size: {PROPERTY_RATING_BATCH_SIZE}, interval: {PROPERTY_RATING_BATCH_INTERVAL}s)")
    
    # Create all batch tasks without waiting for them to complete
    batch_tasks = []
    for i in range(0, len(all_tasks), PROPERTY_RATING_BATCH_SIZE):
        batch = all_tasks[i:i + PROPERTY_RATING_BATCH_SIZE]
        batch_num = i // PROPERTY_RATING_BATCH_SIZE + 1
        total_batches = (len(all_tasks) + PROPERTY_RATING_BATCH_SIZE - 1) // PROPERTY_RATING_BATCH_SIZE
        
        log(f"Starting property batch {batch_num}/{total_batches}: {len(batch)} tasks")
        
        # Create a task for this batch
        batch_task = asyncio.create_task(
            _run_single_property_batch(batch, batch_num)
        )
        batch_tasks.append(batch_task)
        
        # Wait before starting the next batch (except for the last one)
        if i + PROPERTY_RATING_BATCH_SIZE < len(all_tasks):
            await asyncio.sleep(PROPERTY_RATING_BATCH_INTERVAL)
    
    # Now wait for all batches to complete
    log(f"All {len(batch_tasks)} property batches started. Waiting for completion...")
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    # Process all batch results
    all_rating_results = []
    successful_batches = 0
    
    for batch_num, batch_result in enumerate(batch_results, 1):
        if isinstance(batch_result, Exception):
            log(f"Error in property batch {batch_num}: {batch_result}")
            continue
        elif isinstance(batch_result, list):
            all_rating_results.extend(batch_result)
            successful_batches += 1
        else:
            log(f"Property batch {batch_num} returned unexpected result type: {type(batch_result)}")
    
    log(f"Overlapping property batch execution completed: {successful_batches}/{len(batch_tasks)} batches successful")
    
    # Now combine multiple runs for the same property-object pairs (same logic as before)
    combined_results = {}
    
    for rating in all_rating_results:
        if "error" in rating:
            continue
            
        # Create a key for this property-object pair
        virt_obj = rating.get("virtualObject", "unknown")
        property_name = rating.get("property", "unknown")
        obj_id = rating.get("object_id", -1)
        img_id = rating.get("image_id", -1)
                    
        pair_key = f"{virt_obj}:{property_name}:{obj_id}:{img_id}"
        
        # Initialize the combined entry if it doesn't exist
        if pair_key not in combined_results:
            combined_results[pair_key] = {
                "virtualObject": virt_obj,
                "property": property_name,
                "physicalObject": rating.get("physicalObject", "unknown"),
                "object_id": obj_id,
                "image_id": img_id,
                "propertyValue": rating.get("propertyValue", 0.0),
                "explanation": rating.get("explanation", "")
            }
        
        # Add the rating for this run
        if "rating_1" in rating:
            combined_results[pair_key]["rating_1"] = rating["rating_1"]
        elif "rating_2" in rating:
            combined_results[pair_key]["rating_2"] = rating["rating_2"]  
        elif "rating_3" in rating:
            combined_results[pair_key]["rating_3"] = rating["rating_3"]
        else:
            # Handle the case where the rating field doesn't have a run suffix
            # Determine which run this is based on existing data
            if "rating_1" not in combined_results[pair_key]:
                combined_results[pair_key]["rating_1"] = rating.get("rating", 0)
            elif "rating_2" not in combined_results[pair_key]:
                combined_results[pair_key]["rating_2"] = rating.get("rating", 0)
            elif "rating_3" not in combined_results[pair_key]:
                combined_results[pair_key]["rating_3"] = rating.get("rating", 0)
    
    # Convert back to list
    final_results = list(combined_results.values())
    
    # Log summary of results
    log(f"Completed property ratings with {len(all_rating_results)} individual ratings combined into {len(final_results)} final results")
    
    return final_results

# Function to rate a single dimension for relationship rating
async def rate_single_relationship_dimension(relationship_annotation, contact_object, substrate_objects, environment_images, physical_object_database, object_snapshot_map, enhanced_virtual_objects, proxy_matching_results, substrate_utilization_results, dimension_name, group_index=1):
    try:
        virtual_contact_name = relationship_annotation.get("contactObject", "Unknown Contact Object")
        virtual_substrate_name = relationship_annotation.get("substrateObject", "Unknown Substrate Object")
        annotation_text = relationship_annotation.get("annotationText", "No annotation available")
        
        log(f"Rating {dimension_name} for relationship group {group_index}: {virtual_contact_name} -> {virtual_substrate_name}")
        
        # Get pre-generated substrate utilization methods for this contact object
        relevant_substrate_methods = []
        for substrate_result in substrate_utilization_results:
            if (substrate_result.get('contactObject_id') == contact_object.get('object_id') and 
                substrate_result.get('contactImage_id') == contact_object.get('image_id') and
                substrate_result.get('virtualContactObject') == virtual_contact_name and
                substrate_result.get('virtualSubstrateObject') == virtual_substrate_name):
                relevant_substrate_methods.append(substrate_result)
        
        log(f"Found {len(relevant_substrate_methods)} pre-generated substrate utilization methods for this contact object")
        
        # Get the contact object's utilization method from proxy matching results
        contact_utilization_method = "No utilization method available"
        for proxy_result in proxy_matching_results:
            if (proxy_result.get('object_id') == contact_object.get('object_id') and 
                proxy_result.get('image_id') == contact_object.get('image_id') and
                proxy_result.get('virtualObject') == virtual_contact_name):
                contact_utilization_method = proxy_result.get("utilizationMethod", "No utilization method available")
                break
        
        # Check if contact object has a valid utilization method
        has_valid_contact_method = contact_utilization_method != 'No utilization method available'
        
        # Get dimension-specific system prompt
        dimension_system_prompt = get_relationship_rating_system_prompt(dimension_name)
        
        # Build the human message content
        human_message_content = []
        
        # 1. Add the relationship information
        # Get virtual contact object details
        virtual_contact_obj = None
        virtual_substrate_obj = None
        for vobj in enhanced_virtual_objects:
            if vobj.get("objectName") == virtual_contact_name:
                virtual_contact_obj = vobj
            elif vobj.get("objectName") == virtual_substrate_name:
                virtual_substrate_obj = vobj
        
        contact_interaction_deduction = virtual_contact_obj.get("interactionDeduction", "No interaction deduction available") if virtual_contact_obj else "No interaction deduction available"
        contact_dimensions = virtual_contact_obj.get("dimensions_meters", {}) if virtual_contact_obj else {}
        substrate_dimensions = virtual_substrate_obj.get("dimensions_meters", {}) if virtual_substrate_obj else {}
        
        # Format dimensions for display
        def format_dimensions(dims):
            if not dims:
                return "No dimensions available"
            return f"Width: {dims.get('x', 'N/A')}m, Height: {dims.get('y', 'N/A')}m, Depth: {dims.get('z', 'N/A')}m"
        
        relationship_text = f"""# Relationship {dimension_name.title()} Rating Task (Group {group_index})

## Virtual Object Relationship
- **Contact Object**: {virtual_contact_name}
- **Substrate Object**: {virtual_substrate_name}
- **Expected Haptic Feedback**: {annotation_text}

## Virtual Contact Object Details
- **Interaction Deduction**: {contact_interaction_deduction}
- **Dimensions**: {format_dimensions(contact_dimensions)}

## Virtual Substrate Object Details
- **Dimensions**: {format_dimensions(substrate_dimensions)}

## Physical Object Assignment
- **Contact Object**: {contact_object.get('object', 'Unknown')} (ID: {contact_object.get('object_id')}, Image: {contact_object.get('image_id')})
- **Contact Utilization Method**: {contact_utilization_method}
- **Contact Method Available**: {'Yes - this object can serve as a proxy for the virtual contact object' if has_valid_contact_method else 'No - this object cannot serve as a proxy for the virtual contact object, but evaluate it anyway for comprehensive coverage'}
- **Substrate Objects**: All other physical objects listed below

Please rate how well each pair (contact + substrate) can deliver the expected {dimension_name} aspect of the haptic feedback described above.
{"Note: Since this contact object lacks a proper utilization method for the virtual contact object, the ratings may be lower, but this ensures comprehensive coverage of all possible pairs." if not has_valid_contact_method else ""}
"""
        human_message_content.append({
            "type": "text", 
            "text": relationship_text
        })
        
        # 2. Add virtual object snapshots if available
        # Add virtual contact object snapshot
        contact_snapshot_found = False
        normalized_contact_name = normalize_name(virtual_contact_name)
        
        # Try direct match first
        if virtual_contact_name in object_snapshot_map:
            log(f"Found snapshot for contact object: {virtual_contact_name} (direct match)")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Contact Object: {virtual_contact_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[virtual_contact_name]}", 
                    "detail": "high"
                }
            })
            contact_snapshot_found = True
        # Then try normalized match
        elif normalized_contact_name in object_snapshot_map:
            log(f"Found snapshot for contact object: {virtual_contact_name} (normalized as {normalized_contact_name})")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Contact Object: {virtual_contact_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[normalized_contact_name]}", 
                    "detail": "high"
                }
            })
            contact_snapshot_found = True
        # Finally try fuzzy match
        else:
            # Try to find partial matches
            potential_matches = [name for name in object_snapshot_map.keys() 
                               if normalized_contact_name in normalize_name(name) or normalize_name(name) in normalized_contact_name]
            
            if potential_matches:
                best_match = potential_matches[0]  # Take the first match
                log(f"Found snapshot for contact object: {virtual_contact_name} (fuzzy match: {best_match})")
                human_message_content.append({
                    "type": "text",
                    "text": f"\n## Virtual Contact Object: {virtual_contact_name}\n"
                })
                human_message_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{object_snapshot_map[best_match]}", 
                        "detail": "high"
                    }
                })
                contact_snapshot_found = True
        
        if not contact_snapshot_found:
            log(f"No snapshot found for contact object: {virtual_contact_name} (normalized: {normalized_contact_name})")
        
        # Add virtual substrate object snapshot
        substrate_snapshot_found = False
        normalized_substrate_name = normalize_name(virtual_substrate_name)
        
        # Try direct match first
        if virtual_substrate_name in object_snapshot_map:
            log(f"Found snapshot for substrate object: {virtual_substrate_name} (direct match)")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Substrate Object: {virtual_substrate_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[virtual_substrate_name]}", 
                    "detail": "high"
                }
            })
            substrate_snapshot_found = True
        # Then try normalized match
        elif normalized_substrate_name in object_snapshot_map:
            log(f"Found snapshot for substrate object: {virtual_substrate_name} (normalized as {normalized_substrate_name})")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Substrate Object: {virtual_substrate_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[normalized_substrate_name]}", 
                    "detail": "high"
                }
            })
            substrate_snapshot_found = True
        # Finally try fuzzy match
        else:
            # Try to find partial matches
            potential_matches = [name for name in object_snapshot_map.keys() 
                               if normalized_substrate_name in normalize_name(name) or normalize_name(name) in normalized_substrate_name]
            
            if potential_matches:
                best_match = potential_matches[0]  # Take the first match
                log(f"Found snapshot for substrate object: {virtual_substrate_name} (fuzzy match: {best_match})")
                human_message_content.append({
                    "type": "text",
                    "text": f"\n## Virtual Substrate Object: {virtual_substrate_name}\n"
                })
                human_message_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{object_snapshot_map[best_match]}", 
                        "detail": "high"
                    }
                })
                substrate_snapshot_found = True
        
        if not substrate_snapshot_found:
            log(f"No snapshot found for substrate object: {virtual_substrate_name} (normalized: {normalized_substrate_name})")
        
        # 3. Add environment snapshots with their detected objects
        for i, image_base64 in enumerate(environment_images):
            # Add the environment snapshot
            human_message_content.append({
                "type": "text", 
                "text": f"\n## Environment Snapshot {i+1}\n"
            })
            
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}", 
                    "detail": "high"
                }
            })
            
            # Add the detected objects for this snapshot
            objects_in_snapshot = physical_object_database.get(str(i), [])
            objects_text = "\n### Physical Objects in this Snapshot\n"
            
            if objects_in_snapshot:
                for obj in objects_in_snapshot:
                    obj_id = obj['object_id']
                    obj_name = obj['object']
                    obj_position = obj['position']
                    
                    # Highlight the contact object
                    if (obj_id == contact_object.get('object_id') and 
                        obj.get('image_id') == contact_object.get('image_id')):
                        objects_text += f"- **CONTACT OBJECT** - Object ID {obj_id}: {obj_name} ({obj_position})\n"
                        objects_text += f"  Contact Utilization Method: {contact_utilization_method}\n"
                    else:
                        objects_text += f"- Object ID {obj_id}: {obj_name} ({obj_position}) - *Substrate candidate*\n"
                        
                        # Find and attach the substrate utilization method for this object
                        substrate_method_found = False
                        for method in relevant_substrate_methods:
                            method_obj_id = method.get('substrateObject_id')
                            method_img_id = method.get('substrateImage_id')
                            
                            # Convert to int for comparison if needed
                            if isinstance(method_obj_id, str):
                                try:
                                    method_obj_id = int(method_obj_id)
                                except ValueError:
                                    pass
                            if isinstance(method_img_id, str):
                                try:
                                    method_img_id = int(method_img_id)
                                except ValueError:
                                    pass
                            
                            # Check if this method matches the current object
                            if method_obj_id == obj_id and method_img_id == i:
                                substrate_method = method.get('substrateUtilizationMethod', 'No method available')
                                objects_text += f"  **Substrate Utilization Method**: {substrate_method}\n"
                                substrate_method_found = True
                                break
                        
                        if not substrate_method_found:
                            objects_text += f"  **Substrate Utilization Method**: No method available for this object\n"
                    
                    objects_text += f"  Image ID: {i}\n\n"
            else:
                # Check if we should look for objects in a different format
                objects_in_snapshot = physical_object_database.get(i, [])
                if objects_in_snapshot:
                    for obj in objects_in_snapshot:
                        obj_id = obj['object_id']
                        obj_name = obj['object']
                        obj_position = obj['position']
                        
                        if (obj_id == contact_object.get('object_id') and 
                            obj.get('image_id') == contact_object.get('image_id')):
                            objects_text += f"- **CONTACT OBJECT** - Object ID {obj_id}: {obj_name} ({obj_position})\n"
                            objects_text += f"  Contact Utilization Method: {contact_utilization_method}\n"
                        else:
                            objects_text += f"- Object ID {obj_id}: {obj_name} ({obj_position}) - *Substrate candidate*\n"
                            
                            # Find and attach the substrate utilization method for this object
                            substrate_method_found = False
                            for method in relevant_substrate_methods:
                                method_obj_id = method.get('substrateObject_id')
                                method_img_id = method.get('substrateImage_id')
                                
                                # Convert to int for comparison if needed
                                if isinstance(method_obj_id, str):
                                    try:
                                        method_obj_id = int(method_obj_id)
                                    except ValueError:
                                        pass
                                if isinstance(method_img_id, str):
                                    try:
                                        method_img_id = int(method_img_id)
                                    except ValueError:
                                        pass
                                
                                # Check if this method matches the current object
                                if method_obj_id == obj_id and method_img_id == i:
                                    substrate_method = method.get('substrateUtilizationMethod', 'No method available')
                                    objects_text += f"  **Substrate Utilization Method**: {substrate_method}\n"
                                    substrate_method_found = True
                                    break
                            
                            if not substrate_method_found:
                                objects_text += f"  **Substrate Utilization Method**: No method available for this object\n"
                        
                        objects_text += f"  Image ID: {i}\n\n"
                else:
                    objects_text += "- No objects detected in this snapshot\n"
                
            human_message_content.append({
                "type": "text", 
                "text": objects_text
            })
        
        # 4. Substrate utilization methods are now integrated with each object above
        
        # 5. Add final instructions with explicit object ID mapping
        
        # Create a clear object ID mapping table
        object_mapping_text = f"""
# CRITICAL: Object ID Mapping Table

**Contact Object (MUST use these exact IDs):**
- Object ID: {contact_object.get('object_id')}
- Image ID: {contact_object.get('image_id')}  
- Name: {contact_object.get('object')}

**Substrate Objects (MUST use these exact IDs for each substrate):**
"""
        
        # Add all substrate objects with their exact IDs from the database
        for i, image_base64 in enumerate(environment_images):
            img_key = str(i)
            if img_key in physical_object_database:
                objects_in_img = physical_object_database[img_key]
                for obj in objects_in_img:
                    # Skip the contact object itself
                    if not (obj['object_id'] == contact_object.get('object_id') and 
                           obj.get('image_id') == contact_object.get('image_id')):
                        object_mapping_text += f"- Object ID: {obj['object_id']}, Image ID: {obj.get('image_id', i)}, Name: {obj['object']}\n"
        
        human_message_content.append({
            "type": "text", 
            "text": object_mapping_text
        })
        
        human_message_content.append({
            "type": "text", 
            "text": f"""
# Your Task

Each substrate candidate above has been provided with its corresponding substrate utilization method. Your task is to evaluate how well each contact-substrate pair can deliver the expected {dimension_name} aspect of the haptic feedback.

**Contact Object**: {contact_object.get('object')} (ID: {contact_object.get('object_id')}, Image: {contact_object.get('image_id')})
**Contact Utilization Method**: {contact_utilization_method}

For each substrate object and its corresponding substrate utilization method listed above, rate the pair on the {dimension_name} dimension using the 7-point Likert scale provided in the system prompt.

**CRITICAL REQUIREMENTS:**
1. Use ONLY the Object IDs and Image IDs from the mapping table above
2. Do NOT make up or guess any object IDs  
3. The contactObject_id MUST be {contact_object.get('object_id')} and contactImage_id MUST be {contact_object.get('image_id')}
4. For each substrate, use the EXACT object_id and image_id from the mapping table
5. Consider both the contact object's utilization method AND the substrate utilization method listed with each object
6. Rate based on how well the combined utilization methods would deliver the {dimension_name} aspect of the expected haptic feedback: "{annotation_text}"

FORMAT YOUR RESPONSE AS A JSON ARRAY as specified in the system prompt, using the EXACT object IDs from the mapping table above.
"""
        })
        
        # Create the messages
        messages = [
            SystemMessage(content=dimension_system_prompt),
            HumanMessage(content=human_message_content)
        ]
        
        # Get response from the model with retry logic
        log(f"Sending {dimension_name} rating request for group {group_index}")
        
        try:
            # Make direct LLM call to maintain LangSmith tracing context
            response = await relationship_rating_llm.ainvoke(messages)
            log(f"Successfully received {dimension_name} ratings for group {group_index}")
        except Exception as e:
            log(f"Error during {dimension_name} rating LLM call for group {group_index}: {e}")
            raise
        
        # Extract JSON from response
        response_text = extract_response_text(response.content)
        
        # Try to find JSON array
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_content = response_text[json_start:json_end]
        else:
            # Try to find JSON between code blocks
            json_start = response_text.find("```json")
            if json_start != -1:
                json_start += 7  # Length of ```json
                json_end = response_text.find("```", json_start)
                if json_end != -1:
                    json_content = response_text[json_start:json_end].strip()
                else:
                    json_content = response_text[json_start:].strip()
            else:
                # As a fallback, use the entire response
                json_content = response_text
        
        try:
            # Parse the JSON response
            rating_results = json.loads(json_content)
            
            # VALIDATION DISABLED: Using raw results to debug validation issues
            # validated_results = validate_and_normalize_relationship_response(
            #     rating_results, dimension_name, group_index, physical_object_database
            # )
            validated_results = rating_results  # Use raw results without validation
            
            log(f"Using raw {dimension_name} results for group {group_index}: {len(validated_results)} ratings")
            
            # Add group information and dimension to each result
            for result in validated_results:
                result["group_index"] = group_index
                result["dimension"] = dimension_name
                result["virtualContactObject"] = virtual_contact_name
                result["virtualSubstrateObject"] = virtual_substrate_name
                result["expectedHapticFeedback"] = annotation_text
                
            return validated_results
            
        except json.JSONDecodeError as e:
            log(f"Error parsing {dimension_name} rating JSON for group {group_index}: {e}")
            log(f"Raw content: {json_content}")
            
            # Return a basic result with the error
            return [{
                "group_index": group_index,
                "dimension": dimension_name,
                "virtualContactObject": virtual_contact_name,
                "virtualSubstrateObject": virtual_substrate_name,
                "error": f"Failed to parse response: {str(e)}",
                "rawResponse": response_text[:500]  # First 500 chars
            }]
            
    except Exception as e:
        log(f"Error in {dimension_name} rating for group {group_index}: {e}")
        import traceback
        log(traceback.format_exc())
        
        # Return a basic result with the error
        return [{
            "group_index": group_index,
            "error": f"Processing error: {str(e)}"
        }]

# Function to rate a single relationship group with all three dimensions
async def rate_single_relationship_group_simple(relationship_annotation, contact_object, substrate_objects, environment_images, physical_object_database, object_snapshot_map, enhanced_virtual_objects, proxy_matching_results, substrate_utilization_results, group_index=1):
    try:
        virtual_contact_name = relationship_annotation.get("contactObject", "Unknown Contact Object")
        virtual_substrate_name = relationship_annotation.get("substrateObject", "Unknown Substrate Object")
        
        log(f"Rating relationship group {group_index}: {virtual_contact_name} -> {virtual_substrate_name} with all three dimensions")
        
        # Evaluate all three dimensions: harmony, expressivity, realism
        dimensions = ["harmony", "expressivity", "realism"]
        
        log(f"Creating ratings for {len(dimensions)} dimensions for group {group_index}")
        
        # Call the dimension rating function for each dimension sequentially
        all_dimension_results = []
        
        for dimension in dimensions:
            log(f"Processing {dimension} rating for group {group_index}")
            
            dimension_result = await rate_single_relationship_dimension(
                relationship_annotation,
                contact_object,
                substrate_objects,
                environment_images,
                physical_object_database,
                object_snapshot_map,
                enhanced_virtual_objects,
                proxy_matching_results,
                substrate_utilization_results,
                dimension,
                group_index
            )
            
            # Process the dimension result
            if isinstance(dimension_result, Exception):
                log(f"Failed {dimension} rating for group {group_index}: {dimension_result}")
                # Continue with other dimensions even if one fails
                continue
            elif isinstance(dimension_result, list):
                log(f"Completed {dimension} rating for group {group_index} with {len(dimension_result)} results")
                all_dimension_results.extend(dimension_result)
            else:
                log(f"Invalid result type for {dimension} in group {group_index}: {type(dimension_result)}")
                continue
        
        # Combine results from all dimensions
        if all_dimension_results:
            # Group results by contact-substrate pair to combine dimension ratings
            combined_results = {}
            
            for result in all_dimension_results:
                # Create a key for this contact-substrate pair
                contact_id = result.get("contactObject_id", "unknown")
                contact_img_id = result.get("contactImage_id", "unknown")
                substrate_id = result.get("substrateObject_id", "unknown")
                substrate_img_id = result.get("substrateImage_id", "unknown")
                
                pair_key = f"{contact_id}_{contact_img_id}_{substrate_id}_{substrate_img_id}"
                
                # Initialize the combined entry if it doesn't exist
                if pair_key not in combined_results:
                    combined_results[pair_key] = {
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
                        "group_index": result.get("group_index", group_index),
                        "expectedHapticFeedback": result.get("expectedHapticFeedback", "")
                    }
                
                # Add the dimension-specific rating and explanation
                dimension = result.get("dimension", "unknown")
                rating_key = f"{dimension}_rating"
                explanation_key = f"{dimension}_explanation"
                
                if rating_key in result:
                    combined_results[pair_key][rating_key] = result[rating_key]
                if explanation_key in result:
                    combined_results[pair_key][explanation_key] = result[explanation_key]
            
            # Convert back to list
            final_results = list(combined_results.values())
            log(f"Combined all dimensions for group {group_index}: {len(final_results)} final results")
            return final_results
        else:
            log(f"No valid dimension results for group {group_index}")
            return [{
                "group_index": group_index,
                "error": "No valid dimension results"
            }]
        
    except Exception as e:
        log(f"Error in relationship rating for group {group_index}: {e}")
        import traceback
        log(traceback.format_exc())
        
        # Return a basic result with the error
        return [{
            "group_index": group_index,
            "error": f"Processing error: {str(e)}"
        }]

# Function to run relationship ratings for all relationships in parallel
@traceable(run_type="chain", metadata={"process": "relationship_rating_batch"})
async def run_relationship_ratings(haptic_annotation_json, environment_images, physical_object_database, object_snapshot_map, enhanced_virtual_objects, proxy_matching_results, substrate_utilization_results, property_rating_results):
    if not haptic_annotation_json:
        log("No haptic annotation data provided for relationship ratings")
        return []
    
    try:
        # Parse the haptic annotation JSON
        haptic_data = json.loads(haptic_annotation_json)
        relationship_annotations = haptic_data.get("relationshipAnnotations", [])
        
        if not relationship_annotations:
            log("No relationship annotations found in haptic data")
            return []
        
        log(f"Found {len(relationship_annotations)} relationship annotations")
        
        # Get all physical objects from the database
        all_physical_objects = []
        for image_id, objects in physical_object_database.items():
            for obj in objects:
                all_physical_objects.append(obj)
        
        log(f"Found {len(all_physical_objects)} total physical objects for relationship rating")
        
        # Function to get top k physical objects for a virtual contact object based on property rating results
        def get_top_k_contact_objects(virtual_contact_name, k=5):
            # Get property rating results for this virtual contact object
            contact_ratings = []
            for rating in property_rating_results:
                if rating.get("virtualObject") == virtual_contact_name:
                    # Calculate mean rating from the three runs
                    rating_1 = rating.get("rating_1", 0)
                    rating_2 = rating.get("rating_2", 0) 
                    rating_3 = rating.get("rating_3", 0)
                    mean_rating = (rating_1 + rating_2 + rating_3) / 3 if (rating_1 or rating_2 or rating_3) else 0
                    
                    # Get property value for weighting
                    property_value = rating.get("propertyValue", 0.0)
                    
                    # Calculate weighted score
                    weighted_score = mean_rating * property_value
                    
                    contact_ratings.append({
                        "object_id": rating.get("object_id"),
                        "image_id": rating.get("image_id"),
                        "physicalObject": rating.get("physicalObject"),
                        "weighted_score": weighted_score,
                        "mean_rating": mean_rating,
                        "property_value": property_value
                    })
            
            # Group by physical object and sum weighted scores across all properties
            object_scores = {}
            for rating in contact_ratings:
                obj_key = f"{rating['object_id']}_{rating['image_id']}"
                if obj_key not in object_scores:
                    object_scores[obj_key] = {
                        "object_id": rating["object_id"],
                        "image_id": rating["image_id"],
                        "physicalObject": rating["physicalObject"],
                        "total_weighted_score": 0
                    }
                object_scores[obj_key]["total_weighted_score"] += rating["weighted_score"]
            
            # Sort by total weighted score and return top k
            sorted_objects = sorted(object_scores.values(), key=lambda x: x["total_weighted_score"], reverse=True)
            top_k_objects = sorted_objects[:k]
            
            log(f"Top {k} physical objects for virtual contact '{virtual_contact_name}':")
            for i, obj in enumerate(top_k_objects, 1):
                log(f"  {i}. {obj['physicalObject']} (ID: {obj['object_id']}, Image: {obj['image_id']}, Score: {obj['total_weighted_score']:.3f})")
            
            return top_k_objects
        
        # Create rating tasks - Only consider top k physical objects as contact objects
        all_tasks = []
        group_counter = 1
        all_rated_pairs = set()  # Track which pairs will be rated
        
        for relationship in relationship_annotations:
            virtual_contact_name = relationship.get("contactObject", "")
            virtual_substrate_name = relationship.get("substrateObject", "")
            
            log(f"Processing relationship: {virtual_contact_name} -> {virtual_substrate_name}")
            
            # Get top k physical objects for this virtual contact object
            top_k_contact_objects = get_top_k_contact_objects(virtual_contact_name, k=5)
            
            # Create one group for each top k physical object as the contact object
            for top_obj in top_k_contact_objects:
                # Find the actual physical object from the database
                contact_obj = None
                for obj in all_physical_objects:
                    if (obj.get('object_id') == top_obj['object_id'] and 
                        obj.get('image_id') == top_obj['image_id']):
                        contact_obj = obj
                        break
                
                if contact_obj is None:
                    log(f"Warning: Could not find physical object for top-k result: {top_obj}")
                    continue
                # Get the contact object's utilization method from proxy matching results (if available)
                contact_utilization_method = "No utilization method available"
                for proxy_result in proxy_matching_results:
                    if (proxy_result.get('object_id') == contact_obj.get('object_id') and 
                        proxy_result.get('image_id') == contact_obj.get('image_id') and
                        proxy_result.get('virtualObject') == virtual_contact_name):
                        contact_utilization_method = proxy_result.get('utilizationMethod', 'No utilization method available')
                        break
                
                # Add utilization method to the contact object
                contact_obj_with_method = contact_obj.copy()
                contact_obj_with_method['utilizationMethod'] = contact_utilization_method
                
                # Get all other objects as substrate candidates
                substrate_objects = [obj for obj in all_physical_objects 
                                   if not (obj['object_id'] == contact_obj['object_id'] and 
                                          obj.get('image_id') == contact_obj.get('image_id'))]
                
                if len(substrate_objects) > 0:
                    log(f"Creating group {group_counter} for contact object: {contact_obj.get('object')} (ID: {contact_obj.get('object_id')}, Image: {contact_obj.get('image_id')}) with {len(substrate_objects)} substrate candidates")
                    log(f"  Contact utilization method available: {'Yes' if contact_utilization_method != 'No utilization method available' else 'No'}")
                    
                    # Track which pairs will be rated
                    for substrate_obj in substrate_objects:
                        pair_key = f"{virtual_contact_name}_{virtual_substrate_name}_{contact_obj.get('object_id')}_{contact_obj.get('image_id')}_{substrate_obj.get('object_id')}_{substrate_obj.get('image_id')}"
                        all_rated_pairs.add(pair_key)
                    
                    task = rate_single_relationship_group_simple(
                        relationship,
                        contact_obj_with_method,
                        substrate_objects,
                        environment_images,
                        physical_object_database,
                        object_snapshot_map,
                        enhanced_virtual_objects,
                        proxy_matching_results,
                        substrate_utilization_results,
                        group_counter
                    )
                    all_tasks.append(task)
                    group_counter += 1
                else:
                    log(f"Warning: No substrate candidates found for contact object: {contact_obj.get('object')}")
        
        # Run tasks with overlapping batches for maximum throughput (same approach as substrate utilization)
        log(f"Running {len(all_tasks)} relationship rating tasks with overlapping batches (size: {RELATIONSHIP_RATING_BATCH_SIZE}, interval: {RELATIONSHIP_RATING_BATCH_INTERVAL}s)")
        
        # Create all batch tasks without waiting for them to complete
        batch_tasks = []
        for i in range(0, len(all_tasks), RELATIONSHIP_RATING_BATCH_SIZE):
            batch = all_tasks[i:i + RELATIONSHIP_RATING_BATCH_SIZE]
            batch_num = i // RELATIONSHIP_RATING_BATCH_SIZE + 1
            total_batches = (len(all_tasks) + RELATIONSHIP_RATING_BATCH_SIZE - 1) // RELATIONSHIP_RATING_BATCH_SIZE
            
            log(f"Starting relationship batch {batch_num}/{total_batches}: {len(batch)} tasks")
            
            # Create a task for this batch
            batch_task = asyncio.create_task(
                _run_single_relationship_batch(batch, batch_num)
            )
            batch_tasks.append(batch_task)
            
            # Wait before starting the next batch (except for the last one)
            if i + RELATIONSHIP_RATING_BATCH_SIZE < len(all_tasks):
                await asyncio.sleep(RELATIONSHIP_RATING_BATCH_INTERVAL)
        
        # Now wait for all batches to complete
        log(f"All {len(batch_tasks)} relationship batches started. Waiting for completion...")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process all batch results
        all_relationship_results = []
        successful_batches = 0
        
        for batch_num, batch_result in enumerate(batch_results, 1):
            if isinstance(batch_result, Exception):
                log(f"Error in relationship batch {batch_num}: {batch_result}")
                continue
            elif isinstance(batch_result, list):
                all_relationship_results.extend(batch_result)
                successful_batches += 1
            else:
                log(f"Relationship batch {batch_num} returned unexpected result type: {type(batch_result)}")
        
        log(f"Overlapping relationship batch execution completed: {successful_batches}/{len(batch_tasks)} batches successful")
        
        # Generate unrated pairs with 0 scores for comprehensive coverage
        log("Generating unrated pairs with 0 scores for comprehensive coverage")
        
        # Create a set of rated pairs from the results
        rated_pairs_from_results = set()
        for result in all_relationship_results:
            if "error" not in result:
                pair_key = f"{result.get('virtualContactObject', '')}_{result.get('virtualSubstrateObject', '')}_{result.get('contactObject_id', '')}_{result.get('contactImage_id', '')}_{result.get('substrateObject_id', '')}_{result.get('substrateImage_id', '')}"
                rated_pairs_from_results.add(pair_key)
        
        # Generate all possible pairs and add unrated ones with 0 scores
        unrated_pairs = []
        for relationship in relationship_annotations:
            virtual_contact_name = relationship.get("contactObject", "")
            virtual_substrate_name = relationship.get("substrateObject", "")
            annotation_text = relationship.get("annotationText", "No annotation available")
            
            # Create unrated pairs for all possible combinations
            for contact_obj in all_physical_objects:
                for substrate_obj in all_physical_objects:
                    # Skip if same object
                    if (contact_obj.get('object_id') == substrate_obj.get('object_id') and 
                        contact_obj.get('image_id') == substrate_obj.get('image_id')):
                        continue
                    
                    pair_key = f"{virtual_contact_name}_{virtual_substrate_name}_{contact_obj.get('object_id')}_{contact_obj.get('image_id')}_{substrate_obj.get('object_id')}_{substrate_obj.get('image_id')}"
                    
                    # If this pair was not rated, add it with 0 scores
                    if pair_key not in rated_pairs_from_results:
                        unrated_pair = {
                            "virtualContactObject": virtual_contact_name,
                            "virtualSubstrateObject": virtual_substrate_name,
                            "physicalContactObject": contact_obj.get('object', 'Unknown'),
                            "physicalSubstrateObject": substrate_obj.get('object', 'Unknown'),
                            "contactObject_id": contact_obj.get('object_id'),
                            "contactImage_id": contact_obj.get('image_id'),
                            "substrateObject_id": substrate_obj.get('object_id'),
                            "substrateImage_id": substrate_obj.get('image_id'),
                            "contactUtilizationMethod": "Not evaluated - not in top 5",
                            "substrateUtilizationMethod": "Not evaluated - not in top 5",
                            "harmony_rating": 0,
                            "harmony_explanation": "Not evaluated - contact object not in top 5 for this virtual contact object",
                            "expressivity_rating": 0,
                            "expressivity_explanation": "Not evaluated - contact object not in top 5 for this virtual contact object",
                            "realism_rating": 0,
                            "realism_explanation": "Not evaluated - contact object not in top 5 for this virtual contact object",
                            "group_index": 0,
                            "expectedHapticFeedback": annotation_text
                        }
                        unrated_pairs.append(unrated_pair)
        
        # Combine rated and unrated pairs
        all_relationship_results.extend(unrated_pairs)
        
        # Log summary of results
        log(f"Completed relationship ratings with {len(all_relationship_results)} total ratings")
        log(f"  - Rated pairs: {len(all_relationship_results) - len(unrated_pairs)}")
        log(f"  - Unrated pairs (0 scores): {len(unrated_pairs)}")
        
        return all_relationship_results
        
    except Exception as e:
        log(f"Error processing relationship ratings: {e}")
        import traceback
        log(traceback.format_exc())
        return []

# Modify the run_concurrent_tasks function to include proxy matching
@traceable(run_type="chain", metadata={"process": "main_orchestration"})
async def run_concurrent_tasks():
    tasks = []
    results = {}
    
    # Add physical object task if we have environment images
    if environment_image_base64_list:
        log(f"Setting up task to process {len(environment_image_base64_list)} environment images")
        physical_task = process_multiple_images(environment_image_base64_list)
        tasks.append(physical_task)
    
    # Add virtual object task if we have haptic annotation data
    if haptic_annotation_json:
        log("Setting up task to process virtual objects from haptic annotation data")
        virtual_task = process_virtual_objects(haptic_annotation_json)
        tasks.append(virtual_task)
    
    # Run initial tasks concurrently and get results
    if tasks:
        log("Starting concurrent processing of physical and virtual objects")
        task_results = await asyncio.gather(*tasks)
        
        # Process results
        task_index = 0
        
        # Handle physical objects result if that task was included
        if environment_image_base64_list:
            physical_result = task_results[task_index]
            task_index += 1
            results["physical_result"] = physical_result
        
        # Handle virtual objects result if that task was included
        if haptic_annotation_json:
            virtual_result = task_results[task_index]
            results["virtual_result"] = virtual_result
    
    # Create object snapshot map for virtual objects
    object_snapshot_map = {}
    for snapshot in virtual_object_snapshots:
        if 'objectName' in snapshot and 'imageBase64' in snapshot:
            original_name = snapshot['objectName']
            normalized_name = normalize_name(original_name)
            object_snapshot_map[normalized_name] = snapshot['imageBase64']
            object_snapshot_map[original_name] = snapshot['imageBase64']
    
    # Get the actual data from the results
    physical_object_database = results.get("physical_result", {})
    enhanced_virtual_objects = results.get("virtual_result", [])
    
    # Enhance physical object database with YOLO-World segmentation
    if environment_image_base64_list and physical_object_database:
        log("Enhancing physical object database with YOLO-World segmentation...")
        try:
            physical_object_database = await enhance_with_yolo_segmentation(
                physical_object_database, 
                environment_image_base64_list
            )
            log("YOLO-World segmentation enhancement completed successfully")
            
            # Send segmentation data to Quest immediately after enhancement
            try:
                await send_segmentation_data_to_quest(physical_object_database)
            except Exception as segmentation_send_error:
                log(f"Warning: Failed to send segmentation data to Quest: {segmentation_send_error}")
                log("Continuing with normal processing...")
                
        except Exception as e:
            log(f"Error in YOLO-World enhancement: {e}")
            log("Continuing with original object database without segmentation")
    
    # Run proxy matching if both physical and virtual objects are available and proxy matching is enabled
    if environment_image_base64_list and haptic_annotation_json and ENABLE_PROXY_MATCHING:
        log("Setting up proxy matching task")
        
        # Run proxy matching
        proxy_matching_results = await run_proxy_matching(
            enhanced_virtual_objects, 
            environment_image_base64_list, 
            physical_object_database,
            object_snapshot_map
        )
        
        # Add to results
        results["proxy_matching_result"] = proxy_matching_results
        
        # Log sample proxy matching results for debugging
        if len(proxy_matching_results) > 0:
            log("Sample proxy matching result:")
            log(f"- virtualObject: {proxy_matching_results[0].get('virtualObject', 'N/A')}")
            log(f"- physicalObject: {proxy_matching_results[0].get('physicalObject', 'N/A')}")
            log(f"- object_id: {proxy_matching_results[0].get('object_id', 'N/A')}")
            log(f"- image_id: {proxy_matching_results[0].get('image_id', 'N/A')}")
            log(f"- utilizationMethod: {proxy_matching_results[0].get('utilizationMethod', 'N/A')}")
        else:
            log("Warning: No proxy matching results available!")
        
        # Run property rating and substrate utilization concurrently, then relationship rating
        log("Starting new execution sequence: property rating + substrate utilization concurrent, then relationship rating")
        
        # Step 1: Run property rating and substrate utilization concurrently
        log("Step 1: Starting property rating and substrate utilization concurrently")
        
        # Create tasks for concurrent execution
        property_task = run_property_ratings(
            enhanced_virtual_objects,
            environment_image_base64_list,
            physical_object_database,
            object_snapshot_map,
            proxy_matching_results
        )
        
        substrate_task = run_substrate_utilization_methods(
            haptic_annotation_json,
            environment_image_base64_list,
            physical_object_database,
            object_snapshot_map,
            enhanced_virtual_objects,
            proxy_matching_results
        )
        
        # Run both tasks concurrently using asyncio.gather()
        try:
            concurrent_results = await asyncio.gather(property_task, substrate_task, return_exceptions=True)
            
            # Process property rating result (first result)
            if isinstance(concurrent_results[0], Exception):
                log(f"Step 1 error: Property rating failed: {concurrent_results[0]}")
                property_rating_results = []
            else:
                property_rating_results = concurrent_results[0]
                log(f"Step 1 complete: Property rating finished with {len(property_rating_results) if isinstance(property_rating_results, (list, tuple)) else 0} results")
            
            # Process substrate utilization result (second result)
            if isinstance(concurrent_results[1], Exception):
                log(f"Step 1 error: Substrate utilization failed: {concurrent_results[1]}")
                substrate_utilization_results = []
            else:
                substrate_utilization_results = concurrent_results[1]
                log(f"Step 1 complete: Substrate utilization finished with {len(substrate_utilization_results) if isinstance(substrate_utilization_results, (list, tuple)) else 0} results")
                
        except Exception as e:
            log(f"Step 1 error: Concurrent execution failed: {e}")
            property_rating_results = []
            substrate_utilization_results = []
        
        # Step 2: Run relationship rating after both property rating and substrate utilization complete
        if ENABLE_RELATIONSHIP_RATING:
            log("Step 2: Starting relationship rating (after property rating and substrate utilization completion)")
            
            try:
                relationship_rating_results = await run_relationship_ratings(
                    haptic_annotation_json,
                    environment_image_base64_list,
                    physical_object_database,
                    object_snapshot_map,
                    enhanced_virtual_objects,
                    proxy_matching_results,
                    substrate_utilization_results,
                    property_rating_results
                )
                log(f"Step 2 complete: Relationship rating finished with {len(relationship_rating_results) if isinstance(relationship_rating_results, (list, tuple)) else 0} results")
                
            except Exception as e:
                log(f"Step 2 error: Relationship rating failed: {e}")
                relationship_rating_results = []
            
            log("New execution sequence complete: property rating + substrate utilization concurrent, then relationship rating")
            
        else:
            log("Step 2: Relationship rating disabled - skipping")
            
            # Set relationship rating results as empty
            relationship_rating_results = []
            
            log("New execution sequence complete: property rating + substrate utilization concurrent, relationship rating skipped")
        
        # Add to results
        results["property_rating_result"] = property_rating_results
        results["relationship_rating_result"] = relationship_rating_results
        results["substrate_utilization_result"] = substrate_utilization_results
    
    else:
        # Proxy matching is disabled or data is not available
        if not ENABLE_PROXY_MATCHING:
            log("Proxy matching disabled via configuration - skipping proxy matching and all dependent processes")
        else:
            log("No environment images or haptic annotation data available - skipping proxy matching and dependent processes")
        
        # Set empty results for all proxy matching dependent processes
        results["proxy_matching_result"] = []
        results["property_rating_result"] = []
        results["relationship_rating_result"] = []
        results["substrate_utilization_result"] = []
    
    # Store the enhanced physical object database in results
    results["enhanced_physical_result"] = physical_object_database
    
    return results

# Add a utility function to validate and normalize relationship rating responses
def validate_and_normalize_relationship_response(rating_results, dimension, group_index, physical_object_database):
    """Validate and normalize relationship rating responses to ensure consistency - CURRENTLY DISABLED"""
    log(f"VALIDATION DISABLED - returning raw results: {len(rating_results) if rating_results else 0} {dimension} ratings for group {group_index}")
    return rating_results  # Return raw results without validation

# Add a utility function to handle key renaming in JSON structures
def rename_key_in_json(data, old_key, new_key):
    """Recursively rename a key in a JSON-like data structure"""
    if isinstance(data, dict):
        # Create a new dict with updated keys
        new_dict = {}
        for k, v in data.items():
            # Replace the key if it matches
            new_k = new_key if k == old_key else k
            
            # Process the value (which might contain further dict/list structures)
            new_dict[new_k] = rename_key_in_json(v, old_key, new_key)
        return new_dict
    elif isinstance(data, list):
        # Process each item in the list
        return [rename_key_in_json(item, old_key, new_key) for item in data]
    else:
        # Return primitives unchanged
        return data

# Function to calculate final scores and update proxy matching results
def calculate_final_scores(property_rating_results, proxy_matching_results):
    """Calculate final weighted scores for each virtual-physical pair and update proxy matching results"""
    log("Calculating final weighted scores for each virtual-physical pair")
    
    # Create a dictionary to store scores for each virtual-physical pair
    scores = {}
    
    # Process each property rating result
    for rating in property_rating_results:
        if "error" in rating:
            continue
            
        # Create a key for this virtual-physical pair
        virt_obj = rating.get("virtualObject", "unknown")
        obj_id = rating.get("object_id", -1)
        img_id = rating.get("image_id", -1)
        
        # Use object_id and image_id as the primary identifiers
        pair_key = f"{virt_obj}:{obj_id}:{img_id}"
        property_name = rating.get("property", "unknown")
        
        # Calculate the mean of the three ratings
        rating_1 = rating.get("rating_1", 0)
        rating_2 = rating.get("rating_2", 0)
        rating_3 = rating.get("rating_3", 0)
        mean_rating = (rating_1 + rating_2 + rating_3) / 3 if (rating_1 or rating_2 or rating_3) else 0
        
        # Get the property value (significance)
        property_value = rating.get("propertyValue", 0.0)
        
        # Calculate the weighted score for this property
        weighted_score = mean_rating * property_value
        
        # Initialize the entry if it doesn't exist
        if pair_key not in scores:
            scores[pair_key] = {
                "virtual_object": virt_obj,
                "object_id": obj_id,
                "image_id": img_id,
                "total_score": 0,
                "property_scores": {}
            }
        
        # Add the weighted score to the total
        scores[pair_key]["total_score"] += weighted_score
        
        # Store the individual property score
        scores[pair_key]["property_scores"][property_name] = {
            "mean_rating": mean_rating,
            "property_value": property_value,
            "weighted_score": weighted_score
        }
    
    # Update proxy matching results with the calculated scores
    for proxy_result in proxy_matching_results:
        virt_obj = proxy_result.get("virtualObject", "unknown")
        obj_id = proxy_result.get("object_id", -1)
        img_id = proxy_result.get("image_id", -1)
        
        pair_key = f"{virt_obj}:{obj_id}:{img_id}"
        
        if pair_key in scores:
            # Add the total score to the proxy result
            proxy_result["rating_score"] = scores[pair_key]["total_score"]
            
            # Add detailed property scores if desired
            proxy_result["property_scores"] = scores[pair_key]["property_scores"]
    
    return proxy_matching_results

# Function to generate substrate utilization methods for a single contact-substrate relationship
@traceable(run_type="llm", metadata={"process": "substrate_utilization"})
async def generate_substrate_utilization_for_contact(relationship_annotation, contact_object, environment_images, physical_object_database, object_snapshot_map, enhanced_virtual_objects, proxy_matching_results):
    try:
        virtual_contact_name = relationship_annotation.get("contactObject", "Unknown Contact Object")
        virtual_substrate_name = relationship_annotation.get("substrateObject", "Unknown Substrate Object")
        annotation_text = relationship_annotation.get("annotationText", "No annotation available")
        
        log(f"Generating substrate utilization methods for {virtual_contact_name} -> {virtual_substrate_name}")
        
        # Determine the type of virtual substrate object
        virtual_substrate_obj = None
        for vobj in enhanced_virtual_objects:
            if vobj.get("objectName") == virtual_substrate_name:
                virtual_substrate_obj = vobj
                break
        
        # Check if the virtual substrate object is dual-role (both contact and substrate)
        is_dual_role = False
        if virtual_substrate_obj and virtual_substrate_obj.get("involvementType") in ["grasp", "contact"]:
            is_dual_role = True
            log(f"Virtual substrate object '{virtual_substrate_name}' is dual-role (involvementType: {virtual_substrate_obj.get('involvementType')})")
        else:
            log(f"Virtual substrate object '{virtual_substrate_name}' is pure substrate")
        
        # Get existing proxy utilization methods for the virtual substrate object (if dual-role)
        substrate_proxy_methods = []
        if is_dual_role:
            for proxy_result in proxy_matching_results:
                if proxy_result.get('virtualObject') == virtual_substrate_name:
                    substrate_proxy_methods.append(proxy_result)
            log(f"Found {len(substrate_proxy_methods)} existing proxy methods for dual-role substrate object")
        
        # Get the contact object's utilization method from proxy matching results
        contact_utilization_method = 'No utilization method available'
        for proxy_result in proxy_matching_results:
            if (proxy_result.get('object_id') == contact_object.get('object_id') and 
                proxy_result.get('image_id') == contact_object.get('image_id') and
                proxy_result.get('virtualObject') == virtual_contact_name):
                contact_utilization_method = proxy_result.get('utilizationMethod', 'No utilization method available')
                break
        
        has_valid_contact_method = contact_utilization_method != 'No utilization method available'
        
        # If no valid contact method, log warning and skip this task
        if not has_valid_contact_method:
            log(f"WARNING: Contact object {contact_object.get('object')} (ID: {contact_object.get('object_id')}) has no utilization method for {virtual_contact_name}. This should not happen!")
            return []  # Return empty list to skip this invalid task
        
        # Build the human message content
        human_message_content = []
        
        # 1. Add the relationship information
        # Get virtual contact and substrate object details
        virtual_contact_obj = None
        virtual_substrate_obj = None
        for vobj in enhanced_virtual_objects:
            if vobj.get("objectName") == virtual_contact_name:
                virtual_contact_obj = vobj
            elif vobj.get("objectName") == virtual_substrate_name:
                virtual_substrate_obj = vobj
        
        contact_interaction_deduction = virtual_contact_obj.get("interactionDeduction", "No interaction deduction available") if virtual_contact_obj else "No interaction deduction available"
        contact_dimensions = virtual_contact_obj.get("dimensions_meters", {}) if virtual_contact_obj else {}
        substrate_dimensions = virtual_substrate_obj.get("dimensions_meters", {}) if virtual_substrate_obj else {}
        
        # Format dimensions for display
        def format_dimensions(dims):
            if not dims:
                return "No dimensions available"
            return f"Width: {dims.get('x', 'N/A')}m, Height: {dims.get('y', 'N/A')}m, Depth: {dims.get('z', 'N/A')}m"
        
        # Select the appropriate system prompt and build content based on substrate object type
        if is_dual_role:
            substrate_type = "dual_role"
            relationship_text = f"""# Substrate Utilization Method Generation

## Virtual Object Relationship
- **Contact Object**: {virtual_contact_name}
- **Substrate Object**: {virtual_substrate_name} (DUAL-ROLE: also a contact object)
- **Expected Haptic Feedback**: {annotation_text}

## Virtual Contact Object Details
- **Interaction Deduction**: {contact_interaction_deduction}
- **Dimensions**: {format_dimensions(contact_dimensions)}

## Virtual Substrate Object Details (Dual-Role)
- **Involvement Type**: {virtual_substrate_obj.get('involvementType', 'Unknown') if virtual_substrate_obj else 'Unknown'}
- **Dimensions**: {format_dimensions(substrate_dimensions)}
- **Interaction Deduction**: {virtual_substrate_obj.get('interactionDeduction', 'No interaction deduction available') if virtual_substrate_obj else 'No interaction deduction available'}

## Physical Contact Object Assignment
- **Contact Object**: {contact_object.get('object', 'Unknown')} (ID: {contact_object.get('object_id')}, Image: {contact_object.get('image_id')})
- **Contact Utilization Method**: {contact_utilization_method}

Your task is to determine how each physical object in the environment can be utilized as a substrate to work with this specific physical contact object and its utilization method.
"""
        else:
            substrate_type = "pure"
            relationship_text = f"""# Substrate Utilization Method Generation

## Virtual Object Relationship
- **Contact Object**: {virtual_contact_name}
- **Substrate Object**: {virtual_substrate_name} (PURE SUBSTRATE)
- **Expected Haptic Feedback**: {annotation_text}

## Virtual Contact Object Details
- **Interaction Deduction**: {contact_interaction_deduction}
- **Dimensions**: {format_dimensions(contact_dimensions)}

## Virtual Substrate Object Details (Pure Substrate)
- **Dimensions**: {format_dimensions(substrate_dimensions)}

## Physical Contact Object Assignment
- **Contact Object**: {contact_object.get('object', 'Unknown')} (ID: {contact_object.get('object_id')}, Image: {contact_object.get('image_id')})
- **Contact Utilization Method**: {contact_utilization_method}

Your task is to determine how each physical object in the environment can be utilized as a substrate to work with this specific physical contact object and its utilization method.
"""
        
        # Get the appropriate system prompt using the dynamic function
        selected_system_prompt = get_substrate_utilization_system_prompt(substrate_type)
        
        human_message_content.append({
            "type": "text", 
            "text": relationship_text
        })
        
        # 2. Add virtual object snapshots if available
        # Add virtual contact object snapshot
        contact_snapshot_found = False
        normalized_contact_name = normalize_name(virtual_contact_name)
        
        # Try direct match first
        if virtual_contact_name in object_snapshot_map:
            log(f"Found snapshot for contact object: {virtual_contact_name} (direct match)")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Contact Object: {virtual_contact_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[virtual_contact_name]}", 
                    "detail": "high"
                }
            })
            contact_snapshot_found = True
        # Then try normalized match
        elif normalized_contact_name in object_snapshot_map:
            log(f"Found snapshot for contact object: {virtual_contact_name} (normalized as {normalized_contact_name})")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Contact Object: {virtual_contact_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[normalized_contact_name]}", 
                    "detail": "high"
                }
            })
            contact_snapshot_found = True
        # Finally try fuzzy match
        else:
            # Try to find partial matches
            potential_matches = [name for name in object_snapshot_map.keys() 
                               if normalized_contact_name in normalize_name(name) or normalize_name(name) in normalized_contact_name]
            
            if potential_matches:
                best_match = potential_matches[0]  # Take the first match
                log(f"Found snapshot for contact object: {virtual_contact_name} (fuzzy match: {best_match})")
                human_message_content.append({
                    "type": "text",
                    "text": f"\n## Virtual Contact Object: {virtual_contact_name}\n"
                })
                human_message_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{object_snapshot_map[best_match]}", 
                        "detail": "high"
                    }
                })
                contact_snapshot_found = True
        
        if not contact_snapshot_found:
            log(f"No snapshot found for contact object: {virtual_contact_name} (normalized: {normalized_contact_name})")
        
        # Add virtual substrate object snapshot
        substrate_snapshot_found = False
        normalized_substrate_name = normalize_name(virtual_substrate_name)
        
        # Try direct match first
        if virtual_substrate_name in object_snapshot_map:
            log(f"Found snapshot for substrate object: {virtual_substrate_name} (direct match)")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Substrate Object: {virtual_substrate_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[virtual_substrate_name]}", 
                    "detail": "high"
                }
            })
            substrate_snapshot_found = True
        # Then try normalized match
        elif normalized_substrate_name in object_snapshot_map:
            log(f"Found snapshot for substrate object: {virtual_substrate_name} (normalized as {normalized_substrate_name})")
            human_message_content.append({
                "type": "text",
                "text": f"\n## Virtual Substrate Object: {virtual_substrate_name}\n"
            })
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{object_snapshot_map[normalized_substrate_name]}", 
                    "detail": "high"
                }
            })
            substrate_snapshot_found = True
        # Finally try fuzzy match
        else:
            # Try to find partial matches
            potential_matches = [name for name in object_snapshot_map.keys() 
                               if normalized_substrate_name in normalize_name(name) or normalize_name(name) in normalized_substrate_name]
            
            if potential_matches:
                best_match = potential_matches[0]  # Take the first match
                log(f"Found snapshot for substrate object: {virtual_substrate_name} (fuzzy match: {best_match})")
                human_message_content.append({
                    "type": "text",
                    "text": f"\n## Virtual Substrate Object: {virtual_substrate_name}\n"
                })
                human_message_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{object_snapshot_map[best_match]}", 
                        "detail": "high"
                    }
                })
                substrate_snapshot_found = True
        
        if not substrate_snapshot_found:
            log(f"No snapshot found for substrate object: {virtual_substrate_name} (normalized: {normalized_substrate_name})")
        
        # 3. Add environment snapshots with their detected objects
        for i, image_base64 in enumerate(environment_images):
            # Add the environment snapshot
            human_message_content.append({
                "type": "text", 
                "text": f"\n## Environment Snapshot {i+1}\n"
            })
            
            human_message_content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}", 
                    "detail": "high"
                }
            })
            
            # Add the detected objects for this snapshot
            objects_in_snapshot = physical_object_database.get(str(i), [])
            if not objects_in_snapshot:
                # Check if we should look for objects in a different format (integer key)
                objects_in_snapshot = physical_object_database.get(i, [])
            
            objects_text = "\n### Physical Objects in this Snapshot\n"
            
            if objects_in_snapshot:
                for obj in objects_in_snapshot:
                    obj_id = obj.get('object_id', 'Unknown')
                    obj_name = obj.get('object', 'Unknown object')
                    obj_position = obj.get('position', 'Unknown position')
                    
                    # Highlight the contact object
                    if (obj['object_id'] == contact_object.get('object_id') and 
                        i == contact_object.get('image_id')):  # Use i for consistent comparison
                        objects_text += f"- **CONTACT OBJECT** - Object ID: {obj_id} - {obj_name} ({obj_position})\n"
                        objects_text += f"  Contact Utilization Method: {contact_utilization_method}\n"
                        objects_text += f"  Image ID: {i}\n\n"
                    else:
                        objects_text += f"- Object ID: {obj_id} - {obj_name} ({obj_position}) - *Substrate candidate*\n"
                        
                        # Search for all utilization methods for this physical object when used as proxy for the virtual substrate object
                        substrate_utilization_methods = []
                        for proxy_result in proxy_matching_results:
                            # Match by object_id and image_id
                            proxy_obj_id = proxy_result.get('object_id')
                            proxy_img_id = proxy_result.get('image_id')
                            proxy_virtual_obj = proxy_result.get('virtualObject', '')
                            
                            # Convert types for flexible comparison
                            if isinstance(proxy_obj_id, str):
                                try:
                                    proxy_obj_id = int(proxy_obj_id)
                                except ValueError:
                                    pass
                            if isinstance(proxy_img_id, str):
                                try:
                                    proxy_img_id = int(proxy_img_id)
                                except ValueError:
                                    pass
                            
                            # Check if this matches our current physical object and virtual substrate object
                            if (proxy_obj_id == obj_id and 
                                proxy_img_id == i and
                                proxy_virtual_obj == virtual_substrate_name):
                                util_method = proxy_result.get("utilizationMethod", "")
                                if util_method:
                                    substrate_utilization_methods.append(util_method)
                        
                        # Display utilization methods
                        if substrate_utilization_methods:
                            objects_text += f"  **Proxy Utilization Method as {virtual_substrate_name}**:\n"
                            for method in substrate_utilization_methods:
                                # Format the method text with proper indentation
                                objects_text += f"    {method}\n"
                        else:
                            objects_text += f"  No proxy utilization method available as {virtual_substrate_name}\n"
                        
                        objects_text += f"  Image ID: {i}\n\n"
            else:
                objects_text += "- No objects detected in this snapshot\n"
                
            human_message_content.append({
                "type": "text", 
                "text": objects_text
            })
        
        # 4. Add final instructions with explicit object ID mapping
        
        # Create a clear object ID mapping table
        object_mapping_text = f"""
# CRITICAL: Object ID Mapping Table

**Contact Object (MUST use these exact IDs):**
- Object ID: {contact_object.get('object_id')}
- Image ID: {contact_object.get('image_id')}  
- Name: {contact_object.get('object')}

**Substrate Objects (MUST use these exact IDs for each substrate):**
"""
        
        # Add all substrate objects with their exact IDs from the database
        for i, image_base64 in enumerate(environment_images):
            img_key = str(i)
            if img_key in physical_object_database:
                objects_in_img = physical_object_database[img_key]
                for obj in objects_in_img:
                    # Skip the contact object itself
                    if not (obj['object_id'] == contact_object.get('object_id') and 
                           obj.get('image_id') == contact_object.get('image_id')):
                        object_mapping_text += f"- Object ID: {obj['object_id']}, Image ID: {obj.get('image_id', i)}, Name: {obj['object']}\n"
            # Also check if objects are stored with integer keys
            elif i in physical_object_database:
                objects_in_img = physical_object_database[i]
                for obj in objects_in_img:
                    # Skip the contact object itself
                    if not (obj['object_id'] == contact_object.get('object_id') and 
                           i == contact_object.get('image_id')):  # Use i (the actual image index) for comparison
                        object_mapping_text += f"- Object ID: {obj['object_id']}, Image ID: {i}, Name: {obj['object']}\n"
        
        human_message_content.append({
            "type": "text", 
            "text": object_mapping_text
        })
        
        human_message_content.append({
            "type": "text", 
            "text": f"""
# Your Task

For each physical object in the environment (except the contact object itself), determine how it could be utilized as a substrate to work with the physical contact object "{contact_object.get('object')}" using its utilization method: "{contact_utilization_method}"

**Contact Object**: {contact_object.get('object')} (ID: {contact_object.get('object_id')}, Image: {contact_object.get('image_id')})
**Contact Utilization Method**: {contact_utilization_method}

Consider:
- The expected haptic feedback: "{annotation_text}"
- How each substrate candidate should be positioned, oriented, or prepared
- What specific properties or features of each substrate should be utilized
- How each substrate would interact with the contact object's utilization method
- What modifications or setup might be needed for each substrate

Focus on creating substrate utilization methods that would enable the delivery of the expected haptic feedback: "{annotation_text}"

**CRITICAL REQUIREMENTS:**
1. Use ONLY the Object IDs and Image IDs from the mapping table above
2. Do NOT make up or guess any object IDs  
3. The contactObject_id MUST be {contact_object.get('object_id')} and contactImage_id MUST be {contact_object.get('image_id')}
4. For each substrate, use the EXACT object_id and image_id from the mapping table

FORMAT YOUR RESPONSE as specified in the system prompt, using the EXACT object IDs from the mapping table above.
"""
        })
        
        # Create the messages
        messages = [
            SystemMessage(content=selected_system_prompt),
            HumanMessage(content=human_message_content)
        ]
        
        # Get response from the model
        log(f"Sending substrate utilization request for {virtual_contact_name} -> {virtual_substrate_name}")
        # LangChain ChatOpenAI has built-in LangSmith tracing - no extra config needed
        response = await substrate_utilization_llm.ainvoke(messages)
        log(f"Received substrate utilization methods for {virtual_contact_name} -> {virtual_substrate_name}")
        
        # Extract JSON from response
        response_text = extract_response_text(response.content)
        
        # Try to find JSON array
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_content = response_text[json_start:json_end]
        else:
            # Try to find JSON between code blocks
            json_start = response_text.find("```json")
            if json_start != -1:
                json_start += 7  # Length of ```json
                json_end = response_text.find("```", json_start)
                if json_end != -1:
                    json_content = response_text[json_start:json_end].strip()
                else:
                    json_content = response_text[json_start:].strip()
            else:
                # As a fallback, use the entire response
                json_content = response_text
        
        try:
            # Parse the JSON response
            utilization_results = json.loads(json_content)
            
            # Add contact object information to each result
            for result in utilization_results:
                result["relationshipAnnotation"] = relationship_annotation
                
            return utilization_results
            
        except json.JSONDecodeError as e:
            log(f"Error parsing substrate utilization JSON for {virtual_contact_name} -> {virtual_substrate_name}: {e}")
            log(f"Raw content: {json_content}")
            
            # Return a basic result with the error
            return [{
                "virtualContactObject": virtual_contact_name,
                "virtualSubstrateObject": virtual_substrate_name,
                "physicalContactObject": contact_object.get('object', 'Unknown'),
                "error": f"Failed to parse response: {str(e)}",
                "rawResponse": response_text[:500]  # First 500 chars
            }]
            
    except Exception as e:
        log(f"Error in substrate utilization generation for {relationship_annotation.get('contactObject', 'unknown')} -> {relationship_annotation.get('substrateObject', 'unknown')}: {e}")
        import traceback
        log(traceback.format_exc())
        
        # Return a basic result with the error
        return [{
            "virtualContactObject": relationship_annotation.get("contactObject", "unknown"),
            "virtualSubstrateObject": relationship_annotation.get("substrateObject", "unknown"),
            "error": f"Processing error: {str(e)}"
        }]

# Helper function to run a single batch of substrate utilization tasks
async def _run_single_substrate_batch(batch_tasks, batch_num):
    """Run a single batch of substrate utilization tasks and return results"""
    try:
        log(f"Executing substrate batch {batch_num}: {len(batch_tasks)} tasks")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process batch results
        batch_utilization_results = []
        batch_success_count = 0
        
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                log(f"Error in substrate batch {batch_num}, task {j+1}: {result}")
                continue
            elif isinstance(result, list):
                # Each result is an array of utilization results for a single contact-substrate relationship
                batch_utilization_results.extend(result)
                batch_success_count += 1
            else:
                log(f"Substrate batch {batch_num}, task {j+1} returned unexpected result type: {type(result)}")
        
        log(f"Substrate batch {batch_num} completed: {batch_success_count}/{len(batch_tasks)} tasks successful")
        return batch_utilization_results
        
    except Exception as e:
        log(f"Error processing substrate batch {batch_num}: {e}")
        return []

# Helper function to run a single batch of property rating tasks
async def _run_single_property_batch(batch_tasks, batch_num):
    """Run a single batch of property rating tasks and return results"""
    try:
        log(f"Executing property batch {batch_num}: {len(batch_tasks)} tasks")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process batch results
        batch_property_results = []
        batch_success_count = 0
        
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                log(f"Error in property batch {batch_num}, task {j+1}: {result}")
                continue
            elif isinstance(result, list):
                # Each result is an array of property rating results for a single property rating task
                batch_property_results.extend(result)
                batch_success_count += 1
            else:
                log(f"Property batch {batch_num}, task {j+1} returned unexpected result type: {type(result)}")
        
        log(f"Property batch {batch_num} completed: {batch_success_count}/{len(batch_tasks)} tasks successful")
        return batch_property_results
        
    except Exception as e:
        log(f"Error processing property batch {batch_num}: {e}")
        return []

# Helper function to run a single batch of relationship rating tasks
async def _run_single_relationship_batch(batch_tasks, batch_num):
    """Run a single batch of relationship rating tasks and return results"""
    try:
        log(f"Executing relationship batch {batch_num}: {len(batch_tasks)} tasks")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process batch results
        batch_relationship_results = []
        batch_success_count = 0
        
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                log(f"Error in relationship batch {batch_num}, task {j+1}: {result}")
                continue
            elif isinstance(result, list):
                # Each result is an array of relationship rating results for a single relationship group
                batch_relationship_results.extend(result)
                batch_success_count += 1
            else:
                log(f"Relationship batch {batch_num}, task {j+1} returned unexpected result type: {type(result)}")
        
        log(f"Relationship batch {batch_num} completed: {batch_success_count}/{len(batch_tasks)} tasks successful")
        return batch_relationship_results
        
    except Exception as e:
        log(f"Error processing relationship batch {batch_num}: {e}")
        return []

# Function to run substrate utilization methods generation for all relationships
@traceable(run_type="chain", metadata={"process": "substrate_utilization_batch"})
async def run_substrate_utilization_methods(haptic_annotation_json, environment_images, physical_object_database, object_snapshot_map, enhanced_virtual_objects, proxy_matching_results):
    if not haptic_annotation_json:
        log("No haptic annotation data provided for substrate utilization methods")
        return []
    
    try:
        # Parse the haptic annotation JSON
        haptic_data = json.loads(haptic_annotation_json)
        relationship_annotations = haptic_data.get("relationshipAnnotations", [])
        
        if not relationship_annotations:
            log("No relationship annotations found in haptic data")
            return []
        
        log(f"Found {len(relationship_annotations)} relationship annotations")
        
        # Get all physical objects from the database
        all_physical_objects = []
        for image_id, objects in physical_object_database.items():
            for obj in objects:
                all_physical_objects.append(obj)
        
        log(f"Found {len(all_physical_objects)} total physical objects for substrate utilization method generation")
        
        # Create substrate utilization tasks - for each relationship, create one task per physical object as contact
        all_tasks = []
        
        for relationship in relationship_annotations:
            virtual_contact_name = relationship.get("contactObject", "")
            virtual_substrate_name = relationship.get("substrateObject", "")
            
            # For each physical object that could be a contact object (has a utilization method for this virtual contact)
            for contact_obj in all_physical_objects:
                # Check if this physical object has a utilization method for the virtual contact object
                has_contact_utilization = False
                contact_utilization_method = None
                for proxy_result in proxy_matching_results:
                    if (proxy_result.get('object_id') == contact_obj.get('object_id') and 
                        proxy_result.get('image_id') == contact_obj.get('image_id') and
                        proxy_result.get('virtualObject') == virtual_contact_name):
                        has_contact_utilization = True
                        contact_utilization_method = proxy_result.get('utilizationMethod', 'No utilization method available')
                        break
                
                # Only create a task if this physical object can serve as the contact object
                if has_contact_utilization:
                    log(f"Creating substrate utilization task for {virtual_contact_name} -> {virtual_substrate_name} with contact object {contact_obj.get('object')} (ID: {contact_obj.get('object_id')})")
                    log(f"  Contact utilization method: {contact_utilization_method[:50]}..." if contact_utilization_method else "  No utilization method")
                    
                    task = generate_substrate_utilization_for_contact(
                        relationship,
                        contact_obj,
                        environment_images,
                        physical_object_database,
                        object_snapshot_map,
                        enhanced_virtual_objects,
                        proxy_matching_results
                    )
                    all_tasks.append(task)
                else:
                    log(f"Skipping contact object {contact_obj.get('object')} (ID: {contact_obj.get('object_id')}) - no utilization method for {virtual_contact_name}")
        
        # Run tasks with overlapping batches for maximum throughput
        log(f"Running {len(all_tasks)} substrate utilization tasks with overlapping batches (size: {SUBSTRATE_BATCH_SIZE}, interval: {SUBSTRATE_BATCH_INTERVAL}s)")
        
        # Create all batch tasks without waiting for them to complete
        batch_tasks = []
        for i in range(0, len(all_tasks), SUBSTRATE_BATCH_SIZE):
            batch = all_tasks[i:i + SUBSTRATE_BATCH_SIZE]
            batch_num = i // SUBSTRATE_BATCH_SIZE + 1
            total_batches = (len(all_tasks) + SUBSTRATE_BATCH_SIZE - 1) // SUBSTRATE_BATCH_SIZE
            
            log(f"Starting batch {batch_num}/{total_batches}: {len(batch)} tasks")
            
            # Create a task for this batch
            batch_task = asyncio.create_task(
                _run_single_substrate_batch(batch, batch_num)
            )
            batch_tasks.append(batch_task)
            
            # Wait before starting the next batch (except for the last one)
            if i + SUBSTRATE_BATCH_SIZE < len(all_tasks):
                await asyncio.sleep(SUBSTRATE_BATCH_INTERVAL)
        
        # Now wait for all batches to complete
        log(f"All {len(batch_tasks)} batches started. Waiting for completion...")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process all batch results
        all_substrate_utilization_results = []
        successful_batches = 0
        
        for batch_num, batch_result in enumerate(batch_results, 1):
            if isinstance(batch_result, Exception):
                log(f"Error in batch {batch_num}: {batch_result}")
                continue
            elif isinstance(batch_result, list):
                all_substrate_utilization_results.extend(batch_result)
                successful_batches += 1
            else:
                log(f"Batch {batch_num} returned unexpected result type: {type(batch_result)}")
        
        log(f"Overlapping batch execution completed: {successful_batches}/{len(batch_tasks)} batches successful")
        
        # Log summary of results
        log(f"Completed substrate utilization method generation with {len(all_substrate_utilization_results)} total utilization methods")
        
        return all_substrate_utilization_results
        
    except Exception as e:
        log(f"Error processing substrate utilization methods: {e}")
        import traceback
        log(traceback.format_exc())
        return []



def correct_object_ids_in_relationship_results(relationship_results, physical_object_database):
    """
    Corrects the object IDs in relationship rating results by looking up the correct IDs
    based on object names and image IDs from the physical object database.
    Uses fuzzy matching to handle slight variations in object names.
    
    Args:
        relationship_results: List of relationship rating result dictionaries
        physical_object_database: The physical object database containing correct ID mappings
    
    Returns:
        List of corrected relationship rating results
    """
    corrected_results = []
    
    # Create a lookup dictionary for quick ID resolution
    # Format: {image_id: {object_name: object_id}}
    object_lookup = {}
    
    for image_id_str, objects in physical_object_database.items():
        image_id = int(image_id_str)
        object_lookup[image_id] = {}
        
        for obj in objects:
            object_name = obj.get("object", "").strip().lower()
            object_id = obj.get("object_id")
            if object_name and object_id is not None:
                object_lookup[image_id][object_name] = object_id
    
    def find_best_match_object_id(target_name, image_id, threshold=0.8):
        """
        Find the best matching object ID using fuzzy string matching.
        
        Args:
            target_name: The object name to match
            image_id: The image ID to search within
            threshold: Minimum similarity threshold (0.0 to 1.0)
        
        Returns:
            Tuple of (object_id, similarity_score) or (None, 0) if no good match
        """
        if image_id not in object_lookup:
            return None, 0
        
        target_name_clean = target_name.strip().lower()
        
        # First try exact match
        if target_name_clean in object_lookup[image_id]:
            return object_lookup[image_id][target_name_clean], 1.0
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        for db_name, db_id in object_lookup[image_id].items():
            similarity = difflib.SequenceMatcher(None, target_name_clean, db_name).ratio()
            if similarity > best_score and similarity >= threshold:
                best_match = db_id
                best_score = similarity
        
        return best_match, best_score
    
    # Process each relationship result
    for result in relationship_results:
        corrected_result = result.copy()  # Create a copy to avoid modifying the original
        
        # Correct contact object ID
        contact_object_name = result.get("physicalContactObject", "")
        contact_image_id = result.get("contactImage_id")
        
        if contact_object_name and contact_image_id is not None:
            correct_contact_id, similarity = find_best_match_object_id(contact_object_name, contact_image_id)
            if correct_contact_id is not None:
                corrected_result["contactObject_id"] = correct_contact_id
                if similarity < 1.0:
                    log(f"Corrected contact object ID with fuzzy match (similarity: {similarity:.2f}): '{contact_object_name}' in image {contact_image_id} from {result.get('contactObject_id')} to {correct_contact_id}")
                else:
                    log(f"Corrected contact object ID: '{contact_object_name}' in image {contact_image_id} from {result.get('contactObject_id')} to {correct_contact_id}")
            else:
                log(f"Warning: Could not find correct ID for contact object '{contact_object_name}' in image {contact_image_id}")
        
        # Correct substrate object ID
        substrate_object_name = result.get("physicalSubstrateObject", "")
        substrate_image_id = result.get("substrateImage_id")
        
        if substrate_object_name and substrate_image_id is not None:
            correct_substrate_id, similarity = find_best_match_object_id(substrate_object_name, substrate_image_id)
            if correct_substrate_id is not None:
                corrected_result["substrateObject_id"] = correct_substrate_id
                if similarity < 1.0:
                    log(f"Corrected substrate object ID with fuzzy match (similarity: {similarity:.2f}): '{substrate_object_name}' in image {substrate_image_id} from {result.get('substrateObject_id')} to {correct_substrate_id}")
                else:
                    log(f"Corrected substrate object ID: '{substrate_object_name}' in image {substrate_image_id} from {result.get('substrateObject_id')} to {correct_substrate_id}")
            else:
                log(f"Warning: Could not find correct ID for substrate object '{substrate_object_name}' in image {substrate_image_id}")
        
        corrected_results.append(corrected_result)
    
    return corrected_results

# Function to send segmentation data to Quest
async def send_segmentation_data_to_quest(enhanced_physical_database: Dict[int, List[Dict]]) -> bool:
    """Send segmentation data to Quest via the local server"""
    try:
        log("Preparing segmentation data for Quest transmission...")
        
        # Extract only the essential segmentation data
        segmentation_data = {}
        for image_id, objects in enhanced_physical_database.items():
            segmentation_data[str(image_id)] = []
            for obj in objects:
                yolo_segmentation = obj.get("yolo_segmentation")
                if yolo_segmentation and yolo_segmentation.get("bbox"):
                    segmentation_info = {
                        "object_id": obj.get("object_id"),
                        "object": obj.get("object"),
                        "position": obj.get("position"),
                        "image_id": obj.get("image_id"),
                        "yolo_segmentation": {
                            "bbox": yolo_segmentation["bbox"],
                            "mask_center": yolo_segmentation["mask_center"],
                            "confidence": yolo_segmentation["confidence"],
                            "detected_name": yolo_segmentation["detected_name"],
                            "has_mask": yolo_segmentation["has_mask"]
                        }
                    }
                    segmentation_data[str(image_id)].append(segmentation_info)
        
        total_objects = sum(len(objects) for objects in segmentation_data.values())
        log(f"Sending segmentation data for {total_objects} objects across {len(segmentation_data)} images to Quest")
        
        # Prepare the payload for Quest
        quest_payload = {
            "action": "segmentation_data",
            "data": segmentation_data,
            "timestamp": str(uuid.uuid4()),
            "total_objects": total_objects
        }
        
        # Send to Quest via local server (assuming server runs on localhost:5000)
        server_url = "http://localhost:5000/send_to_quest"
        
        response = requests.post(
            server_url,
            json=quest_payload,
            timeout=10  # 10 second timeout
        )
        
        if response.status_code == 200:
            log("Successfully sent segmentation data to Quest")
            return True
        else:
            log(f"Failed to send segmentation data to Quest: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        log(f"Error sending segmentation data to Quest: {e}")
        return False

try:
    # Create a variable to store the processing results
    result: Dict[str, Any] = {"status": "success", "message": "Processing complete"}
    
    # Run all tasks concurrently in a single event loop
    concurrent_results = asyncio.run(run_concurrent_tasks())
    
    # Process physical objects results if available
    if environment_image_base64_list:
        log("Processing completed physical object detection results")
        # Use enhanced database with YOLO bounding boxes if available
        physical_object_database = concurrent_results.get("enhanced_physical_result", 
                                                         concurrent_results.get("physical_result", {}))
        
        # Save physical object database
        output_dir = os.path.join(script_dir, "output")
        physical_output_path = os.path.join(output_dir, "physical_object_database.json")
        physical_saved_path = save_object_database(physical_object_database, physical_output_path)
        
        # Calculate total objects found
        total_physical_objects = sum(len(objects) for objects in physical_object_database.values())
        log(f"Physical object recognition complete. Found {total_physical_objects} objects across {len(physical_object_database)} images.")
        
        # Export annotated images with bounding boxes
        annotated_image_paths = []
        try:
            annotated_image_paths = asyncio.run(export_annotated_images(
                physical_object_database, 
                environment_image_base64_list, 
                output_dir
            ))
            log(f"Exported {len(annotated_image_paths)} annotated images")
        except Exception as e:
            log(f"Error exporting annotated images: {e}")
        
        # Add to result
        result["physical_objects"] = {
            "count": total_physical_objects,
            "database_path": physical_saved_path,
            "object_database": physical_object_database,
            "annotated_images": annotated_image_paths
        }
    else:
        log("No environment images to process")
        result["physical_objects"] = {"status": "error", "message": "No environment images provided"}
    
    # Process virtual objects results if available
    if haptic_annotation_json:
        log("Processing completed virtual object processing results")
        enhanced_virtual_objects = concurrent_results.get("virtual_result", [])
        
        # Save virtual object database
        output_dir = os.path.join(script_dir, "output")
        virtual_output_path = os.path.join(output_dir, "virtual_object_database.json")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save virtual object database
        with open(virtual_output_path, 'w') as f:
            json.dump(enhanced_virtual_objects, f, indent=2)
        
        log(f"Virtual object processing complete. Enhanced {len(enhanced_virtual_objects)} virtual objects with haptic feedback descriptions.")
        
        # Add to result
        result["virtual_objects"] = {
            "count": len(enhanced_virtual_objects),
            "database_path": virtual_output_path,
            "object_database": enhanced_virtual_objects
        }
    else:
        log("No haptic annotation data to process")
        result["virtual_objects"] = {"status": "error", "message": "No haptic annotation data provided"}
    
    # Process proxy matching results if available
    if environment_image_base64_list and haptic_annotation_json and ENABLE_PROXY_MATCHING:
        log("Processing completed proxy matching results")
        proxy_matching_results = concurrent_results.get("proxy_matching_result", [])
        
        # Save proxy matching results
        output_dir = os.path.join(script_dir, "output")
        proxy_output_path = os.path.join(output_dir, "proxy_matching_results.json")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert any imageId keys to image_id before saving
        normalized_proxy_results = rename_key_in_json(proxy_matching_results, "imageId", "image_id")

        # Save proxy matching results with normalized keys
        with open(proxy_output_path, 'w') as f:
            json.dump(normalized_proxy_results, f, indent=2)
        
        log(f"Proxy method proposal complete. Generated proposals for {len(proxy_matching_results)} virtual objects.")
        
        # Add to result
        result["proxy_matching"] = {
            "count": len(proxy_matching_results),
            "database_path": proxy_output_path,
            "matching_results": proxy_matching_results
        }
    else:
        # Handle disabled or unavailable proxy matching
        if not ENABLE_PROXY_MATCHING:
            log("Proxy matching disabled via configuration")
            status_message = "Proxy matching disabled via configuration"
        else:
            log("No data available for proxy matching")
            status_message = "No environment images or haptic annotation data provided"
        
        result["proxy_matching"] = {
            "status": "disabled" if not ENABLE_PROXY_MATCHING else "no_data",
            "message": status_message,
            "count": 0,
            "matching_results": []
        }
    
    # Process property rating results if available
    if environment_image_base64_list and haptic_annotation_json and ENABLE_PROXY_MATCHING:
        log("Processing completed property rating results")
        property_rating_results = concurrent_results.get("property_rating_result", [])
        
        # Save property rating results
        output_dir = os.path.join(script_dir, "output")
        property_rating_output_path = os.path.join(output_dir, "property_rating_results.json")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save property rating results
        with open(property_rating_output_path, 'w') as f:
            json.dump(property_rating_results, f, indent=2)
        
        log(f"Property rating complete. Generated ratings for {len(property_rating_results)} virtual objects.")
        
        # Calculate final scores and update proxy matching results
        if len(property_rating_results) > 0 and len(proxy_matching_results) > 0:
            log("Calculating final scores for proxy matching results")
            updated_proxy_results = calculate_final_scores(property_rating_results, proxy_matching_results)
            
            # Save updated proxy matching results
            with open(proxy_output_path, 'w') as f:
                json.dump(updated_proxy_results, f, indent=2)
            
            log("Updated proxy matching results with final scores")
        
        # Add to result
        result["property_rating"] = {
            "count": len(property_rating_results),
            "database_path": property_rating_output_path,
            "rating_results": property_rating_results
        }
    else:
        # Handle disabled or unavailable property rating
        if not ENABLE_PROXY_MATCHING:
            log("Property rating skipped - proxy matching disabled via configuration")
            status_message = "Property rating skipped - proxy matching disabled via configuration"
        else:
            log("No data available for property rating")
            status_message = "No environment images or haptic annotation data provided"
        
        result["property_rating"] = {
            "status": "disabled" if not ENABLE_PROXY_MATCHING else "no_data",
            "message": status_message,
            "count": 0,
            "rating_results": []
        }
    

    
    # Process relationship rating results if available
    if environment_image_base64_list and haptic_annotation_json and ENABLE_PROXY_MATCHING and ENABLE_RELATIONSHIP_RATING:
        log("Processing completed relationship rating results")
        relationship_rating_results = concurrent_results.get("relationship_rating_result", [])
        
        # Correct object IDs before saving
        if relationship_rating_results and physical_object_database:
            log("Correcting object IDs in relationship rating results...")
            relationship_rating_results = correct_object_ids_in_relationship_results(
                relationship_rating_results, 
                physical_object_database
            )
            log("Object ID correction completed")
        
        # Save relationship rating results
        output_dir = os.path.join(script_dir, "output")
        relationship_rating_output_path = os.path.join(output_dir, "relationship_rating_results.json")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save relationship rating results
        with open(relationship_rating_output_path, 'w') as f:
            json.dump(relationship_rating_results, f, indent=2)
        
        # Create dimension-by-dimension breakdown
        relationship_by_dimension = {
            "harmony": [],
            "expressivity": [],
            "realism": []
        }
        
        # Organize results by dimension
        for result in relationship_rating_results:
            # Create separate entries for each dimension
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
            
            # Add harmony dimension entry
            if "harmony_rating" in result:
                harmony_entry = base_entry.copy()
                harmony_entry.update({
                    "dimension": "harmony",
                    "rating": result.get("harmony_rating", ""),
                    "explanation": result.get("harmony_explanation", "")
                })
                relationship_by_dimension["harmony"].append(harmony_entry)
            
            # Add expressivity dimension entry
            if "expressivity_rating" in result:
                expressivity_entry = base_entry.copy()
                expressivity_entry.update({
                    "dimension": "expressivity",
                    "rating": result.get("expressivity_rating", ""),
                    "explanation": result.get("expressivity_explanation", "")
                })
                relationship_by_dimension["expressivity"].append(expressivity_entry)
            
            # Add realism dimension entry
            if "realism_rating" in result:
                realism_entry = base_entry.copy()
                realism_entry.update({
                    "dimension": "realism",
                    "rating": result.get("realism_rating", ""),
                    "explanation": result.get("realism_explanation", "")
                })
                relationship_by_dimension["realism"].append(realism_entry)
        
        # Save dimension-by-dimension results
        relationship_by_dimension_path = os.path.join(output_dir, "relationship_rating_by_dimension.json")
        with open(relationship_by_dimension_path, 'w') as f:
            json.dump(relationship_by_dimension, f, indent=2)
        
        # Log summary
        harmony_count = len(relationship_by_dimension["harmony"])
        expressivity_count = len(relationship_by_dimension["expressivity"])
        realism_count = len(relationship_by_dimension["realism"])
        
        log(f"Relationship rating complete. Generated ratings for {len(relationship_rating_results)} contact-substrate pairs.")
        log(f"Dimension breakdown: Harmony ({harmony_count}), Expressivity ({expressivity_count}), Realism ({realism_count})")
        log(f"Saved dimension-by-dimension results to: {relationship_by_dimension_path}")
        
        # Add to result
        result["relationship_rating"] = {
            "count": len(relationship_rating_results),
            "database_path": relationship_rating_output_path,
            "by_dimension_path": relationship_by_dimension_path,
            "rating_results": relationship_rating_results,
            "dimension_breakdown": {
                "harmony": harmony_count,
                "expressivity": expressivity_count,
                "realism": realism_count
            }
        }
    else:
        if not ENABLE_PROXY_MATCHING:
            log("Relationship rating skipped - proxy matching disabled via configuration")
            status_message = "Relationship rating skipped - proxy matching disabled via configuration"
            status = "proxy_disabled"
        elif not ENABLE_RELATIONSHIP_RATING:
            log("Relationship rating disabled via configuration")
            status_message = "Relationship rating disabled via configuration"
            status = "disabled"
        else:
            log("No data available for relationship rating")
            status_message = "No environment images or haptic annotation data provided"
            status = "no_data"
        
        result["relationship_rating"] = {
            "status": status,
            "message": status_message,
            "count": 0,
            "rating_results": []
        }
    
    # Process substrate utilization results if available
    if environment_image_base64_list and haptic_annotation_json and ENABLE_PROXY_MATCHING:
        log("Processing completed substrate utilization results")
        substrate_utilization_results = concurrent_results.get("substrate_utilization_result", [])
        
        # Correct object IDs before saving
        if substrate_utilization_results and physical_object_database:
            log("Correcting object IDs in substrate utilization results...")
            substrate_utilization_results = correct_object_ids_in_relationship_results(
                substrate_utilization_results, 
                physical_object_database
            )
            log("Object ID correction completed for substrate utilization")
        
        # Save substrate utilization results
        output_dir = os.path.join(script_dir, "output")
        substrate_utilization_output_path = os.path.join(output_dir, "substrate_utilization_results.json")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save substrate utilization results
        with open(substrate_utilization_output_path, 'w') as f:
            json.dump(substrate_utilization_results, f, indent=2)
        
        log(f"Substrate utilization method generation complete. Generated methods for {len(substrate_utilization_results)} contact-substrate pairs.")
        
        # Add to result
        result["substrate_utilization"] = {
            "count": len(substrate_utilization_results),
            "database_path": substrate_utilization_output_path,
            "utilization_results": substrate_utilization_results
        }
    else:
        # Handle disabled or unavailable substrate utilization
        if not ENABLE_PROXY_MATCHING:
            log("Substrate utilization skipped - proxy matching disabled via configuration")
            status_message = "Substrate utilization skipped - proxy matching disabled via configuration"
        else:
            log("No data available for substrate utilization")
            status_message = "No environment images or haptic annotation data provided"
        
        result["substrate_utilization"] = {
            "status": "disabled" if not ENABLE_PROXY_MATCHING else "no_data",
            "message": status_message,
            "count": 0,
            "utilization_results": []
        }
    
    # Print final result as JSON
    # print(json.dumps(result, indent=2))
        
except Exception as e:
    log(f"Error in processing: {e}")
    import traceback
    log(traceback.format_exc())
    print(json.dumps({"status": "error", "message": str(e)}))

# After loading environment variables and before any LangSmith-enabled LLMs are instantiated
# -----------------------------------------------------------------------------
# SAFETY SWITCH: Disable LangSmith tracing automatically if the request payload
# would exceed the 20-MB upload limit (e.g. many base-64 images).  This prevents
# connection errors like:
#   "content length of XXX bytes exceeds the maximum size limit of 20971520"
# -----------------------------------------------------------------------------
try:
    # Rough estimate – each base-64 char ~0.75 byte of binary data.
    img_chars = sum(len(s) for s in environment_image_base64_list) if 'environment_image_base64_list' in globals() else 0
    if img_chars * 0.75 > 15 * 1024 * 1024:  # >15 MB raw data → will blow past 20 MB when logged
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        log("Large image payload detected – LangSmith tracing disabled to avoid ingest limit")
except Exception as _trace_err:
    log(f"Tracing-safety check failed: {_trace_err}")

