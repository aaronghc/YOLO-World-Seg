import os
import sys
import json
import base64
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import asyncio
from typing import List, Dict, Any
from pydantic import SecretStr
from langsmith import traceable
import uuid
from datetime import datetime
import requests

# Add YOLO-World + EfficientSAM imports
import torch
import cv2
import numpy as np
from inference.models import YOLOWorld
from utils.efficient_sam import load, inference_with_boxes
import supervision as sv
from supervision.detection.core import Detections

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# YOLO-World + EfficientSAM configuration  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.0001
IOU_THRESHOLD = 0.5

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

# Helper function to retry async operations
async def retry_with_backoff(async_func, max_retries=3, base_delay=1.0, backoff_factor=2.0, timeout_seconds=300):
    """Retry async operations with exponential backoff"""
    return await async_func()

log("ProXeek Object Recognition Script started")

# Check if we're running from the server with parameters file
if len(sys.argv) > 1:
    # Running from server with parameters file
    log(f"Running with parameters file: {sys.argv[1]}")
    params_file = sys.argv[1]

    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        log(f"Loaded parameters")

        # Extract data from parameters
        environment_image_base64_list = params.get('environmentImageBase64List', [])

        log(f"Found {len(environment_image_base64_list)} environment images")
        
    except Exception as e:
        log(f"Error reading parameters file: {e}")
        environment_image_base64_list = []
else:
    # Default when running standalone
    log("No parameters file provided, using defaults")
    environment_image_base64_list = []

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

# Set up LangSmith tracing
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "ProXeek-ObjectRecognition"

if langchain_api_key:
    os.environ["LANGSMITH_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    log("LangSmith tracing enabled with API key and project: ProXeek-ObjectRecognition")
else:
    log("Warning: LangSmith API key not found - tracing may not work properly")

# Initialize the physical object recognition LLM
physical_object_recognition_llm = ChatOpenAI(
    model="o4-mini-2025-04-16",
    temperature=0.1,
    base_url="https://api.nuwaapi.com/v1",
    api_key=SecretStr(api_key) if api_key else None
)

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

For each image, create a detailed list of recognizable objects with the following information:

1. Its name with some details (e.g., "white cuboid airpods case")
2. Its position in the image (e.g., "bottom left of the image")

**AVOID INCLUDING:**
- Objects that are too far away or too small to identify reliably
- Objects that are heavily occluded or obscured
- Background elements that are not distinct objects

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

# Function to initialize YOLO-World and EfficientSAM models
def initialize_models():
    """Initialize YOLO-World and EfficientSAM models"""
    global yolo_world_model, efficient_sam_model
    
    if yolo_world_model is not None and efficient_sam_model is not None:
        return yolo_world_model, efficient_sam_model
    
    try:
        log("Initializing YOLO-World and EfficientSAM models...")
        
        # Clear GPU cache before initialization to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log("GPU cache cleared before model initialization")
        
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
                
                # Clear cache after YOLO-World initialization
                torch.cuda.empty_cache()
                
            except Exception as e:
                log(f"Warning: Failed to move YOLO-World model to GPU: {e}")
        
        # Initialize EfficientSAM model
        efficient_sam_model = load(device=DEVICE)
        
        # Final cache clear after all models are loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log(f"GPU memory after initialization: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB reserved")
        
        log(f"YOLO-World and EfficientSAM models initialized successfully on device: {DEVICE}")
        return yolo_world_model, efficient_sam_model
        
    except Exception as e:
        log(f"Error initializing models: {e}")
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        
        # Set classes and run YOLO-World detection with memory management
        log(f"Setting YOLO classes for image {image_id}: {len(object_names)} objects")
        yolo_model.set_classes(object_names)
        
        # Clear GPU cache before inference to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log(f"Running YOLO inference for image {image_id}")
        results = yolo_model.infer(image_rgb, confidence=CONFIDENCE_THRESHOLD)
        log(f"YOLO inference completed for image {image_id}")
        
        detections = Detections.from_inference(results)
        
        # Apply NMS
        detections = detections.with_nms(
            class_agnostic=False,
            threshold=IOU_THRESHOLD
        )
        
        # Run EfficientSAM segmentation on detected bounding boxes
        if len(detections.xyxy) > 0:
            log(f"Running EfficientSAM segmentation for image {image_id} on {len(detections.xyxy)} detections")
            detections.mask = inference_with_boxes(
                image=image_rgb,
                xyxy=detections.xyxy,
                model=sam_model,
                device=DEVICE
            )
            log(f"EfficientSAM segmentation completed for image {image_id}")
            
            # Clear GPU cache after segmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return segmentations
        
    except Exception as e:
        log(f"Error in YOLO segmentation for image {image_id}: {e}")
        
        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log(f"GPU cache cleared after error in image {image_id}")
            
        return []

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
        
        # Define the LLM call function for retry mechanism
        async def call_object_recognition_llm():
            log(f"Sending image {image_id} to object recognition model")
            response = await physical_object_recognition_llm.ainvoke(messages)
            log(f"Received response for image {image_id}")
            return response
        
        # Use retry mechanism with backoff for the LLM call
        try:
            response = await retry_with_backoff(
                call_object_recognition_llm,
                max_retries=3,
                base_delay=2.0,
                backoff_factor=2.0,
                timeout_seconds=120  # 2 minutes timeout per attempt
            )
        except Exception as e:
            log(f"All retry attempts failed for image {image_id}: {e}")
            return {"image_id": image_id, "objects": [], "status": "error", "error": f"Connection error after retries: {str(e)}"}
        
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
        
        # Extract object names for YOLO detection
        object_names = []
        for obj in objects:
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
        
        log(f"Running YOLO segmentation on image {image_id} for objects: {unique_names}")
        
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
            
            # SECOND PASS: ultra-low-threshold YOLO detection for still-unmatched objects
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
            
            # Save annotated image (delete old file first to ensure fresh timestamp)
            output_path = os.path.join(annotated_dir, f"annotated_image_{image_id}.jpg")
            if os.path.exists(output_path):
                os.remove(output_path)
            cv2.imwrite(output_path, image)
            annotated_image_paths.append(output_path)
            
            log(f"Saved annotated image {image_id} to {output_path}")
            
        except Exception as e:
            log(f"Error annotating image {image_id}: {e}")
            continue
    
    log(f"Exported {len(annotated_image_paths)} annotated images to {annotated_dir}")
    return annotated_image_paths

# Function to save object database to JSON file
def save_object_database(object_db: Dict[int, List[Dict]], output_path: str) -> str:
    try:
        # Convert image IDs to strings for JSON compatibility
        json_db = {}
        for img_id, obj_list in object_db.items():
            json_db[str(img_id)] = []
            for obj in obj_list:
                # Create a copy and remove heavy mask contours before save
                obj_copy = json.loads(json.dumps(obj))  # simple deep copy
                yolo_seg = obj_copy.get("yolo_segmentation")
                if yolo_seg and "mask_contours" in yolo_seg:
                    del yolo_seg["mask_contours"]
                json_db[str(img_id)].append(obj_copy)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(json_db, f, indent=2)

        log(f"Object database saved to {output_path}")
        return output_path
    except Exception as e:
        log(f"Error saving object database: {e}")
        return None

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

# Main execution
async def main():
    try:
        result = {
            "status": "success",
            "message": "Physical object recognition completed"
        }
        
        # Process environment images if provided
        if environment_image_base64_list:
            log(f"Processing {len(environment_image_base64_list)} environment images")
            
            # Step 1: Object recognition via LLM
            log("Step 1: Running object recognition...")
            physical_object_database = await process_multiple_images(environment_image_base64_list)
            log(f"Object recognition completed. Found {sum(len(objs) for objs in physical_object_database.values())} objects")
            
            # Step 2: YOLO segmentation (with user approval loop)
            log("Step 2: Running YOLO-World + EfficientSAM segmentation...")
            
            user_approved = False
            segmentation_attempt = 0
            output_dir = os.path.join(script_dir, "output")
            
            while not user_approved:
                segmentation_attempt += 1
                if segmentation_attempt > 1:
                    log(f"\n{'='*60}")
                    log(f"Re-running YOLO-World segmentation (Attempt #{segmentation_attempt})")
                    log(f"{'='*60}\n")
                else:
                    log("Enhancing physical object database with YOLO-World segmentation...")
                
                try:
                    physical_object_database = await enhance_with_yolo_segmentation(
                        physical_object_database, 
                        environment_image_base64_list
                    )
                    log("YOLO-World segmentation enhancement completed successfully")
                    
                    # Step 3: Export annotated images
                    log("Step 3: Exporting annotated images...")
                    annotated_image_paths = await export_annotated_images(
                        physical_object_database, 
                        environment_image_base64_list, 
                        output_dir
                    )
                    log(f"Exported {len(annotated_image_paths)} annotated images")
                    
                    # Display the paths of annotated images for user reference
                    log("\n" + "="*60)
                    log("ANNOTATED IMAGES SAVED TO:")
                    for img_path in annotated_image_paths:
                        log(f"  - {img_path}")
                    log("="*60 + "\n")
                    
                    # Ask user if they want to continue or rerun segmentation
                    log("\n" + "!"*60)
                    log("PLEASE EXAMINE THE ANNOTATED IMAGES IN THE OUTPUT FOLDER")
                    log("!"*60)
                    log("")  # Extra newline for spacing
                    
                    while True:
                        # Use print with flush to ensure prompt appears before input
                        print("\nDo you want to continue with these segmentation results? (yes/no): ", end='', flush=True)
                        user_input = input().strip().lower()
                        
                        if user_input in ['yes', 'y']:
                            log("User approved segmentation results. Continuing to next steps...")
                            user_approved = True
                            break
                        elif user_input in ['no', 'n']:
                            log("User requested to rerun segmentation. Restarting YOLO-World segmentation...")
                            break
                        else:
                            print("Invalid input. Please enter 'yes' or 'no'.", flush=True)
                    
                except Exception as e:
                    log(f"Error in YOLO-World enhancement: {e}")
                    log("Continuing with original object database without segmentation")
                    break  # Exit the loop on critical error
            
            # Step 4: Save physical object database (temporary, will be updated by Quest)
            log("Step 4: Saving initial physical object database...")
            physical_output_path = os.path.join(output_dir, "physical_object_database.json")
            saved_path = save_object_database(physical_object_database, physical_output_path)
            
            # Step 5: Send segmentation data to Quest
            log("Step 5: Sending segmentation data to Quest...")
            await send_segmentation_data_to_quest(physical_object_database)
            
            log("\n" + "="*60)
            log("WAITING FOR QUEST TO SEND BACK WORLD POSITIONS...")
            log("The Quest will process the segmentation data and send back")
            log("updated bounding boxes with world positions to the server.")
            log("The server will automatically save the final database.")
            log("="*60 + "\n")
            
            # Add to result
            total_physical_objects = sum(len(objects) for objects in physical_object_database.values())
            result["physical_objects"] = {
                "count": total_physical_objects,
                "database_path": saved_path,
                "annotated_images": annotated_image_paths
            }
        else:
            log("No environment images provided")
            result["status"] = "error"
            result["message"] = "No environment images provided"
        
        # Print final result
        print(json.dumps(result, indent=2))
        log("Object recognition script completed successfully")
        
    except Exception as e:
        log(f"Error in processing: {e}")
        import traceback
        log(traceback.format_exc())
        print(json.dumps({"status": "error", "message": str(e)}))

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

