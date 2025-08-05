# ProXeek Property Rating VFT (Vision Fine-tuning)

This directory contains the **Vision Fine-Tuning (VFT)** setup for the `gpt-4o-2024-08-06` model for haptic property rating tasks. VFT allows the model to **actually see** the virtual and physical objects instead of relying on text descriptions.

## 🎯 Overview

The VFT system trains a **vision-capable model** to predict haptic property ratings (1-7 Likert scale) by **analyzing actual images** of virtual and physical objects. This represents a major leap forward from text-only approaches.

### **Training Approach:**
- **Input**: System prompt + User question + **Real images** of both objects
- **Output**: Assistant response with the correct rating
- **Learning**: Model learns visual-haptic correlations from image analysis

## 🖼️ Visual Data Integration

### **Image Sources:**
- **Virtual Objects**: `Virtual Objects/` folder (10 objects)
- **Physical Objects**: `Physical Objects/` folder (10 objects)  
- **Format**: PNG/JPEG images converted to base64 data URLs
- **Quality**: High-detail image analysis (`detail: "high"`)

### **Object Matching:**
The system automatically matches object names from your CSV data to image files:
- `Chest` → `Chest.png`
- `Game Controller` → `Game Controller.png`
- `Water bottle` → `Water bottle.png`
- Smart fuzzy matching handles spaces, underscores, etc.

## 🆚 VFT vs Previous Approaches

| **Aspect** | **VFT (Vision)** | **SFT (Text)** | **RFT (Reinforcement)** |
|------------|------------------|----------------|--------------------------|
| **Input** | Text + Images | Text Only | Text Only + Graders |
| **Model** | gpt-4o-2024-08-06 | o4-mini (blocked) | o4-mini (org verification) |
| **Visual Understanding** | ✅ Sees objects | ❌ Text descriptions | ❌ Text descriptions |
| **Accuracy Potential** | Highest | Medium | High (if accessible) |
| **Setup Complexity** | Medium | Simple | Complex |
| **Requirements** | Images + API key | Text + API key | Verification + Graders |

## 📁 Files

### **Core Scripts**
- `create_vft_job.py` - **Main script** (generates datasets + creates VFT job)
- `generate_vft_dataset.py` - **Dataset generator** with image processing
- `monitor_vft_job.py` - **Job monitoring** for vision fine-tuning
- `show_vft_sample.py` - **Data format viewer** (see image integration)

### **Configuration**
- `.env` - API configuration (OPENAI_FINETUNE_API_KEY)
- `requirements_rft.txt` - Python dependencies (includes base64 support)

## 🚀 Quick Start

### **1. Add API Key to .env**
Make sure your `.env` file contains:
```
OPENAI_FINETUNE_API_KEY=your_finetune_key_here
```

### **2. Verify Image Folders**
Ensure you have the image folders with your objects:
```
proxeek/
├── Virtual Objects/
│   ├── Chest.png
│   ├── Keyboard.png
│   ├── Soccer.png
│   └── ... (7 more)
└── Physical Objects/
    ├── Pillow.png
    ├── Game Controller.png
    ├── Water bottle.png
    └── ... (7 more)
```

### **3. Create VFT Fine-tuning Job**
```bash
cd YOLO-World-Seg/proxeek
python create_vft_job.py
```

This will:
- ✅ Generate datasets from Results_feed.csv + images
- ✅ Convert images to base64 data URLs
- ✅ Match virtual/physical objects to their images
- ✅ Split into training (~540 examples) and testing (60 examples)  
- ✅ Upload multimodal files to OpenAI API
- ✅ Create VFT job with gpt-4o-2024-08-06
- ✅ Save job details to `vft_job_info.json`

### **4. Monitor Progress**
```bash
# Check status once
python monitor_vft_job.py

# Continuous monitoring (updates every 30s)
python monitor_vft_job.py --watch
```

### **5. Preview Data Format (Optional)**
```bash
python show_vft_sample.py
```

## 📊 Data Format

### **VFT Training Example:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in haptic design who specializes in evaluating how well physical objects can serve as haptic proxies for virtual objects in VR. Your task is to evaluate how well each physical object can replicate the Inertia aspect..."
    },
    {
      "role": "user", 
      "content": [
        {
          "type": "text",
          "text": "Virtual Object: Chest\nPhysical Object: Pillow\n\nVirtual Object Interaction Activity: The user lifts and moves the chest...\n\nPlease rate how well this physical object can deliver the expected Inertia haptic feedback for the virtual object, based on the images and information provided."
        },
        {
          "type": "text",
          "text": "Virtual Object Image:"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
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
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
            "detail": "high"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": "Based on my analysis of the haptic properties from the visual characteristics, I rate this match as 4 on the 7-point Likert scale."
    }
  ]
}
```

## ⚙️ Configuration

### **Model Settings**
- **Base Model**: `gpt-4o-2024-08-06` (vision-capable)
- **Method**: Supervised Fine-tuning with images
- **API Endpoint**: `https://api.openai.com/v1`
- **Image Quality**: High-detail analysis

### **Hyperparameters**
```json
{
  "n_epochs": 3,
  "batch_size": 1, 
  "learning_rate_multiplier": 1.0
}
```

### **Image Processing**
- **Encoding**: Base64 data URLs
- **Formats**: PNG, JPEG, WEBP supported
- **Detail Level**: "high" for maximum visual fidelity
- **Size Limit**: 10MB per image (OpenAI limit)

## 📈 Expected Results

After successful training, you should see:
- **Improved visual understanding** of object properties
- **Better haptic-visual correlations** (texture, shape, size)
- **More accurate ratings** based on actual object appearance
- **Enhanced spatial reasoning** for VR proxy matching

## 🎯 Success Metrics

Your VFT job is successful if:
- ✅ All images successfully encoded and matched
- ✅ Training loss decreases steadily
- ✅ Model learns visual-haptic patterns
- ✅ Validation performance improves over epochs
- ✅ Visual content moderation passes (no faces/people)

## 🔧 Using Your Vision Fine-tuned Model

Once training completes, use your model ID with **image inputs**:

```python
import requests
import base64

# Encode your test images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

virtual_img = encode_image("test_virtual.png")
physical_img = encode_image("test_physical.png")

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {your_api_key}"},
    json={
        "model": "ft:gpt-4o-2024-08-06:your-org:your-suffix",
        "messages": [
            {"role": "system", "content": "You are an expert in haptic design..."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Rate this haptic match for Texture..."},
                    {"type": "text", "text": "Virtual Object:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{virtual_img}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": "Physical Object:"},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{physical_img}",
                            "detail": "high"  
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
)
```

## 🛠️ Troubleshooting

### **Common Issues**

1. **Missing images**
   ```
   ⚠️  Row 5: Virtual object image not found: Soccer Ball  
   ```
   **Solution**: Check image filenames match CSV object names

2. **Image encoding failed**
   ```
   ❌ Error encoding image Virtual Objects/Large_File.png: File too large
   ```
   **Solution**: Resize images to under 10MB

3. **No dataset generated**
   ```
   ❌ No dataset generated. Check your CSV file and images.
   ```
   **Solution**: Verify CSV file path and image folder structure

4. **Content moderation issues**
   ```
   ❌ Image contains faces/people and will be excluded
   ```
   **Solution**: Ensure object images don't contain people or faces

## 📊 Dataset Statistics

- **Total Examples**: ~600 (100 virtual-physical pairs × 6 properties)
- **Training Set**: ~540 examples **with images**
- **Testing Set**: 60 examples **with images** (10 per property dimension)
- **Properties**: Inertia, Interactivity, Outline, Texture, Hardness, Temperature
- **Images Per Example**: 2 (virtual + physical object)
- **Total Images**: ~1,200 (600 examples × 2 images each)

## 🔄 Integration with ProXeek

After fine-tuning:

1. **Update ProXeek.py** to use your vision fine-tuned model ID
2. **Add image capture** to your haptic evaluation pipeline
3. **Send both text prompts and object images** to the model
4. **Test with real VR scenarios** to verify visual-haptic improvements

## 🎉 Advantages of VFT

✅ **Actual visual understanding** (sees object shapes, textures, colors)  
✅ **Better haptic property assessment** (visual cues inform haptic ratings)  
✅ **No text description limitations** (model sees what you see)  
✅ **Multimodal learning** (combines visual and textual information)  
✅ **Spatial relationship understanding** (size, shape, proportion analysis)  
✅ **Future-proof approach** (vision models are the cutting edge)  

## 🚀 Why VFT is Superior

**Traditional approach**: *"The chest is a rectangular wooden container with metal hinges..."*

**VFT approach**: *Shows actual image of chest + pillow → Model sees texture, size, material properties, spatial relationships*

The model can now make **visual-haptic correlations** that were impossible with text alone!

Ready to create your vision-powered haptic property rating model! 🖼️✨ 