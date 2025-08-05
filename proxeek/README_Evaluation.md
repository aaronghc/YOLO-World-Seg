# ProXeek Model Evaluation System

This system evaluates and compares the performance of your fine-tuned model (`ft:gpt-4o-2024-08-06:mosra::C04ZOYAf`) against the base model (`gpt-4o-2024-08-06`) using OpenAI's Evals API.

## 🎯 Overview

The evaluation system:
- **Tests multiple items per virtual object** from `Results_feed_testing.csv`
- **Uses image snapshots** instead of text descriptions
- **Measures property rating accuracy** across 6 haptic dimensions
- **Compares models** using standardized evaluation criteria
- **Provides detailed performance analysis**

## 📁 Files

| File | Purpose |
|------|---------|
| `generate_eval_dataset.py` | Generate evaluation dataset from CSV with images |
| `create_model_eval.py` | Set up evals and create runs for both models |
| `monitor_eval.py` | Monitor evaluation progress and compare results |
| `README_Evaluation.md` | This documentation |

## 🚀 Quick Start

### 1. **Prepare Requirements**

Ensure you have the required image folders and testing CSV:
```
proxeek/
├── Virtual Objects/     # Virtual object snapshots
├── Physical Objects/    # Physical object snapshots  
├── output/
│   └── Results_feed_testing.csv  # Testing dataset
└── .env                 # Contains OPENAI_FINETUNE_API_KEY
```

### 2. **Generate Evaluation Dataset**

```bash
cd proxeek
python generate_eval_dataset.py
```

This will:
- ✅ Read `Results_feed_testing.csv`
- ✅ Select multiple items per virtual object (default: 3)
- ✅ Find and encode corresponding images
- ✅ Generate `eval_dataset.jsonl`

### 3. **Create and Run Evaluations**

```bash
python create_model_eval.py
```

This will:
- ✅ Create OpenAI eval configuration
- ✅ Upload test dataset
- ✅ Start evaluation runs for both models
- ✅ Save run info to `eval_run_info.json`

### 4. **Monitor Progress**

```bash
# Check status once
python monitor_eval.py

# Continuously monitor (check every 30 seconds)
python monitor_eval.py --watch

# Custom check interval
python monitor_eval.py --watch --interval 60
```

## 📊 Evaluation Metrics

### **Primary Metrics**
- **Rating Accuracy**: How close predicted ratings are to ground truth
- **Standard Deviation**: Consistency of predictions within each dimension

### **Scoring System**
- **Perfect match (distance = 0)**: 1.0 score
- **Off by 1**: 0.7 score  
- **Off by 2**: 0.3 score
- **Off by 3+**: 0.0 score

### **Properties Tested**
- **Inertia**: Weight distribution and movement resistance
- **Interactivity**: Interactive elements and degrees of freedom
- **Outline**: Size and shape compatibility  
- **Texture**: Surface characteristics and tactile properties
- **Hardness**: Material compliance and deformation
- **Temperature**: Thermal sensation and conductivity

## 🧪 Evaluation Process

### **Data Selection**
```python
# From Results_feed_testing.csv:
# - Group by virtual object (all unique objects)
# - Randomly select multiple rows per virtual object (default: 3)
# - Find corresponding image snapshots
# - Generate evaluation items for all valid properties
```

### **Evaluation Configuration**
```json
{
  "name": "ProXeek Property Rating Evaluation",
  "data_source_config": {
    "type": "custom",
    "item_schema": {
      "virtual_object": "string",
      "physical_object": "string", 
      "property_type": "string",
      "ground_truth_rating": "integer",
      "user_message": "array",
      "system_prompt": "string"
    }
  },
  "testing_criteria": [
    {"type": "python", "name": "Rating Accuracy"}
  ]
}
```

### **Model Prompting**
Each evaluation uses:
- **System Prompt**: Property-specific with detailed rubrics
- **User Message**: Multimodal (text + virtual image + physical image)
- **Expected Output**: `"rating: X"` where X is 1-7

## 📈 Results Analysis

### **Performance Comparison**
The system automatically compares:
- **Overall accuracy** (% of correct predictions)
- **Criteria-specific performance** (Rating Accuracy vs Format Check)
- **Per-property analysis** (if sufficient data)
- **Statistical significance** of improvements

### **Example Output**
```
🏆 Overall Performance Comparison:
   Fine-Tuned Accuracy: 75.00%
   Base Model Accuracy: 68.33%
   Improvement: +6.67%

📊 Detailed Dimension Analysis:

Property        FT Acc   FT StdDev  Base Acc  Base StdDev  Improvement 
================================================================================
Inertia         78.9%    0.245      71.1%     0.312        +7.8%       
Interactivity   82.4%    0.189      75.6%     0.289        +6.8%       
Outline         71.3%    0.267      63.8%     0.334        +7.5%       
Texture         73.7%    0.223      68.2%     0.298        +5.5%       
Hardness        76.5%    0.234      69.4%     0.278        +7.1%       
Temperature     67.9%    0.289      58.7%     0.356        +9.2%       

📈 Consistency Analysis:
   Fine-Tuned Avg StdDev: 0.238 (lower = more consistent)
   Base Model Avg StdDev: 0.311
   ✅ Fine-tuned model is more consistent (Δ: 0.073)

✅ Significant improvement! Fine-tuning was successful.
```

## 🔧 Configuration

### **Environment Variables**
```bash
# .env file
OPENAI_FINETUNE_API_KEY=your_finetune_api_key_here
```

### **Model Configuration**
Edit `create_model_eval.py` to change models:
```python
self.fine_tuned_model = "ft:gpt-4o-2024-08-06:mosra::C04ZOYAf"
self.base_model = "gpt-4o-2024-08-06"
```

### **Dataset Size**
Modify `generate_eval_dataset.py`:
```python
# Change items per virtual object (default: 3)
eval_dataset = generate_eval_dataset(items_per_virtual_object=5)
```

## 🐛 Troubleshooting

### **Common Issues**

**1. Missing Images**
```
⚠️  Row X: Virtual object image not found: ObjectName
```
**Solution**: Ensure image files exist in `Virtual Objects/` and `Physical Objects/` folders

**2. CSV Not Found**
```
❌ Testing CSV file not found at output/Results_feed_testing.csv
```
**Solution**: Verify the testing CSV file exists and has correct path

**3. API Authentication**
```
❌ Error creating evaluation: 401 Unauthorized
```
**Solution**: Check `OPENAI_FINETUNE_API_KEY` in `.env` file

**4. Invalid Model Name**
```
❌ Error creating eval run: Invalid model
```
**Solution**: Verify fine-tuned model ID is correct and accessible

### **Debug Commands**

```bash
# Test dataset generation
python generate_eval_dataset.py

# Check run status manually
python -c "
import json
with open('eval_run_info.json') as f:
    info = json.load(f)
print('Eval ID:', info['eval_id'])
print('Fine-tuned Run ID:', info['fine_tuned_run_id'])
print('Base Run ID:', info['base_run_id'])
"

# Monitor specific run
python monitor_eval.py --info-file custom_eval_run_info.json
```

## 📋 Expected Workflow

1. **Fine-tune your model** using VFT (if not done already)
2. **Prepare test data** (`Results_feed_testing.csv` + images)
3. **Generate evaluation dataset** (`generate_eval_dataset.py`)
4. **Create and run evaluations** (`create_model_eval.py`) 
5. **Monitor progress** (`monitor_eval.py --watch`)
6. **Analyze results** (automatic comparison when complete)
7. **Iterate** based on evaluation insights

## 🎉 Success Metrics

**Good Performance Indicators:**
- ✅ **Overall accuracy > 70%**
- ✅ **Fine-tuned > Base by 5%+**
- ✅ **Consistent improvement across dimensions**
- ✅ **Lower standard deviation (better consistency)**

**Areas for Improvement:**
- 🟡 **Fine-tuned accuracy < Base accuracy**
- 🟡 **Overall accuracy < 60%**
- 🟡 **High variance in specific dimensions**
- 🟡 **Inconsistent performance across properties**

This evaluation system provides comprehensive insights into your fine-tuned model's performance, helping you validate the effectiveness of your vision fine-tuning approach for haptic property rating in VR applications. 