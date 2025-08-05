# Manual Model Comparison Guide

This guide explains how to manually evaluate and compare your fine-tuned model (`ft:gpt-4o-2024-08-06:mosra::C04ZOYAf`) against the base model (`gpt-4o-2024-08-06`) using your evaluation dataset.

## Overview

Instead of relying on OpenAI's Evals API, this approach gives you:
- ✅ **Full control** over the evaluation process
- ✅ **Detailed insights** into model responses and failures
- ✅ **Custom analysis** and visualization
- ✅ **No dependency** on OpenAI's evaluation infrastructure
- ✅ **Comprehensive reporting** by property dimension

## Scripts Overview

### 1. `manual_model_comparison.py`
Main evaluation script that runs both models on your evaluation dataset and generates comprehensive comparison results.

### 2. `analyze_comparison_results.py`
Analysis script that creates detailed reports and visualizations from the comparison results.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements_eval.txt
   ```

2. **Ensure API Key is Set**
   Make sure your `.env` file contains:
   ```
   OPENAI_FINETUNE_API_KEY=your_api_key_here
   ```

3. **Verify Evaluation Dataset**
   Ensure `eval_dataset.jsonl` exists in the proxeek directory. If not, generate it:
   ```bash
   python generate_eval_dataset.py
   ```

## Running the Evaluation

### Step 1: Run Model Comparison

```bash
python manual_model_comparison.py
```

**Options:**
- `1` - Quick test (10 items) - Great for testing
- `2` - Small evaluation (100 items) - Good for initial insights
- `3` - Medium evaluation (500 items) - Balanced approach
- `4` - Full evaluation (all items) - Complete analysis

**What it does:**
- Loads your evaluation dataset
- Runs both models on each evaluation item
- Compares predicted ratings with ground truth
- Saves detailed results to `output/comparison_results_final.json`
- Provides real-time analysis and statistics

### Step 2: Analyze Results

After the comparison completes, run the analysis:

```bash
python analyze_comparison_results.py
```

**What it generates:**
- Comprehensive text report with statistics
- 9-panel visualization showing various performance metrics
- CSV export for further analysis
- Identification of interesting cases

## Understanding the Results

### Performance Metrics

**Accuracy Score:** Based on prediction distance from ground truth:
- `1.0` - Perfect match (predicted = ground truth)
- `0.7` - Close (off by 1)
- `0.3` - Somewhat close (off by 2)  
- `0.0` - Far off (off by 3+) or parsing failure

### Key Comparisons

1. **Overall Performance**
   - Mean accuracy scores for both models
   - Success rates (percentage of valid predictions)
   - Statistical significance of improvements

2. **Property Dimension Analysis**
   - Performance breakdown by property type (Inertia, Texture, etc.)
   - Identifies which properties benefit most from fine-tuning
   
3. **Head-to-Head Comparison**
   - Win/loss/tie statistics
   - Cases where each model excels

4. **Error Analysis**
   - Parsing failure rates
   - Common error patterns
   - Response format consistency

## Output Files

All results are saved in the `output/` directory:

### Generated Files:
- `comparison_results_final.json` - Complete raw results
- `comparison_results_partial_*.json` - Intermediate saves during evaluation
- `model_comparison_analysis.png` - 9-panel visualization
- `detailed_comparison_results.csv` - Tabular data for Excel/analysis

## Interpreting Visualizations

The analysis generates a comprehensive 9-panel visualization:

1. **Overall Performance** - Bar chart comparing mean accuracy
2. **Performance by Property** - Property-specific comparison
3. **Head-to-Head** - Pie chart of wins/losses/ties
4. **Score Distribution** - Histogram of accuracy scores
5. **Error Distribution** - How far off predictions are
6. **Performance Correlation** - Scatter plot of model agreement
7. **Parse Success Rate** - Response formatting success
8. **Improvement by Property** - Which properties improved most
9. **Consistency Analysis** - Performance variability

## Best Practices

### 1. Start Small
- Begin with option 1 (10 items) to verify everything works
- Scale up to larger evaluations once confident

### 2. Monitor Rate Limits
- Script includes built-in rate limiting (0.1s delay between requests)
- For large evaluations, consider running during off-peak hours

### 3. Save Intermediate Results
- Script automatically saves partial results every 50 items
- Safe to interrupt and resume if needed

### 4. Review Interesting Cases
- Pay attention to cases where models significantly differ
- Manual inspection can reveal patterns in model behavior

## Troubleshooting

### Common Issues:

**"Results file not found"**
- Run `manual_model_comparison.py` first
- Check that evaluation completed successfully

**"Evaluation dataset not found"**
- Run `generate_eval_dataset.py` to create the dataset
- Verify `eval_dataset.jsonl` exists in the proxeek directory

**Visualization errors**
- Ensure matplotlib and seaborn are installed
- Check that results contain valid data

**API errors**
- Verify your API key has access to the fine-tuned model
- Check rate limits and quotas

### Rate Limiting:
If you encounter rate limit errors:
- Reduce evaluation size temporarily
- Increase delay between requests in the script
- Try again during off-peak hours

## Customizing the Evaluation

### Modify Accuracy Scoring:
Edit the `calculate_accuracy_score()` method in `manual_model_comparison.py` to use different scoring criteria.

### Add New Analysis:
Extend `analyze_comparison_results.py` to include additional metrics or visualizations specific to your needs.

### Filter Results:
Modify the evaluation to focus on specific property types or virtual/physical object combinations.

## Example Workflow

```bash
# 1. Start with a quick test
python manual_model_comparison.py
# Select option 1 (10 items)

# 2. Review results
python analyze_comparison_results.py

# 3. If satisfied, run larger evaluation
python manual_model_comparison.py
# Select option 3 (500 items)

# 4. Generate final analysis
python analyze_comparison_results.py
```

## Expected Runtime

Approximate times (depends on API response speed):
- 10 items: ~2-3 minutes
- 100 items: ~15-20 minutes  
- 500 items: ~1-2 hours
- Full dataset: ~3-6 hours (depends on dataset size)

## Next Steps

After evaluation:
1. Review the improvement metrics by property dimension
2. Identify areas where fine-tuning helped most
3. Consider additional training data for underperforming properties
4. Use insights to guide future fine-tuning iterations

---

🎯 **Pro Tip:** Run evaluations regularly during your model development cycle to track improvements and identify optimization opportunities! 