import os
import json
import re
import time
import asyncio
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# Load environment variables
load_dotenv()

class ManualModelComparison:
    def __init__(self, include_mini=False):
        # Create separate clients for different API keys (sync and async)
        self.finetune_client = OpenAI(api_key=os.getenv('OPENAI_FINETUNE_API_KEY'))
        self.base_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url="https://api.nuwaapi.com/v1")
        
        # Async clients for concurrent requests
        self.async_finetune_client = AsyncOpenAI(api_key=os.getenv('OPENAI_FINETUNE_API_KEY'))
        self.async_base_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url="https://api.nuwaapi.com/v1")
        
        self.fine_tuned_model = "ft:gpt-4o-2024-08-06:mosra::C0WH6GHu"
        self.base_model_4o = "gpt-4o-2024-08-06"
        self.base_model_mini = "o4-mini"  # Default mini model name
        self.include_mini = include_mini
        
        # Check model availability if mini is enabled
        if self.include_mini:
            self.check_model_availability()
        
        self.results = {
            'fine_tuned': [],
            'base_4o': [],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'fine_tuned_model': self.fine_tuned_model,
                'base_model_4o': self.base_model_4o,
                'total_items': 0,
                'completed_items': 0
            }
        }
        
        # Add mini model to results if enabled
        if self.include_mini:
            self.results['base_mini'] = []
            self.results['metadata']['base_model_mini'] = self.base_model_mini
    
    def load_eval_dataset(self, limit=None):
        """Load evaluation dataset with optional limit for testing"""
        print("📂 Loading evaluation dataset...")
        
        script_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(script_dir, 'eval_dataset.jsonl')
        
        if not os.path.exists(dataset_path):
            print(f"❌ Evaluation dataset not found: {dataset_path}")
            return []
        
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                    
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        dataset.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping line {i+1}: Invalid JSON - {e}")
                        continue
        
        print(f"✅ Loaded {len(dataset)} evaluation items")
        return dataset
    
    def check_model_availability(self):
        """Check if the mini model is available"""
        print(f"🔍 Checking availability of {self.base_model_mini}...")
        try:
            # Try a simple test call
            test_messages = [
                {"role": "user", "content": "Hello, please respond with 'test'"}
            ]
            response, error = self.run_model_inference(self.base_model_mini, test_messages, max_retries=1)
            
            if error:
                print(f"❌ Model {self.base_model_mini} not available: {error}")
                print("💡 Trying alternative model names...")
                
                # Try alternative model names
                alternatives = ["o4-mini", "gpt-4o-mini-2024-08-06", "gpt-4o-mini-preview"]
                for alt_model in alternatives:
                    print(f"   Trying {alt_model}...")
                    self.base_model_mini = alt_model
                    response, error = self.run_model_inference(self.base_model_mini, test_messages, max_retries=1)
                    if not error and response:
                        print(f"✅ Found working model: {alt_model}")
                        break
                    else:
                        print(f"   ❌ {alt_model}: {error}")
                else:
                    print("❌ No working mini model found. Disabling mini model.")
                    self.include_mini = False
            else:
                print(f"✅ Model {self.base_model_mini} is available")
                
        except Exception as e:
            print(f"❌ Error checking model availability: {e}")
            print("❌ Disabling mini model.")
            self.include_mini = False
    
    def extract_rating_from_response(self, response_text):
        """Extract rating from model response using multiple patterns"""
        if not response_text:
            return None, "Empty response"
        
        # Clean up response text
        text = response_text.strip().lower()
        
        # Define rating patterns in order of preference
        rating_patterns = [
            (r'rating:\s*(\d+)', 'rating: X format'),
            (r'rating\s+(\d+)', 'rating X format'),
            (r'(\d+)/7', 'X/7 format'),
            (r'score:\s*(\d+)', 'score: X format'),
            (r'rate[d]?\s+(\d+)', 'rate/rated X format'),
            (r'(?:^|\n|\s)(\d+)(?:\s|$|\n)', 'standalone digit'),
            # Additional patterns for o4-mini responses
            (r'(\d+)\s*out\s*of\s*7', 'X out of 7 format'),
            (r'(\d+)\s*/\s*7', 'X/7 format (spaced)'),
            (r'(\d+)\s*points?', 'X points format'),
            (r'(\d+)\s*stars?', 'X stars format'),
        ]
        
        for pattern, pattern_name in rating_patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    rating = int(matches[0])
                    if 1 <= rating <= 7:
                        return rating, f"Extracted via {pattern_name}"
                    else:
                        continue  # Try next pattern
                except ValueError:
                    continue
        
        # Debug: Log the first few failed responses to understand the format
        if len(text) > 0 and not any(re.search(r'\b\d+\b', text)):
            return None, f"No digits found in: {response_text[:100]}..."
        elif len(text) > 0:
            # Find all digits in the response for debugging
            all_digits = re.findall(r'\d+', text)
            return None, f"Digits found but not in 1-7 range: {all_digits} in: {response_text[:100]}..."
        
        return None, f"Could not parse rating from: {response_text[:100]}..."
    
    def run_model_inference(self, model_name, messages, max_retries=3, initial_max_tokens=1000):
        """Run inference on a model with retry logic"""
        # Choose the appropriate client based on model
        if model_name == self.fine_tuned_model:
            client = self.finetune_client
        else:
            client = self.base_client
            
        max_tokens = initial_max_tokens

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,  # allow fallback to larger budget
                    temperature=0.1,   # Deterministic for evaluation
                )
                
                # Check if response content is None or empty
                content = response.choices[0].message.content
                if content is None or content.strip() == "":
                    # First time we get empty content, try once with more tokens
                    if attempt == 0 and max_tokens < 1500:
                        print(f"⚠️  Empty response from {model_name} with max_tokens={max_tokens}. Retrying with higher limit...")
                        max_tokens = 1500  # increase token budget
                        continue  # retry without counting as failure
                    return None, f"Empty response from {model_name}"
                
                return content, None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Attempt {attempt+1} failed for {model_name}: {e}")
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return None, str(e)
        
        return None, "Max retries exceeded"
    
    async def run_model_inference_async(self, model_name, messages, max_retries=3, initial_max_tokens=1000):
        """Run inference on a model asynchronously with retry logic"""
        # Choose the appropriate async client based on model
        if model_name == self.fine_tuned_model:
            client = self.async_finetune_client
        else:
            client = self.async_base_client
            
        max_tokens = initial_max_tokens

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,  # allow fallback to larger budget
                    temperature=0.1,   # Deterministic for evaluation
                )
                
                # Check if response content is None or empty
                content = response.choices[0].message.content
                if content is None or content.strip() == "":
                    # First time we get empty content, try once with more tokens
                    if attempt == 0 and max_tokens < 1500:
                        print(f"⚠️  Empty response from {model_name} with max_tokens={max_tokens}. Retrying with higher limit...")
                        max_tokens = 1500  # increase token budget
                        continue  # retry without counting as failure
                    return None, f"Empty response from {model_name}"
                
                return content, None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Attempt {attempt+1} failed for {model_name}: {e}")
                    print(f"   Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    return None, str(e)
        
        return None, "Max retries exceeded"
    
    def evaluate_single_item(self, item_data, item_index):
        """Evaluate a single item with both models"""
        item = item_data['item']
        
        # Prepare messages for the models
        messages = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": item['user_message']}
        ]
        
        results = {
            'item_index': item_index,
            'virtual_object': item['virtual_object'],
            'physical_object': item['physical_object'],
            'property_type': item['property_type'],
            'ground_truth_rating': item['ground_truth_rating'],
            'interaction_activity': item.get('interaction_activity', ''),
            'utilization_method': item.get('utilization_method', ''),
        }
        
        # Run fine-tuned model
        ft_response, ft_error = self.run_model_inference(self.fine_tuned_model, messages)
        ft_rating, ft_parse_info = self.extract_rating_from_response(ft_response)
        
        results['fine_tuned'] = {
            'raw_response': ft_response,
            'predicted_rating': ft_rating,
            'parse_info': ft_parse_info,
            'error': ft_error,
            'accuracy_score': self.calculate_accuracy_score(ft_rating, item['ground_truth_rating'])
        }
        
        # Run base model 4o
        base_4o_response, base_4o_error = self.run_model_inference(self.base_model_4o, messages)
        base_4o_rating, base_4o_parse_info = self.extract_rating_from_response(base_4o_response)
        
        results['base_4o'] = {
            'raw_response': base_4o_response,
            'predicted_rating': base_4o_rating,
            'parse_info': base_4o_parse_info,
            'error': base_4o_error,
            'accuracy_score': self.calculate_accuracy_score(base_4o_rating, item['ground_truth_rating'])
        }
        
        # Run base model mini (if enabled)
        if self.include_mini:
            base_mini_response, base_mini_error = self.run_model_inference(self.base_model_mini, messages)
            base_mini_rating, base_mini_parse_info = self.extract_rating_from_response(base_mini_response)
            
            # Enhanced debug: Log first few o4-mini responses to understand the format
            if item_index < 3:
                print(f"\n🔍 Debug o4-mini response {item_index}:")
                print(f"   Raw: {base_mini_response}")
                print(f"   Error: {base_mini_error}")
                print(f"   Parse info: {base_mini_parse_info}")
                if base_mini_response:
                    print(f"   Response length: {len(base_mini_response)}")
                    print(f"   Response type: {type(base_mini_response)}")
            
            results['base_mini'] = {
                'raw_response': base_mini_response,
                'predicted_rating': base_mini_rating,
                'parse_info': base_mini_parse_info,
                'error': base_mini_error,
                'accuracy_score': self.calculate_accuracy_score(base_mini_rating, item['ground_truth_rating'])
            }
        else:
            results['base_mini'] = {
                'raw_response': None,
                'predicted_rating': None,
                'parse_info': 'Model disabled',
                'error': None,
                'accuracy_score': 0.0
            }
        
        return results
    
    async def evaluate_single_item_async(self, item_data, item_index):
        """Evaluate a single item with all models concurrently"""
        item = item_data['item']
        
        # Prepare messages for the models
        messages = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": item['user_message']}
        ]
        
        results = {
            'item_index': item_index,
            'virtual_object': item['virtual_object'],
            'physical_object': item['physical_object'],
            'property_type': item['property_type'],
            'ground_truth_rating': item['ground_truth_rating'],
            'interaction_activity': item.get('interaction_activity', ''),
            'utilization_method': item.get('utilization_method', ''),
        }
        
        # Create concurrent tasks for all models
        tasks = []
        model_names = []
        
        # Fine-tuned model
        tasks.append(self.run_model_inference_async(self.fine_tuned_model, messages))
        model_names.append('fine_tuned')
        
        # Base 4o model
        tasks.append(self.run_model_inference_async(self.base_model_4o, messages))
        model_names.append('base_4o')
        
        # Base mini model (if enabled)
        if self.include_mini:
            tasks.append(self.run_model_inference_async(self.base_model_mini, messages))
            model_names.append('base_mini')
        
        # Run all model inferences concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, (response_data, model_key) in enumerate(zip(responses, model_names)):
            if isinstance(response_data, Exception):
                # Handle exceptions
                response, error = None, str(response_data)
            else:
                response, error = response_data
            
            rating, parse_info = self.extract_rating_from_response(response)
            
            results[model_key] = {
                'raw_response': response,
                'predicted_rating': rating,
                'parse_info': parse_info,
                'error': error,
                'accuracy_score': self.calculate_accuracy_score(rating, item['ground_truth_rating'])
            }
            
            # Enhanced debug for first few items
            if item_index < 3 and model_key == 'base_mini':
                print(f"\n🔍 Debug {model_key} response {item_index}:")
                print(f"   Raw: {response}")
                print(f"   Error: {error}")
                print(f"   Parse info: {parse_info}")
                if response:
                    print(f"   Response length: {len(response)}")
                    print(f"   Response type: {type(response)}")
        
        # Add empty base_mini result if not enabled
        if not self.include_mini:
            results['base_mini'] = {
                'raw_response': None,
                'predicted_rating': None,
                'parse_info': 'Model disabled',
                'error': None,
                'accuracy_score': 0.0
            }
        
        return results
    
    def calculate_accuracy_score(self, predicted, ground_truth):
        """Calculate accuracy score based on distance from ground truth"""
        if predicted is None or ground_truth is None:
            return 0.0
        
        distance = abs(predicted - ground_truth)
        
        if distance == 0:
            return 1.0      # Perfect match
        elif distance == 1:
            return 0.9      # Close (off by 1)
        elif distance == 2:
            return 0.5      # Somewhat close (off by 2)
        else:
            return 0.0      # Far off (off by 3+)
    
    def run_comparison(self, limit=None, save_interval=1000):
        """Run complete comparison between models"""
        print("🚀 Starting Manual Model Comparison")
        print("=" * 60)
        print(f"🎯 Fine-Tuned Model: {self.fine_tuned_model}")
        print(f"📈 Base Model 4o: {self.base_model_4o}")
        if self.include_mini:
            print(f"📱 Base Model o4-mini: {self.base_model_mini}")
        else:
            print(f"📱 Base Model o4-mini: Disabled")
        
        # Load dataset
        dataset = self.load_eval_dataset(limit)
        if not dataset:
            return
        
        self.results['metadata']['total_items'] = len(dataset)
        
        print(f"\n🔄 Processing {len(dataset)} evaluation items...")
        
        # Process each item with progress bar
        with tqdm(total=len(dataset), desc="Evaluating") as pbar:
            for i, item_data in enumerate(dataset):
                try:
                    result = self.evaluate_single_item(item_data, i)
                    self.results['fine_tuned'].append(result)
                    self.results['base_4o'].append(result)
                    if self.include_mini:
                        self.results['base_mini'].append(result)
                    self.results['metadata']['completed_items'] += 1
                    
                    if self.include_mini:
                        pbar.set_postfix({
                            'FT': result['fine_tuned']['predicted_rating'],
                            '4o': result['base_4o']['predicted_rating'],
                            'Mini': result['base_mini']['predicted_rating'],
                            'GT': result['ground_truth_rating']
                        })
                    else:
                        pbar.set_postfix({
                            'FT': result['fine_tuned']['predicted_rating'],
                            '4o': result['base_4o']['predicted_rating'],
                            'GT': result['ground_truth_rating']
                        })
                    pbar.update(1)
                    
                    # Save intermediate results
                    if (i + 1) % save_interval == 0:
                        self.save_results(f"comparison_results_partial_{i+1}.json")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"\n❌ Error processing item {i}: {e}")
                    pbar.update(1)
                    continue
        
        print(f"\n✅ Completed evaluation of {self.results['metadata']['completed_items']} items")
        
        # Save final results
        self.save_results("comparison_results_final.json")
        
        # Generate analysis
        self.analyze_results()
    
    async def run_comparison_async(self, limit=None, save_interval=1000, max_concurrent=10):
        """Run complete comparison between models with concurrent processing"""
        print("🚀 Starting Concurrent Model Comparison")
        print("=" * 60)
        print(f"🎯 Fine-Tuned Model: {self.fine_tuned_model}")
        print(f"📈 Base Model 4o: {self.base_model_4o}")
        if self.include_mini:
            print(f"📱 Base Model o4-mini: {self.base_model_mini}")
        else:
            print(f"📱 Base Model o4-mini: Disabled")
        print(f"⚡ Max concurrent requests: {max_concurrent}")
        
        # Load dataset
        dataset = self.load_eval_dataset(limit)
        if not dataset:
            return
        
        self.results['metadata']['total_items'] = len(dataset)
        
        print(f"\n🔄 Processing {len(dataset)} evaluation items concurrently...")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item_with_semaphore(item_data, item_index):
            async with semaphore:
                return await self.evaluate_single_item_async(item_data, item_index)
        
        # Process items with progress bar
        completed_items = 0
        with tqdm(total=len(dataset), desc="Evaluating") as pbar:
            # Process items in batches to avoid overwhelming the API
            batch_size = max_concurrent
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                
                # Create tasks for this batch
                tasks = [
                    process_item_with_semaphore(item_data, i + j)
                    for j, item_data in enumerate(batch)
                ]
                
                # Process batch concurrently
                try:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for j, result in enumerate(batch_results):
                        item_index = i + j
                        
                        if isinstance(result, Exception):
                            print(f"\n❌ Error processing item {item_index}: {result}")
                            pbar.update(1)
                            continue
                        
                        # Store results
                        self.results['fine_tuned'].append(result)
                        self.results['base_4o'].append(result)
                        if self.include_mini:
                            self.results['base_mini'].append(result)
                        
                        completed_items += 1
                        self.results['metadata']['completed_items'] = completed_items
                        
                        # Update progress bar
                        if self.include_mini:
                            pbar.set_postfix({
                                'FT': result['fine_tuned']['predicted_rating'],
                                '4o': result['base_4o']['predicted_rating'],
                                'Mini': result['base_mini']['predicted_rating'],
                                'GT': result['ground_truth_rating']
                            })
                        else:
                            pbar.set_postfix({
                                'FT': result['fine_tuned']['predicted_rating'],
                                '4o': result['base_4o']['predicted_rating'],
                                'GT': result['ground_truth_rating']
                            })
                        pbar.update(1)
                        
                        # Save intermediate results
                        if completed_items % save_interval == 0:
                            self.save_results(f"comparison_results_partial_{completed_items}.json")
                
                except Exception as e:
                    print(f"\n❌ Error processing batch starting at item {i}: {e}")
                    pbar.update(len(batch))
                    continue
                
                # Small delay between batches to be nice to the API
                await asyncio.sleep(0.1)
        
        print(f"\n✅ Completed evaluation of {completed_items} items")
        
        # Save final results
        self.save_results("comparison_results_final.json")
        
        # Generate analysis
        self.analyze_results()
    
    def save_results(self, filename):
        """Save results to JSON file"""
        script_dir = os.path.dirname(__file__)
        filepath = os.path.join(script_dir, 'output', filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filepath}")
    
    def analyze_results(self):
        """Analyze and display comprehensive comparison results"""
        if not self.results['fine_tuned']:
            print("❌ No results to analyze")
            return
        
        print("\n📊 COMPREHENSIVE MODEL COMPARISON ANALYSIS")
        print("=" * 80)
        
        # Overall statistics
        self.print_overall_stats()
        
        # Property dimension analysis
        self.analyze_by_property_dimension()
        
        # Error analysis
        self.analyze_errors()
        
        # Response pattern analysis
        self.analyze_response_patterns()
        
        # Head-to-head comparison
        self.head_to_head_analysis()
    
    def print_overall_stats(self):
        """Print overall performance statistics"""
        print("\n🎯 OVERALL PERFORMANCE")
        print("-" * 40)
        
        ft_scores = [r['fine_tuned']['accuracy_score'] for r in self.results['fine_tuned']]
        base_4o_scores = [r['base_4o']['accuracy_score'] for r in self.results['base_4o']]
        
        ft_mean = np.mean(ft_scores)
        ft_std = np.std(ft_scores)
        base_4o_mean = np.mean(base_4o_scores)
        base_4o_std = np.std(base_4o_scores)
        
        print(f"Fine-Tuned Model:")
        print(f"  Mean Accuracy: {ft_mean:.3f} ± {ft_std:.3f}")
        print(f"  Success Rate: {sum(1 for s in ft_scores if s > 0) / len(ft_scores):.1%}")
        
        print(f"\nBase Model 4o:")
        print(f"  Mean Accuracy: {base_4o_mean:.3f} ± {base_4o_std:.3f}")
        print(f"  Success Rate: {sum(1 for s in base_4o_scores if s > 0) / len(base_4o_scores):.1%}")
        
        if self.include_mini:
            base_mini_scores = [r['base_mini']['accuracy_score'] for r in self.results['base_mini']]
            base_mini_mean = np.mean(base_mini_scores)
            base_mini_std = np.std(base_mini_scores)
            
            print(f"\nBase Model o4-mini:")
            print(f"  Mean Accuracy: {base_mini_mean:.3f} ± {base_mini_std:.3f}")
            print(f"  Success Rate: {sum(1 for s in base_mini_scores if s > 0) / len(base_mini_scores):.1%}")
        
        improvement_4o = ft_mean - base_4o_mean
        
        print(f"\n🏆 Performance Differences:")
        print(f"  FT vs 4o: {improvement_4o:+.3f}")
        
        if self.include_mini:
            improvement_mini = ft_mean - base_mini_mean
            print(f"  FT vs o4-mini: {improvement_mini:+.3f}")
        
        print(f"\n📊 Model Rankings:")
        models = [
            ("Fine-Tuned", ft_mean),
            ("GPT-4o", base_4o_mean),
        ]
        
        if self.include_mini:
            models.append(("o4-mini", base_mini_mean))
        
        models.sort(key=lambda x: x[1], reverse=True)
        for i, (name, score) in enumerate(models, 1):
            print(f"  {i}. {name}: {score:.3f}")
        
        best_improvement = improvement_4o
        if self.include_mini:
            best_improvement = max(improvement_4o, improvement_mini)
            
        if best_improvement > 0.05:
            print("✅ Significant improvement from fine-tuning!")
        elif best_improvement > 0:
            print("🟡 Modest improvement from fine-tuning")
        elif best_improvement < -0.05:
            print("⚠️  Performance regression - investigate!")
        else:
            print("➡️  No significant difference")
    
    def analyze_by_property_dimension(self):
        """Analyze performance by property dimension"""
        print("\n📈 PERFORMANCE BY PROPERTY DIMENSION")
        print("-" * 80)
        
        # Group results by property type
        property_results = defaultdict(lambda: {'fine_tuned': [], 'base_4o': []})
        if self.include_mini:
            for prop_type in property_results:
                property_results[prop_type]['base_mini'] = []
        
        for result in self.results['fine_tuned']:
            prop_type = result['property_type']
            property_results[prop_type]['fine_tuned'].append(result['fine_tuned']['accuracy_score'])
            property_results[prop_type]['base_4o'].append(result['base_4o']['accuracy_score'])
            if self.include_mini:
                property_results[prop_type]['base_mini'].append(result['base_mini']['accuracy_score'])
        
        # Display results table
        if self.include_mini:
            print(f"{'Property':<15} {'FT':<8} {'4o':<8} {'Mini':<8} {'FT-4o':<10} {'FT-Mini':<10}")
            print("=" * 80)
        else:
            print(f"{'Property':<15} {'FT':<8} {'4o':<8} {'FT-4o':<10}")
            print("=" * 50)
        
        for prop_type in sorted(property_results.keys()):
            ft_scores = property_results[prop_type]['fine_tuned']
            base_4o_scores = property_results[prop_type]['base_4o']
            
            if ft_scores and base_4o_scores:
                ft_mean = np.mean(ft_scores)
                base_4o_mean = np.mean(base_4o_scores)
                diff_4o = ft_mean - base_4o_mean
                
                if self.include_mini:
                    base_mini_scores = property_results[prop_type]['base_mini']
                    if base_mini_scores:
                        base_mini_mean = np.mean(base_mini_scores)
                        diff_mini = ft_mean - base_mini_mean
                        print(f"{prop_type:<15} {ft_mean:<8.3f} {base_4o_mean:<8.3f} {base_mini_mean:<8.3f} {diff_4o:<+10.3f} {diff_mini:<+10.3f}")
                else:
                    print(f"{prop_type:<15} {ft_mean:<8.3f} {base_4o_mean:<8.3f} {diff_4o:<+10.3f}")
    
    def analyze_errors(self):
        """Analyze parsing errors and failures"""
        print("\n❌ ERROR ANALYSIS")
        print("-" * 40)
        
        ft_errors = sum(1 for r in self.results['fine_tuned'] if r['fine_tuned']['predicted_rating'] is None)
        base_4o_errors = sum(1 for r in self.results['base_4o'] if r['base_4o']['predicted_rating'] is None)
        total = len(self.results['fine_tuned'])
        
        print(f"Fine-Tuned Model Parse Failures: {ft_errors}/{total} ({ft_errors/total:.1%})")
        print(f"Base 4o Model Parse Failures: {base_4o_errors}/{total} ({base_4o_errors/total:.1%})")
        
        if self.include_mini:
            base_mini_errors = sum(1 for r in self.results['base_mini'] if r['base_mini']['predicted_rating'] is None)
            print(f"Base o4-mini Model Parse Failures: {base_mini_errors}/{total} ({base_mini_errors/total:.1%})")
        
        if ft_errors > 0 or base_4o_errors > 0 or (self.include_mini and base_mini_errors > 0):
            print(f"\n🔍 Sample Parse Failures:")
            count = 0
            for result in self.results['fine_tuned'][:10]:  # Show first 10
                has_error = (result['fine_tuned']['predicted_rating'] is None or 
                           result['base_4o']['predicted_rating'] is None or 
                           (self.include_mini and result['base_mini']['predicted_rating'] is None))
                if has_error:
                    print(f"  Item {result['item_index']}:")
                    if result['fine_tuned']['predicted_rating'] is None:
                        print(f"    FT Error: {result['fine_tuned']['parse_info']}")
                    if result['base_4o']['predicted_rating'] is None:
                        print(f"    4o Error: {result['base_4o']['parse_info']}")
                    if self.include_mini and result['base_mini']['predicted_rating'] is None:
                        print(f"    Mini Error: {result['base_mini']['parse_info']}")
                    count += 1
                    if count >= 3:  # Limit to 3 examples
                        break
    
    def analyze_response_patterns(self):
        """Analyze response format patterns"""
        print("\n📋 RESPONSE PATTERN ANALYSIS")
        print("-" * 40)
        
        ft_patterns = Counter()
        base_4o_patterns = Counter()
        
        for result in self.results['fine_tuned']:
            if result['fine_tuned']['predicted_rating'] is not None:
                ft_patterns[result['fine_tuned']['parse_info']] += 1
            if result['base_4o']['predicted_rating'] is not None:
                base_4o_patterns[result['base_4o']['parse_info']] += 1
        
        print("Fine-Tuned Model Response Patterns:")
        for pattern, count in ft_patterns.most_common():
            print(f"  {pattern}: {count}")
        
        print("\nBase 4o Model Response Patterns:")
        for pattern, count in base_4o_patterns.most_common():
            print(f"  {pattern}: {count}")
        
        if self.include_mini:
            base_mini_patterns = Counter()
            for result in self.results['fine_tuned']:
                if result['base_mini']['predicted_rating'] is not None:
                    base_mini_patterns[result['base_mini']['parse_info']] += 1
            
            print("\nBase o4-mini Model Response Patterns:")
            for pattern, count in base_mini_patterns.most_common():
                print(f"  {pattern}: {count}")
    
    def head_to_head_analysis(self):
        """Analyze head-to-head performance"""
        print("\n⚔️  HEAD-TO-HEAD COMPARISON")
        print("-" * 50)
        
        # Two-way comparison (always available)
        ft_vs_4o_wins = 0
        
        for result in self.results['fine_tuned']:
            ft_score = result['fine_tuned']['accuracy_score']
            base_4o_score = result['base_4o']['accuracy_score']
            
            # Two-way comparison
            if ft_score > base_4o_score:
                ft_vs_4o_wins += 1
        
        total = len(self.results['fine_tuned'])
        
        print("🥊 Fine-Tuned vs GPT-4o:")
        print(f"  FT > 4o: {ft_vs_4o_wins}/{total} ({ft_vs_4o_wins/total:.1%})")
        print(f"  4o > FT: {total - ft_vs_4o_wins}/{total} ({(total - ft_vs_4o_wins)/total:.1%})")
        
        if self.include_mini:
            # Three-way comparison
            ft_wins = 0
            base_4o_wins = 0
            base_mini_wins = 0
            ties = 0
            
            # Additional two-way comparison
            ft_vs_mini_wins = 0
            
            for result in self.results['fine_tuned']:
                ft_score = result['fine_tuned']['accuracy_score']
                base_4o_score = result['base_4o']['accuracy_score']
                base_mini_score = result['base_mini']['accuracy_score']
                
                # Three-way comparison
                max_score = max(ft_score, base_4o_score, base_mini_score)
                winners = []
                if ft_score == max_score:
                    winners.append('ft')
                if base_4o_score == max_score:
                    winners.append('4o')
                if base_mini_score == max_score:
                    winners.append('mini')
                
                if len(winners) == 1:
                    if 'ft' in winners:
                        ft_wins += 1
                    elif '4o' in winners:
                        base_4o_wins += 1
                    else:
                        base_mini_wins += 1
                else:
                    ties += 1
                
                # Additional two-way comparison
                if ft_score > base_mini_score:
                    ft_vs_mini_wins += 1
            
            print(f"\n🏆 Three-Way Comparison (Best Overall):")
            print(f"  Fine-Tuned Wins: {ft_wins}/{total} ({ft_wins/total:.1%})")
            print(f"  GPT-4o Wins: {base_4o_wins}/{total} ({base_4o_wins/total:.1%})")
            print(f"  o4-mini Wins: {base_mini_wins}/{total} ({base_mini_wins/total:.1%})")
            print(f"  Ties: {ties}/{total} ({ties/total:.1%})")
            
            print(f"\n🥊 Fine-Tuned vs o4-mini:")
            print(f"  FT > o4-mini: {ft_vs_mini_wins}/{total} ({ft_vs_mini_wins/total:.1%})")
            print(f"  o4-mini > FT: {total - ft_vs_mini_wins}/{total} ({(total - ft_vs_mini_wins)/total:.1%})")

async def main_async():
    """Async main function with options for different evaluation modes"""
    print("🎯 ProXeek Manual Model Comparison (Concurrent)")
    print("=" * 60)
    
    # Ask user about including o4-mini
    print("\n🔧 Model Configuration:")
    include_mini = input("Include o4-mini model in comparison? (y/n) [default: y]: ").strip().lower() or "y"
    include_mini = include_mini in ['y', 'yes', 'true', '1']
    
    # Initialize comparator
    comparator = ManualModelComparison(include_mini=include_mini)
    
    # Ask user for evaluation scope
    print("\n🔧 Evaluation Options:")
    print("1. Quick test (10 items)")
    print("2. Small evaluation (100 items)")
    print("3. Medium evaluation (500 items)")
    print("4. Full evaluation (all items)")
    
    choice = input("\nSelect option (1-4) [default: 1]: ").strip() or "1"
    
    limits = {"1": 10, "2": 100, "3": 500, "4": None}
    limit = limits.get(choice, 10)
    
    # Ask about concurrency level
    print("\n⚡ Concurrency Options:")
    print("1. Conservative (5 concurrent requests)")
    print("2. Moderate (10 concurrent requests)")
    print("3. Aggressive (20 concurrent requests)")
    
    concurrency_choice = input("\nSelect concurrency (1-3) [default: 2]: ").strip() or "2"
    concurrency_levels = {"1": 5, "2": 10, "3": 20}
    max_concurrent = concurrency_levels.get(concurrency_choice, 10)
    
    if limit:
        print(f"🎯 Running evaluation on {limit} items with {max_concurrent} concurrent requests...")
    else:
        print(f"🎯 Running full evaluation with {max_concurrent} concurrent requests...")
    
    # Run comparison
    try:
        await comparator.run_comparison_async(limit=limit, max_concurrent=max_concurrent)
        print("\n🎉 Evaluation completed successfully!")
        print("📁 Check the 'output' directory for detailed results")
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        print("💾 Partial results may be saved in output directory")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")

def main():
    """Main function with options for different evaluation modes"""
    print("🎯 ProXeek Manual Model Comparison")
    print("=" * 60)
    
    # Ask user about processing mode
    print("\n🔧 Processing Mode:")
    print("1. Sequential (slower, more stable)")
    print("2. Concurrent (faster, uses async)")
    
    mode_choice = input("\nSelect mode (1-2) [default: 2]: ").strip() or "2"
    
    if mode_choice == "2":
        # Run async version
        asyncio.run(main_async())
        return
    
    # Sequential mode (original)
    # Ask user about including o4-mini
    print("\n🔧 Model Configuration:")
    include_mini = input("Include o4-mini model in comparison? (y/n) [default: y]: ").strip().lower() or "y"
    include_mini = include_mini in ['y', 'yes', 'true', '1']
    
    # Initialize comparator
    comparator = ManualModelComparison(include_mini=include_mini)
    
    # Ask user for evaluation scope
    print("\n🔧 Evaluation Options:")
    print("1. Quick test (10 items)")
    print("2. Small evaluation (100 items)")
    print("3. Medium evaluation (500 items)")
    print("4. Full evaluation (all items)")
    
    choice = input("\nSelect option (1-4) [default: 1]: ").strip() or "1"
    
    limits = {"1": 10, "2": 100, "3": 500, "4": None}
    limit = limits.get(choice, 10)
    
    if limit:
        print(f"🎯 Running sequential evaluation on {limit} items...")
    else:
        print("🎯 Running full sequential evaluation...")
    
    # Run comparison
    try:
        comparator.run_comparison(limit=limit)
        print("\n🎉 Evaluation completed successfully!")
        print("📁 Check the 'output' directory for detailed results")
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        print("💾 Partial results may be saved in output directory")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main() 