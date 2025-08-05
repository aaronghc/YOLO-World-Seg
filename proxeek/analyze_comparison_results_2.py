import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.stats import f_oneway

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

class ComparisonAnalyzer2:
    def __init__(self, results_file="comparison_results_final.json"):
        self.results_file = results_file
        self.results = None
        self.df = None
        self.property_types = ['inertia', 'interactivity', 'outline', 'hardness', 'texture', 'temperature']
        
    def load_results(self):
        """Load comparison results from JSON file"""
        script_dir = os.path.dirname(__file__)
        # Use the specific path for the comparison results
        filepath = os.path.join(script_dir, 'output', 'cached', 'Comparison_FT_4o_o4mini', self.results_file)
        
        if not os.path.exists(filepath):
            print(f"❌ Results file not found: {filepath}")
            print("💡 Run manual_model_comparison.py first to generate results")
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        print(f"✅ Loaded results from {filepath}")
        print(f"📊 Total items evaluated: {self.results['metadata']['completed_items']}")
        return True
    
    def create_dataframe(self):
        """Convert results to pandas DataFrame for analysis"""
        if not self.results:
            return False
        
        rows = []
        for result in self.results['fine_tuned']:
            # Base row data
            row = {
                'item_index': result['item_index'],
                'virtual_object': result['virtual_object'],
                'physical_object': result['physical_object'],
                'property_type': result['property_type'],
                'ground_truth_rating': result['ground_truth_rating'],
                'interaction_activity': result['interaction_activity'],
                'utilization_method': result['utilization_method'],
                
                # Fine-tuned model results
                'ft_predicted_rating': result['fine_tuned']['predicted_rating'],
                'ft_parse_success': result['fine_tuned']['predicted_rating'] is not None,
                
                # Base 4o model results
                'base_4o_predicted_rating': result['base_4o']['predicted_rating'],
                'base_4o_parse_success': result['base_4o']['predicted_rating'] is not None,
            }
            
            # Add base_mini results if available
            if 'base_mini' in result:
                row['base_mini_predicted_rating'] = result['base_mini']['predicted_rating']
                row['base_mini_parse_success'] = result['base_mini']['predicted_rating'] is not None
            else:
                row['base_mini_predicted_rating'] = None
                row['base_mini_parse_success'] = False
            
            # Calculate absolute errors for individual dimensions
            if row['ft_predicted_rating'] is not None and row['ground_truth_rating'] is not None:
                row['ft_abs_error'] = abs(row['ft_predicted_rating'] - row['ground_truth_rating'])
            else:
                row['ft_abs_error'] = None
                
            if row['base_4o_predicted_rating'] is not None and row['ground_truth_rating'] is not None:
                row['base_4o_abs_error'] = abs(row['base_4o_predicted_rating'] - row['ground_truth_rating'])
            else:
                row['base_4o_abs_error'] = None
            
            if row['base_mini_predicted_rating'] is not None and row['ground_truth_rating'] is not None:
                row['base_mini_abs_error'] = abs(row['base_mini_predicted_rating'] - row['ground_truth_rating'])
            else:
                row['base_mini_abs_error'] = None
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        print(f"✅ Created DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")
        return True
    
    def calculate_l2_performance(self):
        """Calculate L2 distance for overall model performance"""
        if self.df is None:
            return {}
        
        # Group by virtual-physical object pairs
        pairs = self.df.groupby(['virtual_object', 'physical_object'])
        
        model_l2_distances = {'fine_tuned': [], 'base_4o': [], 'base_mini': []}
        
        print(f"\n🔍 L2 Calculation Debug Info:")
        print(f"Total unique pairs: {len(pairs)}")
        
        valid_pairs_count = 0
        
        for (virtual_obj, physical_obj), group in pairs:
            # We need at least 2 dimensions to calculate meaningful L2 distance
            if len(group) < 2:
                continue
            
            valid_pairs_count += 1
            
            # Calculate L2 distance for each model
            for model_name in ['fine_tuned', 'base_4o', 'base_mini']:
                pred_col = f'{model_name.replace("fine_tuned", "ft")}_predicted_rating'
                
                # Get valid predictions for this model
                valid_predictions = group[group[pred_col].notna() & group['ground_truth_rating'].notna()]
                
                if len(valid_predictions) < 2:  # Need at least 2 dimensions
                    continue
                
                # Calculate L2 distance for available dimensions
                predicted_values = valid_predictions[pred_col].values
                ground_truth_values = valid_predictions['ground_truth_rating'].values
                
                l2_distance = np.sqrt(np.sum([(p - g)**2 for p, g in zip(predicted_values, ground_truth_values)]))
                model_l2_distances[model_name].append(l2_distance)
        
        print(f"Valid pairs processed: {valid_pairs_count}")
        print(f"Fine-tuned L2 calculations: {len(model_l2_distances['fine_tuned'])}")
        print(f"Base 4o L2 calculations: {len(model_l2_distances['base_4o'])}")
        print(f"Base mini L2 calculations: {len(model_l2_distances['base_mini'])}")
        
        # Calculate mean L2 distances
        mean_l2_distances = {}
        for model_name, distances in model_l2_distances.items():
            if distances:
                mean_l2_distances[model_name] = np.mean(distances)
                print(f"{model_name} mean L2: {mean_l2_distances[model_name]:.3f}")
            else:
                mean_l2_distances[model_name] = None
                print(f"{model_name} mean L2: No data")
        
        return mean_l2_distances
    
    def perform_statistical_tests_overall(self):
        """Perform statistical tests for overall model performance (L2 distances)"""
        if self.df is None:
            return {}
        
        print(f"\n📊 STATISTICAL ANALYSIS - OVERALL PERFORMANCE")
        print("-" * 60)
        
        # Group by virtual-physical object pairs and calculate L2 distances
        pairs = self.df.groupby(['virtual_object', 'physical_object'])
        
        l2_data_for_stats = []
        
        for (virtual_obj, physical_obj), group in pairs:
            if len(group) < 2:
                continue
            
            pair_id = f"{virtual_obj}_{physical_obj}"
            
            # Calculate L2 distance for each model
            for model_name in ['fine_tuned', 'base_4o', 'base_mini']:
                pred_col = f'{model_name.replace("fine_tuned", "ft")}_predicted_rating'
                
                valid_predictions = group[group[pred_col].notna() & group['ground_truth_rating'].notna()]
                
                if len(valid_predictions) < 2:
                    continue
                
                predicted_values = valid_predictions[pred_col].values
                ground_truth_values = valid_predictions['ground_truth_rating'].values
                
                l2_distance = np.sqrt(np.sum([(p - g)**2 for p, g in zip(predicted_values, ground_truth_values)]))
                
                l2_data_for_stats.append({
                    'pair_id': pair_id,
                    'model': model_name,
                    'l2_distance': l2_distance
                })
        
        if not l2_data_for_stats:
            print("❌ No data available for statistical testing")
            return {}
        
        stats_df = pd.DataFrame(l2_data_for_stats)
        
        # Check if we have data for all models
        models_with_data = stats_df['model'].unique()
        print(f"Models with data: {models_with_data}")
        print(f"Total L2 distance observations: {len(stats_df)}")
        
        results = {}
        
        if len(models_with_data) >= 2:
            # Perform one-way ANOVA (simplified approach)
            model_groups = [stats_df[stats_df['model'] == model]['l2_distance'].values 
                           for model in models_with_data]
            
            # Remove empty groups
            model_groups = [group for group in model_groups if len(group) > 0]
            
            if len(model_groups) >= 2:
                try:
                    f_stat, p_value = f_oneway(*model_groups)
                    results['anova'] = {'f_statistic': f_stat, 'p_value': p_value}
                    
                    print(f"One-way ANOVA Results:")
                    print(f"  F-statistic: {f_stat:.4f}")
                    print(f"  p-value: {p_value:.6f}")
                    
                    if p_value < 0.05:
                        print(f"  ✅ Significant difference detected (p < 0.05)")
                        
                        # Perform post-hoc tests
                        print(f"\n📈 Post-hoc Analysis:")
                        if HAS_STATSMODELS:
                            try:
                                tukey_result = pairwise_tukeyhsd(stats_df['l2_distance'], stats_df['model'])
                                results['tukey'] = str(tukey_result)
                                print(tukey_result)
                            except Exception as e:
                                print(f"⚠️  Tukey HSD failed: {e}")
                                print("📊 Performing pairwise t-tests instead:")
                                self._perform_pairwise_tests(stats_df, 'l2_distance', models_with_data)
                        else:
                            print("📊 Performing pairwise t-tests (install statsmodels for Tukey HSD):")
                            self._perform_pairwise_tests(stats_df, 'l2_distance', models_with_data)
                    else:
                        print(f"  ❌ No significant difference (p >= 0.05)")
                        
                except Exception as e:
                    print(f"❌ Error performing ANOVA: {e}")
            else:
                print("❌ Not enough groups for ANOVA")
        else:
            print("❌ Need at least 2 models for comparison")
        
        return results
    
    def perform_statistical_tests_by_property(self):
        """Perform statistical tests for each property type"""
        if self.df is None:
            return {}
        
        print(f"\n📊 STATISTICAL ANALYSIS - BY PROPERTY TYPE")
        print("-" * 60)
        
        # Get actual property types from data instead of predefined list
        actual_property_types = self.df['property_type'].unique()
        print(f"Actual property types in data: {actual_property_types}")
        
        results = {}
        
        for property_type in actual_property_types:
            print(f"\n🔍 Testing: {property_type}")
            print("-" * 30)
            
            prop_data = self.df[self.df['property_type'] == property_type].copy()
            
            if len(prop_data) == 0:
                print(f"❌ No data for {property_type}")
                continue
            
            # Prepare data for statistical testing
            test_data = []
            for _, row in prop_data.iterrows():
                pair_id = f"{row['virtual_object']}_{row['physical_object']}"
                
                # Add data for each model if available
                if pd.notna(row['ft_abs_error']):
                    test_data.append({
                        'pair_id': pair_id,
                        'model': 'fine_tuned',
                        'abs_error': row['ft_abs_error']
                    })
                
                if pd.notna(row['base_4o_abs_error']):
                    test_data.append({
                        'pair_id': pair_id,
                        'model': 'base_4o',
                        'abs_error': row['base_4o_abs_error']
                    })
                
                if pd.notna(row['base_mini_abs_error']):
                    test_data.append({
                        'pair_id': pair_id,
                        'model': 'base_mini',
                        'abs_error': row['base_mini_abs_error']
                    })
            
            if not test_data:
                print(f"❌ No valid error data for {property_type}")
                continue
            
            test_df = pd.DataFrame(test_data)
            models_with_data = test_df['model'].unique()
            
            print(f"Models with data: {models_with_data}")
            print(f"Total observations: {len(test_df)}")
            
            if len(models_with_data) >= 2:
                # Perform one-way ANOVA
                model_groups = [test_df[test_df['model'] == model]['abs_error'].values 
                               for model in models_with_data]
                
                # Remove empty groups
                model_groups = [group for group in model_groups if len(group) > 0]
                
                if len(model_groups) >= 2:
                    try:
                        f_stat, p_value = f_oneway(*model_groups)
                        results[property_type] = {'f_statistic': f_stat, 'p_value': p_value}
                        
                        print(f"  F-statistic: {f_stat:.4f}")
                        print(f"  p-value: {p_value:.6f}")
                        
                        if p_value < 0.05:
                            print(f"  ✅ Significant difference detected (p < 0.05)")
                            
                            # Perform post-hoc tests
                            print(f"  📈 Post-hoc Analysis:")
                            if HAS_STATSMODELS:
                                try:
                                    tukey_result = pairwise_tukeyhsd(test_df['abs_error'], test_df['model'])
                                    results[property_type]['tukey'] = str(tukey_result)
                                    print(f"  {tukey_result}")
                                except Exception as e:
                                    print(f"  ⚠️  Tukey HSD failed: {e}")
                                    print(f"  📊 Performing pairwise t-tests instead:")
                                    self._perform_pairwise_tests(test_df, 'abs_error', models_with_data, indent="  ")
                            else:
                                print(f"  📊 Performing pairwise t-tests (install statsmodels for Tukey HSD):")
                                self._perform_pairwise_tests(test_df, 'abs_error', models_with_data, indent="  ")
                        else:
                            print(f"  ❌ No significant difference (p >= 0.05)")
                            
                    except Exception as e:
                        print(f"❌ Error performing ANOVA for {property_type}: {e}")
                else:
                    print(f"❌ Not enough groups for ANOVA")
            else:
                print(f"❌ Need at least 2 models for comparison")
        
        return results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.df is None:
            return
        
        print("\n" + "="*80)
        print("📊 DETAILED COMPARISON ANALYSIS REPORT (L2 Distance Method)")
        print("="*80)
        
        # Basic statistics
        print("\n🎯 BASIC STATISTICS")
        print("-" * 30)
        print(f"Total evaluations: {len(self.df)}")
        print(f"Successful FT predictions: {self.df['ft_parse_success'].sum()}/{len(self.df)} ({self.df['ft_parse_success'].mean():.1%})")
        print(f"Successful Base 4o predictions: {self.df['base_4o_parse_success'].sum()}/{len(self.df)} ({self.df['base_4o_parse_success'].mean():.1%})")
        
        if self.df['base_mini_parse_success'].sum() > 0:
            print(f"Successful Base Mini predictions: {self.df['base_mini_parse_success'].sum()}/{len(self.df)} ({self.df['base_mini_parse_success'].mean():.1%})")
        
        # L2 Distance Performance
        print("\n📏 OVERALL MODEL PERFORMANCE (L2 Distance)")
        print("-" * 50)
        l2_performance = self.calculate_l2_performance()
        for model_name, l2_dist in l2_performance.items():
            if l2_dist is not None:
                model_display = model_name.replace('fine_tuned', 'Fine-Tuned').replace('base_4o', 'GPT-4o').replace('base_mini', 'o4-mini')
                print(f"{model_display}: {l2_dist:.3f} (lower is better)")
            else:
                model_display = model_name.replace('fine_tuned', 'Fine-Tuned').replace('base_4o', 'GPT-4o').replace('base_mini', 'o4-mini')
                print(f"{model_display}: No valid data")
        
        # Performance by property type (absolute error)
        print("\n📈 PERFORMANCE BY PROPERTY TYPE (Mean Absolute Error)")
        print("-" * 60)
        
        property_stats = self.df.groupby('property_type').agg({
            'ft_abs_error': ['mean', 'std', 'count'],
            'base_4o_abs_error': ['mean', 'std', 'count'],
            'base_mini_abs_error': ['mean', 'std', 'count']
        }).round(3)
        
        print(property_stats)
        
        # Error distribution analysis
        print("\n❌ ERROR DISTRIBUTION (Absolute Error)")
        print("-" * 40)
        
        valid_df = self.df.dropna(subset=['ft_abs_error', 'base_4o_abs_error'])
        if len(valid_df) > 0:
            print("Fine-Tuned Model Error Distribution:")
            ft_error_dist = valid_df['ft_abs_error'].value_counts().sort_index()
            for error, count in ft_error_dist.items():
                print(f"  Error {error}: {count} ({count/len(valid_df):.1%})")
            
            print("\nBase 4o Model Error Distribution:")
            base_4o_error_dist = valid_df['base_4o_abs_error'].value_counts().sort_index()
            for error, count in base_4o_error_dist.items():
                print(f"  Error {error}: {count} ({count/len(valid_df):.1%})")
        
        if self.df['base_mini_parse_success'].sum() > 0:
            valid_mini_df = self.df.dropna(subset=['base_mini_abs_error'])
            if len(valid_mini_df) > 0:
                print("\nBase Mini Model Error Distribution:")
                base_mini_error_dist = valid_mini_df['base_mini_abs_error'].value_counts().sort_index()
                for error, count in base_mini_error_dist.items():
                    print(f"  Error {error}: {count} ({count/len(valid_mini_df):.1%})")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.df is None:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Overall L2 Distance Performance
        ax1 = plt.subplot(2, 2, 1)
        l2_performance = self.calculate_l2_performance()
        
        model_names = []
        model_l2_distances = []
        colors = []
        
        if l2_performance['fine_tuned'] is not None:
            model_names.append('Fine-Tuned')
            model_l2_distances.append(l2_performance['fine_tuned'])
            colors.append('#2E86AB')
        
        if l2_performance['base_4o'] is not None:
            model_names.append('GPT-4o')
            model_l2_distances.append(l2_performance['base_4o'])
            colors.append('#A23B72')
        
        if l2_performance['base_mini'] is not None:
            model_names.append('o4-mini')
            model_l2_distances.append(l2_performance['base_mini'])
            colors.append('#F18F01')
        
        if model_names:
            bars = ax1.bar(model_names, model_l2_distances, color=colors, alpha=0.8)
            ax1.set_ylabel('Mean L2 Distance')
            ax1.set_title('Overall Model Performance (L2 Distance)\nLower is Better')
            
            # Add value labels on bars
            for bar, value in zip(bars, model_l2_distances):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            # If no L2 data available, show a message
            ax1.text(0.5, 0.5, 'No complete object pairs\nfor L2 calculation', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Overall Model Performance (L2 Distance)\nLower is Better')
            ax1.set_ylim(0, 1)
        
        # 2. Performance by Property Type (Mean Absolute Error)
        ax2 = plt.subplot(2, 2, 2)
        property_columns = ['ft_abs_error', 'base_4o_abs_error']
        if self.df['base_mini_parse_success'].sum() > 0:
            property_columns.append('base_mini_abs_error')
        
        property_means = self.df.groupby('property_type')[property_columns].mean()
        
        x = np.arange(len(property_means))
        width = 0.8 / len(property_columns)
        
        for i, col in enumerate(property_columns):
            offset = (i - len(property_columns)/2 + 0.5) * width
            color = ['#2E86AB', '#A23B72', '#F18F01'][i]
            label = ['Fine-Tuned', 'GPT-4o', 'o4-mini'][i]
            ax2.bar(x + offset, property_means[col], width, 
                   label=label, color=color, alpha=0.8)
        
        ax2.set_xlabel('Property Type')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Performance by Property Type\nLower is Better')
        ax2.set_xticks(x)
        ax2.set_xticklabels(property_means.index, rotation=45, ha='right')
        ax2.legend()
        
        # 3. Error Distribution
        ax3 = plt.subplot(2, 2, 3)
        valid_df = self.df.dropna(subset=['ft_abs_error', 'base_4o_abs_error'])
        if len(valid_df) > 0:
            max_error = max(valid_df['ft_abs_error'].max(), valid_df['base_4o_abs_error'].max())
            if self.df['base_mini_parse_success'].sum() > 0:
                max_error = max(max_error, valid_df['base_mini_abs_error'].max())
            
            error_range = range(0, int(max_error) + 1)
            ft_error_counts = [sum(valid_df['ft_abs_error'] == e) for e in error_range]
            base_4o_error_counts = [sum(valid_df['base_4o_abs_error'] == e) for e in error_range]
            
            x = np.arange(len(error_range))
            width = 0.8 / (3 if self.df['base_mini_parse_success'].sum() > 0 else 2)
            
            ax3.bar(x - width, ft_error_counts, width, label='Fine-Tuned', 
                   color='#2E86AB', alpha=0.8)
            ax3.bar(x, base_4o_error_counts, width, label='GPT-4o', 
                   color='#A23B72', alpha=0.8)
            
            if self.df['base_mini_parse_success'].sum() > 0:
                base_mini_error_counts = [sum(valid_df['base_mini_abs_error'] == e) for e in error_range]
                ax3.bar(x + width, base_mini_error_counts, width, label='o4-mini', 
                       color='#F18F01', alpha=0.8)
            
            ax3.set_xlabel('Absolute Error')
            ax3.set_ylabel('Count')
            ax3.set_title('Error Distribution')
            ax3.set_xticks(x)
            ax3.set_xticklabels(error_range)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No valid error data', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Error Distribution')
        
        # 4. Improvement by Property Type (Fine-Tuned vs GPT-4o)
        ax4 = plt.subplot(2, 2, 4)
        improvement_by_property = self.df.groupby('property_type').apply(
            lambda x: x['base_4o_abs_error'].mean() - x['ft_abs_error'].mean()  # Positive = FT is better
        ).sort_values(ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in improvement_by_property.values]
        bars = ax4.barh(range(len(improvement_by_property)), improvement_by_property.values, 
                       color=colors, alpha=0.7)
        ax4.set_yticks(range(len(improvement_by_property)))
        ax4.set_yticklabels(improvement_by_property.index)
        ax4.set_xlabel('Error Improvement (GPT-4o Error - FT Error)')
        ax4.set_title('Improvement by Property Type\nPositive = Fine-Tuned Better')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, improvement_by_property.values)):
            ax4.text(value + (0.02 if value >= 0 else -0.02), bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, 'model_comparison_analysis_l2.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {plot_path}")
        
        plt.show()
    
    def export_detailed_csv(self):
        """Export detailed results to CSV for further analysis"""
        if self.df is None:
            return
        
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, 'detailed_comparison_results_l2.csv')
        self.df.to_csv(csv_path, index=False)
        print(f"📋 Detailed results exported to {csv_path}")
    
    def find_interesting_cases(self):
        """Find interesting cases for manual inspection"""
        if self.df is None:
            return
        
        print("\n🔍 INTERESTING CASES FOR MANUAL INSPECTION")
        print("-" * 50)
        
        # Cases where fine-tuned model had much lower error than base 4o model
        ft_much_better = self.df[
            (self.df['base_4o_abs_error'] - self.df['ft_abs_error']) >= 2
        ].head(5)
        
        if len(ft_much_better) > 0:
            print("\n✅ Cases where Fine-Tuned model had much lower error than GPT-4o:")
            for _, row in ft_much_better.iterrows():
                print(f"  Item {row['item_index']}: {row['virtual_object']} → {row['physical_object']} ({row['property_type']})")
                print(f"    GT: {row['ground_truth_rating']}, FT Error: {row['ft_abs_error']}, 4o Error: {row['base_4o_abs_error']}")
        
        # Cases where base 4o model had much lower error than fine-tuned model
        base_4o_much_better = self.df[
            (self.df['ft_abs_error'] - self.df['base_4o_abs_error']) >= 2
        ].head(5)
        
        if len(base_4o_much_better) > 0:
            print("\n⚠️  Cases where GPT-4o model had much lower error than Fine-Tuned:")
            for _, row in base_4o_much_better.iterrows():
                print(f"  Item {row['item_index']}: {row['virtual_object']} → {row['physical_object']} ({row['property_type']})")
                print(f"    GT: {row['ground_truth_rating']}, FT Error: {row['ft_abs_error']}, 4o Error: {row['base_4o_abs_error']}")
        
        # Cases where both models failed to parse
        both_failed = self.df[
            (~self.df['ft_parse_success']) & (~self.df['base_4o_parse_success'])
        ].head(3)
        
        if len(both_failed) > 0:
            print("\n❌ Cases where both models failed to parse:")
            for _, row in both_failed.iterrows():
                print(f"  Item {row['item_index']}: {row['virtual_object']} → {row['physical_object']} ({row['property_type']})")

    def _perform_pairwise_tests(self, data_df, value_column, models, indent=""):
        """Perform pairwise t-tests between models"""
        from itertools import combinations
        
        for model1, model2 in combinations(models, 2):
            group1 = data_df[data_df['model'] == model1][value_column].values
            group2 = data_df[data_df['model'] == model2][value_column].values
            
            if len(group1) > 1 and len(group2) > 1:
                try:
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    significance = "✅ Significant" if p_val < 0.05 else "❌ Not significant"
                    print(f"{indent}  {model1} vs {model2}: t={t_stat:.3f}, p={p_val:.6f} {significance}")
                except Exception as e:
                    print(f"{indent}  {model1} vs {model2}: Error - {e}")

def main():
    """Main analysis function"""
    print("📊 ProXeek Model Comparison Analysis (L2 Distance Method)")
    print("=" * 70)
    
    analyzer = ComparisonAnalyzer2()
    
    # Load results
    if not analyzer.load_results():
        return
    
    # Create DataFrame
    if not analyzer.create_dataframe():
        return
    
    # Generate comprehensive analysis
    analyzer.generate_summary_report()
    analyzer.find_interesting_cases()
    
    # Perform statistical tests
    print("\n" + "="*80)
    print("📈 STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    overall_stats = analyzer.perform_statistical_tests_overall()
    property_stats = analyzer.perform_statistical_tests_by_property()
    
    # Create visualizations
    try:
        analyzer.create_visualizations()
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
        print("💡 Make sure matplotlib and seaborn are installed")
    
    # Export to CSV
    analyzer.export_detailed_csv()
    
    # Summary of statistical findings
    print("\n" + "="*80)
    print("📋 STATISTICAL TESTING SUMMARY")
    print("="*80)
    
    if overall_stats and 'anova' in overall_stats:
        p_val = overall_stats['anova']['p_value']
        print(f"🔍 Overall Performance (L2 Distance):")
        print(f"   ANOVA p-value: {p_val:.6f}")
        if p_val < 0.05:
            print(f"   ✅ Significant differences between models")
        else:
            print(f"   ❌ No significant differences between models")
    else:
        print(f"🔍 Overall Performance: No statistical test performed")
    
    print(f"\n🔍 Property Type Analysis:")
    # Use actual property types from the data instead of predefined list
    actual_property_types = analyzer.df['property_type'].unique() if analyzer.df is not None else []
    for prop in actual_property_types:
        if prop in property_stats and 'p_value' in property_stats[prop]:
            p_val = property_stats[prop]['p_value']
            print(f"   {prop}: p={p_val:.6f} {'✅ Significant' if p_val < 0.05 else '❌ Not significant'}")
        else:
            print(f"   {prop}: No test performed")
    
    print(f"\n💡 Note: Statistical significance tested at α = 0.05 level")
    if not HAS_STATSMODELS:
        print(f"⚠️  Install statsmodels for advanced post-hoc tests: pip install statsmodels")
    
    print(f"\n🎉 Analysis completed!")
    print(f"📁 Check the 'output' directory for all generated files")

if __name__ == "__main__":
    main() 