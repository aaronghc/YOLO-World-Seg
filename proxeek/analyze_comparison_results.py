import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class ComparisonAnalyzer:
    def __init__(self, results_file="comparison_results_final.json"):
        self.results_file = results_file
        self.results = None
        self.df = None
        
    def load_results(self):
        """Load comparison results from JSON file"""
        script_dir = os.path.dirname(__file__)
        filepath = os.path.join(script_dir, 'output', self.results_file)
        
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
                'ft_accuracy_score': result['fine_tuned']['accuracy_score'],
                'ft_parse_success': result['fine_tuned']['predicted_rating'] is not None,
                
                # Base 4o model results
                'base_4o_predicted_rating': result['base_4o']['predicted_rating'],
                'base_4o_accuracy_score': result['base_4o']['accuracy_score'],
                'base_4o_parse_success': result['base_4o']['predicted_rating'] is not None,
            }
            
            # Add base_mini results if available
            if 'base_mini' in result:
                row['base_mini_predicted_rating'] = result['base_mini']['predicted_rating']
                row['base_mini_accuracy_score'] = result['base_mini']['accuracy_score']
                row['base_mini_parse_success'] = result['base_mini']['predicted_rating'] is not None
            else:
                row['base_mini_predicted_rating'] = None
                row['base_mini_accuracy_score'] = 0.0
                row['base_mini_parse_success'] = False
            
            # Calculate derived metrics
            if row['ft_predicted_rating'] is not None and row['ground_truth_rating'] is not None:
                row['ft_error'] = abs(row['ft_predicted_rating'] - row['ground_truth_rating'])
            else:
                row['ft_error'] = None
                
            if row['base_4o_predicted_rating'] is not None and row['ground_truth_rating'] is not None:
                row['base_4o_error'] = abs(row['base_4o_predicted_rating'] - row['ground_truth_rating'])
            else:
                row['base_4o_error'] = None
            
            if row['base_mini_predicted_rating'] is not None and row['ground_truth_rating'] is not None:
                row['base_mini_error'] = abs(row['base_mini_predicted_rating'] - row['ground_truth_rating'])
            else:
                row['base_mini_error'] = None
            
            # Winner determination (2-way or 3-way)
            ft_score = row['ft_accuracy_score']
            base_4o_score = row['base_4o_accuracy_score']
            base_mini_score = row['base_mini_accuracy_score']
            
            if row['base_mini_parse_success']:  # 3-way comparison
                max_score = max(ft_score, base_4o_score, base_mini_score)
                if ft_score == max_score:
                    row['winner'] = 'fine_tuned'
                elif base_4o_score == max_score:
                    row['winner'] = 'base_4o'
                else:
                    row['winner'] = 'base_mini'
            else:  # 2-way comparison
                if ft_score > base_4o_score:
                    row['winner'] = 'fine_tuned'
                elif base_4o_score > ft_score:
                    row['winner'] = 'base_4o'
                else:
                    row['winner'] = 'tie'
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        print(f"✅ Created DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")
        return True
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.df is None:
            return
        
        print("\n" + "="*80)
        print("📊 DETAILED COMPARISON ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        print("\n🎯 BASIC STATISTICS")
        print("-" * 30)
        print(f"Total evaluations: {len(self.df)}")
        print(f"Successful FT predictions: {self.df['ft_parse_success'].sum()}/{len(self.df)} ({self.df['ft_parse_success'].mean():.1%})")
        print(f"Successful Base 4o predictions: {self.df['base_4o_parse_success'].sum()}/{len(self.df)} ({self.df['base_4o_parse_success'].mean():.1%})")
        
        if self.df['base_mini_parse_success'].sum() > 0:
            print(f"Successful Base Mini predictions: {self.df['base_mini_parse_success'].sum()}/{len(self.df)} ({self.df['base_mini_parse_success'].mean():.1%})")
        
        # Performance by property type
        print("\n📈 PERFORMANCE BY PROPERTY TYPE")
        print("-" * 40)
        
        agg_columns = {
            'ft_accuracy_score': ['mean', 'std', 'count'],
            'base_4o_accuracy_score': ['mean', 'std', 'count']
        }
        
        if self.df['base_mini_parse_success'].sum() > 0:
            agg_columns['base_mini_accuracy_score'] = ['mean', 'std', 'count']
        
        property_stats = self.df.groupby('property_type').agg(agg_columns).round(3)
        print(property_stats)
        
        # Winner analysis
        print("\n⚔️  HEAD-TO-HEAD ANALYSIS")
        print("-" * 30)
        winner_counts = self.df['winner'].value_counts()
        total = len(self.df)
        
        for winner, count in winner_counts.items():
            percentage = count / total * 100
            print(f"{winner}: {count}/{total} ({percentage:.1f}%)")
        
        # Error distribution analysis
        print("\n❌ ERROR DISTRIBUTION")
        print("-" * 25)
        
        valid_df = self.df.dropna(subset=['ft_error', 'base_4o_error'])
        if len(valid_df) > 0:
            print("Fine-Tuned Model Error Distribution:")
            ft_error_dist = valid_df['ft_error'].value_counts().sort_index()
            for error, count in ft_error_dist.items():
                print(f"  Error {error}: {count} ({count/len(valid_df):.1%})")
            
            print("\nBase 4o Model Error Distribution:")
            base_4o_error_dist = valid_df['base_4o_error'].value_counts().sort_index()
            for error, count in base_4o_error_dist.items():
                print(f"  Error {error}: {count} ({count/len(valid_df):.1%})")
        
        if self.df['base_mini_parse_success'].sum() > 0:
            valid_mini_df = self.df.dropna(subset=['base_mini_error'])
            if len(valid_mini_df) > 0:
                print("\nBase Mini Model Error Distribution:")
                base_mini_error_dist = valid_mini_df['base_mini_error'].value_counts().sort_index()
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
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall Accuracy Comparison
        ax1 = plt.subplot(3, 3, 1)
        ft_mean = self.df['ft_accuracy_score'].mean()
        base_4o_mean = self.df['base_4o_accuracy_score'].mean()
        
        model_names = ['Fine-Tuned', 'GPT-4o']
        model_means = [ft_mean, base_4o_mean]
        colors = ['#2E86AB', '#A23B72']
        
        if self.df['base_mini_parse_success'].sum() > 0:
            base_mini_mean = self.df['base_mini_accuracy_score'].mean()
            model_names.append('o4-mini')
            model_means.append(base_mini_mean)
            colors.append('#F18F01')
        
        bars = ax1.bar(model_names, model_means, color=colors, alpha=0.8)
        ax1.set_ylabel('Mean Accuracy Score')
        ax1.set_title('Overall Model Performance')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, model_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Performance by Property Type
        ax2 = plt.subplot(3, 3, 2)
        property_columns = ['ft_accuracy_score', 'base_4o_accuracy_score']
        if self.df['base_mini_parse_success'].sum() > 0:
            property_columns.append('base_mini_accuracy_score')
        
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
        ax2.set_ylabel('Mean Accuracy Score')
        ax2.set_title('Performance by Property Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(property_means.index, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. Winner Distribution Pie Chart
        ax3 = plt.subplot(3, 3, 3)
        winner_counts = self.df['winner'].value_counts()
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        ax3.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax3.set_title('Head-to-Head Comparison')
        
        # 4. Accuracy Score Distribution
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(self.df['ft_accuracy_score'].dropna(), bins=20, alpha=0.7, 
                label='Fine-Tuned', color='#2E86AB', density=True)
        ax4.hist(self.df['base_4o_accuracy_score'].dropna(), bins=20, alpha=0.7, 
                label='GPT-4o', color='#A23B72', density=True)
        
        if self.df['base_mini_parse_success'].sum() > 0:
            ax4.hist(self.df['base_mini_accuracy_score'].dropna(), bins=20, alpha=0.7, 
                    label='o4-mini', color='#F18F01', density=True)
        
        ax4.set_xlabel('Accuracy Score')
        ax4.set_ylabel('Density')
        ax4.set_title('Accuracy Score Distribution')
        ax4.legend()
        
        # 5. Error Distribution
        ax5 = plt.subplot(3, 3, 5)
        valid_df = self.df.dropna(subset=['ft_error', 'base_4o_error'])
        if len(valid_df) > 0:
            error_range = range(0, int(max(valid_df['ft_error'].max(), valid_df['base_4o_error'].max())) + 1)
            ft_error_counts = [sum(valid_df['ft_error'] == e) for e in error_range]
            base_4o_error_counts = [sum(valid_df['base_4o_error'] == e) for e in error_range]
            
            x = np.arange(len(error_range))
            width = 0.35
            
            ax5.bar(x - width/2, ft_error_counts, width, label='Fine-Tuned', 
                   color='#2E86AB', alpha=0.8)
            ax5.bar(x + width/2, base_4o_error_counts, width, label='GPT-4o', 
                   color='#A23B72', alpha=0.8)
            
            ax5.set_xlabel('Prediction Error (|predicted - ground_truth|)')
            ax5.set_ylabel('Count')
            ax5.set_title('Error Distribution')
            ax5.set_xticks(x)
            ax5.set_xticklabels(error_range)
            ax5.legend()
        
        # 6. Correlation Plot
        ax6 = plt.subplot(3, 3, 6)
        valid_df = self.df.dropna(subset=['ft_accuracy_score', 'base_4o_accuracy_score'])
        if len(valid_df) > 0:
            ax6.scatter(valid_df['base_4o_accuracy_score'], valid_df['ft_accuracy_score'], 
                       alpha=0.6, color='#F18F01')
            ax6.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
            ax6.set_xlabel('GPT-4o Accuracy Score')
            ax6.set_ylabel('Fine-Tuned Model Accuracy Score')
            ax6.set_title('Model Performance Correlation')
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
        
        # 7. Parse Success Rate
        ax7 = plt.subplot(3, 3, 7)
        ft_success_rate = self.df['ft_parse_success'].mean()
        base_4o_success_rate = self.df['base_4o_parse_success'].mean()
        
        model_names = ['Fine-Tuned', 'GPT-4o']
        success_rates = [ft_success_rate, base_4o_success_rate]
        colors = ['#2E86AB', '#A23B72']
        
        if self.df['base_mini_parse_success'].sum() > 0:
            base_mini_success_rate = self.df['base_mini_parse_success'].mean()
            model_names.append('o4-mini')
            success_rates.append(base_mini_success_rate)
            colors.append('#F18F01')
        
        bars = ax7.bar(model_names, success_rates, color=colors, alpha=0.8)
        ax7.set_ylabel('Parse Success Rate')
        ax7.set_title('Response Parsing Success Rate')
        ax7.set_ylim(0, 1)
        
        for bar, value in zip(bars, success_rates):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Improvement Heatmap by Property
        ax8 = plt.subplot(3, 3, 8)
        improvement_by_property = self.df.groupby('property_type').apply(
            lambda x: x['ft_accuracy_score'].mean() - x['base_4o_accuracy_score'].mean()
        ).sort_values(ascending=False)
        
        colors = ['red' if x < 0 else 'green' for x in improvement_by_property.values]
        bars = ax8.barh(range(len(improvement_by_property)), improvement_by_property.values, 
                       color=colors, alpha=0.7)
        ax8.set_yticks(range(len(improvement_by_property)))
        ax8.set_yticklabels(improvement_by_property.index)
        ax8.set_xlabel('Accuracy Improvement (FT - GPT-4o)')
        ax8.set_title('Improvement by Property Type')
        ax8.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 9. Performance Consistency (Standard Deviation)
        ax9 = plt.subplot(3, 3, 9)
        ft_std = self.df.groupby('property_type')['ft_accuracy_score'].std()
        base_4o_std = self.df.groupby('property_type')['base_4o_accuracy_score'].std()
        
        x = np.arange(len(ft_std))
        width = 0.35
        
        ax9.bar(x - width/2, ft_std.values, width, label='Fine-Tuned', 
               color='#2E86AB', alpha=0.8)
        ax9.bar(x + width/2, base_4o_std.values, width, label='GPT-4o', 
               color='#A23B72', alpha=0.8)
        
        ax9.set_xlabel('Property Type')
        ax9.set_ylabel('Standard Deviation')
        ax9.set_title('Performance Consistency (Lower = More Consistent)')
        ax9.set_xticks(x)
        ax9.set_xticklabels(ft_std.index, rotation=45, ha='right')
        ax9.legend()
        
        plt.tight_layout()
        
        # Save the plot
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.join(script_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, 'model_comparison_analysis.png')
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
        
        csv_path = os.path.join(output_dir, 'detailed_comparison_results.csv')
        self.df.to_csv(csv_path, index=False)
        print(f"📋 Detailed results exported to {csv_path}")
    
    def find_interesting_cases(self):
        """Find interesting cases for manual inspection"""
        if self.df is None:
            return
        
        print("\n🔍 INTERESTING CASES FOR MANUAL INSPECTION")
        print("-" * 50)
        
        # Cases where fine-tuned model significantly outperformed base 4o model
        ft_much_better = self.df[
            (self.df['ft_accuracy_score'] - self.df['base_4o_accuracy_score']) >= 0.7
        ].head(5)
        
        if len(ft_much_better) > 0:
            print("\n✅ Cases where Fine-Tuned model significantly outperformed GPT-4o:")
            for _, row in ft_much_better.iterrows():
                print(f"  Item {row['item_index']}: {row['virtual_object']} → {row['physical_object']} ({row['property_type']})")
                print(f"    GT: {row['ground_truth_rating']}, FT: {row['ft_predicted_rating']}, 4o: {row['base_4o_predicted_rating']}")
        
        # Cases where base 4o model significantly outperformed fine-tuned model
        base_4o_much_better = self.df[
            (self.df['base_4o_accuracy_score'] - self.df['ft_accuracy_score']) >= 0.7
        ].head(5)
        
        if len(base_4o_much_better) > 0:
            print("\n⚠️  Cases where GPT-4o model significantly outperformed Fine-Tuned:")
            for _, row in base_4o_much_better.iterrows():
                print(f"  Item {row['item_index']}: {row['virtual_object']} → {row['physical_object']} ({row['property_type']})")
                print(f"    GT: {row['ground_truth_rating']}, FT: {row['ft_predicted_rating']}, 4o: {row['base_4o_predicted_rating']}")
        
        # Cases where both models failed to parse
        both_failed = self.df[
            (~self.df['ft_parse_success']) & (~self.df['base_4o_parse_success'])
        ].head(3)
        
        if len(both_failed) > 0:
            print("\n❌ Cases where both models failed to parse:")
            for _, row in both_failed.iterrows():
                print(f"  Item {row['item_index']}: {row['virtual_object']} → {row['physical_object']} ({row['property_type']})")
        
        # Add o4-mini analysis if available
        if self.df['base_mini_parse_success'].sum() > 0:
            # Cases where fine-tuned significantly outperformed o4-mini
            ft_vs_mini_better = self.df[
                (self.df['ft_accuracy_score'] - self.df['base_mini_accuracy_score']) >= 0.7
            ].head(3)
            
            if len(ft_vs_mini_better) > 0:
                print("\n✅ Cases where Fine-Tuned significantly outperformed o4-mini:")
                for _, row in ft_vs_mini_better.iterrows():
                    print(f"  Item {row['item_index']}: {row['virtual_object']} → {row['physical_object']} ({row['property_type']})")
                    print(f"    GT: {row['ground_truth_rating']}, FT: {row['ft_predicted_rating']}, Mini: {row['base_mini_predicted_rating']}")

def main():
    """Main analysis function"""
    print("📊 ProXeek Model Comparison Analysis")
    print("=" * 60)
    
    analyzer = ComparisonAnalyzer()
    
    # Load results
    if not analyzer.load_results():
        return
    
    # Create DataFrame
    if not analyzer.create_dataframe():
        return
    
    # Generate comprehensive analysis
    analyzer.generate_summary_report()
    analyzer.find_interesting_cases()
    
    # Create visualizations
    try:
        analyzer.create_visualizations()
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
        print("💡 Make sure matplotlib and seaborn are installed")
    
    # Export to CSV
    analyzer.export_detailed_csv()
    
    print(f"\n🎉 Analysis completed!")
    print(f"📁 Check the 'output' directory for all generated files")

if __name__ == "__main__":
    main() 