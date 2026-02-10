"""
Main Analysis Script - TOPSIS for All NLP Tasks
This script runs TOPSIS analysis for all 5 NLP tasks and generates comprehensive results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from topsis import TOPSIS, create_sample_data
import json
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Define tasks based on roll number endings
TASKS = {
    'summarization': 'Text Summarization (Roll Numbers ending with 0 or 5)',
    'generation': 'Text Generation (Roll Numbers ending with 1 or 6)',
    'classification': 'Text Classification (Roll Numbers ending with 2 or 7)',
    'similarity': 'Text Sentence Similarity (Roll Numbers ending with 3 or 8)',
    'conversational': 'Text Conversational (Roll Numbers ending with 4 or 9)'
}

def create_visualizations(task_type, topsis_obj, results, output_dir):
    """
    Create comprehensive visualizations for TOPSIS results.
    
    Args:
        task_type: Type of NLP task
        topsis_obj: TOPSIS object with analysis results
        results: Results dataframe
        output_dir: Directory to save visualizations
    """
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. TOPSIS Scores Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    bars = ax1.barh(results['Model'], results['TOPSIS_Score'], color=colors)
    ax1.set_xlabel('TOPSIS Score', fontweight='bold')
    ax1.set_ylabel('Model', fontweight='bold')
    ax1.set_title(f'TOPSIS Scores - {task_type.title()}', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 2. Ranking Comparison
    ax2 = plt.subplot(2, 3, 2)
    rank_colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lightcoral']
    ax2.barh(results['Model'], results['Rank'], 
             color=rank_colors[:len(results)])
    ax2.set_xlabel('Rank (1 = Best)', fontweight='bold')
    ax2.set_ylabel('Model', fontweight='bold')
    ax2.set_title('Model Rankings', fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    ax2.invert_xaxis()
    
    # 3. Distance Comparison
    ax3 = plt.subplot(2, 3, 3)
    x = np.arange(len(results))
    width = 0.35
    ax3.bar(x - width/2, results['Distance_to_Best'], width, 
            label='Distance to Ideal Best', alpha=0.8, color='crimson')
    ax3.bar(x + width/2, results['Distance_to_Worst'], width, 
            label='Distance to Ideal Worst', alpha=0.8, color='forestgreen')
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Distance', fontweight='bold')
    ax3.set_title('Distance to Ideal Solutions', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(results['Model'], rotation=45, ha='right')
    ax3.legend()
    
    # 4. Criteria Heatmap (Original Data)
    ax4 = plt.subplot(2, 3, 4)
    sns.heatmap(topsis_obj.data, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Value'}, ax=ax4, linewidths=0.5)
    ax4.set_title('Original Decision Matrix', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Criteria', fontweight='bold')
    ax4.set_ylabel('Model', fontweight='bold')
    
    # 5. Weighted Normalized Matrix Heatmap
    ax5 = plt.subplot(2, 3, 5)
    sns.heatmap(topsis_obj.weighted, annot=True, fmt='.3f', cmap='coolwarm', 
                cbar_kws={'label': 'Weighted Value'}, ax=ax5, linewidths=0.5)
    ax5.set_title('Weighted Normalized Matrix', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Criteria', fontweight='bold')
    ax5.set_ylabel('Model', fontweight='bold')
    
    # 6. Criteria Weights Pie Chart
    ax6 = plt.subplot(2, 3, 6)
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(topsis_obj.weights)))
    wedges, texts, autotexts = ax6.pie(topsis_obj.weights, 
                                         labels=topsis_obj.criteria_names,
                                         autopct='%1.1f%%',
                                         colors=colors_pie,
                                         startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax6.set_title('Criteria Weights Distribution', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{output_dir}/{task_type}_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {filename}")
    plt.close()
    
    # Create additional comparison chart
    create_radar_chart(task_type, topsis_obj, results, output_dir)


def create_radar_chart(task_type, topsis_obj, results, output_dir):
    """Create radar chart for top 3 models."""
    from math import pi
    
    # Get top 3 models
    top_3 = results.head(3)
    
    # Number of criteria
    categories = topsis_obj.criteria_names
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Plot data for top 3 models
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (_, row) in enumerate(top_3.iterrows()):
        model_name = row['Model']
        # Get normalized values for this model
        values = topsis_obj.normalized.loc[model_name].values.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f"{model_name} (Rank {int(row['Rank'])})", 
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    
    # Set title and legend
    ax.set_ylim(0, 1)
    ax.set_title(f'Top 3 Models Comparison - {task_type.title()}', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    filename = f'{output_dir}/{task_type}_radar.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Radar chart saved: {filename}")
    plt.close()


def create_comparison_table(all_results):
    """Create a comparison table showing best model for each task."""
    
    comparison_data = []
    for task, results in all_results.items():
        best_model = results.iloc[0]
        comparison_data.append({
            'Task': TASKS[task],
            'Best Model': best_model['Model'],
            'TOPSIS Score': f"{best_model['TOPSIS_Score']:.4f}",
            'Rank': int(best_model['Rank'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Task', 'Best Model', 'TOPSIS Score'])
    
    for _, row in comparison_df.iterrows():
        table_data.append([row['Task'], row['Best Model'], row['TOPSIS Score']])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F7F7F7')
    
    plt.title('Best Pre-trained Model for Each NLP Task (TOPSIS Analysis)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('visualizations/overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Overall comparison saved: visualizations/overall_comparison.png")
    plt.close()
    
    return comparison_df


def save_results_to_csv(all_results, output_dir):
    """Save all results to CSV files."""
    
    for task, results in all_results.items():
        filename = f'{output_dir}/{task}_results.csv'
        results.to_csv(filename, index=False)
        print(f"✓ Results saved: {filename}")


def generate_report(all_results, topsis_objects):
    """Generate a comprehensive markdown report."""
    
    report = []
    report.append("# TOPSIS Analysis for Pre-trained NLP Model Selection\n")
    report.append("## Assignment Report\n")
    report.append(f"**Date:** February 10, 2026\n")
    report.append("**Method:** TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)\n\n")
    
    report.append("---\n\n")
    report.append("## Executive Summary\n\n")
    report.append("This report presents a comprehensive TOPSIS analysis to identify the best pre-trained NLP models ")
    report.append("for five different tasks: Text Summarization, Text Generation, Text Classification, ")
    report.append("Text Sentence Similarity, and Text Conversational AI.\n\n")
    
    report.append("### Key Findings\n\n")
    for task, results in all_results.items():
        best_model = results.iloc[0]
        report.append(f"- **{TASKS[task]}**: {best_model['Model']} (Score: {best_model['TOPSIS_Score']:.4f})\n")
    
    report.append("\n---\n\n")
    report.append("## Methodology\n\n")
    report.append("### TOPSIS Overview\n\n")
    report.append("TOPSIS is a multi-criteria decision-making method that:\n")
    report.append("1. Normalizes the decision matrix\n")
    report.append("2. Applies weights to criteria\n")
    report.append("3. Identifies ideal best and ideal worst solutions\n")
    report.append("4. Calculates distances from ideal solutions\n")
    report.append("5. Ranks alternatives based on relative closeness to ideal solution\n\n")
    
    report.append("### Models Evaluated\n\n")
    report.append("Six pre-trained models were evaluated across all tasks:\n")
    models = ['BERT', 'GPT-2', 'T5', 'RoBERTa', 'ALBERT', 'DistilBERT']
    for model in models:
        report.append(f"- {model}\n")
    
    report.append("\n---\n\n")
    
    # Detailed analysis for each task
    for task, results in all_results.items():
        topsis = topsis_objects[task]
        
        report.append(f"## {TASKS[task]}\n\n")
        
        report.append("### Evaluation Criteria\n\n")
        report.append("| Criterion | Weight | Impact |\n")
        report.append("|-----------|--------|--------|\n")
        for i, criterion in enumerate(topsis.criteria_names):
            impact_symbol = "📈 Beneficial" if topsis.impacts[i] == '+' else "📉 Non-beneficial"
            report.append(f"| {criterion} | {topsis.weights[i]:.2f} | {impact_symbol} |\n")
        
        report.append("\n### Results\n\n")
        report.append("| Rank | Model | TOPSIS Score | Distance to Best | Distance to Worst |\n")
        report.append("|------|-------|--------------|------------------|-------------------|\n")
        for _, row in results.iterrows():
            report.append(f"| {int(row['Rank'])} | {row['Model']} | {row['TOPSIS_Score']:.4f} | {row['Distance_to_Best']:.4f} | {row['Distance_to_Worst']:.4f} |\n")
        
        report.append("\n### Analysis\n\n")
        best = results.iloc[0]
        second = results.iloc[1]
        worst = results.iloc[-1]
        
        report.append(f"**Best Model:** {best['Model']} achieved the highest TOPSIS score of {best['TOPSIS_Score']:.4f}, ")
        report.append(f"indicating it provides the best overall balance across all evaluation criteria.\n\n")
        
        report.append(f"**Runner-up:** {second['Model']} (Score: {second['TOPSIS_Score']:.4f}) ")
        report.append(f"showed competitive performance.\n\n")
        
        report.append(f"**Least Suitable:** {worst['Model']} ranked last with a score of {worst['TOPSIS_Score']:.4f}.\n\n")
        
        report.append(f"![{task.title()} Analysis](visualizations/{task}_analysis.png)\n\n")
        report.append(f"![{task.title()} Radar Chart](visualizations/{task}_radar.png)\n\n")
        
        report.append("---\n\n")
    
    report.append("## Overall Comparison\n\n")
    report.append("![Overall Comparison](visualizations/overall_comparison.png)\n\n")
    
    report.append("## Conclusion\n\n")
    report.append("The TOPSIS analysis successfully identified optimal pre-trained models for each NLP task ")
    report.append("by considering multiple performance criteria and their relative importance. ")
    report.append("The results provide data-driven recommendations for model selection based on specific use cases.\n\n")
    
    report.append("### Recommendations\n\n")
    report.append("1. **Task-Specific Selection**: Choose models based on the specific NLP task requirements\n")
    report.append("2. **Trade-offs**: Consider the balance between performance metrics and resource constraints\n")
    report.append("3. **Validation**: Validate selected models on your specific dataset before deployment\n")
    report.append("4. **Monitoring**: Continuously monitor model performance in production\n\n")
    
    report.append("---\n\n")
    report.append("## References\n\n")
    report.append("- Hwang, C. L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications\n")
    report.append("- TOPSIS methodology for multi-criteria decision making\n")
    report.append("- Pre-trained NLP model benchmarks and evaluations\n")
    
    # Save report
    with open('README.md', 'w') as f:
        f.write(''.join(report))
    
    print("✓ Report generated: README.md")


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("TOPSIS ANALYSIS FOR PRE-TRAINED NLP MODEL SELECTION")
    print("="*80 + "\n")
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Store all results
    all_results = {}
    topsis_objects = {}
    
    # Run TOPSIS for each task
    for task_key, task_name in TASKS.items():
        print(f"\n{'='*80}")
        print(f"Analyzing: {task_name}")
        print('='*80)
        
        # Get data for this task
        data, weights, impacts = create_sample_data(task_key)
        
        # Run TOPSIS
        topsis = TOPSIS(data, weights, impacts)
        results = topsis.run()
        
        # Store results
        all_results[task_key] = results
        topsis_objects[task_key] = topsis
        
        # Print summary
        topsis.print_summary()
        
        # Create visualizations
        create_visualizations(task_key, topsis, results, 'visualizations')
    
    # Save all results to CSV
    save_results_to_csv(all_results, 'results')
    
    # Create overall comparison
    print(f"\n{'='*80}")
    print("Creating Overall Comparison")
    print('='*80)
    comparison_df = create_comparison_table(all_results)
    comparison_df.to_csv('results/overall_comparison.csv', index=False)
    print("✓ Overall comparison saved: results/overall_comparison.csv")
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print("Generating Comprehensive Report")
    print('='*80)
    generate_report(all_results, topsis_objects)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print('='*80)
    print("\nAll results have been saved to:")
    print("  📁 results/        - CSV files with detailed rankings")
    print("  📁 visualizations/ - PNG files with charts and graphs")
    print("  📄 README.md       - Comprehensive analysis report")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
