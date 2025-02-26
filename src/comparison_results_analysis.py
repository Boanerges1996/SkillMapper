import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

def load_results(file_path: str = "skill_extraction_results.csv") -> pd.DataFrame:
    """Load results from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file {file_path} not found. Run skill_extraction_comparison.py first.")
    
    df = pd.DataFrame(pd.read_csv(file_path))
    return df

def statistical_significance_tests(df: pd.DataFrame):
    """
    Perform statistical significance tests between models
    
    Args:
        df: DataFrame with results
    
    Returns:
        DataFrame with p-values for pairwise comparisons
    """
    # Create a model_shot column for analysis
    df['model_shot'] = df.apply(
        lambda row: f"{row['model']}" if row['shot_setting'] == 'N/A' 
        else f"{row['model']} ({row['shot_setting']}-shot)", axis=1
    )
    
    models = df['model_shot'].unique()
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'mrr']
    
    # Create a DataFrame to store p-values
    p_values = pd.DataFrame(index=pd.MultiIndex.from_product([metrics, models]), 
                           columns=models)
    
    # Perform t-tests for each metric and model pair
    for metric in metrics:
        for model1 in models:
            for model2 in models:
                if model1 != model2:
                    # Get values for model1 and model2
                    values1 = df[df['model_shot'] == model1][metric].values
                    values2 = df[df['model_shot'] == model2][metric].values
                    
                    # Perform t-test
                    t_stat, p_val = ttest_ind(values1, values2, equal_var=False)
                    p_values.loc[(metric, model1), model2] = p_val
    
    return p_values

def create_detailed_visualizations(df: pd.DataFrame, output_dir: str = "analysis_output"):
    """
    Create detailed visualizations of the results
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create a model_shot column for easier plotting
    df['model_shot'] = df.apply(
        lambda row: f"{row['model']}" if row['shot_setting'] == 'N/A' 
        else f"{row['model']} ({row['shot_setting']}-shot)", axis=1
    )
    
    # 1. Performance by Model and Data Type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 Score
    sns.barplot(x='model_shot', y='f1', hue='data_type', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('F1 Score by Model and Data Type')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Precision
    sns.barplot(x='model_shot', y='precision', hue='data_type', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Precision by Model and Data Type')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    
    # Recall
    sns.barplot(x='model_shot', y='recall', hue='data_type', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Recall by Model and Data Type')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # MRR
    sns.barplot(x='model_shot', y='mrr', hue='data_type', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Mean Reciprocal Rank (MRR) by Model and Data Type')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_by_model_data_type.png"), dpi=300)
    
    # 2. Performance across Models (aggregated across data types)
    plt.figure(figsize=(14, 8))
    
    # Reshape data for plotting
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'mrr']
    model_avg = df.groupby('model_shot')[metrics].mean().reset_index()
    model_avg_melted = pd.melt(model_avg, id_vars=['model_shot'], value_vars=metrics, 
                              var_name='Metric', value_name='Value')
    
    # Create plot
    sns.barplot(x='model_shot', y='Value', hue='Metric', data=model_avg_melted)
    plt.title('Average Performance Metrics by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_performance_by_model.png"), dpi=300)
    
    # 3. Performance by Data Type (aggregated across models)
    plt.figure(figsize=(14, 8))
    
    # Reshape data for plotting
    data_type_avg = df.groupby('data_type')[metrics].mean().reset_index()
    data_type_avg_melted = pd.melt(data_type_avg, id_vars=['data_type'], value_vars=metrics, 
                                  var_name='Metric', value_name='Value')
    
    # Create plot
    sns.barplot(x='data_type', y='Value', hue='Metric', data=data_type_avg_melted)
    plt.title('Average Performance Metrics by Data Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_performance_by_data_type.png"), dpi=300)
    
    # 4. Trade-off between Precision and Recall
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with model as hue
    scatter = sns.scatterplot(x='precision', y='recall', hue='model_shot', style='data_type', 
                             size='f1', sizes=(50, 200), data=df)
    
    # Add labels for each point
    for i, row in df.iterrows():
        plt.text(row['precision'], row['recall'], f"{row['model'][:3]}-{row['data_type'][:2]}", 
                fontsize=8, ha='center', va='center')
    
    plt.title('Precision-Recall Trade-off by Model and Data Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_tradeoff.png"), dpi=300)
    
    # 5. Extraction Time Comparison
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    sns.barplot(x='model_shot', y='avg_extraction_time', hue='data_type', data=df)
    plt.title('Average Extraction Time by Model and Data Type')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "extraction_time_comparison.png"), dpi=300)
    
    # 6. Radar Chart for Model Comparison
    plt.figure(figsize=(10, 10))
    
    # Prepare data
    model_avg = df.groupby('model_shot')[metrics].mean()
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable + add labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for model_name in model_avg.index:
        values = model_avg.loc[model_name].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Comparison - Radar Chart')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_radar.png"), dpi=300)
    
    # 7. Zero-shot vs Few-shot Comparison (only for LLM models)
    llm_results = df[df['shot_setting'] != 'N/A']
    
    if not llm_results.empty:
        plt.figure(figsize=(16, 10))
        
        # Reshape data for plotting
        metrics = ['precision', 'recall', 'f1', 'mrr']
        
        # Create a plot with subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            sns.barplot(x='model', y=metric, hue='shot_setting', data=llm_results, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} by Model and Shot Setting')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "zero_vs_few_shot_comparison.png"), dpi=300)
        
        # 8. Performance differences between zero-shot and few-shot
        # Calculate the difference in performance (few-shot - zero-shot)
        shot_pivot = llm_results.pivot_table(
            index=['model', 'data_type'], 
            columns='shot_setting', 
            values=['precision', 'recall', 'f1', 'mrr']
        )
        
        # Calculate differences
        diff_df = pd.DataFrame()
        for metric in metrics:
            try:
                diff_df[f'{metric}_diff'] = shot_pivot[(metric, 'few')] - shot_pivot[(metric, 'zero')]
            except KeyError:
                # In case one of the shot settings is missing
                continue
        
        # Reset index for plotting
        diff_df = diff_df.reset_index()
        
        # Melt for easier plotting
        diff_melted = pd.melt(
            diff_df, 
            id_vars=['model', 'data_type'], 
            var_name='Metric', 
            value_name='Difference (Few-shot - Zero-shot)'
        )
        
        # Clean up metric names
        diff_melted['Metric'] = diff_melted['Metric'].str.replace('_diff', '')
        
        # Create plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='model', y='Difference (Few-shot - Zero-shot)', hue='Metric', data=diff_melted)
        plt.title('Performance Difference: Few-shot vs. Zero-shot')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "zero_vs_few_shot_diff.png"), dpi=300)
        
        # 9. Statistical significance of zero-shot vs few-shot differences
        # Perform t-tests for each model and metric
        pvalues = {}
        for model in llm_results['model'].unique():
            pvalues[model] = {}
            model_data = llm_results[llm_results['model'] == model]
            
            for metric in metrics:
                zero_shot = model_data[model_data['shot_setting'] == 'zero'][metric].values
                few_shot = model_data[model_data['shot_setting'] == 'few'][metric].values
                
                if len(zero_shot) > 0 and len(few_shot) > 0:
                    _, p_val = ttest_ind(zero_shot, few_shot, equal_var=False)
                    pvalues[model][metric] = p_val
        
        # Convert to DataFrame for easier viewing
        pvalues_df = pd.DataFrame(pvalues).T
        pvalues_df.to_csv(os.path.join(output_dir, "zero_vs_few_shot_significance.csv"))

def generate_research_findings(df: pd.DataFrame, output_file: str = "analysis_output/research_findings.md"):
    """
    Generate a markdown report with research findings
    
    Args:
        df: DataFrame with results
        output_file: Output file path
    """
    # Handle case where only BERT model is present (no zero/few-shot comparison)
    bert_only = len(df['model'].unique()) == 1 and 'ModernBERT' in df['model'].unique()
    # Create a model_shot column for analysis
    df['model_shot'] = df.apply(
        lambda row: f"{row['model']}" if row['shot_setting'] == 'N/A' 
        else f"{row['model']} ({row['shot_setting']}-shot)", axis=1
    )
    
    # Calculate aggregated metrics
    model_avg = df.groupby('model_shot')[['precision', 'recall', 'f1', 'accuracy', 'mrr']].mean()
    data_type_avg = df.groupby('data_type')[['precision', 'recall', 'f1', 'accuracy', 'mrr']].mean()
    model_data_type_avg = df.groupby(['model_shot', 'data_type'])[['precision', 'recall', 'f1', 'accuracy', 'mrr']].mean()
    
    # Find best model overall
    best_model_f1 = model_avg['f1'].idxmax()
    best_model_precision = model_avg['precision'].idxmax()
    best_model_recall = model_avg['recall'].idxmax()
    best_model_mrr = model_avg['mrr'].idxmax()
    
    # Find best model for each data type
    best_model_by_data_type = {}
    for data_type in df['data_type'].unique():
        data_type_df = df[df['data_type'] == data_type]
        best_model = data_type_df.groupby('model_shot')['f1'].mean().idxmax()
        best_model_by_data_type[data_type] = best_model
    
    # Analyze zero-shot vs few-shot performance
    llm_results = df[df['shot_setting'] != 'N/A']
    zero_vs_few_analysis = {}
    
    if not llm_results.empty:
        # Group by model and shot_setting
        shot_comparison = llm_results.groupby(['model', 'shot_setting'])[['precision', 'recall', 'f1', 'mrr']].mean()
        
        for model in llm_results['model'].unique():
            try:
                few_f1 = shot_comparison.loc[(model, 'few'), 'f1']
                zero_f1 = shot_comparison.loc[(model, 'zero'), 'f1']
                diff_f1 = few_f1 - zero_f1
                
                few_precision = shot_comparison.loc[(model, 'few'), 'precision']
                zero_precision = shot_comparison.loc[(model, 'zero'), 'precision']
                diff_precision = few_precision - zero_precision
                
                few_recall = shot_comparison.loc[(model, 'few'), 'recall']
                zero_recall = shot_comparison.loc[(model, 'zero'), 'recall']
                diff_recall = few_recall - zero_recall
                
                zero_vs_few_analysis[model] = {
                    'zero_f1': zero_f1,
                    'few_f1': few_f1,
                    'diff_f1': diff_f1,
                    'zero_precision': zero_precision,
                    'few_precision': few_precision,
                    'diff_precision': diff_precision,
                    'zero_recall': zero_recall,
                    'few_recall': few_recall,
                    'diff_recall': diff_recall,
                    'better_shot': 'few' if diff_f1 > 0 else 'zero'
                }
            except KeyError:
                # In case one of the models doesn't have both shot settings
                continue
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write findings to markdown file
    with open(output_file, 'w') as f:
        f.write("# Comparative Analysis of Skill Extraction Methods\n\n")
        
        f.write("## Research Overview\n\n")
        f.write("This research compares the performance of BERT (ModernBERT) and various Generative AI models ")
        f.write("(GPT-4o, LLaMA 3.1, Gemini, Claude) on the task of skill extraction from different types of documents: ")
        f.write("course descriptions, job postings, and CVs. ")
        f.write("The generative AI models were evaluated in both zero-shot and few-shot settings to assess the impact of examples on performance. ")
        f.write("The performance is measured using standard metrics: precision, recall, F1 score, accuracy, and Mean Reciprocal Rank (MRR).\n\n")
        
        f.write("## Key Findings\n\n")
        
        f.write("### Overall Performance\n\n")
        f.write("| Model | Precision | Recall | F1 Score | Accuracy | MRR |\n")
        f.write("|-------|-----------|--------|----------|----------|-----|\n")
        for model, row in model_avg.iterrows():
            f.write(f"| {model} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} | {row['mrr']:.4f} |\n")
        f.write("\n")
        
        f.write("### Best Performing Models\n\n")
        f.write(f"- **Best model by F1 score**: {best_model_f1}\n")
        f.write(f"- **Best model by Precision**: {best_model_precision}\n")
        f.write(f"- **Best model by Recall**: {best_model_recall}\n")
        f.write(f"- **Best model by MRR**: {best_model_mrr}\n\n")
        
        f.write("### Performance by Data Type\n\n")
        f.write("| Data Type | Best Model | F1 Score |\n")
        f.write("|-----------|------------|----------|\n")
        for data_type, model in best_model_by_data_type.items():
            f1_score = model_data_type_avg.loc[(model, data_type), 'f1']
            f.write(f"| {data_type} | {model} | {f1_score:.4f} |\n")
        f.write("\n")
        
        f.write("### Analysis by Data Type\n\n")
        f.write("| Data Type | Precision | Recall | F1 Score | Accuracy | MRR |\n")
        f.write("|-----------|-----------|--------|----------|----------|-----|\n")
        for data_type, row in data_type_avg.iterrows():
            f.write(f"| {data_type} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} | {row['mrr']:.4f} |\n")
        f.write("\n")
        
        # Add Zero-shot vs Few-shot analysis if available
        if zero_vs_few_analysis:
            f.write("### Zero-shot vs Few-shot Performance\n\n")
            f.write("| Model | Zero-shot F1 | Few-shot F1 | Difference | Better Setting |\n")
            f.write("|-------|-------------|-------------|------------|---------------|\n")
            for model, metrics in zero_vs_few_analysis.items():
                f.write(f"| {model} | {metrics['zero_f1']:.4f} | {metrics['few_f1']:.4f} | {metrics['diff_f1']:.4f} | {metrics['better_shot']}-shot |\n")
            f.write("\n")
            
            f.write("#### Impact of Few-shot Examples on Precision and Recall\n\n")
            f.write("| Model | Precision Change | Recall Change |\n")
            f.write("|-------|-----------------|---------------|\n")
            for model, metrics in zero_vs_few_analysis.items():
                f.write(f"| {model} | {metrics['diff_precision']:.4f} | {metrics['diff_recall']:.4f} |\n")
            f.write("\n")
        
        f.write("## Analysis and Discussion\n\n")
        
        f.write("### Trade-offs between Models\n\n")
        f.write("Our analysis reveals several interesting trade-offs between different models:\n\n")
        
        # BERT vs. Generative AI
        bert_results = df[df['model'] == 'ModernBERT']
        if not bert_results.empty:
            bert_metrics = bert_results[['precision', 'recall', 'f1', 'accuracy', 'mrr']].mean()
            
            # For fair comparison, get the best of zero/few-shot for each generative model
            gen_ai_models = df[df['model'] != 'ModernBERT']
            if not gen_ai_models.empty:
                # Group by model and data_type, get the max F1 across shot settings
                gen_ai_best = gen_ai_models.groupby(['model', 'data_type'])[['precision', 'recall', 'f1', 'accuracy', 'mrr']].max()
                gen_ai_avg = gen_ai_best.groupby(level=0).mean().mean()
                
                f.write("#### BERT vs. Best Generative AI Performance\n\n")
                f.write("| Metric | BERT | Average Best Generative AI | Difference |\n")
                f.write("|--------|------|---------------------------|------------|\n")
                for metric in ['precision', 'recall', 'f1', 'accuracy', 'mrr']:
                    diff = bert_metrics[metric] - gen_ai_avg[metric]
                    f.write(f"| {metric.capitalize()} | {bert_metrics[metric]:.4f} | {gen_ai_avg[metric]:.4f} | {diff:.4f} |\n")
                f.write("\n")
                
                # Add analysis of the results
                if bert_metrics['f1'] > gen_ai_avg['f1']:
                    f.write("BERT outperforms generative AI models on average in terms of F1 score. ")
                    f.write("This suggests that for the specific task of skill extraction, ")
                    f.write("the fine-tuned nature of BERT models may provide advantages over generative models.\n\n")
                else:
                    f.write("Generative AI models outperform BERT on average in terms of F1 score. ")
                    f.write("This suggests that for the specific task of skill extraction, ")
                    f.write("the broader knowledge and context understanding of generative models may provide advantages.\n\n")
        
        # Zero-shot vs Few-shot analysis
        if zero_vs_few_analysis and not bert_only:
            f.write("### Impact of Few-shot Examples\n\n")
            
            # Calculate average improvement across models
            avg_f1_improvement = np.mean([metrics['diff_f1'] for metrics in zero_vs_few_analysis.values()])
            avg_precision_improvement = np.mean([metrics['diff_precision'] for metrics in zero_vs_few_analysis.values()])
            avg_recall_improvement = np.mean([metrics['diff_recall'] for metrics in zero_vs_few_analysis.values()])
            
            f.write(f"On average, few-shot prompting resulted in a **{avg_f1_improvement:.4f}** improvement in F1 score ")
            f.write(f"(precision: **{avg_precision_improvement:.4f}**, recall: **{avg_recall_improvement:.4f}**).\n\n")
            
            # Count models that benefited from few-shot
            better_few = sum(1 for metrics in zero_vs_few_analysis.values() if metrics['better_shot'] == 'few')
            better_zero = sum(1 for metrics in zero_vs_few_analysis.values() if metrics['better_shot'] == 'zero')
            
            f.write(f"- {better_few} out of {len(zero_vs_few_analysis)} models performed better with few-shot prompting\n")
            f.write(f"- {better_zero} out of {len(zero_vs_few_analysis)} models performed better with zero-shot prompting\n\n")
            
            # Models with biggest improvement
            if zero_vs_few_analysis:
                best_improvement_model = max(zero_vs_few_analysis.items(), key=lambda x: x[1]['diff_f1'])
                worst_improvement_model = min(zero_vs_few_analysis.items(), key=lambda x: x[1]['diff_f1'])
                
                f.write(f"The model with the largest improvement from few-shot examples was **{best_improvement_model[0]}** ")
                f.write(f"with a **{best_improvement_model[1]['diff_f1']:.4f}** increase in F1 score.\n\n")
                
                if worst_improvement_model[1]['diff_f1'] < 0:
                    f.write(f"Interestingly, **{worst_improvement_model[0]}** performed worse with few-shot examples, ")
                    f.write(f"seeing a **{abs(worst_improvement_model[1]['diff_f1']):.4f}** decrease in F1 score. ")
                    f.write("This suggests that few-shot examples can sometimes constrain the model's reasoning or lead to over-fitting to the examples.\n\n")
        
        f.write("### Performance by Document Type\n\n")
        f.write("Our analysis shows variation in model performance across different document types:\n\n")
        
        # Analyze each data type
        for data_type in df['data_type'].unique():
            f.write(f"#### {data_type.capitalize()} Documents\n\n")
            
            # Get best and worst model for this data type
            data_type_results = df[df['data_type'] == data_type]
            best_model = data_type_results.groupby('model_shot')['f1'].mean().idxmax()
            worst_model = data_type_results.groupby('model_shot')['f1'].mean().idxmin()
            best_f1 = data_type_results[data_type_results['model_shot'] == best_model]['f1'].values[0]
            worst_f1 = data_type_results[data_type_results['model_shot'] == worst_model]['f1'].values[0]
            
            f.write(f"- Best performing model: **{best_model}** (F1: {best_f1:.4f})\n")
            f.write(f"- Worst performing model: **{worst_model}** (F1: {worst_f1:.4f})\n\n")
            
            # Add specific analysis for each data type
            if data_type == 'course':
                f.write("For course descriptions, ")
                if 'ModernBERT' == best_model:
                    f.write("BERT performed best, likely due to its ability to identify structured learning outcomes and course objectives. ")
                else:
                    f.write(f"{best_model} performed best, likely due to its ability to understand educational terminology and context. ")
                f.write("Course descriptions typically contain well-structured learning outcomes that may be easier to match with skill taxonomies.\n\n")
            
            elif data_type == 'job':
                f.write("For job descriptions, ")
                if 'ModernBERT' == best_model:
                    f.write("BERT performed best, possibly due to its training on formal job requirements and industry-specific terminology. ")
                else:
                    f.write(f"{best_model} performed best, possibly due to its broader understanding of job requirements and industry context. ")
                f.write("Job descriptions often contain explicit skill requirements, making them relatively straightforward for skill extraction.\n\n")
            
            elif data_type == 'cv':
                f.write("For CVs, ")
                if 'ModernBERT' == best_model:
                    f.write("BERT performed best, which may be surprising given the unstructured nature of CV content. ")
                else:
                    f.write(f"{best_model} performed best, which aligns with expectations given the unstructured and varied nature of CV content. ")
                f.write("CVs often present skills in varied formats and contexts, requiring models to understand implicit skill mentions.\n\n")
        
        f.write("## Conclusion\n\n")
        
        # Determine overall winner
        if best_model_f1 == 'ModernBERT':
            f.write("Our research demonstrates that **BERT-based models** achieved the highest overall performance for skill extraction across different document types. ")
            f.write("This suggests that specialized models fine-tuned for classification and entity recognition tasks may still hold advantages over general-purpose generative AI for structured information extraction tasks.\n\n")
        else:
            f.write(f"Our research demonstrates that **{best_model_f1}** achieved the highest overall performance for skill extraction across different document types. ")
            f.write("This suggests that recent advances in generative AI are pushing the boundaries of what's possible in structured information extraction tasks.\n\n")
        
        # Add zero-shot vs few-shot conclusion if applicable
        if zero_vs_few_analysis and not bert_only:
            better_shot = "few" if avg_f1_improvement > 0 else "zero"
            f.write(f"Regarding prompting strategies, our analysis shows that **{better_shot}-shot prompting** tends to yield better results on average. ")
            if better_shot == "few":
                f.write("This indicates that providing examples helps guide the models to better understand the skill extraction task. ")
                f.write("However, the benefit varies significantly across models, with some actually performing worse with examples.\n\n")
            else:
                f.write("This is an interesting finding, as it suggests that examples may sometimes constrain the models' ability to identify skills in diverse contexts. ")
                f.write("However, the effect varies significantly across models, with some benefiting substantially from examples.\n\n")
        
        f.write("### Practical Implications\n\n")
        f.write("- **System Design**: For production skill extraction systems, ")
        if best_model_f1 == 'ModernBERT':
            f.write("BERT-based models offer a good balance of accuracy and computational efficiency.\n")
        else:
            f.write(f"generative AI models like {best_model_f1} can significantly improve extraction quality but may require more computational resources.\n")
        
        f.write("- **Document Type Considerations**: Different models excel with different document types, suggesting that hybrid approaches might be optimal for systems processing diverse documents.\n")
        
        f.write("- **Precision vs. Recall**: ")
        if model_avg.loc[best_model_precision, 'precision'] > model_avg.loc[best_model_recall, 'recall']:
            f.write(f"The highest precision ({model_avg.loc[best_model_precision, 'precision']:.4f} from {best_model_precision}) exceeds the highest recall ({model_avg.loc[best_model_recall, 'recall']:.4f} from {best_model_recall}), ")
            f.write("suggesting that false positives are less common than false negatives in skill extraction across all models.\n\n")
        else:
            f.write(f"The highest recall ({model_avg.loc[best_model_recall, 'recall']:.4f} from {best_model_recall}) exceeds the highest precision ({model_avg.loc[best_model_precision, 'precision']:.4f} from {best_model_precision}), ")
            f.write("suggesting that false negatives are less common than false positives in skill extraction across all models.\n\n")
        
        if zero_vs_few_analysis and not bert_only:
            f.write("- **Prompting Strategy**: The choice between zero-shot and few-shot prompting should be made based on the specific model being used and validated empirically, as the optimal strategy varies across models.\n\n")
        
        f.write("### Future Work\n\n")
        f.write("Based on our findings, several promising directions for future research include:\n\n")
        f.write("- Developing ensemble methods that combine the strengths of both BERT and generative AI models\n")
        f.write("- Exploring domain-specific fine-tuning to improve performance on specific document types\n")
        if zero_vs_few_analysis and not bert_only:
            f.write("- Investigating optimal example selection for few-shot prompting across different models\n")
            f.write("- Exploring the relationship between model size and the benefit of few-shot examples\n")
        f.write("- Examining the computational efficiency trade-offs between model types in production environments\n")
        
    print(f"Research findings saved to {output_file}")

def run_analysis():
    """Run the full analysis pipeline"""
    # Load results
    df = load_results()
    
    # Create output directory for visualizations
    os.makedirs("analysis_output", exist_ok=True)
    
    # Create detailed visualizations
    create_detailed_visualizations(df, "analysis_output")
    
    # Generate research findings
    generate_research_findings(df, "analysis_output/research_findings.md")
    
    # Perform statistical significance tests
    p_values = statistical_significance_tests(df)
    p_values.to_csv("analysis_output/statistical_significance.csv")
    print("Statistical significance tests saved to analysis_output/statistical_significance.csv")
    
    print("Analysis complete! Results saved to analysis_output/ directory.")

if __name__ == "__main__":
    run_analysis()