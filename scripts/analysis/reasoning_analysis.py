# %%
# Statistical Analysis: Camstories vs Simplestories Word Tokenisation Evaluation

import json
import pandas as pd
import numpy as np
from scipy import stats

# %%
# Load and parse the word tokenisation evaluation data
def load_word_evaluation_data(file_path):
    """Load and parse the JSONL word evaluation file"""
    camstories_data = []
    simplestories_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                custom_id = data['custom_id']
                
                # Extract the content from the response
                content = data['response']['body']['choices'][0]['message']['content']
                ratings = json.loads(content)
                
                # Add the ID for tracking
                ratings['id'] = custom_id
                
                # Categorize by dataset
                if 'camstories_10k_word_run1-' in custom_id:
                    camstories_data.append(ratings)
                elif 'simplestories_word-' in custom_id:
                    simplestories_data.append(ratings)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line with custom_id {data.get('custom_id', 'unknown')}: {e}")
                continue
    
    return camstories_data, simplestories_data

# %%
# Parameters and paths
file_path = "results/dataset evaluation/word_tokenisation/batch_689703cd75a08190af1b1d27d4d508f6_output.jsonl"

print("Loading word tokenisation evaluation data...")
camstories_data, simplestories_data = load_word_evaluation_data(file_path)

print(f"Found {len(camstories_data)} camstories evaluations")
print(f"Found {len(simplestories_data)} simplestories evaluations")

# %%
# Convert to DataFrames for easier analysis
camstories_df = pd.DataFrame(camstories_data)
simplestories_df = pd.DataFrame(simplestories_data)

# Add dataset labels
camstories_df['dataset'] = 'camstories'
simplestories_df['dataset'] = 'simplestories'

# Combine datasets
combined_df = pd.concat([camstories_df, simplestories_df], ignore_index=True)

print("\nCamstories sample:")
print(camstories_df[['id', 'originality', 'coherence', 'grammar', 'quality']].head())
print("\nSimplestories sample:")
print(simplestories_df[['id', 'originality', 'coherence', 'grammar', 'quality']].head())

# %%
# Calculate descriptive statistics
metrics = ['originality', 'coherence', 'grammar', 'quality']

print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)

for metric in metrics:
    print(f"\n{metric.upper()}:")
    print("Camstories:")
    print(f"  Mean: {camstories_df[metric].mean():.2f}")
    print(f"  Std:  {camstories_df[metric].std():.2f}")
    print(f"  N:    {len(camstories_df)}")
    
    print("Simplestories:")
    print(f"  Mean: {simplestories_df[metric].mean():.2f}")
    print(f"  Std:  {simplestories_df[metric].std():.2f}")
    print(f"  N:    {len(simplestories_df)}")

# %%
# Statistical tests and p-values
print("\n" + "="*50)
print("STATISTICAL TESTS (Two-sample t-tests)")
print("="*50)

results = {}

for metric in metrics:
    # Perform independent samples t-test
    camstories_values = camstories_df[metric].values
    simplestories_values = simplestories_df[metric].values
    
    # Check for normality (Shapiro-Wilk test)
    _, p_norm_cam = stats.shapiro(camstories_values)
    _, p_norm_simple = stats.shapiro(simplestories_values)
    
    # Two-sample t-test (assuming unequal variances - Welch's t-test)
    t_stat, p_value = stats.ttest_ind(camstories_values, simplestories_values, equal_var=False)
    
    # Mann-Whitney U test (non-parametric alternative)
    u_stat, p_value_mw = stats.mannwhitneyu(camstories_values, simplestories_values, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(camstories_values) - 1) * np.var(camstories_values, ddof=1) + 
                         (len(simplestories_values) - 1) * np.var(simplestories_values, ddof=1)) / 
                        (len(camstories_values) + len(simplestories_values) - 2))
    cohens_d = (np.mean(camstories_values) - np.mean(simplestories_values)) / pooled_std
    
    results[metric] = {
        't_statistic': t_stat,
        'p_value_ttest': p_value,
        'p_value_mannwhitney': p_value_mw,
        'cohens_d': cohens_d,
        'p_norm_camstories': p_norm_cam,
        'p_norm_simplestories': p_norm_simple
    }
    
    print(f"\n{metric.upper()}:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value (t-test): {p_value:.6f}")
    print(f"  p-value (Mann-Whitney): {p_value_mw:.6f}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  Normality p-values: Camstories={p_norm_cam:.4f}, Simplestories={p_norm_simple:.4f}")
    
    # Interpretation
    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "ns (not significant)"
    
    direction = "Camstories > Simplestories" if np.mean(camstories_values) > np.mean(simplestories_values) else "Simplestories > Camstories"
    print(f"  Significance: {significance}")
    print(f"  Direction: {direction}")

# %%
# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE - WORD TOKENISATION EVALUATION")
print("="*70)

summary_data = []
for metric in metrics:
    cam_mean = camstories_df[metric].mean()
    simple_mean = simplestories_df[metric].mean()
    p_val = results[metric]['p_value_ttest']
    cohens_d = results[metric]['cohens_d']
    
    summary_data.append({
        'Metric': metric.capitalize(),
        'Camstories_Mean': f"{cam_mean:.2f}",
        'Simplestories_Mean': f"{simple_mean:.2f}",
        'Difference': f"{cam_mean - simple_mean:+.2f}",
        'p_value': f"{p_val:.6f}",
        'Cohens_d': f"{cohens_d:.3f}",
        'Significance': "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# %%
# Meaningful interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

significant_differences = []
for metric in metrics:
    p_val = results[metric]['p_value_ttest']
    if p_val < 0.05:
        cam_mean = camstories_df[metric].mean()
        simple_mean = simplestories_df[metric].mean()
        direction = "higher" if cam_mean > simple_mean else "lower"
        winner = "Camstories" if cam_mean > simple_mean else "Simplestories"
        significant_differences.append(f"{metric.capitalize()}: {winner} scored significantly {direction}")

if significant_differences:
    print("SIGNIFICANT DIFFERENCES:")
    for diff in significant_differences:
        print(f"- {diff}")
else:
    print("No statistically significant differences found between camstories and simplestories.")

# %%
# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('word_tokenisation_camstories_vs_simplestories_results.csv')
combined_df.to_csv('word_tokenisation_ratings_data.csv', index=False)

print(f"\nResults saved to:")
print("- word_tokenisation_camstories_vs_simplestories_results.csv")
print("- word_tokenisation_ratings_data.csv")


