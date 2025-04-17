#!/usr/bin/env python3
"""
Token Count Visualization

This script creates visualizations of token counts for different models
and training stages in the math-reasoning-in-language-models project.

Requirements:
    - matplotlib
    - numpy
    - pandas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Token counts from calculations (in billions)
token_data = {
    "Model": ["GPT-2", "GPT-2", "GPT-2", 
              "GPT-2 Large", "GPT-2 Large", "GPT-2 Large",
              "Mistral-7B", "Mistral-7B", "Mistral-7B"],
    "Training Stage": ["Pre-training", "Curriculum Learning", "Instruction Tuning"] * 3,
    "Tokens (B)": [
        # GPT-2 tokens (in billions)
        3.1457,  # Pre-training
        0.9830,  # Curriculum Learning
        1.3107,  # Instruction Tuning
        
        # GPT-2 Large tokens (in billions)
        0.4915,  # Pre-training
        0.9830,  # Curriculum Learning
        0.6554,  # Instruction Tuning
        
        # Mistral-7B tokens (in billions)
        0.9830,  # Pre-training
        1.9661,  # Curriculum Learning
        0.6554   # Instruction Tuning
    ]
}

df = pd.DataFrame(token_data)

# Calculate totals
model_totals = df.groupby('Model')['Tokens (B)'].sum().reset_index()
stage_totals = df.groupby('Training Stage')['Tokens (B)'].sum().reset_index()

# Set up the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Token Count Analysis for Math Reasoning Models', fontsize=16)

# Plot 1: Bar chart by model
axs[0, 0].bar(df['Model'], df['Tokens (B)'], color=['blue', 'orange', 'green'])
axs[0, 0].set_title('Total Tokens by Model')
axs[0, 0].set_ylabel('Tokens (Billions)')
axs[0, 0].set_xticklabels(df['Model'], rotation=45, ha='right')

# Plot 2: Stacked bar chart by model and training stage
models = df['Model'].unique()
stages = df['Training Stage'].unique()
bottom = np.zeros(len(models))

for stage in stages:
    stage_data = df[df['Training Stage'] == stage]
    values = []
    for model in models:
        model_value = stage_data[stage_data['Model'] == model]['Tokens (B)'].values
        values.append(model_value[0] if len(model_value) > 0 else 0)
    
    axs[0, 1].bar(models, values, bottom=bottom, label=stage)
    bottom += values

axs[0, 1].set_title('Tokens by Model and Training Stage')
axs[0, 1].set_ylabel('Tokens (Billions)')
axs[0, 1].set_xticklabels(models, rotation=45, ha='right')
axs[0, 1].legend()

# Plot 3: Pie chart of total tokens by model
axs[1, 0].pie(model_totals['Tokens (B)'], labels=model_totals['Model'], autopct='%1.1f%%')
axs[1, 0].set_title('Token Distribution by Model')

# Plot 4: Pie chart of total tokens by training stage
axs[1, 1].pie(stage_totals['Tokens (B)'], labels=stage_totals['Training Stage'], autopct='%1.1f%%')
axs[1, 1].set_title('Token Distribution by Training Stage')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('token_distribution.png', dpi=300)
plt.show()

# Create a more detailed comparison table
pivot_table = df.pivot(index='Model', columns='Training Stage', values='Tokens (B)')
pivot_table['Total'] = pivot_table.sum(axis=1)
pivot_table.loc['Total'] = pivot_table.sum()

print("\nDetailed Token Comparison (Billions):")
print(pivot_table.round(4))

# Calculate percentages
print("\nToken Distribution Percentages:")
percentage_table = (pivot_table / pivot_table.loc['Total', 'Total'] * 100).round(2)
print(percentage_table)

print(f"\nGrand Total: {pivot_table.loc['Total', 'Total']:.4f} Billion Tokens")