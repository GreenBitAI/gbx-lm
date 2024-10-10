import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_token_distribution(news_data):
    sns.histplot(news_data["tokens"], kde=False)
    plt.title('Distribution of chunk sizes')
    plt.xlabel('Token count')
    plt.ylabel('Frequency')
    plt.show()

def plot_entity_count_vs_token_count(entity_dist_df):
    sns.lmplot(
        x="token_count", y="entity_count", data=entity_dist_df, line_kws={"color": "red"}
    )
    plt.title("Entity Count vs Token Count Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Entity Count")
    plt.show()

def plot_node_degree_distribution(degree_dist_df):
    mean_degree = np.mean(degree_dist_df['node_degree'])
    percentiles = np.percentile(degree_dist_df['node_degree'], [25, 50, 75, 90])
    
    plt.figure(figsize=(12, 6))
    sns.histplot(degree_dist_df['node_degree'], bins=50, kde=False, color='blue')
    plt.yscale('log')
    plt.xlabel('Node Degree')
    plt.ylabel('Count (log scale)')
    plt.title('Node Degree Distribution')
    
    plt.axvline(mean_degree, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_degree:.2f}')
    plt.axvline(percentiles[0], color='purple', linestyle='dashed', linewidth=1, label=f'25th Percentile: {percentiles[0]:.2f}')
    plt.axvline(percentiles[1], color='orange', linestyle='dashed', linewidth=1, label=f'50th Percentile: {percentiles[1]:.2f}')
    plt.axvline(percentiles[2], color='yellow', linestyle='dashed', linewidth=1, label=f'75th Percentile: {percentiles[2]:.2f}')
    plt.axvline(percentiles[3], color='brown', linestyle='dashed', linewidth=1, label=f'90th Percentile: {percentiles[3]:.2f}')
    
    plt.legend()
    plt.show()

# Additional visualization functions can be added here
