import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pie_chart(col: pd.Series, title: str) -> None:
    # Get value counts
    value_counts = col.value_counts()
    
    # Generate a color palette with enough colors for the number of unique items
    num_unique = len(value_counts)  # Number of unique items
    colors = plt.cm.Paired(np.linspace(0, 1, num_unique))  # Generate a color palette
    
    # Plot pie chart with dynamic color palette
    plt.figure(figsize=(7, 7))  # Set a larger figure size for better visualization
    ax = value_counts.plot(kind='pie', 
                           autopct='%1.1f%%', 
                           startangle=90, 
                           colors=colors, 
                           wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
                           textprops={'fontsize': 12, 'fontweight': 'bold'}, 
                           figsize=(8, 8))
    
    # Set axis to be equal (so pie chart is circular)
    plt.axis('equal')
    
    # Add a title
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Remove the legend (optional, since we have labels on the pie chart)
    plt.legend().set_visible(False)
    
    # Display the chart
    plt.show()


def calculate_statistics(series: pd.Series) -> dict:
    """Calculate mean, median, and standard deviation for a given series."""
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std()
    }

def plot_histogram(series: pd.Series, stats: dict, title: str, xlabel: str, color: str) -> None:
    """Plot histogram with KDE and statistical lines."""
    sns.histplot(series, kde=True, color=color, bins=10)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2f}')
    plt.axvline(stats['median'], color='green', linestyle='-', label=f'Median: {stats["median"]:.2f}')
    plt.axvline(stats['mean'] + stats['std'], color='orange', linestyle=':', label=f'+1 Std Dev: {stats["mean"] + stats["std"]:.2f}')
    plt.axvline(stats['mean'] - stats['std'], color='orange', linestyle=':', label=f'-1 Std Dev: {stats["mean"] - stats["std"]:.2f}')
    plt.legend()

def plot_distribution_text_length(col: pd.Series, title: str) -> None:
    """Plot distribution of text lengths in terms of characters and words."""
    col_name = col.name
    col = col.astype(str)
    text_lengths_chars = col.apply(len)  # Length in characters
    text_lengths_words = col.apply(lambda x: len(x.split()))  # Length in words
    
    # Calculate statistics
    stats_chars = calculate_statistics(text_lengths_chars)
    stats_words = calculate_statistics(text_lengths_words)
    
    # Set up the plot
    plt.figure(figsize=(14, 7))
    
    # Plot histogram for text lengths in characters
    plt.subplot(1, 2, 1)
    plot_histogram(text_lengths_chars, stats_chars, f"Distribution of {col_name} Lengths (Characters)", f"{col_name} Length (Characters)", "skyblue")
    
    # Plot histogram for text lengths in words
    plt.subplot(1, 2, 2)
    plot_histogram(text_lengths_words, stats_words, f"Distribution of {col_name} Lengths (Words)", f"{col_name} Length (Words)", "lightgreen")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Print the statistics for reference
    print(f"Character-based stats: Mean = {stats_chars['mean']:.2f}, Median = {stats_chars['median']:.2f}, Std Dev = {stats_chars['std']:.2f}")
    print(f"Word-based stats: Mean = {stats_words['mean']:.2f}, Median = {stats_words['median']:.2f}, Std Dev = {stats_words['std']:.2f}")

if __name__ == '__main__':
    pass