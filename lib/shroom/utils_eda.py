import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pie_chart(col: pd.Series, title: str) -> None:
    """Plot a pie chart for the given column."""
    value_counts = col.value_counts()
    num_unique = len(value_counts)
    colors = plt.cm.Paired(np.linspace(0, 1, num_unique))
    
    plt.figure(figsize=(7, 7))
    value_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors,
                      wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
                      textprops={'fontsize': 12, 'fontweight': 'bold'}, figsize=(8, 8))
    plt.axis('equal')
    plt.title(title, fontsize=15, fontweight='bold', color='darkblue')
    plt.legend().set_visible(False)
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
    plt.title(title, fontsize=15, fontweight='bold', color='darkblue')
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel("Frequency", fontsize=14, fontweight='bold')
    plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2f}')
    plt.axvline(stats['median'], color='green', linestyle='-', label=f'Median: {stats["median"]:.2f}')
    plt.axvline(stats['mean'] + stats['std'], color='orange', linestyle=':', label=f'+1 Std Dev: {stats["mean"] + stats["std"]:.2f}')
    plt.axvline(stats['mean'] - stats['std'], color='orange', linestyle=':', label=f'-1 Std Dev: {stats["mean"] - stats["std"]:.2f}')
    plt.legend()

def plot_distribution_text_length(col: pd.Series, title: str) -> None:
    """Plot distribution of text lengths in terms of characters and words."""
    col_name = col.name
    col = col.astype(str)
    text_lengths_chars = col.apply(len)
    text_lengths_words = col.apply(lambda x: len(x.split()))
    
    stats_chars = calculate_statistics(text_lengths_chars)
    stats_words = calculate_statistics(text_lengths_words)
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plot_histogram(text_lengths_chars, stats_chars, f"Distribution of {col_name} Lengths (Characters)", f"{col_name} Length (Characters)", "skyblue")
    
    plt.subplot(1, 2, 2)
    plot_histogram(text_lengths_words, stats_words, f"Distribution of {col_name} Lengths (Words)", f"{col_name} Length (Words)", "lightgreen")

    plt.tight_layout()
    plt.show()

    print(f"Character-based stats: Mean = {stats_chars['mean']:.2f}, Median = {stats_chars['median']:.2f}, Std Dev = {stats_chars['std']:.2f}")
    print(f"Word-based stats: Mean = {stats_words['mean']:.2f}, Median = {stats_words['median']:.2f}, Std Dev = {stats_words['std']:.2f}")

def extract_hallucinations(text: str, hallucination_positions: list) -> list:
    """
    Extract hallucinated spans from the provided text based on the given positions.
    
    Parameters:
    - text (str): The text from which hallucinations will be extracted.
    - hallucination_positions (list of list): A list of pairs [start, end] representing the spans of hallucinated text.

    Returns:
    - list of str: A list of extracted hallucinated text spans.
    """
    return [text[start:end] for start, end in hallucination_positions]

def plot_line_chart(df: pd.DataFrame) -> None:
    """Plot a line chart showing the number of hallucination spans vs. text length."""
    df['hallucination_spans'] = [extract_hallucinations(x, y) for x, y in zip(df['model_output_text'], df['hard_labels'])]
    df['hallucination_spans_number'] = df['hallucination_spans'].apply(len)
    df['text_length'] = df['model_output_text'].apply(lambda x: len(x.split()))

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.regplot(x='text_length', y='hallucination_spans_number', data=df, 
                scatter_kws={'s': 100, 'color': 'dodgerblue', 'alpha': 0.7}, 
                line_kws={'color': 'red', 'linewidth': 2})

    plt.title("Hallucination Spans vs. Generation Text Length", fontsize=18, fontweight='bold', color='darkblue')
    plt.xlabel("Length of Generation Text (Number of Words)", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Hallucination Spans", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pass