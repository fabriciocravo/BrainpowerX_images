import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


def generate_inference_heatmaps(seed=None, save_path=None):
    """
    Generate 7 heatmaps showing p-values from different statistical inference methods:
    - 4 methods with 4 p-values each (edge-level): Parametric FWER, Parametric FDR, Size, TFCE
    - 2 methods with 2 p-values each (network-level): cNBS FWER, cNBS FDR
    - 1 method with 1 p-value (whole-brain): Omnibus

    First variable always significant (p < 0.05), second varies across methods.

    Parameters:
    - seed: Random seed for reproducibility
    - save_path: Path to save the figure (optional)
    """

    if seed is not None:
        np.random.seed(seed)

    # Define significance results for each method (1 = significant, 0 = non-significant)
    # First variable always significant (1), second varies across methods
    edge_methods = {
        'Parametric\nFWER': [1, 0, 0, 0],  # Conservative - only first significant
        'Parametric\nFDR': [1, 1, 0, 0],  # Less conservative - first two significant
        'Size\n(Cluster)': [1, 1, 0, 0],  # Good sensitivity - first two significant
        'TFCE': [1, 1, 1, 0]  # Best sensitivity - first three significant
    }

    network_methods = {
        'cNBS\nFWER': [1, 0],  # Conservative - only first significant
        'cNBS\nFDR': [1, 1]  # Less conservative - both significant
    }

    wholebrain_methods = {
        'Omnibus': [1]  # Single significant result
    }

    # Create variable labels
    edge_labels = [f'var_{i + 1}' for i in range(4)]
    network_labels = [f'net_{i + 1}' for i in range(2)]
    wholebrain_labels = ['whole-brain']

    # Set up figure with 7 subplots arranged vertically (rows)
    fig = plt.figure(figsize=(6, 14))  # Tall and narrow for vertical layout
    gs = gridspec.GridSpec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1], hspace=0.6)  # 7 rows

    # Color parameters - use log scale for better visualization
    # p < 0.05 = significant (dark colors), p > 0.05 = non-significant (light colors)

    plot_idx = 0

    # Plot edge-level methods (4 significance values each)
    for method_name, sig_vals in edge_methods.items():
        ax = fig.add_subplot(gs[plot_idx])

        # Convert to matrix format for heatmap (horizontal row)
        sig_matrix = np.array(sig_vals).reshape(1, -1)  # 1 row, multiple columns

        sns.heatmap(sig_matrix,
                    cmap='Reds',  # Red and white scale
                    xticklabels=edge_labels,
                    yticklabels=[method_name.replace('\n', ' ')],
                    square=True,
                    linewidths=0.5,
                    linecolor='white',
                    cbar=False,
                    vmin=0, vmax=1,  # 0 = white, 1 = red
                    ax=ax)

        ax.set_title('', fontsize=12, pad=5)  # No title to save space
        ax.set_xlabel('', fontsize=10)
        ax.tick_params(axis='x', rotation=0, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=12)  # Increased from 8 to 12

        plot_idx += 1

    # Plot network-level methods (2 significance values each)
    for method_name, sig_vals in network_methods.items():
        ax = fig.add_subplot(gs[plot_idx])

        sig_matrix = np.array(sig_vals).reshape(1, -1)  # 1 row, multiple columns

        sns.heatmap(sig_matrix,
                    cmap='Reds',  # Red and white scale
                    xticklabels=network_labels,
                    yticklabels=[method_name.replace('\n', ' ')],
                    square=True,
                    linewidths=0.5,
                    linecolor='white',
                    cbar=False,
                    vmin=0, vmax=1,
                    ax=ax)

        ax.set_title('', fontsize=12, pad=5)
        ax.set_xlabel('', fontsize=10)
        ax.tick_params(axis='x', rotation=0, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=12)  # Increased from 8 to 12

        plot_idx += 1

    # Plot whole-brain method (1 significance value)
    method_name = 'Omnibus'
    sig_vals = wholebrain_methods[method_name]
    ax = fig.add_subplot(gs[plot_idx])

    sig_matrix = np.array(sig_vals).reshape(1, -1)  # 1 row, 1 column

    im = sns.heatmap(sig_matrix,
                     cmap='Reds',  # Red and white scale
                     xticklabels=wholebrain_labels,
                     yticklabels=[method_name],
                     square=True,
                     linewidths=0.5,
                     linecolor='white',
                     cbar=False,  # Remove colorbar
                     vmin=0, vmax=1,
                     ax=ax)

    ax.set_title('', fontsize=12, pad=5)
    ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=12)  # Increased from 8 to 12


    # Add significance explanation
    plt.figtext(0.02, 0.02, 'Red = Significant (1), White = Non-significant (0). Each row shows one method.',
                fontsize=9, ha='left')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inference methods heatmaps saved to: {save_path}")

    plt.show()

    return edge_methods, network_methods, wholebrain_methods


def print_significance_summary(edge_methods, network_methods, wholebrain_methods):
    """Print summary of significance results for each method"""

    print("\nSignificance Summary:")
    print("=" * 50)

    print("\nEdge-level methods (4 variables):")
    for method, sig_vals in edge_methods.items():
        method_clean = method.replace('\n', ' ')
        sig_count = sum(sig_vals)
        print(f"{method_clean:15}: {sig_vals} ({sig_count}/4 significant)")

    print("\nNetwork-level methods (2 networks):")
    for method, sig_vals in network_methods.items():
        method_clean = method.replace('\n', ' ')
        sig_count = sum(sig_vals)
        print(f"{method_clean:15}: {sig_vals} ({sig_count}/2 significant)")

    print("\nWhole-brain method (1 test):")
    for method, sig_vals in wholebrain_methods.items():
        sig_count = sum(sig_vals)
        print(f"{method:15}: {sig_vals} ({sig_count}/1 significant)")


# Example usage
if __name__ == "__main__":
    # Generate inference method heatmaps
    edge, network, wholebrain = generate_inference_heatmaps(
        seed=42
    )

    # Print summary
    print_significance_summary(edge, network, wholebrain)