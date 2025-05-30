import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from controlling_variables import *


def generate_tstat_heatmaps(n_variables=4, n_permutations=10, seed=None, save_path=None):
    """
    Generate four t-statistics heatmaps showing GLM results:
    1. Original t-stats (variables × single column)
    2. Permutation t-stats (variables × permutations)
    3. Averaged t-stats (2 averaged variables × single column)
    4. Averaged permutation t-stats (2 averaged variables × permutations)

    Parameters:
    - n_variables: Number of variables (4 for original, 2 for averaged)
    - n_permutations: Number of permutations to show
    - seed: Random seed for reproducibility
    - save_path: Path to save the figure (optional)
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate original t-statistics (variables × 1 regressor)
    tstat_original = np.random.normal(0, 2, (n_variables, 1))

    # Generate permutation t-statistics (variables × permutations)
    tstat_permutations = np.random.normal(0, 1, (n_variables, n_permutations))

    # Generate averaged t-statistics (average pairs of variables)
    n_avg_vars = n_variables // 2
    tstat_avg_original = np.zeros((n_avg_vars, 1))
    tstat_avg_permutations = np.zeros((n_avg_vars, n_permutations))

    # Average pairs of variables
    for i in range(n_avg_vars):
        # Average variables i*2 and i*2+1
        tstat_avg_original[i, 0] = np.mean([tstat_original[i * 2, 0], tstat_original[i * 2 + 1, 0]])
        tstat_avg_permutations[i, :] = np.mean([tstat_permutations[i * 2, :], tstat_permutations[i * 2 + 1, :]],
                                               axis=0)

    # Create labels
    var_labels = [f'var_{i + 1}' for i in range(n_variables)]
    avg_var_labels = [f'net_{i + 1}' for i in range(n_avg_vars)]
    perm_labels = [f'perm_{i + 1}' for i in range(n_permutations)]

    # Set up figure with 4 subplots, keeping square sizes constant
    fig = plt.figure(figsize=(16, max(8, n_variables * 0.4)))
    gs = gridspec.GridSpec(1, 5, width_ratios=[0.5, n_permutations / 2, 0.2, 0.4, n_permutations / 2],
                           wspace=0.8)  # Increase from 0.5 to 0.8 or higher

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])  # ADD THIS LINE
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])

    # Set consistent color scale for all plots
    vmin, vmax = -3, 3

    # Plot 1: Original t-statistics (variables × 1)
    sns.heatmap(tstat_original,
                cmap='RdBu_r',
                center=0,
                vmin=vmin, vmax=vmax,
                xticklabels=['t-stat'],
                yticklabels=var_labels,
                square=True,
                linewidths=0.5,
                linecolor='white',
                cbar=False,
                ax=ax1)

    ax1.set_title('Original\nt-statistics', fontsize=title_fonts, pad=15)
    ax1.set_xlabel('Regressors', fontsize=x_label_fonts)
    ax1.tick_params(axis='x', rotation=0, labelsize=x_label_fonts)
    ax1.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)

    # Plot 2: Permutation t-statistics (variables × permutations)
    sns.heatmap(tstat_permutations,
                cmap='RdBu_r',
                center=0,
                vmin=vmin, vmax=vmax,
                xticklabels=perm_labels,
                yticklabels=[],
                square=True,
                linewidths=0.5,
                linecolor='white',
                cbar=False,
                cbar_kws={'label': 't-statistic'},
                ax=ax2)

    ax2.set_title('Permutation\nt-statistics', fontsize=x_label_fonts, pad=15)
    ax2.set_xlabel('Permutations', fontsize=y_label_fonts)
    ax2.tick_params(axis='x', rotation=45, labelsize=x_label_fonts)

    # Plot 3: Averaged original t-statistics (averaged variables × 1)
    sns.heatmap(tstat_avg_original,
                cmap='RdBu_r',
                center=0,
                vmin=vmin, vmax=vmax,
                xticklabels=['t-stat'],
                yticklabels=avg_var_labels,
                square=True,
                linewidths=0.5,
                linecolor='white',
                cbar=False,
                ax=ax3)

    ax3.set_title('Averaged\nt-statistics', fontsize=title_fonts, pad=15)
    ax3.set_xlabel('Regressor', fontsize=x_label_fonts)
    ax3.tick_params(axis='x', rotation=0, labelsize=x_label_fonts)
    ax3.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)

    # Add this after the ax2.tick_params line:
    cbar = plt.colorbar(ax2.collections[0], cax=ax_cbar)

    # Plot 4: Averaged permutation t-statistics (averaged variables × permutations)
    sns.heatmap(tstat_avg_permutations,
                cmap='RdBu_r',
                center=0,
                vmin=vmin, vmax=vmax,
                xticklabels=perm_labels,
                yticklabels=[],
                square=True,
                linewidths=0.5,
                linecolor='white',
                cbar=False,
                ax=ax4)

    ax4.set_title('Averaged Permutation\nt-statistics', fontsize=14, pad=15)
    ax4.set_xlabel('Permutations', fontsize=x_label_fonts)
    ax4.set_ylabel('', fontsize=y_label_fonts)
    ax4.tick_params(axis='x', rotation=45, labelsize=x_label_fonts)

    plt.tight_layout()

    plt.show()

    return tstat_original, tstat_permutations, tstat_avg_original, tstat_avg_permutations


# Example usage
if __name__ == "__main__":
    # Generate t-statistics heatmaps
    orig, perm, avg_orig, avg_perm = generate_tstat_heatmaps(
        n_variables=4,
        n_permutations=8,
        seed=42
    )