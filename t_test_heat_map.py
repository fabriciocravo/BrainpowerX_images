import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from controlling_variables import *


def generate_triple_fc_heatmap(n_variables, sub_labels_1, sub_labels_2,
                               title_label_1, title_label_2, title_label_3,
                               seed=None, save_path=None):
    """
    Generate three side-by-side heatmaps: two original matrices and their difference.

    Parameters:
    - n_variables: Number of variables/connections (rows)
    - sub_labels_1: Subject labels for first heatmap
    - sub_labels_2: Subject labels for second heatmap
    - title_label_1: Title for first heatmap
    - title_label_2: Title for second heatmap
    - title_label_3: Title for third heatmap (difference)
    - seed: Random seed for reproducibility
    - save_path: Path to save the figure (optional)
    """

    n_subjects_1 = len(sub_labels_1)
    n_subjects_2 = len(sub_labels_2)

    if seed is not None:
        np.random.seed(seed)

    # Generate random group-level data matrices
    group_matrix_1 = np.random.randn(n_variables, n_subjects_1)
    group_matrix_2 = np.random.randn(n_variables, n_subjects_2)

    # Normalize to reasonable range [-1, 1]
    fc_matrix_1 = np.tanh(group_matrix_1)
    fc_matrix_2 = np.tanh(group_matrix_2)

    # Create labels
    var_labels = [f'var{i + 1}' for i in range(n_variables)]
    sub_labels_1_formatted = [f'sub{x}' for x in sub_labels_1]
    sub_labels_2_formatted = [f'sub{x}' for x in sub_labels_2]

    # Find common subjects for subtraction
    sub_labels_3 = list(set(sub_labels_1) & set(sub_labels_2))
    sub_labels_3_formatted = [f'sub{x}' for x in sub_labels_3]

    # Get indexes for common subjects
    indexes_in_1 = [sub_labels_1.index(x) for x in sub_labels_3]
    indexes_in_2 = [sub_labels_2.index(x) for x in sub_labels_3]

    # Create difference matrix (Task - Rest for common subjects)
    fc_matrix_3 = fc_matrix_1[:, indexes_in_1] - fc_matrix_2[:, indexes_in_2]

    # TO:
    fig = plt.figure(figsize=(20, max(6, n_variables * 0.15)))  # Slightly wider for colorbar
    max_subjects = max(n_subjects_1, n_subjects_2)
    gs = gridspec.GridSpec(3, 3, width_ratios=[max_subjects, max_subjects, 0.1],
                           height_ratios=[1, 0.05, 0.8], wspace=0.3, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Rest Data
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right: Task Data
    ax3 = fig.add_subplot(gs[0, 2])  # Colorbar (much narrower)
    ax4 = fig.add_subplot(gs[2, :2])  # Bottom: Difference plot

    # First heatmap (no colorbar)
    sns.heatmap(fc_matrix_1,
                cmap='Reds',
                center=0,
                square=False,
                cbar=False,
                vmin=-1, vmax=1,
                xticklabels=sub_labels_1_formatted,
                yticklabels=var_labels,
                ax=ax1)

    ax1.set_title(title_label_1, fontsize=title_fonts, pad=20)
    ax1.set_ylabel('Variables (Flattened Connections)', fontsize=y_label_fonts)
    ax1.tick_params(axis='x', rotation=45, labelsize=x_label_fonts)
    ax1.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)
    ax1.set_aspect('equal', adjustable='box')

    # Second heatmap (with colorbar)
    im = sns.heatmap(fc_matrix_2,
                     cmap='Reds',
                     center=0,
                     square=False,
                     cbar=False,  # No automatic colorbar
                     vmin=-1, vmax=1,
                     xticklabels=sub_labels_2_formatted,
                     yticklabels=False,
                     ax=ax2)


    ax2.set_title(title_label_2, fontsize=title_fonts, pad=20)
    ax2.set_ylabel('', fontsize=y_label_fonts)
    ax2.tick_params(axis='x', rotation=45, labelsize=x_label_fonts)
    ax2.tick_params(axis='y', labelsize=y_label_fonts)
    ax2.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im.collections[0], cax=ax3, shrink=0.5)
    cbar.set_label('Functional Connectivity')

    # Third heatmap (difference - narrower)
    sns.heatmap(fc_matrix_3,
                cmap='RdBu_r',
                center=0,
                square=False,
                cbar=False,
                vmin=-2, vmax=2,
                xticklabels=sub_labels_3_formatted,
                yticklabels=var_labels,
                ax=ax4)  # Use ax3

    ax4.set_title(title_label_3, fontsize=title_fonts, pad=20)
    ax4.set_ylabel('', fontsize=y_label_fonts)
    ax4.tick_params(axis='x', rotation=45, labelsize=x_label_fonts)
    ax4.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)
    ax4.set_aspect('equal', adjustable='box')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Triple heatmap saved to: {save_path}")

    plt.show()

    return fc_matrix_1, fc_matrix_2, fc_matrix_3


# Example usage
if __name__ == "__main__":
    # Task vs Rest comparison with difference
    fc_data_1, fc_data_2, fc_diff = generate_triple_fc_heatmap(
        n_variables=4,
        sub_labels_1=[1, 2, 3, 4, 5, 6],
        sub_labels_2=[2, 3, 5, 6, 7, 8],
        title_label_1='Rest Data (Variables × Subjects)',
        title_label_2='Task Data (Variables × Subjects)',
        title_label_3=r'$\mathbf{Y}$ = Rest - Task',
        seed=60
    )