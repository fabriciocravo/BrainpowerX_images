import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


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
    var_labels = [f'var_{i + 1}' for i in range(n_variables)]
    sub_labels_1_formatted = [f'sub_{x}' for x in sub_labels_1]
    sub_labels_2_formatted = [f'sub_{x}' for x in sub_labels_2]

    # Find common subjects for subtraction
    sub_labels_3 = list(set(sub_labels_1) & set(sub_labels_2))
    sub_labels_3_formatted = [f'sub_{x}' for x in sub_labels_3]

    # Get indexes for common subjects
    indexes_in_1 = [sub_labels_1.index(x) for x in sub_labels_3]
    indexes_in_2 = [sub_labels_2.index(x) for x in sub_labels_3]

    # Create difference matrix (Task - Rest for common subjects)
    fc_matrix_3 = fc_matrix_1[:, indexes_in_1] - fc_matrix_2[:, indexes_in_2]

    # Create figure with three subplots - third one narrower
    fig = plt.figure(figsize=(18, max(6, n_variables * 0.15)))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 2], wspace=0.15)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

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

    ax1.set_title(title_label_1, fontsize=14, pad=20)
    ax1.set_xlabel('Subjects', fontsize=12)
    ax1.set_ylabel('Variables (Flattened Connections)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_aspect('equal', adjustable='box')

    # Second heatmap (with colorbar)
    sns.heatmap(fc_matrix_2,
                cmap='Reds',
                center=0,
                square=False,
                cbar=True,
                cbar_kws={'label': 'Data Values'},
                vmin=-1, vmax=1,
                xticklabels=sub_labels_2_formatted,
                yticklabels=False,
                ax=ax2)

    ax2.set_title(title_label_2, fontsize=14, pad=20)
    ax2.set_xlabel('Subjects', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_aspect('equal', adjustable='box')

    # Third heatmap (difference - narrower)
    sns.heatmap(fc_matrix_3,
                cmap='RdBu_r',  # Red-blue for positive/negative differences
                center=0,
                square=False,
                cbar=False,
                vmin=-2, vmax=2,
                xticklabels=sub_labels_3_formatted,
                yticklabels=False,
                ax=ax3)

    ax3.set_title(title_label_3, fontsize=14, pad=20)
    ax3.set_xlabel('Subjects', fontsize=12)
    ax3.set_ylabel('', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Triple heatmap saved to: {save_path}")

    plt.show()

    return fc_matrix_1, fc_matrix_2, fc_matrix_3


# Example usage
if __name__ == "__main__":
    # Task vs Rest comparison with difference
    print("Generating Task vs Rest vs Difference comparison...")
    fc_data_1, fc_data_2, fc_diff = generate_triple_fc_heatmap(
        n_variables=15,
        sub_labels_1=[1, 2, 5, 6, 7, 8, 9],
        sub_labels_2=[2, 3, 5, 6, 7, 9, 10],
        title_label_1='Task Data (Variables × Subjects)',
        title_label_2='Rest Data (Variables × Subjects)',
        title_label_3='Y = Task - Rest',
        seed=42
    )

    print("\nAll triple heatmaps generated!")
    print(f"Task data shape: {fc_data_1.shape}")
    print(f"Rest data shape: {fc_data_2.shape}")
    print(f"Difference shape: {fc_diff.shape}")