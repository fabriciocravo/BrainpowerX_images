import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


def generate_design_matrices(sub_labels_1, sub_labels_2, sub_labels_3, seed=None, save_path=None):
    """
    Generate three design matrices for different statistical tests:
    1. One-sample t-test: Column of ones
    2. Two-sample t-test: Binary group indicators
    3. Correlation test: Continuous values (age, behavioral scores, etc.)

    Parameters:
    - sub_labels_1: Subject list for one-sample t-test
    - sub_labels_2: Subject list for two-sample t-test
    - sub_labels_3: Subject list for correlation test
    - seed: Random seed for reproducibility
    - save_path: Path to save the figure (optional)
    """

    label_font_size = 14
    title_font_size = 16

    if seed is not None:
        np.random.seed(seed)

    n_subjects_1 = len(sub_labels_1)
    n_subjects_2 = len(sub_labels_2)
    n_subjects_3 = len(sub_labels_3)

    # Design Matrix 1: One-sample t-test (column of ones)
    design_1 = np.ones((n_subjects_1, 1))

    # Design Matrix 2: Two-sample t-test (binary groups)
    design_2 = np.zeros((n_subjects_2, 2))
    mid_point = n_subjects_2 // 2
    design_2[:mid_point, 0] = 1  # First half: Group A (1,0)
    design_2[mid_point:, 1] = 1  # Second half: Group B (0,1)

    # Design Matrix 3: Correlation test (continuous values - e.g., age)
    category_values = [1, 12, 24, 11, 10, 10]  # Your specific values
    design_3 = np.array([[category_values[i]] for i in range(min(len(category_values), n_subjects_3))])

    # Create subject labels
    sub_labels_1_formatted = [f'sub_{x}' for x in sub_labels_1]
    sub_labels_2_formatted = [f'sub_{x}' for x in sub_labels_2]
    sub_labels_3_formatted = [f'sub_{x}' for x in sub_labels_3]

    # Calculate max subjects for consistent figure height
    max_subjects = max(n_subjects_1, n_subjects_2, n_subjects_3)

    # Set up the figure with much thicker columns (as thick as 3 letters)
    fig = plt.figure(figsize=(6, max(6, max_subjects * 0.25)))  # Reduced width for thicker columns
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.4, 0.8, 0.4], wspace=0.6)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Design Matrix 1: One-sample t-test
    sns.heatmap(design_1,
                cmap='Reds',
                cbar=False,
                vmin=0, vmax=1,
                xticklabels=['Intercept'],
                yticklabels=sub_labels_1_formatted,
                linewidths=0.5,  # Add lines between cells
                linecolor='white',  # White lines to separate subjects
                ax=ax1)

    ax1.set_title(r'$\mathbf{X}$ t-test', fontsize=title_font_size, pad=15)
    ax1.tick_params(axis='both', which='major', labelsize=label_font_size)  # ADD THIS LINE

    # Design Matrix 2: Two-sample t-test
    sns.heatmap(design_2,
                cmap='Reds',
                cbar=False,
                vmin=0, vmax=1,
                xticklabels=['Group A', 'Group B'],
                yticklabels=sub_labels_2_formatted,
                linewidths=0.5,  # Add lines between cells
                linecolor='white',  # White lines to separate subjects/groups
                ax=ax2)

    ax2.set_title(r'$\mathbf{X}$ t2-test', fontsize=title_font_size, pad=15)
    ax2.tick_params(axis='both', which='major', labelsize=label_font_size)  # ADD THIS LINE

    # Design Matrix 3: Correlation test
    sns.heatmap(design_3,
                cmap='viridis',
                cbar=True,
                cbar_kws={'label': 'Score'},
                xticklabels=[''],
                yticklabels=sub_labels_3_formatted,
                linewidths = 0.5,  # Add lines between cells
                linecolor = 'white',  # White lines to separate subjects/groups
                ax=ax3)

    ax3.set_title(r'$\mathbf{X}$ corr-test', fontsize=title_font_size, pad=15)
    ax3.tick_params(axis='both', which='major', labelsize=label_font_size)  # ADD THIS LINE

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Design matrices saved to: {save_path}")

    plt.show()

    return design_1, design_2, design_3


# Example usage
if __name__ == "__main__":
    # Different subject lists for each test type
    sub_list_1 = [2, 5, 6, 7]  # 5 subjects for one-sample
    sub_list_2 = [1, 6, 2, 3]  # 10 subjects for two-sample
    sub_list_3 = [1, 2, 5, 7, 8, 9]  # 5 subjects for correlation

    design_1, design_2, design_3 = generate_design_matrices(
        sub_labels_1=sub_list_1,
        sub_labels_2=sub_list_2,
        sub_labels_3=sub_list_3,
        seed=42
    )