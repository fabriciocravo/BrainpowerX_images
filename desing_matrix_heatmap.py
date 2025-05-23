import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


def generate_design_matrices(n_subjects=20, seed=None, save_path=None):
    """
    Generate three design matrices for different statistical tests:
    1. One-sample t-test: Column of ones
    2. Two-sample t-test: Binary group indicators
    3. Correlation test: Continuous values (age, behavioral scores, etc.)

    Parameters:
    - n_subjects: Number of subjects (rows)
    - seed: Random seed for reproducibility
    - save_path: Path to save the figure (optional)
    """

    if seed is not None:
        np.random.seed(seed)

    # Design Matrix 1: One-sample t-test (column of ones)
    design_1 = np.ones((n_subjects, 1))

    # Design Matrix 2: Two-sample t-test (binary groups)
    design_2 = np.zeros((n_subjects, 2))
    mid_point = n_subjects // 2
    design_2[:mid_point, 0] = 1  # First half: Group A (1,0)
    design_2[mid_point:, 1] = 1  # Second half: Group B (0,1)

    # Design Matrix 3: Correlation test (continuous values - e.g., age)
    # Generate realistic age-like data (20-80 years)
    design_3 = np.random.normal(45, 15, (n_subjects, 1))  # Mean=45, SD=15
    design_3 = np.clip(design_3, 18, 80)  # Clip to reasonable age range

    # Create subject labels
    sub_labels = [f'sub_{i + 1}' for i in range(n_subjects)]

    # Set up the figure with custom width ratios - make columns thinner
    fig = plt.figure(figsize=(12, max(8, n_subjects * 0.3)))  # Reduced width
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.5, 1, 0.5], wspace=0.4)  # Thinner columns

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Design Matrix 1: One-sample t-test
    sns.heatmap(design_1,
                cmap='Reds',
                cbar=False,
                vmin=0, vmax=1,  # Set explicit range so 1s appear red
                xticklabels=['Intercept'],
                yticklabels=sub_labels,
                square=False,
                ax=ax1)

    ax1.set_title('One-Sample t-test\n(Task vs Rest)', fontsize=12, pad=15)
    ax1.set_xlabel('Regressor', fontsize=10)
    ax1.set_ylabel('Subjects', fontsize=10)

    # Design Matrix 2: Two-sample t-test
    sns.heatmap(design_2,
                cmap='Reds',
                cbar=False,
                vmin=0, vmax=1,  # Same scale as matrix 1
                xticklabels=['Group A', 'Group B'],
                yticklabels=sub_labels,
                square=False,
                ax=ax2)

    ax2.set_title('Two-Sample t-test\n(Group A vs Group B)', fontsize=12, pad=15)
    ax2.set_xlabel('Group Indicators', fontsize=10)
    ax2.set_ylabel('', fontsize=10)

    # Design Matrix 3: Correlation test
    im3 = sns.heatmap(design_3,
                      cmap='viridis',
                      cbar=True,
                      cbar_kws={'label': 'Age (years)'},
                      xticklabels=['Age'],
                      yticklabels=False,  # Too many labels
                      square=False,
                      ax=ax3)

    ax3.set_title('Correlation Test\n(Age vs Connectivity)', fontsize=12, pad=15)
    ax3.set_xlabel('Continuous Variable', fontsize=10)
    ax3.set_ylabel('', fontsize=10)

    plt.suptitle('Three Statistical Test Frameworks Supported by BrainPowerX',
                 fontsize=16, y=0.95)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Design matrices saved to: {save_path}")

    plt.show()

    return design_1, design_2, design_3


def generate_single_design_matrix(design_type='one_sample', n_subjects=20, seed=None):
    """
    Generate a single design matrix for specific test type.

    Parameters:
    - design_type: 'one_sample', 'two_sample', or 'correlation'
    - n_subjects: Number of subjects
    - seed: Random seed
    """

    if seed is not None:
        np.random.seed(seed)

    if design_type == 'one_sample':
        return np.ones((n_subjects, 1))

    elif design_type == 'two_sample':
        design = np.zeros((n_subjects, 2))
        mid_point = n_subjects // 2
        design[:mid_point, 0] = 1
        design[mid_point:, 1] = 1
        return design

    elif design_type == 'correlation':
        # Generate continuous values (age-like)
        values = np.random.normal(45, 15, (n_subjects, 1))
        return np.clip(values, 18, 80)

    else:
        raise ValueError("design_type must be 'one_sample', 'two_sample', or 'correlation'")


# Example usage
if __name__ == "__main__":
    print("Generating all three design matrices...")

    design_1, design_2, design_3 = generate_design_matrices(
        n_subjects=30,
        seed=42
    )

    print(f"\nDesign matrix shapes:")
    print(f"One-sample t-test: {design_1.shape}")
    print(f"Two-sample t-test: {design_2.shape}")
    print(f"Correlation test: {design_3.shape}")

    print(f"\nSample values:")
    print(f"One-sample (first 5): {design_1[:5].flatten()}")
    print(f"Two-sample (first 5): {design_2[:5]}")
    print(f"Correlation (first 5): {design_3[:5].flatten()}")

    print("\nAll design matrices generated successfully!")