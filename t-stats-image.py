import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from controlling_variables import *


def generate_tstat_power_heatmaps(n_variables=4, n_repetitions=8, seed=None, save_path=None):
    """
    Generate two simple heatmaps matching the existing figure style:
    1. Ground truth t-statistics (single column)
    2. Power analysis across repetitions (multiple columns showing detection rates)

    Parameters:
    - n_variables: Number of variables (default 4 to match figure)
    - n_repetitions: Number of repetitions for power analysis
    - seed: Random seed for reproducibility
    - save_path: Path to save the figure
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate ground truth t-statistics (4 variables only)
    ground_truth_t = np.array([
        2.1,  # Strong positive
        -0.5,  # Weak negative
        1.8,  # Moderate positive
        -2.3  # Strong negative
    ])[:n_variables]

    gt_positive = ground_truth_t > 0
    gt_positive = gt_positive.astype(int).reshape(-1, 1)  # Make it a column vector
    gt_positive = 3*gt_positive


    # Generate power simulation data
    # Higher absolute t-stats should have higher detection rates
    power_data = np.zeros((n_variables, n_repetitions))

    for i, true_t in enumerate(ground_truth_t):
        # REPLACE the existing power_data generation with this:
        for i, true_t in enumerate(ground_truth_t):
            abs_t = abs(true_t)
            if abs_t > 2.0:
                detection_prob = 0.85
            elif abs_t > 1.0:
                detection_prob = 0.60
            else:
                detection_prob = 0.20

            # Generate detections for both positive and negative effects
            detections = np.random.binomial(1, detection_prob, n_repetitions)

            if true_t > 0:
                # Positive effects: use +3 when detected
                power_data[i, :] = 3 * detections
            else:
                # Negative effects: use -3 when detected
                power_data[i, :] = -3 * detections


    power_data = 3*power_data

    # Create variable labels (match your figure)
    var_labels = [f'var{i + 1}' for i in range(n_variables)]
    rep_labels = [f'rep{i + 1}' for i in range(n_repetitions)]

    # Create simple side-by-side plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(13, 4))

    # Panel G: Ground truth t-statistics (single column, match your colors)
    gt_matrix = ground_truth_t.reshape(-1, 1)

    sns.heatmap(gt_matrix,
                cmap='RdBu_r',  # Match your blue-red scheme
                center=0,
                cbar=True,
                xticklabels=['t-stat'],
                yticklabels=var_labels,
                vmin=-3, vmax=3,
                square=True,
                ax=ax1)

    ax1.set_title('Ground Truth\nT-statistics',
                  fontsize=title_fonts)
    ax1.tick_params(axis='x', rotation=0, labelsize=x_label_fonts)
    ax1.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)
    ax1.set_ylabel('')

    sns.heatmap(gt_positive,
                cmap='RdBu_r',  # Match your blue-red scheme
                cbar=False,
                xticklabels=['t-stat'],
                yticklabels=[''],
                vmin=-3, vmax=3,
                square=True,
                ax=ax2)

    ax2.set_title('Ground Truth\n' +
                  r't $>$ 0',
                  fontsize=title_fonts)

    ax2.tick_params(axis='x', rotation=0, labelsize=x_label_fonts)
    ax2.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)
    ax2.set_ylabel('')

    # ADD THIS NEW SECTION after the ax2 heatmap:
    # Ground truth negative effects (t < 0)
    gt_negative = ground_truth_t < 0
    gt_negative = gt_negative.astype(int).reshape(-1, 1)
    gt_negative = -3 * gt_negative  # Use -3 for negative effects

    sns.heatmap(gt_negative,
                cmap='RdBu_r',
                cbar=False,
                xticklabels=['t-stat'],
                yticklabels=[''],
                vmin=-3, vmax=3,
                square=True,
                ax=ax3)

    ax3.set_title('Ground Truth\n' + r't < 0', fontsize=title_fonts)
    ax3.tick_params(axis='x', rotation=0, labelsize=x_label_fonts)
    ax3.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)
    ax3.set_ylabel('')

    # Panel H: Power analysis (match your colors)
    sns.heatmap(power_data,
                cmap='RdBu_r',  # Match your red scheme from original heatmaps
                cbar=False,
                xticklabels=rep_labels,
                yticklabels=var_labels,
                vmin=-3, vmax=3,
                square=True,
                ax=ax4)

    ax4.set_title('Detection across repetitions\n' +
                  r'P-value $\leq$ 0.05',
                  fontsize=title_fonts)
    ax4.set_ylabel('')
    ax4.tick_params(axis='x', rotation=45, labelsize=x_label_fonts)
    ax4.tick_params(axis='y', rotation=0, labelsize=y_label_fonts)
    ax4.set_xlabel('Repetitions')

    plt.tight_layout()

    plt.show()

    # Calculate and print power summary
    power_rates = np.mean(power_data / 3, axis=1)
    print("\nPower Analysis Summary:")
    print("Variable | Ground Truth t-stat | Power Rate")
    print("-" * 45)
    for i, (t_val, power) in enumerate(zip(ground_truth_t, power_rates)):
        print(f"var_{i + 1:2d}   | {t_val:8.1f}          | {power:6.1%}")

    return ground_truth_t, power_data, power_rates


# Example usage
if __name__ == "__main__":
    gt_stats, power_matrix, power_summary = generate_tstat_power_heatmaps(
        n_variables=4,  # Match your figure
        n_repetitions=10,
        seed=42
    )