import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from controlling_variables import *

# Set style parameters for large, impressive visualization
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def generate_big_fc_matrix(n_variables=100, n_subjects=50, seed=42, save_path=None):
    """
    Generate a single large, impressive functional connectivity matrix
    with Variables × Subjects organization to show generalizability

    Parameters:
    - n_variables: Number of variables (connections/edges) - make it big for wow factor
    - n_subjects: Number of subjects
    - seed: Random seed for reproducibility
    - save_path: Path to save the figure
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate realistic FC-like data with some structure
    # Create blocks of correlated variables to show some organization
    fc_matrix = np.random.uniform(-1, 1, (n_variables, n_subjects))

    # Create the big impressive figure
    fig = plt.figure(figsize=(4, 4))

    # Main title
    fig.suptitle('Generalizable Flattened Organization\nVariables × Subjects Data Matrix',
                 fontsize=title_fonts, y=0.98)

    # Create the main heatmap - no ticks for clean look
    ax = plt.gca()

    # Use a professional colormap that looks impressive
    im = ax.imshow(fc_matrix,
                   cmap='RdBu_r',  # Classic neuroscience colormap
                   aspect='auto',  # Allow rectangular
                   interpolation='nearest',
                   vmin=-1, vmax=1)

    # Remove all ticks for clean, impressive look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add axis labels
    ax.set_xlabel('Subjects', fontsize=20)
    ax.set_ylabel('Variables', fontsize=20)

    # Add colorbar with proper scaling
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Functional Connectivity Strength', fontsize=14)
    cbar.ax.tick_params(labelsize=12)


    plt.tight_layout()

    plt.show()

    return fc_matrix


def generate_wow_factor_matrix(save_path=None):
    """
    Generate an even more impressive version with larger dimensions
    """
    print("Generating impressive Variables × Subjects matrix...")
    print("This demonstrates the generic, scalable nature of your framework!")

    # Make it big for maximum impact
    matrix = generate_big_fc_matrix(
        n_variables=300,  # Big number for wow factor
        n_subjects=150,  # Reasonable subject count
        seed=42,
    )

    print(f"\nGenerated matrix dimensions: {matrix.shape}")
    print(f"Total data points: {matrix.shape[0] * matrix.shape[1]:,}")
    print("This single visualization shows the power of your generalizable framework!")

    return matrix


# Example usage
if __name__ == "__main__":
    # Generate the wow factor version
    big_matrix = generate_wow_factor_matrix(save_path="generalizable_dataset_organization.png")