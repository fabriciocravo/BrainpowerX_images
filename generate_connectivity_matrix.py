import numpy as np

# Create 268x268 zero matrix
matrix = np.zeros((268, 268))

# Add 4 connections between distant brain regions across hemispheres
# Connection 1: Left frontal (node 10) <-> Right frontal (node 150)
matrix[9, 149] = matrix[149, 9] = 1  # Changed to binary (1 or 0)

# Connection 2: Left parietal (node 50) <-> Right parietal (node 190)
matrix[49, 189] = matrix[189, 49] = 1  # Changed to binary (1 or 0)

# Connection 3: Left temporal (node 80) <-> Right occipital (node 240)
matrix[79, 239] = matrix[239, 79] = 1  # Changed to binary (1 or 0)

# Connection 4: Left occipital (node 30) <-> Right temporal (node 220)
matrix[29, 219] = matrix[219, 29] = 1  # Changed to binary (1 or 0)

# Save as CSV file (not text file) - BioImage Suite prefers CSV
np.savetxt('connectivity.csv', matrix, fmt='%d', delimiter=',')  # Changed to CSV format with integer format
