
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

def compute_half_sums(data, half_size):
    """
    Splits the data into two halves and computes the sum for each half along the first dimension.

    Parameters:
        data (numpy array): The input data of shape (n_steps, 2).
        half_size (int): The size of each half.

    Returns:
        tuple: (first_half_sum, second_half_sum) for the two halves.
    """
    first_half = data[:half_size]
    second_half = data[half_size:]

    first_half_sum = np.sum(first_half, axis=0)
    second_half_sum = np.sum(second_half, axis=0)

    return first_half_sum, second_half_sum

def classify_sums(first_half_sum, second_half_sum):
    """
    Classifies the sums based on the given conditions and returns the label.

    Parameters:
        first_half_sum (numpy array): Sum of the first half.
        second_half_sum (numpy array): Sum of the second half.

    Returns:
        int: Classification label (0 or 1).
    """
    if first_half_sum[0] < second_half_sum[0]:
        return 0
    else:
        return 1

def process_trials(x, num_trials, n_steps):
    """
    Processes multiple trials, computes half sums, and classifies them.

    Parameters:
        x (numpy array): Input data of shape (num_trials, n_steps, 2).
        num_trials (int): Number of trials to process.
        n_steps (int): Number of steps in each trial.

    Returns:
        int: Classification label for the processed trial.
    """
    half_size = n_steps // 2

    # Compute sums and classify the trial
    first_half_sum, second_half_sum = compute_half_sums(x, half_size)
    print("first_half_sum", first_half_sum)
    print("second_half_sum", second_half_sum)
    label = classify_sums(first_half_sum, second_half_sum)

    return label

def generate_low_correlation_cov_matrix(M, var_range=(1, 2), corr_range=(-0.1, 0.1)):
    """
    Generates an MxM covariance matrix with low correlations across nodes.

    Parameters:
        M (int): Size of the covariance matrix (MxM).
        var_range (tuple): Range for variances (diagonal elements).
        corr_range (tuple): Range for low correlations (off-diagonal elements).

    Returns:
        np.ndarray: Generated covariance matrix.
    """
    # Step 1: Generate variances (diagonal elements)
    variances = np.random.uniform(var_range[0], var_range[1], size=M)

    # Step 2: Generate a matrix with random low correlations (off-diagonal)
    low_corr_matrix = np.random.uniform(corr_range[0], corr_range[1], size=(M, M))

    # Step 3: Make the matrix symmetric
    low_corr_matrix = (low_corr_matrix + low_corr_matrix.T) / 2

    # Step 4: Set the diagonal elements to the variances
    np.fill_diagonal(low_corr_matrix, variances)

    # Visualize the covariance matrix
    sns.heatmap(low_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Covariance Matrix with Low Correlations (Size: {M}x{M})")
    plt.show()
    return low_corr_matrix

def mvnrand(mean, covariance_matrix, dim, n=1):
    """
    Generate random samples from a multivariate normal distribution.

    Parameters:
        mean (array-like or scalar): Mean vector. If scalar, it is expanded into a vector of zeros.
        covariance_matrix (array-like or scalar): Covariance matrix (R).
        dim (int): Dimension of the data.
        n (int): Number of samples to generate. Default is 1.

    Returns:
        np.ndarray: Random samples from the multivariate normal distribution.
    """
    if dim == 1:  # 1D case
        mean = np.array([mean]) if np.isscalar(mean) else mean
        samples = np.random.normal(mean, math.sqrt(covariance_matrix), size=n)
    else:  # Multidimensional case
        mean = np.zeros(len(covariance_matrix)) if np.isscalar(mean) and mean == 0 else mean
        samples = np.random.multivariate_normal(mean, covariance_matrix, size=n)
    return samples
