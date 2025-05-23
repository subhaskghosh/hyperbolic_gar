import os
import csv
import numpy as np
import time
import itertools
import pandas as pd
from tqdm import tqdm
import threading
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from matplotlib import rc

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
rc('text', usetex=True)
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-v0_8-ticks")

# Global shared data structures
lock = threading.Lock()
shared_weights = None  # Shared weights among nodes
local_gradients = []  # Stores gradients computed by each node

def compute_accuracy(w, X, y, dataset="mnist"):
    """
    Computes accuracy given the weights, features, and labels.
    Args:
        w (ndarray): Weight matrix or vector.
        X (ndarray): Input features.
        y (ndarray): True labels.
        dataset (str): Dataset type ("mnist" or "spambase").
    Returns:
        float: Accuracy in percentage.
    """
    logits = X @ w
    if dataset in ["mnist", "fashion_mnist", "cifar10"]:
        predictions = np.argmax(logits, axis=1)
        true_labels = np.argmax(y, axis=1)
    elif dataset == "spambase":
        predictions = (logits > 0.5).astype(int)
        true_labels = y
    return np.mean(predictions == true_labels) * 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_noise_2d(n, m, mean=0.0, variance=200):
    std_dev = np.sqrt(variance)
    noise_array = np.random.normal(loc=mean, scale=std_dev, size=(n, m))
    return noise_array

def preprocess_data(dataset="mnist"):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255.0
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255.0
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255.0  # Flatten 32x32x3
        X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255.0
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    elif dataset == "spambase":
        data = fetch_openml(name="spambase", version=1, as_frame=False, data_home='./data')
        X, y = data.data, data.target.astype(int)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    else:
        raise ValueError("Unsupported dataset. Choose 'mnist', 'fashion_mnist', 'cifar10', or 'spambase'.")

    return X_train, y_train, X_test, y_test

def loss_function(w, X, y, dataset="mnist"):
    logits = X @ w
    if dataset in ["mnist", "fashion_mnist", "cifar10"]:
        probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return -np.mean(np.sum(y * np.log(probabilities + 1e-9), axis=1))
    elif dataset == "spambase":
        probabilities = sigmoid(logits)
        return -np.mean(y * np.log(probabilities + 1e-9) + (1 - y) * np.log(1 - probabilities + 1e-9))

def compute_gradient(w, X, y, dataset="mnist"):
    logits = X @ w
    if dataset in ["mnist", "fashion_mnist", "cifar10"]:
        probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return X.T @ (probabilities - y) / X.shape[0]
    elif dataset == "spambase":
        probabilities = sigmoid(logits)
        return X.T @ (probabilities - y) / X.shape[0]

def compute_ipm_gradient(gradients, own_grad):
    """
    Manipulates gradients using the Inner Product Manipulation (IPM) attack.
    Byzantine nodes maximize their influence in the aggregated gradient.
    """
    # Filter out None values
    valid_gradients = [g for g in gradients if g is not None]

    # Ensure we have valid gradients before computing mean
    if len(valid_gradients) == 0:
        return own_grad

    valid_gradients = np.array(valid_gradients)

    honest_gradient = np.mean(valid_gradients, axis=0)  # Estimate true gradient direction
    ipm_gradient = -2.0 * honest_gradient  # Reverse and amplify honest direction
    return ipm_gradient

def compute_local_gradient(node_id, X_batch, y_batch, dataset, byzantine_nodes, noise_variance, add_noise, ipm_attack):
    global shared_weights, local_gradients
    grad = compute_gradient(shared_weights, X_batch, y_batch, dataset)
    if (add_noise or ipm_attack) and node_id in byzantine_nodes:
        if ipm_attack:
            manipulated_grad = compute_ipm_gradient(local_gradients, grad)
            grad = manipulated_grad
        else:
            noise = generate_noise_2d(grad.shape[0], grad.shape[1], variance=noise_variance)
            grad += noise
    with lock:
        local_gradients[node_id] = grad

def normalize_vector(v, norm_type="l2"):
    """
    Normalizes the input vector based on the specified normalization type.
    Args:
        v (ndarray): Input vector or matrix.
        norm_type (str): Type of normalization ("l2", "l1", "max").
    Returns:
        ndarray: Normalized vector or matrix.
    """
    if norm_type == "l2":
        norm = np.linalg.norm(v, axis=0, keepdims=True)
    elif norm_type == "l1":
        norm = np.sum(np.abs(v), axis=0, keepdims=True)
    elif norm_type == "max":
        norm = np.max(np.abs(v), axis=0, keepdims=True)
    else:
        raise ValueError("Unsupported norm_type. Choose 'l2', 'l1', or 'max'.")
    return v / (norm + 1e-9)  # Add epsilon to avoid division by zero

def l2_normalize(v, epsilon=1e-9):
    """
    Applies L2 normalization to a vector or matrix.
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + epsilon)  # Avoid division by zero

def adaptive_scaling_factor(gradients, epsilon=1e-9):
    """
    Computes adaptive λ based on the variance of the gradient norms.
    """
    norm_variance = np.var(np.linalg.norm(gradients, axis=-1))
    return 1 / (1 + np.sqrt(1 + norm_variance + epsilon))

def map_to_hyperbolic_adaptive(v, lambda_adaptive):
    """
    Maps a Euclidean vector to the Poincaré disk (hyperbolic space) with adaptive scaling.
    """
    return (lambda_adaptive * v) / (1 + np.sqrt(1 + lambda_adaptive ** 2 * np.linalg.norm(v, axis=-1) ** 2))

def map_to_euclidean_adaptive(v_h, lambda_adaptive):
    """
    Maps a vector from the Poincaré disk (hyperbolic space) back to Euclidean space.
    """
    norm_h_sq = np.linalg.norm(v_h, axis=-1, keepdims=True) ** 2
    return (lambda_adaptive * v_h) / (1 - norm_h_sq + 1e-9)  # Avoid division by zero

def map_to_hyperbolic_standard(v):
    norm_v_sq = np.linalg.norm(v, axis=-1, keepdims=True) ** 2
    # For v=0, norm_v_sq = 0, denominator is 1 + sqrt(1) = 2. Result is 0. Correct.
    return v / (1 + np.sqrt(1 + norm_v_sq))

def map_to_euclidean_standard(v_h):
    norm_h_sq = np.linalg.norm(v_h, axis=-1, keepdims=True) ** 2
    # Ensure norm_h_sq < 1 for stability
    return (2 * v_h) / (1 - norm_h_sq + 1e-9)

def hyperbolic_geometric_median(gradients, scaling_factor=2.0, norm_type="l2", max_iter=100, tol=1e-5):
    """
    Computes the hyperbolic geometric median of a set of gradients.
    Args:
        gradients (ndarray): Array of gradients (nodes x dimensions).
        scaling_factor (float): Scaling factor for hyperbolic mapping.
        norm_type (str): Type of normalization ("l2").
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
    Returns:
        ndarray: Aggregated gradient using the hyperbolic geometric median.
    """
    gradients = np.array(gradients)  # Ensure it's a NumPy array

    # Check for empty gradients
    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients shape (num_nodes, d), got {gradients.shape}")

    num_nodes, d = gradients.shape

    # Map all gradients to hyperbolic space
    hyperbolic_gradients = np.array([map_to_hyperbolic_standard(g) for g in gradients])

    # Initialize geometric median as the hyperbolic mean
    g_med_h = np.mean(hyperbolic_gradients, axis=0)  # Shape: (d,)

    for _ in range(max_iter):
        dists = np.array([poincare_distance(g_med_h, g) for g in hyperbolic_gradients])
        weights = 1 / (dists + 1e-9)  # Shape: (num_nodes, 1)
        new_median = np.sum(weights[:, np.newaxis] * hyperbolic_gradients, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - g_med_h) < tol:
            break
        g_med_h = new_median

    aggregated_gradient = map_to_euclidean_standard(g_med_h)
    return aggregated_gradient.reshape(-1, 1)  # Shape: (d, 1)

def hyperbolic_geometric_median_adaptive(gradients, max_iter=100, tol=1e-5, epsilon=1e-9):
    """
    Computes the hyperbolic geometric median of a set of gradients with adaptive scaling.

    Args:
        gradients (ndarray): Array of gradients (num_nodes, dimensions).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        ndarray: Aggregated gradient using the hyperbolic geometric median.
    """
    gradients = np.array(gradients)  # Ensure it's a NumPy array

    if gradients.ndim != 2:
        raise ValueError(f"Expected gradients shape (num_nodes, d), got {gradients.shape}")

    num_nodes, d = gradients.shape

    lambda_adaptive = adaptive_scaling_factor(gradients, epsilon)
    hyperbolic_gradients = np.array([map_to_hyperbolic_adaptive(g, lambda_adaptive) for g in gradients])
    g_med_h = np.mean(hyperbolic_gradients, axis=0)

    for _ in range(max_iter):
        dists = np.array([poincare_distance(g_med_h, g) for g in hyperbolic_gradients])
        weights = 1 / (dists + epsilon)
        new_median = np.sum(weights[:, np.newaxis] * hyperbolic_gradients, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - g_med_h) < tol:
            break
        g_med_h = new_median

    aggregated_gradient = map_to_euclidean_adaptive(g_med_h, lambda_adaptive)
    return aggregated_gradient.reshape(-1, d)  # Ensure correct shape

def compute_hyperbolic_aggregation():
    """
    Computes the global gradient using hyperbolic geometric median aggregation.
    Returns:
        ndarray: Aggregated gradient in Euclidean space.
    """
    global local_gradients
    gradients = np.array(local_gradients)
    return hyperbolic_geometric_median(gradients)

def poincare_map(vector, epsilon=1e-2):
    norm_sq = np.linalg.norm(vector) ** 2
    if norm_sq >= 1 - epsilon:
        norm_sq = 1 - epsilon
    return vector / (1 + np.sqrt(1 + norm_sq))

def poincare_distance(u, v, epsilon=1e-5):
    """
    Computes the hyperbolic (Poincaré) distance between two vectors.
    """
    norm_u_sq = np.clip(np.linalg.norm(u, axis=-1) ** 2, 0, 1 - epsilon)
    norm_v_sq = np.clip(np.linalg.norm(v, axis=-1) ** 2, 0, 1 - epsilon)
    num = 2 * np.linalg.norm(u - v, axis=-1) ** 2
    denom = (1 - norm_u_sq) * (1 - norm_v_sq)
    return np.arccosh(1 + num / (denom + epsilon))

def euclidean_map(hyperbolic_vector):
    norm_sq = np.linalg.norm(hyperbolic_vector)**2
    return 2 * hyperbolic_vector / (1 - norm_sq)

def curvature_aware_weighted_krum(local_gradients, beta, curvature_factor=0.1):
    n = len(local_gradients)
    k = n - beta - 2
    scores = []
    for i, grad_i in enumerate(local_gradients):
        distances = []
        for j, grad_j in enumerate(local_gradients):
            if i != j:
                d = poincare_distance(grad_i, grad_j)
                weighted_distance = d * (1 + curvature_factor * np.linalg.norm(grad_j))
                distances.append(weighted_distance)
        scores.append((i, np.sum(sorted(distances)[:k])))
    best_index = min(scores, key=lambda x: x[1])[0]
    return local_gradients[best_index]


def euclidean_geometric_median_standalone(gradients_flat, max_iter=100, tol=1e-5, epsilon=1e-9):
    """
    Computes the Euclidean geometric median of a set of flattened gradients.
    Args:
        gradients_flat (ndarray): Array of flattened gradients (num_nodes x flattened_dim).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        epsilon (float): Small constant to avoid division by zero in weights.
    Returns:
        ndarray: Aggregated flattened gradient (flattened_dim, 1) using Euclidean geometric median.
    """
    # gradients_flat is already a NumPy array, e.g., (num_nodes, d_flat)
    if gradients_flat.ndim != 2:
        raise ValueError(f"Expected gradients_flat shape (num_nodes, d_flat), got {gradients_flat.shape}")

    num_nodes, d_flat = gradients_flat.shape
    if num_nodes == 0:
        return np.zeros((d_flat, 1))  # Or handle as error

    g_med_e = np.mean(gradients_flat, axis=0)  # Initialize with mean, shape (d_flat,)

    for _ in range(max_iter):
        # Distances from current median estimate to each gradient vector
        # gradients_flat is (num_nodes, d_flat), g_med_e is (d_flat,)
        # Broadcasting will handle (gradients_flat - g_med_e) correctly
        distances = np.linalg.norm(gradients_flat - g_med_e, axis=1)  # shape (num_nodes,)

        # Check for points at the median
        at_median_indices = np.where(distances < epsilon)[0]
        if len(at_median_indices) > 0:
            # If one of the points is the median, return it
            # For simplicity, if any point is very close, we consider it converged to that point
            # or that point is the median.
            # A more robust check: if g_med_e itself is one of the input points.
            # However, if multiple points are at g_med_e, this is fine.
            # If g_med_e coincides with one of the data points, Weiszfeld might have issues with 0 distance.
            # For now, we consider it as the median if this happens.
            is_one_of_points = False
            for i_idx in at_median_indices:
                if np.allclose(g_med_e, gradients_flat[i_idx], atol=epsilon):
                    g_med_e = gradients_flat[i_idx]  # Ensure it's exactly one of the points
                    is_one_of_points = True
                    break
            if is_one_of_points:  # if g_med_e is one of the input points, it's the solution
                break

        weights = 1.0 / (distances + epsilon)  # shape (num_nodes,)

        # new_median = sum (w_i * grad_i) / sum (w_i)
        # weights[:, np.newaxis] makes weights (num_nodes, 1) for broadcasting with gradients_flat (num_nodes, d_flat)
        new_median = np.sum(weights[:, np.newaxis] * gradients_flat, axis=0) / np.sum(weights)  # shape (d_flat,)

        if np.linalg.norm(new_median - g_med_e) < tol:
            break
        g_med_e = new_median

    return g_med_e.reshape(-1, 1)  # Ensure shape (d_flat, 1)

def trimmed_mean_standalone(gradients_flat, trim_fraction=0.1):
    """
    Computes the trimmed mean of flattened gradients.
    Trims a fraction from both ends based on L2 norms of the flattened gradient vectors.
    Args:
        gradients_flat (ndarray): Array of flattened gradients (num_nodes x flattened_dim).
        trim_fraction (float): Fraction of gradients to trim from each end (0.0 to <0.5).
    Returns:
        ndarray: Aggregated flattened gradient (flattened_dim, 1) using trimmed mean.
    """
    if gradients_flat.ndim != 2:
        raise ValueError(f"Expected gradients_flat shape (num_nodes, d_flat), got {gradients_flat.shape}")

    num_nodes, d_flat = gradients_flat.shape
    if num_nodes == 0:
        return np.zeros((d_flat, 1))

    if not (0 <= trim_fraction < 0.5):
        print(f"Warning: trim_fraction {trim_fraction} is out of valid range [0, 0.5). Using 0 (mean).")
        trim_fraction = 0.0
        # return np.mean(gradients_flat, axis=0).reshape(-1, 1) # Or proceed with trim_fraction = 0

    num_to_trim_each_end = int(np.floor(num_nodes * trim_fraction))

    if num_to_trim_each_end * 2 >= num_nodes and num_nodes > 0:  # Check num_nodes > 0 to avoid issues if it was 0
        # Avoid trimming all gradients if num_nodes is small and trim_fraction is large
        # print(f"Warning: Trimming ({num_to_trim_each_end} from each end) would remove all gradients or leave too few for {num_nodes} nodes. Defaulting to mean.")
        # Return mean of all if trimming is not possible
        return np.mean(gradients_flat, axis=0).reshape(-1, 1)

    # Calculate L2 norms of each flattened gradient vector
    norms = np.linalg.norm(gradients_flat, axis=1)  # shape (num_nodes,)

    # Get indices that would sort the norms
    sorted_indices = np.argsort(norms)

    # Select gradients to keep (trimming from both ends)
    start_index = num_to_trim_each_end
    end_index = num_nodes - num_to_trim_each_end  # Exclusive end for slicing

    indices_to_keep = sorted_indices[start_index:end_index]

    if len(indices_to_keep) == 0:
        # This should ideally be caught by the check above, but as a safeguard
        # print(f"Warning: No gradients left after trimming for {num_nodes} nodes and trim_fraction {trim_fraction}. Defaulting to mean.")
        return np.mean(gradients_flat, axis=0).reshape(-1, 1)

    trimmed_gradients_subset = gradients_flat[indices_to_keep, :]  # shape (num_kept, d_flat)

    return np.mean(trimmed_gradients_subset, axis=0).reshape(-1, 1)  # shape (d_flat, 1)

def centered_clipping(
        gradients_flat,
        initial_center_v,
        clipping_threshold_tau=10.0,
        num_inner_iterations_L=1,
        epsilon=1e-9
):
    """
    Implements the Centered Clipping algorithm as described by Karimireddy et al.

    Args:
        gradients_flat (ndarray): Array of flattened gradients (num_nodes x flattened_dim).
        clipping_threshold_tau (float): The clipping threshold τ for the norm of deviation.
        initial_center_v (ndarray): Initial guess for the center v (flattened_dim,).
                                      This could be the previous round's aggregate or
                                      a robust estimate like coordinate-wise median of current gradients.
        num_inner_iterations_L (int): Number of inner iterations L.
        epsilon (float): Small constant for numerical stability (e.g., in norm calculation).

    Returns:
        ndarray: Aggregated flattened gradient (flattened_dim, 1).
    """
    if gradients_flat.ndim != 2:
        raise ValueError(f"Expected gradients_flat shape (num_nodes, d_flat), got {gradients_flat.shape}")

    num_nodes, d_flat = gradients_flat.shape
    if num_nodes == 0:
        return np.zeros((d_flat, 1))

    if initial_center_v.shape != (d_flat,):
        raise ValueError(f"Expected initial_center_v shape ({d_flat},), got {initial_center_v.shape}")

    current_center_v = np.copy(initial_center_v)  # Work with a copy

    for iteration_l in range(num_inner_iterations_L):
        clipped_deviations_ci = []
        for i in range(num_nodes):
            grad_i = gradients_flat[i, :]  # shape (d_flat,)
            deviation_i = grad_i - current_center_v  # shape (d_flat,)
            norm_deviation_i = np.linalg.norm(deviation_i)

            if norm_deviation_i > epsilon:  # Avoid division by zero if deviation is zero
                scale_factor = min(1.0, clipping_threshold_tau / norm_deviation_i)
            else:
                scale_factor = 1.0  # No scaling if deviation is zero

            ci = deviation_i * scale_factor
            clipped_deviations_ci.append(ci)

        if not clipped_deviations_ci:  # Should not happen if num_nodes > 0
            # This would mean an issue, perhaps all gradients were identical to center
            # and norms were zero.
            break

            # Sum of clipped deviations
        sum_clipped_deviations = np.sum(np.array(clipped_deviations_ci), axis=0)  # shape (d_flat,)

        # Update the center v
        current_center_v = current_center_v + (1.0 / num_nodes) * sum_clipped_deviations
        # Note: The pseudocode might imply this for L > 1.
        # If L=1, the output is v_initial + mean(clipped_deviations).
        # If L>1, v is iteratively refined.

    return current_center_v.reshape(-1, 1)  # shape (d_flat, 1)

def hyperbolic_krum_aggregate(gradients, beta):
    """
    Given a 2D NumPy array of flattened gradients (shape: [num_nodes, d]),
    maps each gradient into hyperbolic space (with scaling_factor=1.0),
    computes pairwise Poincaré distances (squared) between gradients,
    and then performs a KRUM-style selection using the hyperbolic distances.
    Returns the selected gradient mapped back to Euclidean space.

    Parameters:
        gradients (np.ndarray): Array of shape [num_nodes, d]
        beta (int): Number of Byzantine nodes assumed

    Returns:
        np.ndarray: The aggregated (selected) gradient (flattened, shape: [d])
    """
    # Map each gradient into hyperbolic space.
    hyper_gradients = np.array([map_to_hyperbolic_standard(g) for g in gradients])
    n = hyper_gradients.shape[0]
    k = n - beta - 2  # as in standard KRUM
    scores = []
    # Compute a score for each candidate.
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                d = poincare_distance(hyper_gradients[i], hyper_gradients[j])
                distances.append(d ** 2)
        distances = np.array(distances)
        score = np.sum(np.sort(distances)[:k])
        scores.append(score)
    best_index = np.argmin(scores)
    best_grad_h = hyper_gradients[best_index]
    # Map back to Euclidean space.
    best_grad_euclid = map_to_euclidean_standard(best_grad_h)
    return best_grad_euclid

def aggregate_gradients(aggregation_type="average",
                        beta=0,  # Used by Krum
                        trim_fraction_for_trimmed_mean=0.1,
                        clipping_threshold_for_norm_clipping=1.0,
                        curvature_factor_for_cak=0.1):  # Parameter for curvature_aware_krum
    global shared_weights, local_gradients  # Make sure these are accessible if not passed as args

    # No lock needed here if local_gradients is fully populated before this call
    # and shared_weights is only read for its shape.
    # with lock: # If local_gradients could be modified concurrently, lock it.
    #             # For now, assuming it's stable when this function is called.

    valid_local_gradients = [g for g in local_gradients if g is not None and isinstance(g, np.ndarray)]

    if not valid_local_gradients:
        if shared_weights is not None and isinstance(shared_weights, np.ndarray):
            print("Warning: No valid local gradients to aggregate. Returning zeros.")
            return np.zeros_like(shared_weights)
        else:
            # This is a critical error if shared_weights is also None or not an array
            raise ValueError(
                "Cannot aggregate: No valid local gradients and shared_weights is not a valid NumPy array to infer shape.")

    # All valid_local_gradients are expected to be NumPy arrays of the same shape
    # This shape is assumed to be the `original_shape`
    original_shape = shared_weights.shape
    flattened_dim = np.prod(original_shape)
    num_valid_nodes = len(valid_local_gradients)

    avg_gradient_np = None  # Initialize

    if aggregation_type == "average":
        avg_gradient_np = np.mean(np.array(valid_local_gradients), axis=0)

    elif aggregation_type == "median":  # Coordinate-wise median
        avg_gradient_np = np.median(np.array(valid_local_gradients), axis=0)

    elif aggregation_type == "hyperbolic_krum":
        # New branch: first flatten each node’s gradient, then run hyperbolic KRUM.
        gradients_array = np.array([g.flatten() for g in local_gradients])
        avg_gradient_np = hyperbolic_krum_aggregate(gradients_array, beta)

    elif aggregation_type == "krum":
        # Krum requires at least 2*beta + 3 workers in some definitions, or more generally n > 2*beta
        # Your Krum condition was num_valid_nodes <= 2 * beta + 2 (i.e., num_valid_nodes < 2*beta + 3)
        if num_valid_nodes <= 2 * beta + 2:
            print(f"Warning: Not enough nodes ({num_valid_nodes}) for Krum with beta={beta}. Defaulting to mean.")
            avg_gradient_np = np.mean(np.array(valid_local_gradients), axis=0)
        else:
            k_krum = num_valid_nodes - beta - 2
            if k_krum <= 0:  # Ensure k_krum is positive
                print(
                    f"Warning: k_krum for Krum is <=0 ({k_krum}) with {num_valid_nodes} nodes and beta={beta}. Defaulting to mean.")
                avg_gradient_np = np.mean(np.array(valid_local_gradients), axis=0)
            else:
                scores = []
                for i in range(num_valid_nodes):
                    grad_i_arr = valid_local_gradients[i]
                    distances = []
                    for j in range(num_valid_nodes):
                        if i == j:
                            continue
                        grad_j_arr = valid_local_gradients[j]
                        distances.append(np.linalg.norm(grad_i_arr - grad_j_arr) ** 2)

                    distances.sort()
                    scores.append(sum(distances[:k_krum]))

                best_index = np.argmin(scores)
                avg_gradient_np = valid_local_gradients[best_index]

    elif aggregation_type == "hyperbolic":
        # Assumes hyperbolic_geometric_median handles (num_nodes, flattened_dim)
        # and returns (flattened_dim, 1) or (flattened_dim,)
        gradients_for_aggregation_flat = np.array([g.flatten() for g in valid_local_gradients])
        if gradients_for_aggregation_flat.shape[0] > 0:  # Check if any gradients are left
            avg_gradient_flat = hyperbolic_geometric_median(gradients_for_aggregation_flat)
            avg_gradient_np = avg_gradient_flat.reshape(original_shape)
        else:  # Should be caught by earlier check on valid_local_gradients
            avg_gradient_np = np.zeros(original_shape)

    elif aggregation_type == "hyperbolic_adaptive":
        # Ensure hyperbolic_geometric_median_adaptive exists and handles input/output similarly
        gradients_for_aggregation_flat = np.array([g.flatten() for g in valid_local_gradients])
        if gradients_for_aggregation_flat.shape[0] > 0:
            avg_gradient_flat = hyperbolic_geometric_median_adaptive(gradients_for_aggregation_flat)
            # Check shape of avg_gradient_flat from adaptive version: if (d_flat, d_flat) that's an issue.
            # Assuming it's (d_flat,) or (d_flat, 1)
            if avg_gradient_flat.ndim > 1 and avg_gradient_flat.shape[1] != 1 and avg_gradient_flat.shape[
                0] != flattened_dim:
                print(
                    f"Warning: hyperbolic_geometric_median_adaptive returned unexpected shape {avg_gradient_flat.shape}. Expected ({flattened_dim},) or ({flattened_dim},1).")
                # Fallback or error
                avg_gradient_np = np.zeros(original_shape)  # Fallback to zeros
            else:
                avg_gradient_np = avg_gradient_flat.reshape(original_shape)
        else:
            avg_gradient_np = np.zeros(original_shape)

    elif aggregation_type == "curvature_aware_krum":
        # This function needs the list of original shaped gradients.
        # Ensure curvature_aware_weighted_krum exists and handles input/output as expected.
        # It also needs 'beta'
        if num_valid_nodes > 0:
            # map valid_local_gradients (Euclidean) to hyperbolic space first for this method
            # This requires map_to_hyperbolic_standard to be defined
            hyperbolic_grads_for_cak = [map_to_hyperbolic_standard(g) for g in valid_local_gradients]
            selected_hyperbolic_grad = curvature_aware_weighted_krum(hyperbolic_grads_for_cak, beta,
                                                                     curvature_factor=curvature_factor_for_cak)
            avg_gradient_np = map_to_euclidean_standard(selected_hyperbolic_grad)  # map back
        else:
            avg_gradient_np = np.zeros(original_shape)

    elif aggregation_type == "euclidean_median":
        gradients_for_aggregation_flat = np.array([g.flatten() for g in valid_local_gradients])
        if gradients_for_aggregation_flat.shape[0] > 0:
            avg_gradient_flat = euclidean_geometric_median_standalone(gradients_for_aggregation_flat)
            avg_gradient_np = avg_gradient_flat.reshape(original_shape)
        else:
            avg_gradient_np = np.zeros(original_shape)

    elif aggregation_type == "trimmed_mean":
        gradients_for_aggregation_flat = np.array([g.flatten() for g in valid_local_gradients])
        if gradients_for_aggregation_flat.shape[0] > 0:
            avg_gradient_flat = trimmed_mean_standalone(gradients_for_aggregation_flat,
                                                        trim_fraction=trim_fraction_for_trimmed_mean)
            avg_gradient_np = avg_gradient_flat.reshape(original_shape)
        else:
            avg_gradient_np = np.zeros(original_shape)


    elif aggregation_type == "centered_clipping":
        gradients_for_aggregation_flat = np.array([g.flatten() for g in valid_local_gradients])
        if gradients_for_aggregation_flat.shape[0] > 0:
            initial_v = np.median(gradients_for_aggregation_flat, axis=0)
        else:  # Should be caught by earlier checks
            initial_v = np.zeros(flattened_dim)
        if gradients_for_aggregation_flat.shape[0] > 0:
            avg_gradient_flat = centered_clipping(gradients_for_aggregation_flat, initial_v)
            avg_gradient_np = avg_gradient_flat.reshape(original_shape)
        else:
            avg_gradient_np = np.zeros(original_shape)

    else:
        raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    # Final check to ensure the output gradient has the correct shape
    if avg_gradient_np is None:  # Should not happen if all types are handled
        print(f"Error: avg_gradient_np is None for aggregation_type {aggregation_type}. Defaulting to zeros.")
        return np.zeros(original_shape)

    if avg_gradient_np.shape != original_shape:
        try:
            avg_gradient_np = avg_gradient_np.reshape(original_shape)
        except ValueError as e:
            print(
                f"Critical Error: Could not reshape aggregated gradient for type '{aggregation_type}'. Original shape: {original_shape}, Agg grad shape: {avg_gradient_np.shape}. Error: {e}")
            # Fallback to a zero gradient of the correct shape to prevent crash, but this indicates a problem.
            return np.zeros(original_shape)

    return avg_gradient_np

def distributed_sgd(X, y, X_test, y_test, num_nodes, byzantine_nodes, noise_variance, lr=0.01, epochs=10, batch_size=128, dataset="mnist", ipm_attack=False, add_noise=False, aggregation_type="average"):
    global shared_weights, local_gradients
    n_samples, d = X.shape
    num_classes = y.shape[1] if dataset in ["mnist", "fashion_mnist", "cifar10"] else 1
    shared_weights = np.zeros((d, num_classes))
    local_gradients = [None] * num_nodes
    loss = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            node_batch_size = len(X_batch) // num_nodes
            threads = []

            for node_id in range(num_nodes):
                node_start = node_id * node_batch_size
                node_end = (node_id + 1) * node_batch_size if node_id < num_nodes - 1 else len(X_batch)
                X_node_batch = X_batch[node_start:node_end]
                y_node_batch = y_batch[node_start:node_end]

                thread = threading.Thread(target=compute_local_gradient,
                                          args=(node_id, X_node_batch, y_node_batch, dataset, byzantine_nodes, noise_variance, add_noise, ipm_attack))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            avg_gradient = aggregate_gradients(aggregation_type, beta=len(byzantine_nodes))
            if avg_gradient.shape != shared_weights.shape:
                avg_gradient = avg_gradient.reshape(shared_weights.shape)  # Ensure correct shape

            shared_weights -= lr * avg_gradient

        current_loss = loss_function(shared_weights, X, y, dataset)
        # Accuracy measured after each epoch; the final values after 500 epochs will be used.
        train_accuracy = compute_accuracy(shared_weights, X, y, dataset)
        test_accuracy = compute_accuracy(shared_weights, X_test, y_test, dataset)
        loss.append(current_loss)
        # Uncomment the following line to print progress per epoch:
        print(f"{aggregation_type}: Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    return shared_weights, loss

def run_experiments():
    """
    Run experiments with varying datasets, Byzantine fractions, noise variances, attack modes,
    and now different numbers of samples per node. Accuracy is measured after 500 epochs.
    """
    datasets = ["spambase", "mnist", "fashion_mnist", "cifar10"]
    byzantine_fractions = [0.1, 0.2, 0.3, 0.4]
    noise_variances = [1.0, 10.0, 100.0, 200.0]
    attack_modes = ["add_noise"]  # Only one can be True at a time

    # New: Different number of samples per node to experiment with
    sample_per_node_options = [16, 32, 64, 128]

    results_file = "results.csv"

    # Ensure results file exists with headers (including samples_per_node)
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "byzantine_fraction", "noise_variance", "attack_mode", "samples_per_node",
                "noisy_train_accuracy", "noisy_test_accuracy",
                "noiseless_train_accuracy", "noiseless_test_accuracy",
                "median_train_accuracy", "median_test_accuracy",
                "krum_train_accuracy", "krum_test_accuracy",
                "hyperbolic_train_accuracy", "hyperbolic_test_accuracy",
                "euclidean_median_train_accuracy", "euclidean_median_test_accuracy",
                "trimmed_mean_train_accuracy", "trimmed_mean_test_accuracy",
                "centered_clipping_train_accuracy", "centered_clipping_test_accuracy",
                "hyperbolic_krum_train_accuracy", "hyperbolic_krum_test_accuracy"
            ])

    # Load existing results to avoid redundant runs
    existing_results = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
    if not existing_results.empty:
        existing_configs = set(existing_results.apply(
            lambda row: (row.dataset, row.byzantine_fraction, row.noise_variance, row.attack_mode, row.samples_per_node), axis=1))
    else:
        existing_configs = set()

    # Generate valid experiment configurations
    experiment_configs = []
    for dataset, byzantine_fraction, attack_mode, samples_per_node in itertools.product(datasets, byzantine_fractions, attack_modes, sample_per_node_options):
        if attack_mode == "add_noise":
            for noise_variance in noise_variances:
                experiment_configs.append((dataset, byzantine_fraction, noise_variance, attack_mode, samples_per_node))
        else:  # ipm_attack, noise variance is not applicable
            experiment_configs.append((dataset, byzantine_fraction, 0.0, attack_mode, samples_per_node))

    for dataset, byzantine_fraction, noise_variance, attack_mode, samples_per_node in tqdm(experiment_configs,
                                                                                          desc="Running experiments"):
        config_tuple = (dataset, byzantine_fraction, noise_variance, attack_mode, samples_per_node)
        if config_tuple in existing_configs:
            continue  # Skip completed runs

        # Load dataset
        X_train, y_train, X_test, y_test = preprocess_data(dataset)
        num_nodes = 20 if dataset == "spambase" else 32
        lr = 0.01
        epochs = 500  # Final accuracy is measured after 500 epochs
        batch_size = num_nodes * samples_per_node
        num_byzantine_nodes = max(1, int(byzantine_fraction * num_nodes))
        byzantine_nodes = set(np.random.choice(num_nodes, num_byzantine_nodes, replace=False))

        ipm_attack = (attack_mode == "ipm_attack")
        add_noise = (attack_mode == "add_noise")

        print(f"Running {dataset}, Byzantine Fraction: {byzantine_fraction}, Noise Variance: {noise_variance}, Attack: {attack_mode}, Samples per Node: {samples_per_node}")

        # Run distributed SGD for different aggregation methods
        noisy_weights, noisy_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes, byzantine_nodes, noise_variance,
                                                           lr, epochs, batch_size, dataset, ipm_attack, add_noise)
        noiseless_weights, noiseless_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes, set(), noise_variance,
                                                                   lr, epochs, batch_size, dataset, False, False)
        median_weights, median_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes,
                                                             byzantine_nodes, noise_variance, lr, epochs, batch_size, dataset,
                                                             ipm_attack, add_noise, aggregation_type="median")
        krum_weights, krum_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes, byzantine_nodes, noise_variance,
                                                         lr, epochs, batch_size, dataset, ipm_attack, add_noise,
                                                         aggregation_type="krum")
        hyperbolic_weights, hyperbolic_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes,
                                                                     byzantine_nodes, noise_variance, lr, epochs, batch_size, dataset,
                                                                     ipm_attack, add_noise,
                                                                     aggregation_type="hyperbolic")
        euclidean_median_weights, euclidean_median_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes,
                                                                          byzantine_nodes, noise_variance, lr, epochs,
                                                                          batch_size, dataset,
                                                                          ipm_attack, add_noise,
                                                                          aggregation_type="euclidean_median")
        trimmed_mean_weights, trimmed_mean_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes,
                                                                  byzantine_nodes, noise_variance, lr, epochs,
                                                                  batch_size, dataset,
                                                                  ipm_attack, add_noise,
                                                                  aggregation_type="trimmed_mean")
        centered_clipping_weights, centered_clipping_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes,
                                                                            byzantine_nodes, noise_variance, lr, epochs,
                                                                            batch_size, dataset,
                                                                            ipm_attack, add_noise,
                                                                            aggregation_type="centered_clipping")
        hyperbolic_krum_weights, hyperbolic_krum_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes,
                                                                        byzantine_nodes, noise_variance, lr, epochs,
                                                                        batch_size,
                                                                        dataset,
                                                                        ipm_attack, add_noise,
                                                                        aggregation_type="hyperbolic_krum")

        if attack_mode == "add_noise":
            attack_type = 'Gaussian Byzantine'
        else:
            attack_type = 'Inner Product Manipulation'

        if dataset == 'spambase':
            dataset_name = 'Spambase'
        elif dataset == 'mnist':
            dataset_name = 'MNIST'
        elif dataset == 'fashion_mnist':
            dataset_name = 'Fashion-MNIST'
        elif dataset == 'cifar10':
            dataset_name = 'CIFAR-10'

        # Compute accuracy after 500 epochs
        noisy_train_acc = compute_accuracy(noisy_weights, X_train, y_train, dataset)
        noisy_test_acc = compute_accuracy(noisy_weights, X_test, y_test, dataset)
        noiseless_train_acc = compute_accuracy(noiseless_weights, X_train, y_train, dataset)
        noiseless_test_acc = compute_accuracy(noiseless_weights, X_test, y_test, dataset)
        median_train_acc = compute_accuracy(median_weights, X_train, y_train, dataset)
        median_test_acc = compute_accuracy(median_weights, X_test, y_test, dataset)
        krum_train_acc = compute_accuracy(krum_weights, X_train, y_train, dataset)
        krum_test_acc = compute_accuracy(krum_weights, X_test, y_test, dataset)
        hyperbolic_train_acc = compute_accuracy(hyperbolic_weights, X_train, y_train, dataset)
        hyperbolic_test_acc = compute_accuracy(hyperbolic_weights, X_test, y_test, dataset)
        euclidean_median_train_acc = compute_accuracy(euclidean_median_weights, X_train, y_train, dataset)
        euclidean_median_test_acc = compute_accuracy(euclidean_median_weights, X_test, y_test, dataset)
        trimmed_mean_train_acc = compute_accuracy(trimmed_mean_weights, X_train, y_train, dataset)
        trimmed_mean_test_acc = compute_accuracy(trimmed_mean_weights, X_test, y_test, dataset)
        centered_clipping_train_acc = compute_accuracy(centered_clipping_weights, X_train, y_train, dataset)
        centered_clipping_test_acc = compute_accuracy(centered_clipping_weights, X_test, y_test, dataset)
        hyperbolic_krum_train_acc = compute_accuracy(hyperbolic_krum_weights, X_train, y_train, dataset)
        hyperbolic_krum_test_acc = compute_accuracy(hyperbolic_krum_weights, X_test, y_test, dataset)

        # Save loss curves
        loss_curve_file = f'loss_curves/{dataset}_frac_{byzantine_fraction}_noise_{noise_variance}_attack_{attack_mode}_samples_{samples_per_node}.csv'
        os.makedirs("loss_curves", exist_ok=True)
        pd.DataFrame({
            "epoch": range(epochs),
            "noisy_loss": noisy_loss,
            "noiseless_loss": noiseless_loss,
            "median_loss": median_loss,
            "krum_loss": krum_loss,
            "hyperbolic_loss": hyperbolic_loss,
            "euclidean_median_loss": euclidean_median_loss,
            "trimmed_mean_loss": trimmed_mean_loss,
            "centered_clipping_loss": centered_clipping_loss,
            "hyperbolic_krum_loss": hyperbolic_krum_loss
        }).to_csv(loss_curve_file, index=False)

        # Generate and save the plot
        os.makedirs("sgd_plots", exist_ok=True)
        sns.set_theme(style="whitegrid")
        colors = sns.color_palette("Spectral")
        plt.figure(figsize=(12, 8))
        plt.plot(range(epochs), noisy_loss, label="With Noise", color=colors[0], linewidth=2)
        plt.plot(range(epochs), noiseless_loss, label="Without Noise", color=colors[1], linewidth=2)
        plt.plot(range(epochs), median_loss, label="Coordinate-Wise Median", color=colors[2], linewidth=2)
        plt.plot(range(epochs), krum_loss, label="Krum", color=colors[3], linewidth=2)
        plt.plot(range(epochs), hyperbolic_loss, label="Hyperbolic Aggregation", color=colors[4], linewidth=2)
        plt.plot(range(epochs), euclidean_median_loss, label="Euclidean Median", color=colors[0], linewidth=2)
        plt.plot(range(epochs), trimmed_mean_loss, label="Trimmed Mean", color=colors[1], linewidth=2)
        plt.plot(range(epochs), centered_clipping_loss, label="Centered Clipping", color=colors[2], linewidth=2)
        plt.plot(range(epochs), hyperbolic_krum_loss, label="Hyperbolic Krum Aggregation", color=colors[5], linewidth=2)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(
            f"Loss Curve ({dataset_name})\nByzantine Fraction: {byzantine_fraction}, Noise Variance: {noise_variance}, Attack: {attack_type}, Samples/Node: {samples_per_node}",
            fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12, loc="upper right")
        plt.savefig(f'sgd_plots/{dataset}_frac_{byzantine_fraction}_noise_{noise_variance}_attack_{attack_mode}_samples_{samples_per_node}.pdf',
                    dpi=300, bbox_inches="tight")
        plt.close()

        # Save results to CSV
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset, byzantine_fraction, noise_variance, attack_mode, samples_per_node,
                noisy_train_acc, noisy_test_acc,
                noiseless_train_acc, noiseless_test_acc,
                median_train_acc, median_test_acc,
                krum_train_acc, krum_test_acc,
                hyperbolic_train_acc, hyperbolic_test_acc,
                euclidean_median_train_acc, euclidean_median_test_acc,
                trimmed_mean_train_acc, trimmed_mean_test_acc,
                centered_clipping_train_acc, centered_clipping_test_acc,
                hyperbolic_krum_train_acc, hyperbolic_krum_test_acc
            ])

        print(f"Saved results and plots for {dataset}, Byzantine Fraction: {byzantine_fraction}, Noise Variance: {noise_variance}, Attack: {attack_mode}, Samples per Node: {samples_per_node}")

if __name__ == "__main__":
    run_experiments()
