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

def map_to_hyperbolic(v, scaling_factor=2.0, norm_type="l2"):
    """
    Maps a Euclidean vector to the Poincaré disk (hyperbolic space).
    Uses specified normalization and scaling factor.
    """
    return scaling_factor * v / (1 + np.sqrt(1 + np.linalg.norm(v, axis=-1) ** 2))

def map_to_euclidean(v_h, scaling_factor=2.0):
    """
    Maps a vector from the Poincaré disk (hyperbolic space) back to Euclidean space.
    Uses the specified scaling factor.
    """
    norm_h_sq = np.linalg.norm(v_h, axis=-1, keepdims=True) ** 2
    return (scaling_factor * v_h) / (1 - norm_h_sq + 1e-9)  # Avoid division by zero

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
    hyperbolic_gradients = np.array([map_to_hyperbolic(g, scaling_factor, norm_type) for g in gradients])

    # Initialize geometric median as the hyperbolic mean
    g_med_h = np.mean(hyperbolic_gradients, axis=0)  # Shape: (d,)

    for _ in range(max_iter):
        dists = np.array([poincare_distance(g_med_h, g) for g in hyperbolic_gradients])
        weights = 1 / (dists + 1e-9)  # Shape: (num_nodes, 1)
        new_median = np.sum(weights[:, np.newaxis] * hyperbolic_gradients, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - g_med_h) < tol:
            break
        g_med_h = new_median

    aggregated_gradient = map_to_euclidean(g_med_h, scaling_factor)
    return aggregated_gradient.reshape(-1, 1)  # Shape: (d, 1)

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
    hyper_gradients = np.array([map_to_hyperbolic(g, scaling_factor=1.0) for g in gradients])
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
    best_grad_euclid = map_to_euclidean(best_grad_h, scaling_factor=1.0)
    return best_grad_euclid

def aggregate_gradients(aggregation_type="average", beta=0):
    global shared_weights, local_gradients
    with lock:
        if aggregation_type == "average":
            avg_gradient = np.mean(local_gradients, axis=0)
        elif aggregation_type == "median":
            avg_gradient = np.median(local_gradients, axis=0)  # Coordinate-wise median
        elif aggregation_type == "krum":
            n = len(local_gradients)
            k = n - beta - 2
            scores = []
            for i, grad in enumerate(local_gradients):
                distances = [np.linalg.norm(grad - other_grad) ** 2 for j, other_grad in enumerate(local_gradients) if j != i]
                scores.append((i, np.sum(sorted(distances)[:k])))
            best_index = min(scores, key=lambda x: x[1])[0]
            avg_gradient = local_gradients[best_index]
        elif aggregation_type == "hyperbolic":
            gradients_array = np.array([g.flatten() for g in local_gradients])  # Shape (num_nodes, d)
            avg_gradient = hyperbolic_geometric_median(gradients_array, scaling_factor=2.0, norm_type="l2")
        elif aggregation_type == "hyperbolic_krum":
            # New branch: first flatten each node’s gradient, then run hyperbolic KRUM.
            gradients_array = np.array([g.flatten() for g in local_gradients])
            avg_gradient = hyperbolic_krum_aggregate(gradients_array, beta)
        else:
            raise ValueError("Unsupported aggregation type. Choose 'average', 'median', 'krum', 'hyperbolic', or 'hyperbolic_adaptive'.")
        return avg_gradient

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
        print(f"aggregation_type: {aggregation_type}, Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    return shared_weights, loss

def run_experiments():
    """
    Run experiments with varying datasets, Byzantine fractions, noise variances, attack modes,
    and now different numbers of samples per node. Accuracy is measured after 500 epochs.
    """
    datasets = ["spambase", "mnist", "fashion_mnist", "cifar10"]
    byzantine_fractions = [0.1, 0.2, 0.3, 0.4]
    noise_variances = [0.01, 0.1, 1.0, 10.0, 100.0, 200.0]
    attack_modes = ["add_noise", "ipm_attack"]  # Only one can be True at a time

    # New: Different number of samples per node to experiment with
    sample_per_node_options = [16, 32, 64, 128]

    results_file = "results_hk.csv"

    # Ensure results file exists with headers (including samples_per_node)
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "byzantine_fraction", "noise_variance", "attack_mode", "samples_per_node",
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
        hyperbolic_krum_weights, hyperbolic_krum_loss = distributed_sgd(X_train, y_train, X_test, y_test, num_nodes,
                                                              byzantine_nodes, noise_variance, lr, epochs, batch_size,
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
        hyperbolic_krum_train_acc = compute_accuracy(hyperbolic_krum_weights, X_train, y_train, dataset)
        hyperbolic_krum_test_acc = compute_accuracy(hyperbolic_krum_weights, X_test, y_test, dataset)

        # Save loss curves
        loss_curve_file = f'loss_curves_hk/{dataset}_frac_{byzantine_fraction}_noise_{noise_variance}_attack_{attack_mode}_samples_{samples_per_node}.csv'
        os.makedirs("loss_curves_hk", exist_ok=True)
        pd.DataFrame({
            "epoch": range(epochs),
            "hyperbolic_krum_loss": hyperbolic_krum_loss
        }).to_csv(loss_curve_file, index=False)

        # Generate and save the plot
        os.makedirs("sgd_plots_hk", exist_ok=True)
        sns.set_theme(style="whitegrid")
        colors = sns.color_palette("Spectral")
        plt.figure(figsize=(12, 8))
        plt.plot(range(epochs), hyperbolic_krum_loss, label="Hyperbolic Krum Aggregation", color=colors[5], linewidth=2)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(
            f"Loss Curve ({dataset_name})\nByzantine Fraction: {byzantine_fraction}, Noise Variance: {noise_variance}, Attack: {attack_type}, Samples/Node: {samples_per_node}",
            fontsize=16, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12, loc="upper right")
        plt.savefig(f'sgd_plots_hk/{dataset}_frac_{byzantine_fraction}_noise_{noise_variance}_attack_{attack_mode}_samples_{samples_per_node}.pdf',
                    dpi=300, bbox_inches="tight")
        plt.close()

        # Save results to CSV
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset, byzantine_fraction, noise_variance, attack_mode, samples_per_node,
                hyperbolic_krum_train_acc, hyperbolic_krum_test_acc
            ])

        print(f"Saved results and plots for {dataset}, Byzantine Fraction: {byzantine_fraction}, Noise Variance: {noise_variance}, Attack: {attack_mode}, Samples per Node: {samples_per_node}")

if __name__ == "__main__":
    run_experiments()