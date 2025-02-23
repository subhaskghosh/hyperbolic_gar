import os
import csv
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from matplotlib import rc

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
rc('text', usetex=True)
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-v0_8-ticks")

# Set device and batch_size for DataLoaders
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
shuffle = True


#############################################
# Model: TemporalGNN using A3TGCN2
#############################################
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # A3TGCN2 cell: in_channels=node_features, out_channels=32, periods=number of timesteps in input
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods, batch_size=batch_size)
        # Linear layer for final prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        # x shape: [B, num_nodes, num_features, T]
        h = self.tgnn(x, edge_index)  # returns h: [B, num_nodes, T]
        h = F.relu(h)
        h = self.linear(h)  # output: [B, num_nodes, T]
        return h

#############################################
# Loss Function
#############################################
loss_fn = torch.nn.MSELoss()

#############################################
# Evaluation Metrics
#############################################
def evaluate_model(model, data_loader, static_edge_index, loss_fn):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mape = 0.0
    total_count = 0
    with torch.no_grad():
        for encoder_inputs, labels in data_loader:
            y_hat = model(encoder_inputs, static_edge_index)
            loss = loss_fn(y_hat, labels)
            batch_count = encoder_inputs.size(0) * labels.size(1) * labels.size(2)
            total_loss += loss.item() * batch_count
            mae = F.l1_loss(y_hat, labels, reduction='sum')
            total_mae += mae.item()
            epsilon = 1e-6
            mape = torch.abs((y_hat - labels) / (labels + epsilon)).sum().item()
            total_mape += mape
            total_count += batch_count
    avg_mse = total_loss / total_count
    rmse = np.sqrt(avg_mse)
    avg_mae = total_mae / total_count
    avg_mape = (total_mape / total_count) * 100
    return avg_mse, avg_mae, avg_mape, rmse


#############################################
# Robust Aggregation Helpers (NumPy implementations)
#############################################
def normalize_and_aggregate(flat_gradients, agg_func, beta=0):
    """
    Normalize each row of flat_gradients (shape [num_nodes, d]) to unit norm,
    aggregate them via agg_func (e.g. np.mean, np.median, or functions like krum_aggregate),
    and then rescale by the median norm.
    """
    norms = np.linalg.norm(flat_gradients, axis=1, keepdims=True)
    normalized = flat_gradients / (norms + 1e-9)
    agg_normalized = agg_func(normalized, axis=0)
    median_norm = np.median(norms)
    return agg_normalized * median_norm


def map_to_hyperbolic(v, scaling_factor=2.0):
    return scaling_factor * v / (1 + np.sqrt(1 + np.linalg.norm(v, axis=-1) ** 2))


def map_to_euclidean(v_h, scaling_factor=1.0):
    norm_h_sq = np.linalg.norm(v_h, axis=-1, keepdims=True) ** 2
    return (scaling_factor * v_h) / (1 - norm_h_sq + 1e-9)  # Avoid division by zero


def poincare_distance(u, v, epsilon=1e-5):
    norm_u_sq = np.clip(np.linalg.norm(u) ** 2, 0, 1 - epsilon)
    norm_v_sq = np.clip(np.linalg.norm(v) ** 2, 0, 1 - epsilon)
    num = 2 * np.linalg.norm(u - v) ** 2
    denom = (1 - norm_u_sq) * (1 - norm_v_sq)
    return np.arccosh(1 + num / (denom + epsilon))


def krum_aggregate(gradients, beta):
    n = gradients.shape[0]
    k = n - beta - 2
    scores = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                distances.append(np.linalg.norm(gradients[i] - gradients[j]) ** 2)
        scores.append(np.sum(np.sort(distances)[:k]))
    best_index = np.argmin(scores)
    return gradients[best_index]


def hyperbolic_geometric_median(gradients, scaling_factor=2.0, max_iter=100, tol=1e-5):
    gradients = np.array(gradients)
    hyperbolic_gradients = np.array([map_to_hyperbolic(g, scaling_factor) for g in gradients])
    g_med_h = np.mean(hyperbolic_gradients, axis=0)
    for _ in range(max_iter):
        dists = np.array([poincare_distance(g_med_h, g) for g in hyperbolic_gradients])
        weights = 1 / (dists + 1e-9)
        new_median = np.sum(weights[:, None] * hyperbolic_gradients, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - g_med_h) < tol:
            break
        g_med_h = new_median
    aggregated = map_to_euclidean(g_med_h, scaling_factor)
    return aggregated


def robust_aggregate_vector(gradients, aggregation_type="average", beta=0):
    """
    Given a list/array of node gradients (each with shape e.g. d1, d2, ...),
    flatten them to shape [num_nodes, d], normalize each row to unit norm,
    aggregate via the chosen aggregator, rescale by the median norm, and reshape back.
    Supported aggregation_type: 'average', 'median', 'krum', 'hyperbolic'
    """
    gradients = np.array(gradients)
    expected_shape = gradients.shape[1:]
    flat_gradients = np.reshape(gradients, (gradients.shape[0], -1))
    norms = np.linalg.norm(flat_gradients, axis=1, keepdims=True)
    median_norm = np.median(norms)
    normalized_gradients = flat_gradients / (norms + 1e-9)

    if aggregation_type == "average":
        agg_flat = np.mean(normalized_gradients, axis=0) * median_norm
    elif aggregation_type == "median":
        agg_flat = np.median(normalized_gradients, axis=0) * median_norm
    elif aggregation_type == "krum":
        agg_flat = krum_aggregate(normalized_gradients, beta) * median_norm
    elif aggregation_type == "hyperbolic":
        agg_normalized = hyperbolic_geometric_median(normalized_gradients, scaling_factor=2.0)
        agg_flat = agg_normalized * median_norm
    else:
        raise ValueError("Unsupported aggregation type. Use 'average', 'median', 'krum', or 'hyperbolic'.")

    return np.reshape(agg_flat, expected_shape)


#############################################
# Distributed SGD for Temporal GNN using robust aggregation (with SGD update)
#############################################
def distributed_sgd_temporal(model, optimizer, train_loader, test_loader, static_edge_index,
                             num_nodes, byzantine_nodes, noise_variance, aggregation_type,
                             ipm_attack, num_epochs):
    model.train()
    epoch_loss_list = []
    epoch_metrics_list = []  # Each element: (train_metrics, test_metrics)
    for epoch in range(num_epochs):
        batch_losses = []
        # For each batch in the training data:
        for encoder_inputs, labels in train_loader:
            batch_size = encoder_inputs.size(0)
            node_batch_size = batch_size // num_nodes
            node_gradients = []
            # Simulate each node’s gradient computation:
            for node in range(num_nodes):
                start = node * node_batch_size
                end = (node + 1) * node_batch_size if node < num_nodes - 1 else batch_size
                x_node = encoder_inputs[start:end]
                y_node = labels[start:end]
                optimizer.zero_grad()
                y_hat = model(x_node, static_edge_index)
                loss = loss_fn(y_hat, y_node)
                loss.backward()
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1))
                flat_grad = torch.cat(grads).detach().cpu().numpy()
                # Apply Byzantine attack if this node is Byzantine:
                if node in byzantine_nodes:
                    if ipm_attack:
                        flat_grad = -2.0 * flat_grad
                    else:
                        noise = np.random.normal(0, np.sqrt(noise_variance), size=flat_grad.shape)
                        flat_grad = flat_grad + noise
                node_gradients.append(flat_grad)
            # Robustly aggregate node gradients using the chosen aggregator:
            agg_grad = robust_aggregate_vector(node_gradients, aggregation_type, beta=len(byzantine_nodes))
            # Unflatten aggregated gradient and update model parameters using vanilla SGD:
            pointer = 0
            aggregated_grads = []
            for p in model.parameters():
                numel = p.numel()
                grad_segment = agg_grad[pointer:pointer + numel].reshape(p.shape)
                aggregated_grads.append(torch.tensor(grad_segment, dtype=p.dtype, device=p.device))
                pointer += numel
            # Set gradients and perform SGD update
            for p, g in zip(model.parameters(), aggregated_grads):
                p.grad = g
            optimizer.step()
            batch_losses.append(loss.item())
        avg_epoch_loss = np.mean(batch_losses)
        epoch_loss_list.append(avg_epoch_loss)
        # Evaluate model on train and test sets
        train_metrics = evaluate_model(model, train_loader, static_edge_index, loss_fn)
        test_metrics = evaluate_model(model, test_loader, static_edge_index, loss_fn)
        epoch_metrics_list.append((train_metrics, test_metrics))
        print(f"Epoch {epoch + 1} Loss = {avg_epoch_loss:.4f}")
        print(
            f"Train: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, MAPE={train_metrics[2]:.2f}%, RMSE={train_metrics[3]:.4f}")
        print(
            f"Test:  MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, MAPE={test_metrics[2]:.2f}%, RMSE={test_metrics[3]:.4f}")
    return model, epoch_loss_list, epoch_metrics_list


#############################################
# Experiment Runner
#############################################
def run_experiments():
    # Load METR-LA dataset
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    # Create DataLoaders for training and testing
    train_input = np.array(train_dataset.features)  # shape: (num_samples, num_nodes, num_features, T)
    train_target = np.array(train_dataset.targets)  # shape: (num_samples, num_nodes, T)
    train_x_tensor = torch.from_numpy(train_input).float().to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target).float().to(DEVICE)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle,
                                               drop_last=True)

    test_input = np.array(test_dataset.features)
    test_target = np.array(test_dataset.targets)
    test_x_tensor = torch.from_numpy(test_input).float().to(DEVICE)
    test_target_tensor = torch.from_numpy(test_target).float().to(DEVICE)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    # Load the static graph (edge_index) from the first snapshot
    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(DEVICE)
        break

    # Parameter grid
    byzantine_fractions = [0.2, 0.3]
    noise_variances = [100.0, 200.0]
    aggregation_methods = ['hyperbolic', 'krum', 'average', 'median']
    sample_per_node_options = [4]  # fixed (batch_size for temporal data is fixed)
    num_nodes = 8  # simulated nodes

    results_file = "results_temporal.csv"
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["byzantine_fraction", "noise_variance", "aggregation_method", "baseline",
                             "epoch", "train_loss", "train_MSE", "train_MAE", "train_MAPE", "train_RMSE",
                             "test_MSE", "test_MAE", "test_MAPE", "test_RMSE"])

    existing_results = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
    if not existing_results.empty:
        existing_configs = set(existing_results.apply(
            lambda row: (row["byzantine_fraction"], row["noise_variance"], row["aggregation_method"], row["baseline"]),
            axis=1))
    else:
        existing_configs = set()

    # Create experiment configurations as product of grid parameters.
    experiment_configs = list(
        itertools.product(byzantine_fractions, noise_variances, aggregation_methods, sample_per_node_options))

    # For each configuration:
    for byz_frac, noise_var, agg_method, _ in experiment_configs:
        # For aggregator 'average', we run two experiments:
        #   baseline: no Byzantine attack (byzantine_nodes empty, ipm_attack=False, add_noise=False)
        #   noisy: with Byzantine nodes as computed.
        baseline_flags = [True] if agg_method == "average" else [False]
        # For non-average aggregators, only run the "noisy" experiment.
        if agg_method != "average":
            baseline_flags = [False]

        for baseline in baseline_flags:
            config_tuple = (byz_frac, noise_var, agg_method, baseline)
            if config_tuple in existing_configs:
                print(f"Skipping config {config_tuple} as already exists")
                continue

            # Determine Byzantine nodes and attack flags:
            if baseline:
                byzantine_nodes = set()  # No Byzantine nodes for baseline
                ipm_attack = False
                add_noise = False
            else:
                num_byz = max(1, int(byz_frac * num_nodes))
                byzantine_nodes = set(np.random.choice(num_nodes, num_byz, replace=False))
                # If noise_var is 0.0, we use ipm_attack; else we add noise.
                ipm_attack = (noise_var == 0.0)
                add_noise = (noise_var > 0.0)

            # Create a fresh model and use SGD optimizer (vanilla SGD)
            model = TemporalGNN(node_features=2, periods=12, batch_size=batch_size).to(DEVICE)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

            print(f'Using: {DEVICE}')
            print('Net\'s state_dict:')
            total_param = 0
            for param_tensor in model.state_dict():
                print(param_tensor, '\t', model.state_dict()[param_tensor].size())
                total_param += np.prod(model.state_dict()[param_tensor].size())
            print('Net\'s total params:', total_param)
            # --------------------------------------------------
            print('Optimizer\'s state_dict:')  # If you notice here the Attention is a trainable parameter
            for var_name in optimizer.state_dict():
                print(var_name, '\t', optimizer.state_dict()[var_name])

            num_epochs = 25
            print(f"Running config: Byzantine Fraction = {byz_frac}, Noise Variance = {noise_var}, "
                  f"Aggregation = {agg_method}, Baseline = {baseline}")
            model, train_loss_list, epoch_metrics_list = distributed_sgd_temporal(
                model, optimizer, train_loader, test_loader, static_edge_index,
                num_nodes, byzantine_nodes, noise_var, aggregation_type=agg_method,
                ipm_attack=ipm_attack, num_epochs=num_epochs
            )

            # Save per-epoch results to CSV
            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                for epoch, train_loss in enumerate(train_loss_list, start=1):
                    train_metrics, test_metrics = epoch_metrics_list[epoch - 1]
                    writer.writerow([byz_frac, noise_var, agg_method, baseline, epoch,
                                     train_loss,
                                     train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3],
                                     test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]])
            # Plot training loss curve
            os.makedirs("sgd_plots_temporal", exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_epochs + 1), train_loss_list, label="Train Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.title(f"Temporal GNN Distributed SGD\nByz Fraction: {byz_frac}, Noise: {noise_var}, "
                      f"Agg: {agg_method}, Baseline: {baseline}")
            plt.legend()
            plot_file = f"sgd_plots_temporal/byz_{byz_frac}_noise_{noise_var}_agg_{agg_method}_baseline_{baseline}.pdf"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved results and plot for config: {config_tuple}")


if __name__ == "__main__":
    run_experiments()