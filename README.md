# Distributed Optimization with Faulty Nodes

This repository implements a robust distributed stochastic gradient descent (SGD) framework designed for Internet-of-Things (IoT) applications. Our framework is particularly geared toward scenarios where a fraction of worker nodes may behave arbitrarily (Byzantine faults). To mitigate the impact of adversarial updates, we implement robust gradient aggregation methods, including a novel hyperbolic geometric median aggregator. The codebase includes experiments on both fully-connected models (using benchmark datasets such as MNIST, Fashion-MNIST, CIFAR-10, and Spambase) and a temporal graph neural network for traffic prediction based on the METR-LA dataset.

## Overview

In distributed learning, each worker node computes a local gradient on its mini-batch and sends it to a central aggregator. However, in adversarial settings some nodes may report corrupted gradients. Our approach employs robust aggregation rules to filter out these outliers. In particular, we compare:
- **Average Aggregation** (with a baseline using honest nodes and a variant with adversarial Gaussian noise),
- **Hyperbolic Geometric Median Aggregation** (which maps gradients into hyperbolic space using the Poincaré ball model, aggregates robustly, and maps back to Euclidean space).

We evaluate these methods on standard datasets as well as on a traffic forecasting task using a Temporal Graph Neural Network (based on the A3T-GCN cell).

## Dependencies

The project is implemented in Python (3.8+). The main libraries used include:
- **NumPy** and **Pandas** for numerical computations and data manipulation.
- **TensorFlow** (with Keras) for constructing and training fully-connected models.
- **PyTorch** and **torch_geometric_temporal** for temporal graph neural network models.
- **scikit-learn** for data preprocessing and dataset splitting.
- **Matplotlib** and **Seaborn** for visualization.
- **tqdm** for progress bars.
- **threading** (from the Python standard library) to simulate distributed gradient computation.
