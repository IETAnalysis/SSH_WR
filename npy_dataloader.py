import os
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

# === Seed Control for Reproducibility ===
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_final_features(single_stats, A_dist, mixed_features, hist_features, selected_keys):
    """
    Construct final feature vector by horizontally stacking selected feature sets.

    Args:
        single_stats (np.ndarray): Statistics of individual signals.
        A_dist (np.ndarray): Distribution of amplitude bins.
        mixed_features (np.ndarray): Signal-level global features.
        hist_features (np.ndarray): 100-bin histogram features.
        selected_keys (tuple): Short codes indicating selected feature types (e.g., 'Sc', 'Ac').

    Returns:
        np.ndarray: Combined feature vector for a burst.
    """
    features_map = {
        'Sc': single_stats,
        'Ac': A_dist,
        'Sm': mixed_features,
        'Tm': hist_features,
    }
    selected_features = [features_map[key] for key in selected_keys if key in features_map]
    return np.hstack(selected_features)


def compute_signal_features(signal):
    """
    Compute basic signal-domain features.

    Returns:
        list: Max value, variance, and signal energy.
    """
    return [
        np.max(signal),
        np.var(signal),
        np.sum(signal ** 2),
    ]


def extract_features(pkt_lengths, pkt_times, sample_num, beta, final_dim, combination):
    """
    Compute full feature vectors for each burst in the dataset.

    Args:
        pkt_lengths (ndarray): Packet lengths per burst.
        pkt_times (ndarray): Packet timestamps per burst.
        sample_num (int): Sampling resolution for signal construction.
        beta (float): Soft-sigmoid sharpness.
        final_dim (int): Final feature vector size.
        combination (tuple): Feature combination codes (e.g., 'Sc', 'Tm').

    Returns:
        np.ndarray: Extracted features of shape [N, M, final_dim]
    """
    num_samples, num_bursts = pkt_lengths.shape
    results = np.zeros((num_samples, num_bursts, final_dim))

    for i in tqdm(range(num_samples), desc="Extracting features"):
        for burst_idx, (lengths, times) in enumerate(zip(pkt_lengths[i], pkt_times[i])):
            single_feats = []
            A_dist = np.zeros(4, dtype=int)
            x = np.linspace(0, 2 * math.pi, sample_num)
            bins = np.array([40 / 1515, 80 / 1515, 160 / 1515, 320 / 1515, 640 / 1515])
            combined_signal = np.zeros(sample_num)

            for A, W in zip(lengths, times):
                sign = np.sign(A)
                A, W = A_W_transform(A, W, beta)
                idx = np.searchsorted(bins, A, side='right') - 1
                if 0 <= idx < len(A_dist):
                    A_dist[idx] += 1

                if sign == 1:
                    signal = A * np.sin(W * x)
                elif sign == -1:
                    signal = A * np.cos(W * x)
                else:
                    raise ValueError("Invalid direction (must be ±1)")

                energy = np.sum(signal ** 2)
                combined_signal += signal
                single_feats.append([W, A, energy])

            single_feats = np.array(single_feats)
            single_stats = np.hstack([
                np.max(single_feats, axis=0),
                np.mean(single_feats, axis=0),
                np.var(single_feats, axis=0),
            ])
            mixed_features = compute_signal_features(combined_signal)
            hist_features, _ = np.histogram(combined_signal, bins=100, density=True)
            final = create_final_features(single_stats, A_dist, mixed_features, hist_features, combination)
            results[i, burst_idx] = final

    return results


def A_W_transform(A, W, beta):
    """
    Normalize amplitude and transform frequency using soft-sigmoid.
    """
    A = abs(A) / 1515
    W_tanh = np.maximum(np.tanh(W), 5e-5)
    W_arctanh = arctanh(np.clip(2 * (W_tanh - 0.5), -0.99999, 0.99999))
    W_sig = 1 / (1 + np.exp(-beta * W_arctanh))
    return A, 1 / W_sig


def arctanh(x):
    return 0.5 * np.log((1 + x) / (1 - x))


def label_distribution(samples):
    """
    Count label distribution in the dataset.
    """
    counter = Counter(sample["flow_label"] for sample in samples)
    return counter


def data_loader_fre(dataset_name, pkt_num, beta, data_save, combination, final_dim, sample_num=100, batch_size=256,
                    split_ratio=0.8, pin_memory=True):
    """
    Loads or creates dataset features, splits into train/test, and returns DataLoader objects.

    Args:
        dataset_name (str): Dataset identifier
        pkt_num (int): Number of packets per sample.
        beta (float): Sigmoid beta value.
        data_save (str): Directory to save processed arrays.
        combination (tuple): Selected feature codes.
        final_dim (int): Feature vector dimension.
        sample_num (int): Signal resolution.
        batch_size (int): DataLoader batch size.
        split_ratio (float): Ratio of train samples.
        pin_memory (bool): Whether to pin memory in CUDA loader.

    Returns:
        Tuple[DataLoader, DataLoader]: train_loader, test_loader
    """
    os.makedirs(data_save, exist_ok=True)
    base_filename = f"{dataset_name}{pkt_num}_arrays_{final_dim}"
    data_path = os.path.join(data_save, base_filename + ".npy")

    if os.path.exists(data_path):
        print(f"[INFO] Feature file exists: {data_path}")
    else:
        json_path = f"{dataset_name}{pkt_num}_data_mix.json"
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        print("Label distribution:", label_distribution(raw_data))

        pkt_length = np.array([x["pkt_length"] for x in raw_data], dtype=object)
        pkt_time = np.array([x["pkt_time"] for x in raw_data], dtype=object)
        label = np.array([x["flow_label"] for x in raw_data])
        burst_label = np.array([x["burst_label"] for x in raw_data])

        fre_array = extract_features(pkt_length, pkt_time, sample_num, beta, final_dim, combination)

        np.save(data_path, fre_array)
        np.save(os.path.join(data_save, f"{dataset_name}{pkt_num}_labels_{final_dim}.npy"), label)
        np.save(os.path.join(data_save, f"{dataset_name}{pkt_num}_burstlabels_{final_dim}.npy"), burst_label)

    # Load feature arrays
    features = np.load(data_path)
    labels = np.load(os.path.join(data_save, f"{dataset_name}{pkt_num}_labels_{final_dim}.npy"))
    burst_labels = np.load(os.path.join(data_save, f"{dataset_name}{pkt_num}_burstlabels_{final_dim}.npy"))

    print(f"[SHAPE] Features: {features.shape}, Labels: {labels.shape}")

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels.squeeze()).long()
    burst_labels = torch.from_numpy(burst_labels).long()

    # Train/test split
    total_samples = labels.shape[0]
    split_idx = int(total_samples * split_ratio)

    train_data = features[:split_idx]
    train_labels = labels[:split_idx]
    train_bursts = burst_labels[:split_idx]

    test_data = features[split_idx:]
    test_labels = labels[split_idx:]
    test_bursts = burst_labels[split_idx:]

    # Wrap into DataLoader
    train_loader = DataLoader(
        dataset=torch.utils.data.TensorDataset(train_data, train_labels, train_bursts),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=1
    )

    test_loader = DataLoader(
        dataset=torch.utils.data.TensorDataset(test_data, test_labels, test_bursts),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=1
    )

    return train_loader, test_loader
