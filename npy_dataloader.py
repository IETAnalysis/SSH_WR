import json
import math
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_feature_vector(single_component_statistics,
                         amplitude_distribution,
                         superposed_waveform_statistics,
                         burst_waveform_histogram,
                         selected_keys):
    feature_map = {
        'Sc': single_component_statistics,
        'Ac': amplitude_distribution,
        'Sm': superposed_waveform_statistics,
        'Tm': burst_waveform_histogram,
    }
    return np.hstack([feature_map[key] for key in selected_keys if key in feature_map])


def compute_superposed_waveform_statistics(signal):
    return [
        np.max(signal),
        np.sum(signal ** 2),
        np.var(signal),
    ]


def inverse_hyperbolic_tangent(x):
    return 0.5 * np.log((1 + x) / (1 - x))


def transform_amplitude_and_temporal_frequency(packet_length_with_direction, inter_arrival_time, beta):
    amplitude = abs(packet_length_with_direction) / config.PACKET_LENGTH_MAX_BYTES
    bounded_time = np.maximum(np.tanh(inter_arrival_time), config.TIME_NORMALIZATION_EPSILON)
    centered_time = np.clip(2 * (bounded_time - 0.5), -0.99999, 0.99999)
    stretched_time = inverse_hyperbolic_tangent(centered_time)
    soft_time = 1.0 / (1.0 + np.exp(-beta * stretched_time))
    temporal_frequency = 1.0 / soft_time
    return amplitude, temporal_frequency


def extract_burst_features(packet_lengths, inter_arrival_times, sample_num, beta, final_dim, combination):
    num_samples, num_bursts = packet_lengths.shape
    results = np.zeros((num_samples, num_bursts, final_dim), dtype=np.float32)
    sample_points = np.linspace(0, 2 * math.pi, sample_num)
    amplitude_bin_edges = np.array(config.AMPLITUDE_BIN_EDGES)

    for sample_idx in tqdm(range(num_samples), desc='Extracting features'):
        for burst_idx, (burst_packet_lengths, burst_inter_arrival_times) in enumerate(
            zip(packet_lengths[sample_idx], inter_arrival_times[sample_idx])
        ):
            component_statistics = []
            amplitude_distribution = np.zeros(4, dtype=np.float32)
            superposed_waveform = np.zeros(sample_num, dtype=np.float32)

            for signed_packet_length, inter_arrival_time in zip(burst_packet_lengths, burst_inter_arrival_times):
                direction = np.sign(signed_packet_length)
                amplitude, temporal_frequency = transform_amplitude_and_temporal_frequency(
                    signed_packet_length,
                    inter_arrival_time,
                    beta,
                )
                bin_index = np.searchsorted(amplitude_bin_edges, amplitude, side='right')
                bin_index = min(bin_index, len(amplitude_distribution) - 1)
                amplitude_distribution[bin_index] += 1

                if direction == 1:
                    component_waveform = amplitude * np.sin(temporal_frequency * sample_points)
                elif direction == -1:
                    component_waveform = amplitude * np.cos(temporal_frequency * sample_points)
                else:
                    raise ValueError('Invalid packet direction; expected +1 or -1.')

                component_energy = np.sum(component_waveform ** 2)
                component_statistics.append([amplitude, temporal_frequency, component_energy])
                superposed_waveform += component_waveform

            component_statistics = np.asarray(component_statistics, dtype=np.float32)
            single_component_statistics = np.hstack([
                np.max(component_statistics, axis=0),
                np.min(component_statistics, axis=0),
                np.var(component_statistics, axis=0),
            ])
            superposed_waveform_statistics = compute_superposed_waveform_statistics(superposed_waveform)
            burst_waveform_histogram, _ = np.histogram(
                superposed_waveform,
                bins=config.HISTOGRAM_BIN_COUNT,
                density=True,
            )
            final_features = build_feature_vector(
                single_component_statistics,
                amplitude_distribution,
                superposed_waveform_statistics,
                burst_waveform_histogram,
                combination,
            )
            results[sample_idx, burst_idx] = final_features

    return results


def label_distribution(samples):
    return Counter(sample['flow_label'] for sample in samples)


def _resolve_dataset_layout(dataset_name):
    if dataset_name not in config.DATASET_LAYOUTS:
        valid_names = ', '.join(sorted(config.DATASET_LAYOUTS))
        raise ValueError('Unsupported dataset_name: {}. Expected one of: {}.'.format(dataset_name, valid_names))
    return config.DATASET_LAYOUTS[dataset_name]


def _resolve_split_json_path(data_root, dataset_name, observation_window_size, split_name):
    layout = _resolve_dataset_layout(dataset_name)
    base_name = '{}{}'.format(layout['prefix'], observation_window_size)
    split_dir = os.path.join(data_root, layout['directory'], base_name, split_name)
    json_path = os.path.join(split_dir, '{}_{}.json'.format(base_name, split_name))
    return base_name, split_dir, json_path


def _resolve_cache_file(cache_dir, base_name, split_name, suffix, final_dim):
    return os.path.join(cache_dir, '{}_{}_{}_{}.npy'.format(base_name, split_name, suffix, final_dim))


def _load_or_build_split_arrays(data_root,
                                cache_dir,
                                dataset_name,
                                observation_window_size,
                                split_name,
                                beta,
                                combination,
                                final_dim,
                                sample_num):
    base_name, _, json_path = _resolve_split_json_path(data_root, dataset_name, observation_window_size, split_name)
    if not os.path.exists(json_path):
        raise FileNotFoundError('Missing {} dataset file: {}'.format(split_name, json_path))

    feature_path = _resolve_cache_file(cache_dir, base_name, split_name, 'arrays', final_dim)
    flow_label_path = _resolve_cache_file(cache_dir, base_name, split_name, 'labels', final_dim)
    burst_label_path = _resolve_cache_file(cache_dir, base_name, split_name, 'burstlabels', final_dim)

    if not os.path.exists(feature_path):
        with open(json_path, 'r', encoding='utf-8') as file_obj:
            raw_data = json.load(file_obj)

        print('{} [{}] label distribution:'.format(base_name, split_name), label_distribution(raw_data))
        packet_lengths = np.array([sample['pkt_length'] for sample in raw_data], dtype=object)
        inter_arrival_times = np.array([sample['pkt_time'] for sample in raw_data], dtype=object)
        flow_labels = np.array([sample['flow_label'] for sample in raw_data], dtype=np.int64)
        burst_labels = np.array([sample['burst_label'] for sample in raw_data], dtype=np.int64)

        feature_array = extract_burst_features(
            packet_lengths,
            inter_arrival_times,
            sample_num,
            beta,
            final_dim,
            combination,
        )
        np.save(feature_path, feature_array)
        np.save(flow_label_path, flow_labels)
        np.save(burst_label_path, burst_labels)

    features = np.load(feature_path)
    flow_labels = np.load(flow_label_path)
    burst_labels = np.load(burst_label_path)
    return features, flow_labels, burst_labels


def _build_loader(features, flow_labels, burst_labels, batch_size, shuffle, pin_memory):
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(flow_labels.squeeze()).float(),
        torch.from_numpy(burst_labels).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory and torch.cuda.is_available(),
        num_workers=0,
    )


def data_loader_fre(dataset_name,
                    pkt_num,
                    beta,
                    data_root,
                    cache_dir,
                    combination,
                    final_dim,
                    sample_num=600,
                    batch_size=256,
                    pin_memory=True):
    os.makedirs(cache_dir, exist_ok=True)

    train_features, train_flow_labels, train_burst_labels = _load_or_build_split_arrays(
        data_root=data_root,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        observation_window_size=pkt_num,
        split_name='train',
        beta=beta,
        combination=combination,
        final_dim=final_dim,
        sample_num=sample_num,
    )
    test_features, test_flow_labels, test_burst_labels = _load_or_build_split_arrays(
        data_root=data_root,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        observation_window_size=pkt_num,
        split_name='test',
        beta=beta,
        combination=combination,
        final_dim=final_dim,
        sample_num=sample_num,
    )

    train_loader = _build_loader(
        train_features,
        train_flow_labels,
        train_burst_labels,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    test_loader = _build_loader(
        test_features,
        test_flow_labels,
        test_burst_labels,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader




