import os

# === Directory and Logging Configuration ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_DIR, '..', '..', 'dataset', 'open', 'SSH-WR Dataset'))
DATA_SAVE_DIR = os.path.join(PROJECT_DIR, 'data')
BASE_LOG_DIR = os.path.join(PROJECT_DIR, 'log')
CSV_PATH = 'results.csv'

LOGGING_CONFIG = {
    'log_file': os.path.join(BASE_LOG_DIR, CSV_PATH),
    'logger_name': 'ssh_wr'
}

# === Training Hyperparameters ===
TRAINING_CONFIG = {
    'num_epochs': 400,
    'lr': 0.0005,
    'train_teacher_forcing_ratio': 0.5,
    'test_teacher_forcing_ratio': 0.0,
    'grad_clip': 10.0,
    'flow_loss_weight': 0.2,
    'positive_weight': 1,
    'patience': 40,
    'observation_window_sizes': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}

# Backward-compatible aliases used by older scripts
TRAINING_CONFIG['lambd'] = TRAINING_CONFIG['flow_loss_weight']
TRAINING_CONFIG['n'] = TRAINING_CONFIG['positive_weight']
TRAINING_CONFIG['pkt_num'] = TRAINING_CONFIG['observation_window_sizes']

# === Dataset Configuration ===
DATASET_CONFIG = {
    'dataset_name': None,
    'batch_size': 256,
    'beta': 0.05,
    'sample_num': 600,
}

DATASET_LAYOUTS = {
    'Hysteria2': {'directory': 'Hysteria2', 'prefix': 'Hysteria2_'},
    'Trojan': {'directory': 'Trojan', 'prefix': 'Trojan_'},
    'Vless': {'directory': 'vless', 'prefix': 'Vless_'},
}

# === Model Architecture ===
MODEL_CONFIG = {
    'decoder_start_token_id': 2,
    'decoder_num_embeddings': 3,
    'decoder_input_dim': 512,
    'hidden_size': 64,
    'n_layers': 4,
    'dropout': 0.5,
}

# === Feature Configuration ===
FEATURES_CONFIG = {
    'single_component_statistics': 9,
    'amplitude_distribution': 4,
    'superposed_waveform_statistics': 3,
    'burst_waveform_histogram': 100,
}

FEATURES_SHORTNAMES = {
    'Sc': 'single_component_statistics',
    'Ac': 'amplitude_distribution',
    'Sm': 'superposed_waveform_statistics',
    'Tm': 'burst_waveform_histogram',
}

FEATURE_COMBINATION = ('Sc', 'Ac', 'Sm', 'Tm')

# === Constants derived from the manuscript ===
PACKET_LENGTH_MAX_BYTES = 1515.0
TIME_NORMALIZATION_EPSILON = 0.005
AMPLITUDE_BIN_EDGES = [80 / PACKET_LENGTH_MAX_BYTES,
                       160 / PACKET_LENGTH_MAX_BYTES,
                       320 / PACKET_LENGTH_MAX_BYTES,
                       640 / PACKET_LENGTH_MAX_BYTES]
HISTOGRAM_BIN_COUNT = 100
FLOW_DECISION_THRESHOLD = 0.5
BURST_DECISION_THRESHOLD = 0.5




