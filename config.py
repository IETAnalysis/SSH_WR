import os

# === Directory and Logging Configuration ===
DATA_DIR = 'data'                     # Directory containing raw or processed data
DATA_SAVE_DIR = 'data'                # Directory for saving transformed or intermediate data
BASE_LOG_DIR = 'log'                  # Root directory for logs
CSV_PATH = 'results.csv'              # Default CSV log filename

LOGGING_CONFIG = {
    'log_file': os.path.join(BASE_LOG_DIR, CSV_PATH),
    'logger_name': 'logger'
}

# === Training Hyperparameters ===
TRAINING_CONFIG = {
    'num_epochs': 400,                      # Maximum number of training epochs
    'lr': 0.0005,                           # Initial learning rate
    'train_teacher_forcing_ratio': 0.5,     # Teacher forcing ratio during training
    'test_teacher_forcing_ratio': 0.0,      # Teacher forcing ratio during inference
    'grad_clip': 10.0,                      # Gradient clipping threshold to prevent explosion
    'lambd': 0.2,                           # Loss balancing coefficient (burst vs classification)
    'n': 1,                                 # Weight multiplier for burst class imbalance
    'patience': 40,                         # Early stopping patience (epochs without improvement)
    'pkt_num': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # burst window lengths to evaluate
}

# === Dataset Configuration ===
DATASET_CONFIG = {
    'dataset_name': None,     # Name of the dataset to be used
    'batch_size': 256,            # Mini-batch size for training/evaluation
    'beta': 0.05,                 # Beta value for soft-sigmoid activation (if applicable)
    'sample_num': 600             # Number of samples
}

# === Model Architecture ===
MODEL_CONFIG = {
    'encoder_input_dim': 113,     # Input dimension for the encoder
    'decoder_input_dim': 512,     # Input dimension for the decoder
    'hidden_size': 64,            # Hidden layer size in both encoder/decoder
    'n_layers': 4,                # Number of recurrent layers
    'dropout': 0.5,               # Dropout probability for regularization
    'grad_clip': 10.0             # Redundant for safety; used in training
}

# === Learning Rate Scheduler (Optional) ===
LEARNING_RATE_SCHEDULER = {
    'use_scheduler': False,       # Whether to use a learning rate scheduler
    'scheduler_step_size': 30,    # Step interval to reduce LR (if enabled)
    'scheduler_gamma': 0.5        # LR decay factor
}

# === Feature Configuration ===

# Mapping of full feature names to their dimensionality
FEATURES_CONFIG = {
    'single_features_stats': 9,
    'A_distribution': 4,
    'mixed_signal_features': 3,
    'hist_features': 100
}

# Shortcodes for each feature (used for selection logic and UI)
FEATURES_SHORTNAMES = {
    'Sc': 'single_features_stats',
    'Ac': 'A_distribution',
    'Sm': 'mixed_signal_features',
    'Tm': 'hist_features'
}

# Active feature combination (set manually or dynamically)
# Choose from combinations like:
# ('Sc', 'Ac'), ('Sc', 'Sm'), ('Sc', 'Tm'), ('Ac', 'Sm'), ...
# ('Sc', 'Ac', 'Sm', 'Tm') — Full feature set
FEATURE_COMBINATION = ('Sc', 'Ac', 'Tm')

