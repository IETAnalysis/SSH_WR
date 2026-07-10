# SSH-WR: Detecting SSH Traffic in Encrypted Tunnels with Traffic Waveform Representation

## Files and Modules

### `config.py`
Central configuration for training, feature construction, and manuscript-aligned constants. The current keys use the paper terminology, including `positive_weight`, `flow_loss_weight`, `observation_window_sizes`, `sample_num`, and waveform-related constants.

### `model.py`
Implements the encoder-decoder architecture with additive attention. The encoder consumes burst-level feature sequences, and the decoder produces burst-level SSH probabilities. Flow-level probabilities are obtained by global max pooling over burst-level predictions, consistent with the paper.

### `npy_dataloader.py`
Implements burst waveform feature extraction and dataset loading. Packet length, packet direction, and inter-arrival time are transformed into waveform descriptors. The exported feature groups match the paper-level notation:
- `Sc`: single component statistics
- `Ac`: amplitude distribution
- `Sm`: superposed waveform statistics
- `Tm`: burst waveform histogram

### `train.py`
Implements training and evaluation. The burst-level loss uses a positive-weighted binary cross-entropy style weighting mask controlled by `positive_weight`, while the flow-level loss is weighted by `flow_loss_weight`.

### `SSH_WR_main.py`
Main entry for training and evaluation. It reads the configuration, prepares the dataset, instantiates the encoder-decoder model, and records burst-level and flow-level metrics.

### `util.py`
Utility functions for logging and GPU memory cleanup.

## Dataset
The code expects pre-split mixed-traffic JSON files under `../../dataset/open/SSH-WR Dataset`, relative to this code directory. The dataset should follow this structure:

```text
SSH-WR Dataset/
  Hysteria2/
    Hysteria2_10/
      train/Hysteria2_10_train.json
      test/Hysteria2_10_test.json
  Trojan/
    Trojan_10/
      train/Trojan_10_train.json
      test/Trojan_10_test.json
  vless/
    Vless_10/
      train/Vless_10_train.json
      test/Vless_10_test.json
```

The same structure is used for all observation window sizes in `10, 20, ..., 100`. The generated NumPy feature caches are saved under the local `data` directory and do not need to be provided with the released dataset.

Use the following dataset names in `config.py`:
- `Hysteria2`
- `Trojan`
- `Vless`

## Installation

### Prerequisites
- Python 3.7+
- torch 1.8.1+
- numpy
- tqdm

### Install
```bash
pip install -r requirements.txt
```

## Usage
Set `DATASET_CONFIG['dataset_name']` in `config.py`, ensure the pre-split dataset is placed under `../../dataset/open/SSH-WR Dataset`, then run:

```bash
python SSH_WR_main.py
```

