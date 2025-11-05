# **SSH-WR: Detecting SSH Traffic in Encrypted Tunnels with Traffic Waveform Representation**

---

## **SSH-WR Overview**

![Overall SSH-WR Architecture](https://github.com/user-attachments/assets/1957a238-3695-4d00-bca5-3a97ad1e5746)

---

## **Files and Modules**

### **config.py**

The `config.py` module acts as the **centralized configuration hub** for the entire training and evaluation pipeline. It contains all the critical hyperparameters, architectural settings, dataset paths, logging configurations, and feature selection parameters in a structured and readable format. This file serves as the foundation for configuring various components, ensuring flexibility during experimentation.

### **model.py**

The `model.py` module defines the complete **Seq2Seq model with attention**, central to the architecture of SSH-WR. It comprises the following components:

* **Encoder**: A bidirectional LSTM-based encoder that processes the input sequence.
* **Attention**: An additive attention mechanism inspired by Bahdanau et al., which enables the model to focus on relevant parts of the input sequence.
* **Decoder**: An attentive LSTM-based decoder that generates the output sequence by combining embeddings and context information from the attention mechanism.

### **npy\_dataloader.py**

The `npy_dataloader.py` module handles the **feature extraction and data loading** process. It is responsible for:

* Dynamically selecting feature combinations based on the specific experiment or task.
* Generating and saving structured `.npy` feature arrays from raw **JSON data**.
* Preparing PyTorch **DataLoader objects** for both training and evaluation, ensuring efficient and seamless data handling during model training.

### **SSH\_WR\_main.py**

The `SSH_WR_main.py` script orchestrates the **training pipeline** for the Seq2Seq model applied to **burst detection** and **flow detection** tasks. Key components of the script include:

* **Encoder and Decoder Models**: Defines the architecture of the sequence-to-sequence model.
* **Training Pipeline**: Manages the training loop, validation process, and **early stopping** mechanisms to prevent overfitting.
* **Feature Combination**: Dynamically handles different feature combinations, enabling the model to adapt to various configurations.
* **Configuration and Logging**: The script integrates with the `config.py` file for configuration management and supports detailed logging of key events throughout the training process.

### **train.py**

The `train.py` script is dedicated to **training and evaluating** the deep learning model for burst and flow detection. It incorporates the following key features:

* **Optimization with Adam**: The model is optimized using the Adam optimizer for stable and efficient training.
* **Multiple Loss Functions**: The script incorporates **binary cross-entropy losses** for both burst and flow detection tasks.
* **Gradient Clipping**: To prevent gradient explosion, gradient clipping is employed during training.
* **Early Stopping**: Implements early stopping to halt training when performance ceases to improve, preventing overfitting.
* **Evaluation Metrics**: Utility functions for evaluating the model's performance using **F1-score, precision, recall**, and **accuracy** for both burst and flow detection tasks.

### **util.py**

The `util.py` script provides essential **utility functions** for managing GPU memory and setting up logging systems:

* **GPU Memory Management**: Clears unused GPU memory during training to prevent memory overflow.
* **Logger Setup**: Configures a logging system to capture important events and track progress throughout the training and evaluation processes.

---

## **Dataset**

The dataset used for training and evaluation, which can be downloaded from the following Google Drive link, is publicly available.

[Download Dataset](https://drive.google.com/drive/folders/1zigriMV5VXjQu3NR7XNrKYHM6cAP2Epl?usp=sharing)

The dataset consists of features such as packet lengths and timestamps, which are processed to extract traffic waveformn features for model training.

---

## **Installation**

### Prerequisites

To run the code, the following Python libraries are required:

* Python 3.x
* PyTorch (recommended version 1.9.0 or later)
* NumPy
* Pandas
* scikit-learn
* tqdm

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/IETAnalysis/SSH_WR.git
   cd SSH_WR
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### **Usage**


To train the model, run the `SSH_WR_main.py` script. It will automatically load the configuration from `config.py` and start the training process.

```bash
python SSH_WR_main.py

