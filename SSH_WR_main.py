import os
import torch
from torch import optim
from model import Encoder, Decoder, Seq2Seq
from npy_dataloader import data_loader_fre
from util import get_logger, clear_gpu_memory
import time
from datetime import datetime
import torch.nn as nn
import config  # Importing configuration file
from train import train, split_evaluate_wise_flow

# Loss function: Binary Cross Entropy Loss
criterion = nn.BCELoss()
criterion_fine = nn.BCELoss(reduction='none')

def calculate_final_feature_dim(combination):
    """
    Calculate the final dimension of features based on the feature combination selected.
    :param combination: Tuple of feature combinations, e.g., ('Sc', 'Ac')
    :return: total dimension of the selected feature combination.
    """
    total_dim = 0
    for feature in combination:
        # Fetch the feature's actual name and its corresponding dimension from config
        feature_name = config.FEATURES_SHORTNAMES[feature]
        total_dim += config.FEATURES_CONFIG[feature_name]
    return total_dim

def main():
    # Fetch configuration parameters for training from the config file
    config_dict = config.TRAINING_CONFIG
    grad_clip = config_dict['grad_clip']
    train_teacher_forcing_ratio = config_dict['train_teacher_forcing_ratio']
    test_teacher_forcing_ratio = config_dict['test_teacher_forcing_ratio']
    pkt_num = config_dict['pkt_num']
    lr = config_dict['lr']
    num_epochs = config_dict['num_epochs']
    lambd = config_dict['lambd']
    n = config_dict['n']
    patience = config_dict['patience']

    config_model = config.MODEL_CONFIG
    hidden_size = config_model['hidden_size']
    decoder_input_dim = config_model['decoder_input_dim']
    layer_num = config_model['n_layers']
    dropout = config_model['dropout']

    config_dataset = config.DATASET_CONFIG
    dataset_name = config.DATASET_CONFIG['dataset_name']
    batch_size = config_dataset['batch_size']
    sample_num = config_dataset['sample_num']
    beta = config_dataset['beta']

    combination = config.FEATURE_COMBINATION

    # Calculate final feature dimension based on selected feature combination
    final_dim = calculate_final_feature_dim(combination)
    print(f"Feature Combination: {combination} -> Final Feature Dimension: {final_dim}")

    # Ensure CUDA is available for training
    assert torch.cuda.is_available()

    # Iterate over different packet numbers (pkt_num)
    for pkt_N in pkt_num:
        clear_gpu_memory()

        # Prepare log file and timestamp for results
        formatted_start_time = datetime.fromtimestamp(time.time()).strftime("%m_%d_%H")
        log_path = os.path.join(config.BASE_LOG_DIR, f"logfre_{dataset_name}_{pkt_N}")
        csv_path = f'results_{formatted_start_time}.csv'

        # Create directory if it doesn't exist and prepare logger
        os.makedirs(log_path, exist_ok=True)
        logfile = os.path.join(log_path, csv_path)
        logger = get_logger(log_path, logfile)
        logger.info(config_dict)
        logger.info(config_model)
        logger.info(config_dataset)
        logger.info(combination)

        # Prepare the dataset for training and testing
        print("[!] preparing dataset...")
        train_iter, test_iter = data_loader_fre(dataset_name=dataset_name,
                                                batch_size=batch_size,
                                                pkt_num=pkt_N,
                                                beta=beta,
                                                sample_num=sample_num,
                                                data_save=config.data_SAVE,
                                                combination=combination,
                                                final_dim=final_dim)

        print(f"[TRAIN]: {len(train_iter)} (dataset: {len(train_iter.dataset)}) "
              f"[TEST]: {len(test_iter)} (dataset: {len(test_iter.dataset)})")

        # Instantiate encoder and decoder models based on configuration
        encoder = Encoder(embed_size=final_dim,
                          hidden_size=hidden_size,
                          n_layers=layer_num,
                          dropout=dropout)

        decoder = Decoder(embed_size=decoder_input_dim,
                          hidden_size=hidden_size,
                          n_layers=layer_num,
                          dropout=dropout)

        # Instantiate Seq2Seq model by passing encoder and decoder
        seq2seq = Seq2Seq(encoder, decoder).cuda()
        print(seq2seq)

        # Optimizer setup: Adam optimizer with the learning rate
        optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

        # Log header for tracking performance metrics
        logger.info("Epoch, Time, train_b_loss, train_c_loss, test_b_loss, test_c_loss, "
                    "train_acc_b, train_prec_b, train_rec_b, train_f1_b, "
                    "train_acc_f, train_prec_f, train_rec_f, train_f1_f, "
                    "test_acc_b, test_prec_b, test_rec_b, test_f1_b, "
                    "test_acc_f, test_prec_f, test_rec_f, test_f1_f")

        # Initialize early stopping and tracking variables
        patience_counter = 0
        best_train_b_loss = float('inf')
        best_train_c_loss = float('inf')

        # Start the training loop
        for e in range(1, num_epochs + 1):
            start_time = time.time()

            # Training and testing for each epoch
            (train_b_loss, train_c_loss,
             train_acc_burst, train_precision_burst, train_recall_burst, train_f1_burst,
             train_acc_flow, train_precision_flow, train_recall_flow, train_f1_flow) = train(seq2seq, optimizer, train_iter,
                                                                                 grad_clip, train_teacher_forcing_ratio,
                                                                                 n, lambd)
            (test_b_loss, test_c_loss,
             test_acc_burst, test_precision_burst, test_recall_burst, test_f1_burst,
             test_acc_flow, test_precision_flow, test_recall_flow, test_f1_flow) = split_evaluate_wise_flow(seq2seq, test_iter,
                                                                                                    n,
                                                                                                    test_teacher_forcing_ratio)

            print(f"Epoch:{e}, Train burst f1:{train_f1_burst}, Train flow f1:{train_f1_flow}, Test burst f1:{test_f1_burst}, Test flow f1:{test_f1_flow}")

            # Log the results of each epoch
            train_time = time.time()
            logger.info('%d, %.3f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, '
                        '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f',
                        e, train_time - start_time,
                        train_b_loss, train_c_loss, test_b_loss, test_c_loss,
                        train_acc_burst, train_precision_burst, train_recall_burst, train_f1_burst,
                        train_acc_flow, train_precision_flow, train_recall_flow, train_f1_flow,
                        test_acc_burst, test_precision_burst, test_recall_burst, test_f1_burst,
                        test_acc_flow, test_precision_flow, test_recall_flow, test_f1_flow)

            # Early stopping based on validation loss
            if train_b_loss + lambd * train_c_loss < best_train_b_loss + lambd * best_train_c_loss:
                patience_counter = 0
                best_train_b_loss = train_b_loss
                best_train_c_loss = train_c_loss
                best_model_state = seq2seq.state_dict()
                best_epoch = e
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stop] No improvement in {patience} epochs. Stopping early at epoch {e}.")
                break

# Entry point for running the training
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
