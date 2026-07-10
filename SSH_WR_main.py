import os
import time
from datetime import datetime

import torch
from torch import optim

import config
from model import Decoder, Encoder, Seq2Seq
from npy_dataloader import data_loader_fre
from train import split_evaluate_wise_flow, train
from util import clear_gpu_memory, get_logger


def calculate_final_feature_dim(combination):
    total_dim = 0
    for feature_code in combination:
        feature_name = config.FEATURES_SHORTNAMES[feature_code]
        total_dim += config.FEATURES_CONFIG[feature_name]
    return total_dim


def main():
    training_config = config.TRAINING_CONFIG
    model_config = config.MODEL_CONFIG
    dataset_config = config.DATASET_CONFIG

    grad_clip = training_config['grad_clip']
    train_teacher_forcing_ratio = training_config['train_teacher_forcing_ratio']
    test_teacher_forcing_ratio = training_config['test_teacher_forcing_ratio']
    observation_window_sizes = training_config['observation_window_sizes']
    learning_rate = training_config['lr']
    num_epochs = training_config['num_epochs']
    flow_loss_weight = training_config['flow_loss_weight']
    positive_weight = training_config['positive_weight']
    patience = training_config['patience']

    hidden_size = model_config['hidden_size']
    decoder_input_dim = model_config['decoder_input_dim']
    num_layers = model_config['n_layers']
    dropout = model_config['dropout']

    dataset_name = dataset_config['dataset_name']
    batch_size = dataset_config['batch_size']
    sample_num = dataset_config['sample_num']
    beta = dataset_config['beta']

    if not dataset_name:
        raise ValueError('DATASET_CONFIG["dataset_name"] must be set before training.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_combination = config.FEATURE_COMBINATION
    final_feature_dim = calculate_final_feature_dim(feature_combination)
    print(f'Feature Combination: {feature_combination} -> Final Feature Dimension: {final_feature_dim}')

    for observation_window_size in observation_window_sizes:
        clear_gpu_memory()
        timestamp = datetime.fromtimestamp(time.time()).strftime('%m_%d_%H')
        log_dir = os.path.join(config.BASE_LOG_DIR, f'logfre_{dataset_name}_{observation_window_size}')
        log_file = os.path.join(log_dir, f'results_{timestamp}.csv')
        logger = get_logger(log_dir, log_file)
        logger.info(training_config)
        logger.info(model_config)
        logger.info(dataset_config)
        logger.info(feature_combination)

        print('[!] preparing dataset...')
        train_iter, test_iter = data_loader_fre(
            dataset_name=dataset_name,
            pkt_num=observation_window_size,
            beta=beta,
            data_root=config.DATA_DIR,
            cache_dir=config.DATA_SAVE_DIR,
            combination=feature_combination,
            final_dim=final_feature_dim,
            sample_num=sample_num,
            batch_size=batch_size,
        )
        print(f'[TRAIN]: {len(train_iter)} (dataset: {len(train_iter.dataset)}) [TEST]: {len(test_iter)} (dataset: {len(test_iter.dataset)})')

        encoder = Encoder(
            input_dim=final_feature_dim,
            hidden_size=hidden_size,
            n_layers=num_layers,
            dropout=dropout,
            cell_type='lstm',
        )
        decoder = Decoder(
            embedding_dim=decoder_input_dim,
            hidden_size=hidden_size,
            n_layers=num_layers,
            dropout=dropout,
            cell_type='lstm',
        )
        seq2seq = Seq2Seq(encoder, decoder).to(device)
        optimizer = optim.Adam(seq2seq.parameters(), lr=learning_rate)

        logger.info('Epoch, Time, train_b_loss, train_c_loss, test_b_loss, test_c_loss, train_acc_b, train_prec_b, train_rec_b, train_f1_b, train_acc_f, train_prec_f, train_rec_f, train_f1_f, test_acc_b, test_prec_b, test_rec_b, test_f1_b, test_acc_f, test_prec_f, test_rec_f, test_f1_f')

        patience_counter = 0
        best_train_burst_loss = float('inf')
        best_train_flow_loss = float('inf')
        best_epoch = 0
        best_model_state = None

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_metrics = train(
                seq2seq,
                optimizer,
                train_iter,
                grad_clip,
                train_teacher_forcing_ratio,
                positive_weight,
                flow_loss_weight,
            )
            test_metrics = split_evaluate_wise_flow(
                seq2seq,
                test_iter,
                positive_weight,
                test_teacher_forcing_ratio,
            )
            elapsed = time.time() - start_time

            print(f'Epoch:{epoch}, Train burst f1:{train_metrics[5]:.4f}, Train flow f1:{train_metrics[9]:.4f}, Test burst f1:{test_metrics[5]:.4f}, Test flow f1:{test_metrics[9]:.4f}')
            logger.info('%d, %.3f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f', epoch, elapsed, *train_metrics, *test_metrics)

            train_burst_loss, train_flow_loss = train_metrics[0], train_metrics[1]
            if train_burst_loss + flow_loss_weight * train_flow_loss < best_train_burst_loss + flow_loss_weight * best_train_flow_loss:
                patience_counter = 0
                best_train_burst_loss = train_burst_loss
                best_train_flow_loss = train_flow_loss
                best_model_state = seq2seq.state_dict()
                best_epoch = epoch
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'[Early Stop] No improvement in {patience} epochs. Stopping early at epoch {epoch}.')
                break

        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(log_dir, f'best_model_epoch_{best_epoch}.pt'))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as exc:
        print('[STOP]', exc)
