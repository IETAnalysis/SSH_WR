import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import config

burst_criterion = nn.BCELoss(reduction='none')
flow_criterion = nn.BCELoss()


def build_positive_weight_mask(burst_labels, positive_weight):
    weight_mask = burst_labels.float() * positive_weight
    weight_mask[weight_mask == 0] = 1.0
    return weight_mask


def collapse_burst_labels(burst_labels):
    binary_burst_labels = burst_labels.clone()
    binary_burst_labels[binary_burst_labels != 0] = 1
    return binary_burst_labels


def compute_flow_predictions_from_bursts(predicted_burst_labels):
    return (predicted_burst_labels.max(axis=1) > 0).astype(np.int64)


def calculate_flow_score(predictions, ground_truth):
    predictions = np.asarray(predictions).astype(np.int64)
    ground_truth = np.asarray(ground_truth).astype(np.int64)
    pred_inv = np.logical_not(predictions)
    gt_inv = np.logical_not(ground_truth)
    true_pos = float(np.logical_and(predictions, ground_truth).sum())
    false_pos = float(np.logical_and(predictions, gt_inv).sum())
    false_neg = float(np.logical_and(pred_inv, ground_truth).sum())
    true_neg = float(np.logical_and(pred_inv, gt_inv).sum())
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1, precision, recall, accuracy


def calculate_burst_f1(predictions, ground_truth):
    if np.max(predictions) == np.max(ground_truth) and np.max(predictions) == 0:
        return None, None, None, None

    pred_inv = np.logical_not(predictions)
    gt_inv = np.logical_not(ground_truth)
    true_pos = float(np.logical_and(predictions, ground_truth).sum())
    false_pos = float(np.logical_and(predictions, gt_inv).sum())
    false_neg = float(np.logical_and(pred_inv, ground_truth).sum())
    true_neg = float(np.logical_and(pred_inv, gt_inv).sum())

    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    return accuracy, precision, recall, f1


def _run_epoch(model, data_iter, optimizer, grad_clip, teacher_forcing_ratio, positive_weight, flow_loss_weight, training):
    device = next(model.parameters()).device
    if training:
        model.train()
    else:
        model.eval()

    total_burst_loss = 0.0
    total_flow_loss = 0.0
    burst_accuracy_list, burst_f1_list, burst_precision_list, burst_recall_list = [], [], [], []
    flow_true_all, flow_pred_all = [], []

    context_manager = torch.enable_grad if training else torch.no_grad
    with context_manager():
        for features, flow_labels, burst_labels_raw in data_iter:
            src = features.to(device).transpose(0, 1)
            flow_labels = flow_labels.to(device).float()
            burst_labels_raw = burst_labels_raw.to(device)
            burst_labels = collapse_burst_labels(burst_labels_raw)
            decoder_burst_labels = burst_labels.transpose(0, 1)

            if training:
                optimizer.zero_grad()

            predicted_bursts, predicted_flows = model(src, decoder_burst_labels, teacher_forcing_ratio=teacher_forcing_ratio)
            burst_weight_mask = build_positive_weight_mask(burst_labels.contiguous().view(-1), positive_weight)

            burst_loss = burst_criterion(predicted_bursts.contiguous(), burst_labels.float())
            burst_loss = (burst_loss.view(-1) * burst_weight_mask).mean()
            flow_loss = flow_criterion(predicted_flows, flow_labels)
            total_loss = burst_loss + flow_loss_weight * flow_loss

            if training:
                total_loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_burst_loss += burst_loss.item()
            total_flow_loss += flow_loss.item()

            pred_burst_labels = (predicted_bursts > config.BURST_DECISION_THRESHOLD).long().cpu().numpy()
            true_burst_labels = burst_labels.cpu().numpy()
            pred_flow_labels = (predicted_flows > config.FLOW_DECISION_THRESHOLD).long().cpu().numpy()
            true_flow_labels = flow_labels.long().cpu().numpy()

            batch_size = true_burst_labels.shape[0]
            for idx in range(batch_size):
                acc_b, precision_b, recall_b, f1_b = calculate_burst_f1(pred_burst_labels[idx], true_burst_labels[idx])
                if f1_b is not None:
                    burst_accuracy_list.append(acc_b)
                    burst_f1_list.append(f1_b)
                    burst_precision_list.append(precision_b)
                    burst_recall_list.append(recall_b)

                flow_true_all.append(int(true_flow_labels[idx]))
                flow_pred_all.append(int(pred_flow_labels[idx]))

    f1_flow, precision_flow, recall_flow, acc_flow = calculate_flow_score(flow_pred_all, flow_true_all)
    average_burst_accuracy = sum(burst_accuracy_list) / len(burst_accuracy_list)
    average_burst_precision = sum(burst_precision_list) / len(burst_precision_list)
    average_burst_recall = sum(burst_recall_list) / len(burst_recall_list)
    average_burst_f1 = sum(burst_f1_list) / len(burst_f1_list)

    return (
        total_burst_loss / len(data_iter),
        total_flow_loss / len(data_iter),
        average_burst_accuracy,
        average_burst_precision,
        average_burst_recall,
        average_burst_f1,
        acc_flow,
        precision_flow,
        recall_flow,
        f1_flow,
    )


def train(model, optimizer, train_iter, grad_clip, teacher_forcing_ratio, n, lambd):
    return _run_epoch(model, train_iter, optimizer, grad_clip, teacher_forcing_ratio, n, lambd, True)


def split_evaluate_wise_flow(model, val_iter, n, test_teacher_forcing_ratio):
    return _run_epoch(model, val_iter, None, 0.0, test_teacher_forcing_ratio, n, 1.0, False)


