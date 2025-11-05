import os
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch.nn as nn

# Loss functions: Binary Cross-Entropy Loss
criterion = nn.BCELoss()
criterion_fine = nn.BCELoss(reduction='none')


def n_burst(A, n):
    """
    Function to scale the burst labels with a given factor 'n'.
    This is used to adjust the contribution of burst labels in the loss computation.

    Args:
        A: Tensor containing burst labels (0 or 1)
        n: The scaling factor

    Returns:
        A: Tensor after scaling
    """
    A = A * n
    A[A == 0] = 1  # Replace zero with 1 to avoid division by zero in loss calculation
    return A


def train(model, optimizer, train_iter, grad_clip, teacher_forcing_ratio, n, lambd):
    """
    Train the Seq2Seq model using the training data.

    Args:
        model: The Seq2Seq model to be trained.
        optimizer: Optimizer for the model's parameters.
        train_iter: DataLoader for the training dataset.
        grad_clip: Gradient clipping value for model stability.
        teacher_forcing_ratio: Ratio of teacher forcing for training.
        n: Scaling factor for burst labels.
        lambd: Weighting factor for the classification loss.

    Returns:
        Tuple containing training loss values and metrics.
    """
    model.train()
    total_b_loss = 0
    total_c_loss = 0
    flow_correct_count, total_flow = 0, 0
    acc_b_list, f1_b_list, precision_b_list, recall_b_list = [], [], [], []
    flow_true_all, flow_pred_all = [], []

    # Loop through the training data
    for b, batch in enumerate(train_iter):
        data, flow_label, burst_fine_label = batch
        src, burst_fine_label = data.cuda(), burst_fine_label.cuda()
        flow_label = flow_label.cuda()

        # Process burst labels: convert non-zero to 1
        burst_labels = burst_fine_label.clone()
        burst_labels[burst_labels != 0] = 1
        burst_labels = burst_labels.transpose(0, 1).cuda()
        src = src.transpose(0, 1)

        # Zero gradients, perform forward pass, and calculate loss
        optimizer.zero_grad()
        output, sen_res = model(src, burst_labels, teacher_forcing_ratio=teacher_forcing_ratio)

        # Weight the burst labels by 'n' factor
        weight_burst = burst_fine_label.contiguous().view(-1).cuda()
        weight_burst = n_burst(weight_burst.clone(), n)

        # Calculate burst loss
        burst_loss = criterion_fine(output.contiguous(), burst_labels.float())
        burst_loss = burst_loss.view(-1) * weight_burst
        burst_loss = burst_loss.mean()

        # Calculate flow classification loss
        clf_loss = criterion(sen_res, flow_label.float())

        # Total loss
        all_loss = burst_loss + lambd * clf_loss
        all_loss.backward()  # Backpropagation
        clip_grad_norm_(model.parameters(), grad_clip)  # Gradient clipping
        optimizer.step()

        # Accumulate losses
        total_b_loss += burst_loss.data.item()
        total_c_loss += clf_loss.data.item()

        # Metrics calculation for burst detection
        preds_b = (output > 0.5).long().cpu().numpy()
        trues_b = burst_labels.cpu().numpy()
        pred_f = (sen_res > 0.5).long().cpu().numpy()
        trues_f = flow_label.cpu().numpy()

        batch_size = burst_labels.shape[0]
        for i in range(batch_size):
            true_labels = trues_b[i, :]
            pred_labels = preds_b[i, :]

            # Calculate burst metrics (accuracy, precision, recall, F1-score)
            acc_b, precision_b, recall_b, f1_b = calculate_burst_f1(pred_labels, true_labels)
            if f1_b is not None:
                acc_b_list.append(acc_b)
                f1_b_list.append(f1_b)
                precision_b_list.append(precision_b)
                recall_b_list.append(recall_b)

            # Flow classification metrics
            flow_true = trues_f[i]
            flow_pred = pred_f[i]
            flow_true_all.append(flow_true)
            flow_pred_all.append(flow_pred)

            # Count correct flow predictions
            if all(true_labels == 0):
                if all(pred_labels == 0):
                    flow_correct_count += 1
            else:
                if not all(pred_labels == 0):
                    flow_correct_count += 1

            total_flow += 1

    # Flow accuracy and F1-score
    flow_accuracy = flow_correct_count / total_flow
    f1_flow, precision_flow, recall_flow, acc_flow = calculate_flow_score(flow_pred_all, flow_true_all)

    # Average burst metrics
    average_f1_b = sum(f1_b_list) / len(f1_b_list)
    average_precision_b = sum(precision_b_list) / len(precision_b_list)
    average_recall_b = sum(recall_b_list) / len(recall_b_list)
    average_acc_b = sum(acc_b_list) / len(acc_b_list)

    return total_b_loss / len(train_iter), total_c_loss / len(
        train_iter), average_acc_b, average_precision_b, average_recall_b, average_f1_b, acc_flow, precision_flow, recall_flow, f1_flow


def split_evaluate_wise_flow(model, val_iter, n, test_teacher_forcing_ratio):
    """
    Evaluate the model on the validation set.

    Args:
        model: The trained model to evaluate.
        val_iter: DataLoader for the validation dataset.
        n: Scaling factor for burst labels.
        test_teacher_forcing_ratio: Ratio of teacher forcing during testing.

    Returns:
        Tuple containing validation loss values and metrics.
    """
    with torch.no_grad():
        model.eval()
        total_b_loss = 0
        total_c_loss = 0
        flow_correct_count, total_flow = 0, 0
        acc_b_list, f1_b_list, precision_b_list, recall_b_list = [], [], [], []
        flow_true_all, flow_pred_all = [], []

        # Loop through the validation data
        for b, batch in enumerate(val_iter):
            data, flow_label, burst_fine_label = batch
            src, burst_fine_label = data.cuda(), burst_fine_label.cuda()
            flow_label = flow_label.cuda()

            # Process burst labels and flow labels
            burst_labels = burst_fine_label.clone()
            burst_labels[burst_labels != 0] = 1
            burst_labels = burst_labels.transpose(0, 1).cuda()
            src = src.transpose(0, 1)

            # Forward pass through the model
            output, sen_res = model(src, burst_labels, teacher_forcing_ratio=test_teacher_forcing_ratio)

            # Weight the burst labels
            weight_burst = burst_fine_label.contiguous().view(-1).cuda()
            weight_burst = n_burst(weight_burst.clone(), n)

            # Calculate burst loss
            burst_loss = criterion_fine(output.contiguous(), burst_labels.float())
            burst_loss = burst_loss.view(-1) * weight_burst
            burst_loss = burst_loss.mean()

            # Calculate classification loss
            clf_loss = criterion(sen_res, flow_label.float())
            total_b_loss += burst_loss.data.item()
            total_c_loss += clf_loss.data.item()

            # Calculate metrics for burst detection and flow classification
            preds_b = (output > 0.5).long().cpu().numpy()
            trues_b = burst_labels.cpu().numpy()
            pred_f = (sen_res > 0.5).long().cpu().numpy()
            trues_f = flow_label.cpu().numpy()

            batch_size = burst_labels.shape[0]
            for i in range(batch_size):
                true_labels = trues_b[i, :]
                pred_labels = preds_b[i, :]
                acc_b, precision_b, recall_b, f1_b = calculate_burst_f1(pred_labels, true_labels)
                if f1_b is not None:
                    acc_b_list.append(acc_b)
                    f1_b_list.append(f1_b)
                    precision_b_list.append(precision_b)
                    recall_b_list.append(recall_b)

                flow_true = trues_f[i]
                flow_pred = pred_f[i]
                flow_true_all.append(flow_true)
                flow_pred_all.append(flow_pred)

                # Count correct flow predictions
                if all(true_labels == 0):
                    if all(pred_labels == 0):
                        flow_correct_count += 1
                else:
                    if not all(pred_labels == 0):
                        flow_correct_count += 1

                total_flow += 1

        # Flow accuracy and F1-score
        flow_accuracy = flow_correct_count / total_flow
        f1_flow, precision_flow, recall_flow, acc_flow = calculate_flow_score(flow_pred_all, flow_true_all)

        # Average burst metrics
        average_f1_b = sum(f1_b_list) / len(f1_b_list)
        average_precision_b = sum(precision_b_list) / len(precision_b_list)
        average_recall_b = sum(recall_b_list) / len(recall_b_list)
        average_acc_b = sum(acc_b_list) / len(acc_b_list)

        return total_b_loss / len(val_iter), total_c_loss / len(
            val_iter), average_acc_b, average_precision_b, average_recall_b, average_f1_b, acc_flow, precision_flow, recall_flow, f1_flow


def calculate_flow_score(pd, gt):
    """
    Calculate evaluation metrics (F1, Precision, Recall, Accuracy) for flow prediction.

    Args:
        pd: Predicted flow values.
        gt: Ground truth flow values.

    Returns:
        f1, precision, recall, accuracy: Evaluation metrics.
    """
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1, precision, recall, acc


def calculate_burst_f1(pd, gt):
    """
    Calculate evaluation metrics (Accuracy, Precision, Recall, F1-score) for burst prediction.

    Args:
        pd: Predicted burst labels.
        gt: Ground truth burst labels.

    Returns:
        accuracy, precision, recall, f1: Evaluation metrics.
    """
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        return None, None, None, None

    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)

    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = np.logical_and(seg_inv, gt_inv).sum()

    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)

    return accuracy, precision, recall, f1
