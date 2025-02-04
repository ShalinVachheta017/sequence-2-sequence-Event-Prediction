import torch
import torch.nn as nn


def cross_entropy_loss(predictions, targets, mask):
    """
    CrossEntropyLoss, ignoring positions where mask=0.
    logist: Unnormalized scores output by the mode
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    B, T, F = predictions.shape

    # Convert last-dim targets to class indices if needed
    if targets.dim() == 3 and targets.size(2) > 1:
        targets = targets.argmax(dim=2) # .argmax(dim=2) converts one-hot encoded targets to class indices

    predictions = predictions.view(B * T, F)
    targets = targets.view(B * T).long() # .long() converts it to the required integer type
    mask = mask.view(B * T)

    loss = criterion(predictions, targets)
    masked_loss = loss * mask
    # 1e-8 is a very small value (0.00000001) that Prevents division-by-zero errors when the mask sum is 0.
    return masked_loss.sum() / (mask.sum() + 1e-8)


def accuracy(predictions, targets, mask):
    """
       Args:
        predictions (torch.Tensor): Predicted logits with shape (B, T, F).
        targets (torch.Tensor): True labels, either as class indices (B, T) or one-hot (B, T, F).
        mask (torch.Tensor): Binary mask indicating valid positions with shape (B, T).

    """
    B, T, _ = predictions.shape  # e.g. (batch_size, seq_len, num_classes)
    
    if targets.dim() == 3 and targets.size(2) > 1:
        targets = targets.argmax(dim=2)

    preds = predictions.argmax(dim=2)

    # Move tensors to CPU and convert to NumPy
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    mask_np = mask.cpu().numpy()

    # Compute per-sequence unique-event-based accuracy
    accuracies = []
    # Iterate over sequences/per sequence approach
    for pred_i, label_i, mask_i in zip(preds_np, targets_np, mask_np):
        masked_preds_i = pred_i[mask_i == 1]
        masked_labels_i = label_i[mask_i == 1]

        unique_preds_i = set(masked_preds_i.tolist())
        unique_labels_i = set(masked_labels_i.tolist())

        if len(unique_labels_i) == 0:
            accuracies.append(0.0)
            continue

        intersection = unique_preds_i.intersection(unique_labels_i)
        acc_i = len(intersection) / len(unique_labels_i)
        accuracies.append(acc_i)
# Warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Remove the division by len(accuracies),just the sum ;then divide by the total number of examples (or total number of sequences) later in your loop.
    return float(sum(accuracies) ) if accuracies else 0.0

def coverage(predictions, targets, mask):
    B, T, _ = predictions.shape # shape: (B, T, F)

    if targets.dim() == 3 and targets.size(2) > 1:
        targets = targets.argmax(dim=2)

    preds = predictions.argmax(dim=2)    # (B, T)
    correct = (preds == targets).float()  # (B, T)
    mask = mask.float()                  # (B, T)

    # sum(correct_i * mask_i) => # correct positions in sample i
    sum_correct_per_sample = (correct * mask).sum(dim=1)   # shape: (B,)
    sum_mask_per_sample = mask.sum(dim=1)                  # shape: (B,)

    coverage_per_sample = sum_correct_per_sample / \
        (sum_mask_per_sample + 1e-8)  # shape: (B,)
# Warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Removed mean
    coverage = coverage_per_sample 
    return coverage


def get_metrics_fn():
    """
    Returns a dictionary mapping metric names to their functions.
    """
    return {
        'loss': cross_entropy_loss,
        'accuracy': accuracy,
        'coverage': coverage
    }
