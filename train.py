import torch
import torch.nn as nn


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    metrics_fn,
    device
):
    """
    Trains the model for one epoch.
    Computes and returns the average loss over the training set.
    """
    model.train()
    total_loss = 0.0  # Initialize total loss
    total_batches = 0  # Count batches
    #$total_seqs = len(train_loader.dataset)  # Total number of sequences in the dataset

    # Iterate over each mini-batch in the training set
    for batch in train_loader:
        X = batch['input'].to(device)
        Y = batch['target'].to(device)
        mask1 = batch['mask'].to(device)  # Internal "Mask 1" for training

        optimizer.zero_grad()  # Reset gradients
        predictions = model(X, target=Y)  # Forward pass

        # Compute loss using Mask 1
        loss_val = metrics_fn['loss'](predictions, Y, mask1)
        

        # Backpropagation and optimization
        loss_val.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate total loss
        total_loss += loss_val.item()
        total_batches += 1

    # Compute the epoch-average loss
    avg_loss = total_loss / total_batches
    return {'loss': avg_loss}


def validate_one_epoch(
    model,
    val_loader,
    metrics_fn,
    device
):
    """
    Validates the model for one epoch.
    Computes and returns the average loss over the validation set.
    """
    model.eval()
    total_loss = 0.0  # Initialize total loss
    total_batches = 0  # Count batches

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in val_loader:
            X = batch['input'].to(device)
            Y = batch['target'].to(device)
            # Internal "Mask 1" for validation
            mask1 = batch['mask'].to(device)

            predictions = model(X, target=None)  # Forward pass

            # Compute loss using Mask 1
            loss_val = metrics_fn['loss'](predictions, Y, mask1)

            # Accumulate total loss
            total_loss += loss_val.item()
            total_batches += 1

    # Compute the epoch-average loss
    avg_loss = total_loss / total_batches
    return {'loss': avg_loss}


def run_training(
    model,
    train_loader,
    val_loader,
    metrics_fn,
    device,
    epochs,
    optimizer,
):
    """
    Runs the training loop for the specified number of epochs.
    Returns the training history and final model state.
    """
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Train for the specified number of epochs
    for epoch in range(1, epochs + 1):
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, metrics_fn, device)

        # Validate one epoch
        val_metrics = validate_one_epoch(model, val_loader, metrics_fn, device)

        # Store epoch metrics
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])

        # Print progress
        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )

    # Return the model's state at the final epoch
    final_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return history, final_model_state


def evaluate_for_cv(
    model,
    val_loader,
    metrics_fn,
    device
):
    """
    Evaluates the model with Mask 2 for final cross-validation.
    Computes coverage and accuracy using Mask 2.
    """
    model.eval()
    total_cov, total_acc = 0.0, 0.0  # Initialize totals

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in val_loader:
            X = batch['input'].to(device)
            Y = batch['target'].to(device)
            mask2 = batch['mask2'].to(device)  # Mask 2 for evaluation

            predictions = model(X, target=None)  # Forward pass

            # Compute coverage and accuracy using Mask 2
            cov_val = metrics_fn['coverage'](predictions, Y, mask2)
            acc_val = metrics_fn['accuracy'](predictions, Y, mask2)

            # Accumulate totals
            total_cov += cov_val.sum().item()
            total_acc += acc_val

    total_seqs = len(val_loader.dataset)
    # Dividing the summed accuracy and coverage by the total number of sequences
    avg_cov = total_cov / total_seqs
    avg_acc = total_acc / total_seqs

    # Print evaluation results
    print(
        f"Final CV Evaluation => Coverage: {avg_cov:.4f}, Accuracy: {avg_acc:.4f}"
    )

    return {
        'coverage': avg_cov,
        'accuracy': avg_acc
    }
