import os
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau #4overfit
from utils import load_config, set_seed, get_device, get_logger, plot_all_metrics, save_all_jobs_results
from data_loader import load_raw_data, shuffle_and_split_cycles, get_data_loaders_for_fold
from model_definition import Encoder, Decoder, Seq2Seq
from train import run_training, evaluate_for_cv
from metrics import get_metrics_fn


def average_fold_histories(fold_histories):
    """
    Computes the average of metrics (e.g., loss, accuracy, coverage) across all folds.
    """
    if not fold_histories:
        raise ValueError("fold_histories cannot be empty.")

    # Ensure all dictionaries have the same keys
    keys = fold_histories[0].keys()
    if not all(set(hist.keys()) == set(keys) for hist in fold_histories):
        raise ValueError("Inconsistent keys in fold histories.")

    num_folds = len(fold_histories)
    avg_history = {}

    for key in keys:
        # Each hist[key] is a list of metrics for the entire epoch range
        # shape: (num_folds, num_epochs)
        metrics = np.array([hist[key] for hist in fold_histories])
        avg_metrics = metrics.mean(axis=0).tolist()  # average across folds
        avg_history[key] = avg_metrics

    return avg_history

def main(config_path, data_path):
    """
    Main function orchestrating the k-fold cross-validation and evaluation workflow.
    """
    # 1) Load config and initialize seed/device
    config = load_config(config_path)
    general = config['general']
    jobs = config.get('jobs', [])
    set_seed(general['seed'])
    device = get_device()

    # 2) Load and split data into train/val/test
    X, y, mask, mask2, cycles = load_raw_data(data_path)
    (X_train_val, y_train_val, mask_train_val, mask2_train_val), \
        (X_test, y_test, mask_test, mask2_test) = shuffle_and_split_cycles(
            X, y, mask, mask2, cycles,
            train_val_ratio=general['train_val_split_ratio'],
            seed=general['seed']
    )

    metrics_fn = get_metrics_fn()
    batch_size = general['batch_size']
    folds = general['folds']
    all_jobs_metrics = []

    # 3) Iterate over jobs (each job is a set of hyperparameters)
    for job in jobs:
        job_name = job['job_name']
        lr = job['learning_rate']
        epochs = job['epochs']
        hidden_size = job['hidden_size']
        num_layers = job['num_layers']
        bidirectional = job['bidirectional']

        # Setup logger
        logger = get_logger(
            os.path.join(general['log_directory'], job_name),
            job_name
        )

        logger.info(f"Starting K-Fold Cross-Validation for: {job_name}")
        logger.info(
            f"Hyperparameters => LR={lr}, Epochs={epochs}, "
            f"Hidden Size={hidden_size}, Num Layers={num_layers}, "
            f"Bidirectional={bidirectional}"
        )

        model_save_path = os.path.join(general['model_save_path'], job_name)
        os.makedirs(model_save_path, exist_ok=True)

        input_size = 34
        output_size = 34

        fold_histories = []
        fold_final_states = []
        fold_mask2_coverages = []
        fold_mask2_accuracies = []

        # 4) Perform k-fold cross-validation
        for fold_index, (train_loader, val_loader) in enumerate(
            get_data_loaders_for_fold(
                X_train_val, y_train_val, mask_train_val, mask2_train_val,
                batch_size=batch_size, folds=folds, seed=general['seed']
            ),
            start=1
        ):
            logger.info(f"===== Fold {fold_index}/{folds} =====\n")

            # Build the model fresh for each fold
            encoder = Encoder(input_size, hidden_size, num_layers, bidirectional).to(device)
            decoder = Decoder(input_size, hidden_size, output_size, num_layers, bidirectional).to(device)
            model = Seq2Seq(encoder, decoder, device).to(device)

            # Setup optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            # Train and validate (Mask 1)
            history, final_model_state_fold = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                metrics_fn=metrics_fn,
                device=device,
                epochs=epochs,
                optimizer=optimizer
            )

            fold_histories.append(history)
            fold_final_states.append(final_model_state_fold)

            # Step the scheduler using validation loss
            val_loss = history['val_loss'][-1]  # Extract final validation loss
            scheduler.step(val_loss)

            logger.info(
                f"Fold {fold_index}: Final Training Loss={history['train_loss'][-1]: .4f}, Final Validation Loss={val_loss: .4f}"
            )

            # Evaluate final model on this fold with Mask 2
            model.load_state_dict(final_model_state_fold)
            cv_metrics = evaluate_for_cv(model, val_loader, metrics_fn, device)
            fold_mask2_coverages.append(cv_metrics['coverage'])
            fold_mask2_accuracies.append(cv_metrics['accuracy'])

            logger.info(
                f"Fold {fold_index} => Mask2 Coverage: {cv_metrics['coverage']:.4f}, "
                f"Accuracy: {cv_metrics['accuracy']:.4f}"
            )
            logger.info(f"Fold {fold_index} completed.\n")

        # 5) Average the Mask 1 metrics across folds
        avg_history = {
            key: np.mean([fold[key] for fold in fold_histories], axis=0).tolist()
            for key in fold_histories[0]
        }

        # Plot average train/val curves
        plot_all_metrics(avg_history, model_save_path)

        # 6) Aggregate Mask 2 coverage/accuracy across folds
        avg_mask2_cov = sum(fold_mask2_coverages) / len(fold_mask2_coverages)
        avg_mask2_acc = sum(fold_mask2_accuracies) / len(fold_mask2_accuracies)
        logger.info(
            f"Averaged Mask 2 coverage: {avg_mask2_cov:.4f}, accuracy: {avg_mask2_acc:.4f}")

        # 7) Save final metrics for the job
        job_metrics = {
            'job_name': job_name,
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'avg_train_loss': avg_history['train_loss'][-1],
            'avg_val_loss': avg_history['val_loss'][-1],
            'avg_mask2_coverage': avg_mask2_cov,
            'avg_mask2_accuracy': avg_mask2_acc,
        }
        all_jobs_metrics.append(job_metrics)

    # 8) Save all jobs’ results to CSV
    # Save the all_jobs_results.csv directly in the main folder
    all_jobs_csv_path = "all_jobs_results.csv"
    
    save_all_jobs_results(all_jobs_metrics, all_jobs_csv_path)


    # 9) Identify the best job metrics
    if all_jobs_metrics:
        best_job = max(all_jobs_metrics, key=lambda x: x['avg_mask2_coverage'])
        print(
            f"Best Job: {best_job['job_name']} "    
            f"with Mask2 Coverage = {best_job['avg_mask2_coverage']:.4f}"
        )
    else:
        print("No job metrics available.")

    print("All jobs completed.")


if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    DATA_PATH = "Events.csv"
    main(CONFIG_PATH, DATA_PATH)
