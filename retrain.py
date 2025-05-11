import os
import csv
import torch
from torch.utils.data import DataLoader
from utils import load_config, set_seed, get_device, save_model
from data_loader import load_raw_data, shuffle_and_split_cycles, SequenceDataset
from model_definition import Encoder, Decoder, Seq2Seq
from metrics import get_metrics_fn


def retrain_and_test(best_job, config_path, data_path, runs=5):
    """
    Retrain the best job configuration multiple times and evaluate on the test set.

    Args:
        best_job (dict): The best model configuration.
        config_path (str): Path to the configuration file.
        data_path (str): Path to the dataset file.
        runs (int): Number of retraining runs.
    """
    print("Loading configuration...")
    config = load_config(config_path)
    general = config['general']
    set_seed(general['seed'])
    device = get_device()

    print("Loading and splitting data...")
    X, y, mask, mask2, cycles = load_raw_data(data_path)
    (X_train_val, y_train_val, mask_train_val, mask2_train_val), \
        (X_test, y_test, mask_test, mask2_test) = shuffle_and_split_cycles(
        X, y, mask, mask2, cycles,
        train_val_ratio=general['train_val_split_ratio'],
        seed=general['seed']
    )

    print("Preparing test data loader...")
    test_dataset = SequenceDataset(X_test, y_test, mask_test, mask2_test)
    test_loader = DataLoader(
        test_dataset, batch_size=best_job['batch_size'], shuffle=False)

    # Extract hyperparameters from the best job
    input_size = 34
    output_size = 34
    hidden_size = best_job['hidden_size']
    num_layers = best_job['num_layers']
    bidirectional = best_job['bidirectional']
    lr = best_job['learning_rate']
    epochs = best_job['epochs']

    metrics_fn = get_metrics_fn()
    model_save_paths = []

    print("Starting retraining...")
    for run_id in range(1, runs + 1):
        print(f"Retraining Run {run_id}/{runs}")
        encoder = Encoder(input_size, hidden_size,
                          num_layers, bidirectional).to(device)
        decoder = Decoder(input_size, hidden_size, output_size,
                          num_layers, bidirectional).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-5)  # Add weight decay


        for epoch in range(1, epochs + 1):
            model.train()
            train_loader = DataLoader(SequenceDataset(X_train_val, y_train_val, mask_train_val, mask2_train_val),
                                      batch_size=best_job['batch_size'], shuffle=True)

            for batch in train_loader:
                X_batch = batch['input'].to(device)
                Y_batch = batch['target'].to(device)
                mask_batch = batch['mask'].to(device)

                optimizer.zero_grad()
                predictions = model(X_batch, target=Y_batch)
                loss = metrics_fn['loss'](predictions, Y_batch, mask_batch)
                loss.backward()
                optimizer.step()

        # Save the model state for the current run
        model_save_path = os.path.join(
            general['model_save_path'], f"model_run_{run_id}.pth")
        save_model(model, model_save_path, epoch=epochs, loss=loss.item())
        model_save_paths.append(model_save_path)

    print("Retraining complete. Starting testing...")

    test_results = []
    for run_id, model_path in enumerate(model_save_paths, 1):
        print(f"Testing Model {run_id}/{runs}")
        encoder = Encoder(input_size, hidden_size,
                          num_layers, bidirectional).to(device)
        decoder = Decoder(input_size, hidden_size, output_size,
                          num_layers, bidirectional).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.eval()

        total_test_loss, total_test_coverage, total_test_accuracy = 0.0, 0.0, 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                X_batch = batch['input'].to(device)
                Y_batch = batch['target'].to(device)
                mask_batch = batch['mask2'].to(device)

                # print("X_batch shape:", X_batch.shape)
                # print("Y_batch shape:", Y_batch.shape)
                # print("mask_batch shape:", mask_batch.shape)    

                predictions = model(X_batch)
                loss = metrics_fn['loss'](predictions, Y_batch, mask_batch)
                coverage = metrics_fn['coverage'](predictions, Y_batch, mask_batch)
                accuracy = metrics_fn['accuracy'](predictions, Y_batch, mask_batch)

                    
                total_test_loss += loss.item()
                total_test_coverage += coverage.sum().item()
                total_test_accuracy += accuracy
                total_batches += 1
                
        total_seqs = len(test_loader.dataset)
        avg_test_loss = total_test_loss / total_batches
        avg_test_coverage = total_test_coverage / total_seqs
        avg_test_accuracy = total_test_accuracy / total_seqs

        print(
            f"Run {run_id}: Test Loss = {avg_test_loss:.4f}, Test Coverage = {avg_test_coverage:.4f},Test Accuracy = {avg_test_accuracy:.4f} ")

        test_results.append({
            'run_id': run_id,
            'test_loss': avg_test_loss,
            'test_coverage': avg_test_coverage,
            'test_accuracy': avg_test_accuracy
        })

    # Save test results
    results_path = "final_test_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=['run_id', 'test_loss', 'test_coverage','test_accuracy'])
        writer.writeheader()
        writer.writerows(test_results)

    print(f"Test results saved to {results_path}")


if __name__ == "__main__":
    try:
        print("Starting retrain.py...")
        CSV_PATH = "all_jobs_results.csv"
        CONFIG_PATH = "config.yaml"
        DATA_PATH = "Events.csv"

        # Select the best job
        print("Selecting the best job...")
        from post_analysis import select_best_job
        best_job = select_best_job(CSV_PATH)
        print(f"Best Job: {best_job}")

        # Retrain and test the best job
        retrain_and_test(best_job, CONFIG_PATH, DATA_PATH)
        print("Retraining and testing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")


