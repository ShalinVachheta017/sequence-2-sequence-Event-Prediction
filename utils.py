# utils.py

import os
import csv
import torch
import random
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt


def load_config(config_path):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def save_model(model, path, epoch, loss):
    """
    Save the model state and additional metadata.
    Creates the directory (if not present) and saves the model parameters.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path to save the model.
        epoch (int): Epoch at which the model is saved.
        loss (float): Validation loss at the time of saving.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, path)


def load_model(model, optimizer, path, device):
    """
    Loads the model state from the specified path.

    Args:
        model (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer): Optimizer to load the state into.
        path (str): Path to the saved model.
        device (torch.device): Device to map the model to.

    Returns:
        int: Epoch number.
        float: Loss value.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def get_device():
    """
    Returns the available device (GPU if available, else CPU).

    Returns:
        torch.device: The device to use.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_logger(log_directory, job_name):
    """
    Sets up and returns a logger.

    Args:
        log_directory (str): Directory to save log files.
        job_name (str): Name of the job for log file naming.

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(log_directory, exist_ok=True)
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(
            log_directory, f"{job_name}_log.txt"))
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def plot_metric_over_epochs(history, metric_name, output_path):
    """
    Plots a single metric over epochs.

    Args:
        history (dict): Dictionary containing metric history.
        metric_name (str): Name of the metric to plot.
        output_path (str): Directory to save the plot.
    """
    train_key = f"train_{metric_name}"
    val_key = f"val_{metric_name}"

    plt.figure()
    plt.plot(history[train_key], label=f"Train {metric_name.capitalize()}")
    plt.plot(history[val_key], label=f"Val {metric_name.capitalize()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Over Epochs")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{metric_name}_over_epochs.png"))
    plt.close()


def plot_all_metrics(history, output_path):
    """
    Plots all relevant metrics over epochs.

    Args:
        history (dict): Dictionary containing metric history.
        output_path (str): Directory to save the plots.
    """
    for metric in ['loss']:  # , 'accuracy', 'coverage'
        plot_metric_over_epochs(history, metric, output_path)
        

def save_all_jobs_results(all_jobs_metrics, output_path):
    """
    Saves the results of all jobs (hyperparameters and final averaged metrics) into a CSV.
    """
    if not all_jobs_metrics:
        print("No job metrics to save.")
        return

    fieldnames = [
        'job_name', 'learning_rate', 'epochs', 'batch_size',
        'hidden_size', 'num_layers', 'bidirectional',
        'avg_train_loss', 'avg_val_loss',
        'avg_mask2_coverage', 'avg_mask2_accuracy'
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job_metric in all_jobs_metrics:
            writer.writerow(job_metric)
