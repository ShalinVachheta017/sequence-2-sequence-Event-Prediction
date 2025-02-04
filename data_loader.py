import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

class SequenceDataset(Dataset):
    """
    A PyTorch Dataset for our sequences.
    Each item is a dictionary containing:
        'input': The input feature matrix (sequence length T x input dimension).
        'target': The target sequence.
        'mask': Mask 1 (used during training/validation).
        'mask2': Mask 2 (used for evaluation).
    """

    def __init__(self, X, y, mask, mask2):
        self.X = X
        self.y = y
        self.mask = mask
        self.mask2 = mask2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.X[idx], dtype=torch.float),
            'target': torch.tensor(self.y[idx], dtype=torch.float),
            'mask': torch.tensor(self.mask[idx], dtype=torch.float),
            'mask2': torch.tensor(self.mask2[idx], dtype=torch.float)
        }


def load_raw_data(data_path):
    """
    Loads raw data from CSV (or another source).
    Returns X, y, mask, mask2, cycles.
    """
    data = pd.read_csv(data_path)  
    mask = data['mask'].values.reshape(-1, 81)
    mask2 = data['mask2'].values.reshape(-1, 81)

    # X and y: shape (num_cycles, 81, 34)
    X = data[[str(i) for i in range(34)]].values.reshape(-1, 81, 34)
    # If y is the same as X or shifted, depends on your dataset
    y = data[[str(i) for i in range(34)]].values.reshape(-1, 81, 34)

    cycles = data['cycle'].values.reshape(-1, 81)[:, 0]  # e.g. the cycle ID

    return X, y, mask, mask2, cycles


def shuffle_and_split_cycles(X, y, mask, mask2, cycles, train_val_ratio=0.7, seed=42):
    """
    Shuffle cycles and split them into train+val vs. test sets by cycle ID.

    For example, 70% train+val, 30% test.
    """
    np.random.seed(seed)
    unique_cycles = np.unique(cycles)
    np.random.shuffle(unique_cycles)

    num_train_val = int(len(unique_cycles) * train_val_ratio) #from configuration file
    train_val_cycles = unique_cycles[:num_train_val]
    test_cycles = unique_cycles[num_train_val:]

    # Creates boolean masks Marks rows belonging to train+val cycles & Marks rows belonging to test cycles.
    train_val_mask = np.isin(cycles, train_val_cycles)
    test_mask = np.isin(cycles, test_cycles)

    # Split
    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    mask_train_val = mask[train_val_mask]
    mask2_train_val = mask2[train_val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    mask_test = mask[test_mask]
    mask2_test = mask2[test_mask]

    return (X_train_val, y_train_val, mask_train_val, mask2_train_val), \
           (X_test, y_test, mask_test, mask2_test)


def create_datasets(X, y, mask, mask2):
    """
    Creates a PyTorch Dataset from arrays.
    """
    return SequenceDataset(X, y, mask, mask2)


def get_data_loaders_for_fold(
    X_train_val,
    y_train_val,
    mask_train_val,
    mask2_train_val,
    batch_size,
    folds=5,
    seed=42
):
    """
    Generates (train_loader, val_loader) for each fold in k-fold cross-validation.
    Each fold merges 4 subsets for training and the 1 subset for validation.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    indices = np.arange(len(X_train_val))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        # Training set = 4 folds combined
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        mask_train = mask_train_val[train_idx]
        mask2_train = mask2_train_val[train_idx]

        # Validation set = the 1 remaining fold
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]
        mask_val = mask_train_val[val_idx]
        mask2_val = mask2_train_val[val_idx]

        train_dataset = create_datasets(X_train, y_train, mask_train, mask2_train)
        val_dataset = create_datasets(X_val, y_val, mask_val, mask2_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        yield train_loader, val_loader
