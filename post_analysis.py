import pandas as pd


def select_best_job(csv_path):
    """
    Selects the best job based on validation coverage and ensures it is not overfitted.

    Args:
        csv_path (str): Path to the all_jobs_results.csv file.

    Returns:
        dict: Best job's parameters as a dictionary.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

  
    filtered_df = df[df['avg_train_loss'] >= df['avg_val_loss']]

    if filtered_df.empty:
        raise ValueError(
            "No valid jobs found with avg_train_loss => avg_val_loss.")

    # Select the job with the highest avg_val_coverage
    best_job_row = filtered_df.loc[filtered_df['avg_mask2_coverage'].idxmax()]

    # Convert the row into a dictionary
    best_job = {
        'job_name': best_job_row['job_name'],
        'learning_rate': best_job_row['learning_rate'],
        'epochs': int(best_job_row['epochs']),
        'batch_size': int(best_job_row['batch_size']),
        'hidden_size': int(best_job_row['hidden_size']),
        'num_layers': int(best_job_row['num_layers']),
        'bidirectional': bool(best_job_row['bidirectional']),
        'avg_mask2_coverage': best_job_row['avg_mask2_coverage']
    }

    return best_job


if __name__ == "__main__":
    CSV_PATH = "all_jobs_results.csv"
    best_job = select_best_job(CSV_PATH)
    print(f"Best Job Selected: {best_job}")
