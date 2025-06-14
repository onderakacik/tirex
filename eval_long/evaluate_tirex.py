import torch
from tirex import load_model, ForecastModel
from datamodules.custom_forecasting_datamodule import CustomForecastingDataModule
from datamodules.ett_datamodule import ETTDataModule
import numpy as np
import torchmetrics
import pandas as pd
import os
from datetime import datetime
import argparse
from typing import List
import torch.cuda.amp as amp # Import for mixed precision
from tqdm import tqdm # Import tqdm


def evaluate_tirex_on_dataset(
    dataset_name: str,
    seq_len: int,
    pred_len: int,
    features: str = "M",
    target: str = "OT",
    batch_size: int = 512,
    num_workers: int = 4,
    results_file: str = "tirex_evaluation_results.csv"
) -> dict:
    """
    Evaluate TiRex model on a specific dataset with given sequence and prediction lengths.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ETTh1', 'ETTm1', 'weather', 'illness', 'exchange_rate', 'traffic', 'electricity')
        seq_len: Input sequence length
        pred_len: Prediction length
        features: Feature type ('M' for multivariate or 'S' for single)
        target: Target variable name
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        results_file: Path to CSV file to store results
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load TiRex model and move it to the device
    model: ForecastModel = load_model("NX-AI/TiRex")
    model.to(device) # Move model to GPU/CPU
    
    # 2. Setup appropriate datamodule based on dataset name
    if dataset_name.lower().startswith('ett'):
        datamodule = ETTDataModule(
            data_dir="./data",
            batch_size=batch_size,
            test_batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            dataset_name=dataset_name,
            pred_len=pred_len,
            seq_len=seq_len,
            features=features,
            target=target,
            scale=True
        )
    else:  # Use CustomForecastingDataModule for other datasets
        datamodule = CustomForecastingDataModule(
            data_dir="./data",
            dataset_name=dataset_name.lower(),  # Ensure lowercase for matching config
            batch_size=batch_size,
            test_batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            pred_len=pred_len,
            seq_len=seq_len,
            features=features,
            target=target,
            scale=True
        )
    
    print("Preparing data...")
    datamodule.prepare_data(test_only=True)
    print("Setting up datamodule...")
    datamodule.setup(stage="test")
    print("Creating test loader...")
    test_loader = datamodule.test_dataloader()
    
    # Initialize metrics and move them to device
    test_mse = torchmetrics.MeanSquaredError().to(device)
    test_mae = torchmetrics.MeanAbsoluteError().to(device)
    
    # Wrap the test_loader with tqdm for a progress bar
    progress_bar_desc = f"{dataset_name} (seq={seq_len}, pred={pred_len})"
    for batch_x, batch_y in tqdm(test_loader, desc=progress_bar_desc):
        # Move data batches to the device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        batch_size, n_vars, seq_len = batch_x.shape
        
        # Reshape to 2D for TiRex
        x_2d = batch_x.reshape(-1, seq_len)
        
        # Get forecasts using mixed precision (if on GPU)
        with amp.autocast(enabled=device.type == 'cuda'): # Enable autocast only if using CUDA
            quantiles, means = model.forecast(
                context=x_2d,
                output_type="torch",
                prediction_length=pred_len,
            )
        
        # Reshape predictions back
        means = means.reshape(batch_size, n_vars, pred_len)
        
        # Ensure means is on the correct device before updating metrics
        means = means.to(device)
        
        # Update metrics
        test_mse.update(means, batch_y)
        test_mae.update(means, batch_y)
    
    # Compute final metrics
    final_mse = test_mse.compute().item()
    final_mae = test_mae.compute().item()
    
    # Prepare results
    results = {
        'dataset': dataset_name,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'features': features,
        'mae': final_mae,
        'mse': final_mse,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to CSV
    df_row = pd.DataFrame([results])
    if os.path.exists(results_file):
        df_row.to_csv(results_file, mode='a', header=False, index=False)
    else:
        df_row.to_csv(results_file, mode='w', header=True, index=False)
    
    return results


def evaluate_multiple_configurations(
    dataset_names: List[str],
    seq_lengths: List[int],
    pred_lengths: List[int],
    features: str = "M",
    target: str = "OT",
    results_file: str = "tirex_evaluation_results.csv"
):
    """
    Evaluate TiRex model on multiple datasets with different sequence and prediction lengths.
    
    Args:
        dataset_names: List of dataset names
        seq_lengths: List of input sequence lengths
        pred_lengths: List of prediction lengths
        features: Feature type ('M' for multivariate or 'S' for single)
        target: Target variable name
        results_file: Path to CSV file to store results
    """

    num_workers = os.cpu_count() // 4
    batch_size = 128

    for dataset_name in dataset_names:
        for pred_len in pred_lengths:
            for seq_len in seq_lengths:
                print(f"\nEvaluating {dataset_name} with seq_len={seq_len}, pred_len={pred_len}")
                try:
                    results = evaluate_tirex_on_dataset(
                        dataset_name=dataset_name,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        features=features,
                        target=target,
                        results_file=dataset_name + "_" + results_file,
                        num_workers=num_workers,
                        batch_size=batch_size
                    )
                    print(f"Results: MAE={results['mae']:.4f}, MSE={results['mse']:.4f}")
                except Exception as e:
                    print(f"Error evaluating {dataset_name} with seq_len={seq_len}, pred_len={pred_len}: {str(e)}")
                    continue


def parse_list_arg(arg_str: str) -> List:
    """Parse comma-separated string into a list, handling both strings and integers."""
    items = [item.strip() for item in arg_str.split(',')]
    # Try to convert to integers if possible
    try:
        return [int(item) for item in items]
    except ValueError:
        return items


def main():
    parser = argparse.ArgumentParser(description='Evaluate TiRex model on multiple datasets and configurations')
    
    # Required arguments
    parser.add_argument('--datasets', type=str, required=True,
                      help='Comma-separated list of dataset names (e.g., "ETTh1,ETTm1,weather")')
    parser.add_argument('--seq-lengths', type=str, required=True,
                      help='Comma-separated list of sequence lengths (e.g., "96,336,720")')
    parser.add_argument('--pred-lengths', type=str, required=True,
                      help='Comma-separated list of prediction lengths (e.g., "24,48,96")')
    
    # Optional arguments
    parser.add_argument('--features', type=str, default="M",
                      help='Feature type: "M" for multivariate or "S" for single (default: "M")')
    parser.add_argument('--target', type=str, default="OT",
                      help='Target variable name (default: "OT")')
    parser.add_argument('--results-file', type=str, default="tirex_evaluation_results.csv",
                      help='Path to CSV file to store results (default: "tirex_evaluation_results.csv")')
    
    args = parser.parse_args()
    
    # Parse list arguments
    dataset_names = parse_list_arg(args.datasets)
    seq_lengths = parse_list_arg(args.seq_lengths)
    pred_lengths = parse_list_arg(args.pred_lengths)
    
    # Run evaluation
    evaluate_multiple_configurations(
        dataset_names=dataset_names,
        seq_lengths=seq_lengths,
        pred_lengths=pred_lengths,
        features=args.features,
        target=args.target,
        results_file=args.results_file
    )


if __name__ == "__main__":
    main() 