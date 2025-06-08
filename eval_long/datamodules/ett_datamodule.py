"""
ETT (Electricity Transformer Temperature) dataset module for ccnn_v2.
Adapted from TCCN's data handling for ETT datasets.
"""

import os
import pathlib
import zipfile
import shutil
import requests
import gdown
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import gc

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from hydra import utils

# Import for deterministic worker init
from .deterministic_utils import worker_init_fn as deterministic_worker_init_fn
from .utils import normalise_data, split_data, load_data_from_partition, save_data


class ETTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        dataset_name: str = "ETTh1",
        pred_len: int = 96,
        seq_len: int = 336,
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        use_deterministic_worker_init: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()

        # Save parameters to self
        self.data_dir = data_dir  # Store path directly without Hydra
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_name = dataset_name  # ETTh1, ETTh2, ETTm1, or ETTm2
        self.pred_len = pred_len  # prediction length
        self.seq_len = seq_len  # input sequence length
        self.features = features  # 'S' for single target, 'M' for multiple features
        self.target = target  # target column name, default is 'OT'
        self.scale = scale  # whether to normalize data

        self.use_deterministic_worker_init = use_deterministic_worker_init
        self.seed = seed
        
        self.generator = torch.Generator().manual_seed(self.seed)
        self.worker_init_fn = deterministic_worker_init_fn if self.use_deterministic_worker_init else None

        # Set paths - use os.path.join to properly handle relative paths
        if self.data_dir.startswith("."):
            # Handle relative path
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            root = os.path.join(current_dir, self.data_dir)
        else:
            # Use as is for absolute paths
            root = self.data_dir
            
        root = pathlib.Path(root)
        self.download_location = root / "ETT"
        self.csv_path = self.download_location / f"{dataset_name}.csv"
        self.data_processed_location = root / "ETT" / "processed_data" / dataset_name / f"seq_{seq_len}_pred_{pred_len}_{features}"

        
        # Create directory if it doesn't exist
        os.makedirs(self.download_location, exist_ok=True)
        os.makedirs(self.data_processed_location, exist_ok=True)
        
        # Set data characteristics
        self.data_type = "sequence"
        self.freq = 'h' if 'h' in dataset_name else 't'  # h for hourly, t for minute
        
        # Determine input and output dimensions based on features
        self.input_channels = 7 if features == "M" else 1  # ETT datasets have 7 features
        self.output_channels = 7 if features == "M" else 1

        # Initialize tensors to None
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None
        self.test_X = None
        self.test_Y = None

    def prepare_data(self):
        # Check if processed data exists, if not, download and process
        if not os.path.exists(self.data_processed_location / "train_X.pt"):
            # Download data if not already downloaded
            self._download()
            # Process and save data
            self.train_X, self.val_X, self.test_X, self.train_Y, self.val_Y, self.test_Y = self._process_data()
            # Save processed data
            save_data(
                self.data_processed_location,
                train_X=self.train_X,
                val_X=self.val_X,
                test_X=self.test_X,
                train_y=self.train_Y,
                val_y=self.val_Y,
                test_y=self.test_Y,
            )
            gc.collect()

    def setup(self, stage=None):
        # Load datasets for appropriate stages
        if stage == "fit" or stage is None:
            # Train data
            if self.train_X is None:
                self.train_X, self.train_Y = load_data_from_partition(self.data_processed_location, partition="train")
            self.train_dataset = TensorDataset(self.train_X, self.train_Y)
            gc.collect()

            # Validation data
            if self.val_X is None:
                self.val_X, self.val_Y = load_data_from_partition(self.data_processed_location, partition="val")
            self.val_dataset = TensorDataset(self.val_X, self.val_Y)
            gc.collect()
        
        if stage == "test" or stage is None:
            # Test data
            if self.test_X is None:
                self.test_X, self.test_Y = load_data_from_partition(self.data_processed_location, partition="test")
            self.test_dataset = TensorDataset(self.test_X, self.test_Y)
            gc.collect()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            persistent_workers=self.num_workers > 0,
        )

    def _download(self):
        """Download ETT dataset if it doesn't exist"""
        # Check if data file already exists
        if os.path.exists(self.csv_path):
            print(f"{self.dataset_name} dataset already exists at {self.csv_path}")
            return
        
        # URL for the Google Drive ETT dataset
        file_id = "1bnrv7gpn27yO54WJI-vuXP5NclE5BlBx" 
        zip_path = self.download_location / "ETT-small.zip"
        
        # Download the zip file from Google Drive
        print(f"Downloading ETT dataset from Google Drive...")
        gdown.download(
            id=file_id,
            output=str(zip_path),
            quiet=False
        )
        
        # Extract the zip file
        print(f"Extracting ETT dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.download_location)
        
        # Move files to the correct location if needed
        extracted_dir = self.download_location / "ETT-small"
        if os.path.exists(extracted_dir):
            for file in os.listdir(extracted_dir):
                src_path = extracted_dir / file
                dest_path = self.download_location / file
                shutil.copy(src_path, dest_path)
        
        # Clean up
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        print(f"ETT dataset downloaded successfully to {self.download_location}")

    def _process_data(self):
        """Process the ETT dataset for forecasting tasks with memory efficiency"""
        print(f"Processing {self.dataset_name} dataset...")
        
        # Read the CSV file
        df_raw = pd.read_csv(self.csv_path)
        
        # Define data boundaries for train/val/test splits
        if 'h' in self.dataset_name:
            # For hourly data
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            # For minute-level data
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        
        # Extract features based on setting
        if self.features == 'M':
            # Use all features except the date
            cols_data = df_raw.columns[1:]
            data = df_raw[cols_data].values
        else:  # 'S'
            # Use only the target column
            data = df_raw[[self.target]].values
        
        # Convert to float32 early to save memory
        data = data.astype(np.float32)
        
        # Normalize data
        if self.scale:
            scaler = StandardScaler()
            # Fit scaler on training data
            train_data = data[border1s[0]:border2s[0]]
            scaler.fit(train_data)
            data_normalized = scaler.transform(data)
            del train_data
            gc.collect()
        else:
            data_normalized = data
        
        def create_sequences(data, seq_len, pred_len):
            X, Y = [], []
            # Ensure enough data for at least one sequence
            if len(data) < seq_len + pred_len:
                raise ValueError(f"Not enough data to create sequences. Data length: {len(data)}, Required: {seq_len + pred_len}")
            
            for i in range(len(data) - seq_len - pred_len + 1):
                X.append(data[i:i + seq_len])
                Y.append(data[i + seq_len:i + seq_len + pred_len])
            return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
        
        # Process Train Data
        print("Creating training sequences...")
        train_X_np, train_Y_np = create_sequences(
            data_normalized[border1s[0]:border2s[0]], 
            self.seq_len, 
            self.pred_len
        )
        train_X = torch.FloatTensor(train_X_np).transpose(1, 2)
        train_Y = torch.FloatTensor(train_Y_np).transpose(1, 2)
        del train_X_np, train_Y_np
        gc.collect()

        # Process Validation Data
        print("Creating validation sequences...")
        val_X_np, val_Y_np = create_sequences(
            data_normalized[border1s[1]:border2s[1]], 
            self.seq_len, 
            self.pred_len
        )
        val_X = torch.FloatTensor(val_X_np).transpose(1, 2)
        val_Y = torch.FloatTensor(val_Y_np).transpose(1, 2)
        del val_X_np, val_Y_np
        gc.collect()

        # Process Test Data
        print("Creating testing sequences...")
        test_X_np, test_Y_np = create_sequences(
            data_normalized[border1s[2]:border2s[2]], 
            self.seq_len, 
            self.pred_len
        )
        test_X = torch.FloatTensor(test_X_np).transpose(1, 2)
        test_Y = torch.FloatTensor(test_Y_np).transpose(1, 2)
        del test_X_np, test_Y_np
        gc.collect()

        # Delete the large normalized data array
        del data_normalized, data
        gc.collect()

        print(f"Processed data shapes:")
        print(f"Train X: {train_X.shape}, Train Y: {train_Y.shape}")
        print(f"Val X: {val_X.shape}, Val Y: {val_Y.shape}")
        print(f"Test X: {test_X.shape}, Test Y: {test_Y.shape}")
        
        return train_X, val_X, test_X, train_Y, val_Y, test_Y