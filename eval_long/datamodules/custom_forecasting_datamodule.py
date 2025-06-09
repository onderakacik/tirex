"""
A unified data module for handling various time series forecasting datasets (Weather, ILI, Exchange rate, etc.).
Adapted from TCCN's data handling patterns.
"""

import os
import pathlib
import zipfile
import shutil
import gdown
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import gc

# Import for deterministic worker init
from .deterministic_utils import worker_init_fn as deterministic_worker_init_fn
from .utils import load_data_from_partition, save_data


class CustomForecastingDataModule(pl.LightningDataModule):
    # Dataset configurations
    DATASET_CONFIGS = {
        'weather': {
            'file_name': 'weather.csv',
            'zip_name': 'weather.zip',
            'drive_id': '1nXdMIJ7K201Bx3IBGNiaNFQ6FzeDEzIr',
            'num_features': 21,
            'date_col': 'date'
        },
        'illness': {
            'file_name': 'national_illness.csv',
            'zip_name': 'illness.zip',
            'drive_id': '1WMKg7KevVEEd9jrfLG8mcpOequZMbjlM',
            'num_features': 7,
            'date_col': 'date'
        },
        'exchange_rate': {
            'file_name': 'exchange_rate.csv',
            'zip_name': 'exchange_rate.zip',
            'drive_id': '1rN79CxW3Vldp-WDuSoG0bKq9tYQR79UK',
            'num_features': 8,
            'date_col': 'date'
        },
        'traffic': {
            'file_name': 'traffic.csv',
            'zip_name': 'traffic.zip',
            'drive_id': '17t49bEbuhVI5v_q5mEINGRgMEf_kjwLg',
            'num_features': 862,
            'date_col': 'date'
        },
        'electricity': {
            'file_name': 'electricity.csv',
            'zip_name': 'electricity.zip',
            'drive_id': '1FHH0S3d6IK_UOpg6taBRavx4MragRLo1',
            'num_features': 321,
            'date_col': 'date'
        },
    }

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        pred_len: int = 96,
        seq_len: int = 336,
        features: str = "M",
        target: str = "OT",  # Only used for single-feature mode
        scale: bool = True,
        use_deterministic_worker_init: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()

        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(self.DATASET_CONFIGS.keys())}")

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.features = features
        self.target = target
        self.scale = scale
        
        # Get dataset specific configuration
        self.dataset_config = self.DATASET_CONFIGS[dataset_name]
        self.num_data_features = self.dataset_config['num_features']
        
        self.use_deterministic_worker_init = use_deterministic_worker_init
        self.seed = seed
        
        self.generator = torch.Generator().manual_seed(self.seed)
        self.worker_init_fn = deterministic_worker_init_fn if self.use_deterministic_worker_init else None

        # Set up paths
        if self.data_dir.startswith("."):
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            root = os.path.join(current_dir, self.data_dir)
        else:
            root = self.data_dir
            
        root = pathlib.Path(root)
        self.download_location = root / dataset_name.capitalize()
        self.csv_path = self.download_location / self.dataset_config['file_name']
        self.data_processed_location = root / dataset_name.capitalize() / "processed_data" / self.dataset_name / f"seq_{seq_len}_pred_{pred_len}_{features}"

        os.makedirs(self.download_location, exist_ok=True)
        # os.makedirs(self.data_processed_location, exist_ok=True)
        
        self.data_type = "sequence"
        
        self.input_channels = self.num_data_features if features == "M" else 1
        self.output_channels = self.num_data_features if features == "M" else 1

        # Initialize tensors to None
        self.train_X, self.train_Y = None, None
        self.val_X, self.val_Y = None, None
        self.test_X, self.test_Y = None, None


    def prepare_data(self):
        # Check if raw data exists, download if not
        if not self.csv_path.exists():
             self._download()
        
        # # Check if processed data exists. If not, process and save it.
        # if not (self.data_processed_location / "train_X.pt").exists():
        #     print("Processed data not found. Processing and saving...")
        #     self.train_X, self.val_X, self.test_X, self.train_Y, self.val_Y, self.test_Y = self._process_data()
        #     save_data(
        #         self.data_processed_location,
        #         train_X=self.train_X,
        #         val_X=self.val_X,
        #         test_X=self.test_X,
        #         train_y=self.train_Y,
        #         val_y=self.val_Y,
        #         test_y=self.test_Y,
        #     )
        #     gc.collect()
        # else:
        #     print("Processed data found. Skipping processing in prepare_data.")


        # Always process data and store in memory - writing to disk and reading later creates memory bottlenecks for large datasets e.g. electricity, traffic, etc.
        print("Processing data and storing in memory...")
        self.train_X, self.val_X, self.test_X, self.train_Y, self.val_Y, self.test_Y = self._process_data()
        gc.collect()



    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            if self.train_X is None:
                self.train_X, self.train_Y = load_data_from_partition(self.data_processed_location, partition="train")
            self.train_dataset = TensorDataset(self.train_X, self.train_Y)
            gc.collect()

            if self.val_X is None:
                self.val_X, self.val_Y = load_data_from_partition(self.data_processed_location, partition="val")
            self.val_dataset = TensorDataset(self.val_X, self.val_Y)
            gc.collect()

        if stage == "test" or stage is None:
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
        """Downloads the dataset from Google Drive if it doesn't exist."""
        if self.csv_path.exists():
            print(f"{self.dataset_name} dataset already exists at {self.csv_path}")
            return
        
        file_id = self.dataset_config['drive_id']
        zip_path = self.download_location / self.dataset_config['zip_name']
        
        print(f"Downloading {self.dataset_name} dataset from Google Drive...")
        gdown.download(
            id=file_id,
            output=str(zip_path),
            quiet=False
        )
        
        print(f"Extracting {self.dataset_name} dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.download_location)
        
        # Try to move CSV if it's in a subfolder
        extracted_subfolder = self.download_location / self.dataset_name
        expected_csv_in_subfolder = extracted_subfolder / self.dataset_config['file_name']

        if expected_csv_in_subfolder.exists():
            print(f"Moving '{self.dataset_config['file_name']}' from subfolder {extracted_subfolder} to {self.download_location}")
            shutil.move(str(expected_csv_in_subfolder), str(self.csv_path))
        elif (self.download_location / self.dataset_config['file_name']).exists():
            print(f"'{self.dataset_config['file_name']}' found at {self.csv_path}")
        else:
            print(f"Warning: Could not find '{self.dataset_config['file_name']}' directly or in subfolder '{self.dataset_name}'.")

        # Clean up
        if extracted_subfolder.exists() and not os.listdir(extracted_subfolder):
            shutil.rmtree(extracted_subfolder)
        elif extracted_subfolder.exists():
            print(f"Subfolder {extracted_subfolder} still contains files, not removing.")

        if zip_path.exists():
            os.remove(zip_path)
        
        if self.csv_path.exists():
            print(f"{self.dataset_name} dataset downloaded and available at {self.csv_path}")
        else:
            raise FileNotFoundError(f"Failed to find or place {self.dataset_config['file_name']} at {self.csv_path}")

    def _process_data(self):
        """
        Processes the dataset into sequences for time series forecasting sequentially
        for train, validation, and test splits to reduce peak memory usage.
        Intermediate numpy arrays are deleted after tensor creation.
        """
        print(f"Processing {self.dataset_name} dataset...")
        
        df_raw = pd.read_csv(self.csv_path)
        
        # Handle data selection based on features mode
        if self.features == 'M':
            # For multivariate case, use all numeric columns except date
            if self.dataset_config['date_col'] is not None:
                cols_data = df_raw.columns.difference([self.dataset_config['date_col']])
                data_values = df_raw[cols_data].values
            else:
                data_values = df_raw.values
        else:  # 'S' for univariate
            data_values = df_raw[[self.target]].values
        
        data_values = data_values.astype(np.float32)
        
        # Calculate split points (70/10/20 split)
        df_len = len(data_values)
        num_train = int(df_len * 0.7)  # 70% for training
        num_test = int(df_len * 0.2)   # 20% for testing
        num_vali = df_len - num_train - num_test  # Remaining for validation
        
        # Define boundaries for each split
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        # Normalize data
        if self.scale:
            scaler = StandardScaler()
            # Fit scaler only on training data
            train_data = data_values[border1s[0]:border2s[0]]
            scaler.fit(train_data)
            data_normalized = scaler.transform(data_values)
        else:
            data_normalized = data_values

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
        del train_X_np, train_Y_np # Free up memory
        gc.collect() # Explicitly call garbage collector

        # Process Validation Data
        print("Creating validation sequences...")
        val_X_np, val_Y_np = create_sequences(
            data_normalized[border1s[1]:border2s[1]], 
            self.seq_len, 
            self.pred_len
        )
        val_X = torch.FloatTensor(val_X_np).transpose(1, 2)
        val_Y = torch.FloatTensor(val_Y_np).transpose(1, 2)
        del val_X_np, val_Y_np # Free up memory
        gc.collect() # Explicitly call garbage collector

        # Process Test Data
        print("Creating testing sequences...")
        test_X_np, test_Y_np = create_sequences(
            data_normalized[border1s[2]:border2s[2]], 
            self.seq_len, 
            self.pred_len
        )
        test_X = torch.FloatTensor(test_X_np).transpose(1, 2)
        test_Y = torch.FloatTensor(test_Y_np).transpose(1, 2)
        del test_X_np, test_Y_np # Free up memory
        gc.collect() # Explicitly call garbage collector

        # Delete the large normalized data array
        del data_normalized
        gc.collect()

        print(f"Processed data shapes:")
        print(f"Train X: {train_X.shape}, Train Y: {train_Y.shape}")
        print(f"Val X: {val_X.shape}, Val Y: {val_Y.shape}")
        print(f"Test X: {test_X.shape}, Test Y: {test_Y.shape}")
        
        return train_X, val_X, test_X, train_Y, val_Y, test_Y 