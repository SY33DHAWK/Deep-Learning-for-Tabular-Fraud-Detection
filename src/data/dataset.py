import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

class FraudDetectionDataset(Dataset):
    """
    PyTorch Dataset for IEEE Fraud Detection with categorical embeddings.
    """
    
    def __init__(self, data_path: str, is_train: bool = True):
        """
        Args:
            data_path: Path to parquet file
            is_train: Whether this is training data (loads target variable)
        """
        self.data_path = Path(data_path)
        self.is_train = is_train
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pl.read_parquet(data_path)
        
        # Identify column types
        self.target_col = "isFraud"
        self.id_col = "TransactionID"
        
        # Categorical columns (these will use embeddings)
        self.categorical_cols = [
            'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
        ]
        
        # Add card columns as categorical (they have high cardinality)
        self.categorical_cols += [f'card{i}' for i in range(1, 7)]
        
        # Strictly filter numerical columns by Polars numeric dtypes
        numeric_dtypes = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
        self.numerical_cols = [
            col for col, dtype in self.df.schema.items()
            if dtype in numeric_dtypes and col not in [self.target_col, self.id_col]
        ]
        
        print(f"   ✓ Numerical features: {len(self.numerical_cols)}")
        print(f"   ✓ Categorical features: {len(self.categorical_cols)}")
        
        # Preprocess data
        self._preprocess()
        
        # Create categorical mappings (for embeddings)
        self.cat_mappings = self._create_categorical_mappings()
        
        # Convert to numpy for faster access
        self._convert_to_numpy()
        
    def _preprocess(self):
        """Handle missing values and basic preprocessing."""
        print("Preprocessing data...")
        
        # Fill missing numerical values with median
        for col in self.numerical_cols:
            median_val = self.df[col].median()
            if median_val is not None:
                self.df = self.df.with_columns(
                    pl.col(col).fill_null(median_val)
                )
            else:
                # If all NaN, fill with 0
                self.df = self.df.with_columns(
                    pl.col(col).fill_null(0.0)
                )
        
        # Fill missing categorical values with "UNKNOWN"
        for col in self.categorical_cols:
            self.df = self.df.with_columns(
                pl.col(col).fill_null("UNKNOWN")
            )
        
        print("   ✓ Preprocessing complete")
    
    def _create_categorical_mappings(self) -> Dict[str, Dict]:
        """
        Create mappings from categorical values to indices.
        Returns dict of {column_name: {value: index}}
        """
        print("Creating categorical mappings...")
        mappings = {}
        
        for col in self.categorical_cols:
            unique_vals = self.df[col].unique().sort()
            # Create mapping: value -> index (0 is reserved for padding/unknown)
            mappings[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            # Add 0 for unknown/padding
            mappings[col]["UNKNOWN"] = 0
            
            cat_count = len(mappings[col])
            print(f"   ✓ {col}: {cat_count} unique categories")
        
        return mappings
    
    def _convert_to_numpy(self):
        """Convert Polars DataFrame to numpy arrays for fast access."""
        print("Converting to numpy arrays...")
        
        # FIX: Cast all numerical columns to Float32 before converting to numpy.
        # This prevents Polars from returning an 'object' array when mixing 
        # Int64 and Float64 columns.
        self.numerical_data = self.df.select(self.numerical_cols).cast(pl.Float32).to_numpy()
        
        # Categorical features (as indices)
        self.categorical_data = np.zeros((len(self.df), len(self.categorical_cols)), dtype=np.int64)
        
        for idx, col in enumerate(self.categorical_cols):
            # Map values to indices, default to 0 (unknown)
            mapping = self.cat_mappings[col]
            self.categorical_data[:, idx] = [
                mapping.get(val, 0) for val in self.df[col].to_list()
            ]
        
        # Target variable
        if self.is_train:
            self.target = self.df[self.target_col].to_numpy()
        else:
            self.target = None
        
        print(f"   ✓ Numerical array shape: {self.numerical_data.shape}")
        print(f"   ✓ Categorical array shape: {self.categorical_data.shape}")
        if self.is_train:
            print(f"   ✓ Target array shape: {self.target.shape}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'numerical': torch.Tensor (num_numerical_features,)
                - 'categorical': torch.Tensor (num_categorical_features,)
                - 'target': torch.Tensor (1,) [only if is_train]
        """
        numerical = torch.FloatTensor(self.numerical_data[idx])
        categorical = torch.LongTensor(self.categorical_data[idx])
        
        if self.is_train:
            target = torch.FloatTensor([self.target[idx]])
            return {
                'numerical': numerical,
                'categorical': categorical,
                'target': target
            }
        else:
            return {
                'numerical': numerical,
                'categorical': categorical,
                'transaction_id': self.df[self.id_col][idx]
            }


def create_dataloaders(
    train_path: str,
    batch_size: int = 4096,
    val_split: float = 0.1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    """
    # Load full dataset
    full_dataset = FraudDetectionDataset(train_path, is_train=True)
    
    # Split into train/val
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(val_split * dataset_size))
    
    # Shuffle indices
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True    # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Train loader: {len(train_loader)} batches ({len(train_indices)} samples)")
    print(f"✓ Val loader: {len(val_loader)} batches ({len(val_indices)} samples)")
    
    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATASET & DATALOADER")
    print("=" * 60)
    
    train_path = "data/processed/train_merged.parquet"
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path,
        batch_size=4096,
        val_split=0.1,
        num_workers=0  # Set to 2 or 4 if you have multiple CPU cores
    )
    
    # Test one batch
    print("\n" + "=" * 60)
    print("TESTING ONE BATCH")
    print("=" * 60)
    
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Numerical shape: {batch['numerical'].shape}")
    print(f"Categorical shape: {batch['categorical'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    print(f"Target distribution: 0s={sum(batch['target'] == 0)}, 1s={sum(batch['target'] == 1)}")
    
    print("\n✅ Dataset and DataLoader working correctly!")