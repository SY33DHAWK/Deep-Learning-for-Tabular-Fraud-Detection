import torch
import torch.nn as nn
from typing import List

class TabularMLP(nn.Module):
    """
    A robust Multi-Layer Perceptron designed for mixed tabular data.
    Uses embeddings for categorical variables and skip connections for stability.
    """
    
    def __init__(
        self,
        num_numerical_features: int,
        cat_cardinalities: List[int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3
    ):
        super(TabularMLP, self).__init__()
        
        self.num_numerical_features = num_numerical_features
        self.num_categorical_features = len(cat_cardinalities)
        self.embedding_dim = embedding_dim
        
        # 1. Categorical Embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dim)
            for cardinality in cat_cardinalities
        ])
        
        # Calculate the total input size for the MLP
        input_dim = num_numerical_features + (self.num_categorical_features * embedding_dim)
        
        # 2. MLP Layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
        # 3. Output Layer (Binary Classification -> 1 logit)
        self.output_layer = nn.Linear(current_dim, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        # 1. Process Categorical Features
        cat_embeds = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            cat_embeds.append(emb_layer(categorical[:, i]))
            
        cat_embeds = torch.stack(cat_embeds, dim=1)
        cat_embeds = cat_embeds.view(cat_embeds.size(0), -1)
        
        # 2. Concatenate Numerical and Categorical features
        x = torch.cat([numerical, cat_embeds], dim=1)
        
        # 3. Pass through MLP
        x = self.mlp(x)
        
        # 4. Output Layer
        logits = self.output_layer(x)
        
        return logits


# --- TEST THE MODEL ---
if __name__ == "__main__":
    from src.data.dataset import FraudDetectionDataset, create_dataloaders

    print("=" * 60)
    print("TESTING TABULAR MLP MODEL")
    print("=" * 60)
    
    # Load dataset to get actual dimensions
    train_path = "data/processed/train_merged.parquet"
    dataset = FraudDetectionDataset(train_path, is_train=True)
    
    # Get cardinalities (number of unique categories per categorical column)
    # FIX: Added +1 as a safety buffer to prevent off-by-one IndexError in embeddings
    cat_cardinalities = [len(dataset.cat_mappings[col]) + 1 for col in dataset.categorical_cols]
    
    print(f"\nModel Configuration:")
    print(f"   ✓ Numerical features: {dataset.numerical_data.shape[1]}")
    print(f"   ✓ Categorical features: {len(cat_cardinalities)}")
    
    # Initialize the model
    model = TabularMLP(
        num_numerical_features=dataset.numerical_data.shape[1],
        cat_cardinalities=cat_cardinalities,
        embedding_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    # Test a forward pass
    print("\nTesting Forward Pass...")
    train_loader, _ = create_dataloaders(train_path, batch_size=64, val_split=0.1)
    batch = next(iter(train_loader))
    
    numerical = batch['numerical']
    categorical = batch['categorical']
    target = batch['target']
    
    print(f"   Input Numerical shape: {numerical.shape}")
    print(f"   Input Categorical shape: {categorical.shape}")
    
    # Forward pass
    logits = model(numerical, categorical)
    print(f"   Output Logits shape: {logits.shape}")
    
    # Calculate dummy loss to ensure gradients flow
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, target)
    print(f"   Dummy Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("   ✓ Gradients calculated successfully!")
    
    print("\n✅ Tabular MLP Model working correctly!")