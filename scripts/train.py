import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import wandb
from pathlib import Path

from src.data.dataset import create_dataloaders
from src.models.mlp import TabularMLP

# --- Configuration ---
CONFIG = {
    "batch_size": 4096,
    "epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "embedding_dim": 16,
    "hidden_dims": [256, 128, 64],
    "dropout": 0.3,
    "val_split": 0.1,
    "project_name": "ieee-fraud-detection",
}

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        numerical = batch['numerical'].to(device)
        categorical = batch['categorical'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=(device.type == 'cuda')):
            logits = model(numerical, categorical).squeeze()
            loss = criterion(logits, target.squeeze())
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            target = batch['target'].to(device)
            
            logits = model(numerical, categorical).squeeze()
            loss = criterion(logits, target.squeeze())
            
            total_loss += loss.item()
            
            # Get probabilities for ROC-AUC
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(target.squeeze().cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    roc_auc = roc_auc_score(all_targets, all_preds)
    
    return avg_loss, roc_auc

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # 2. Initialize W&B
    wandb.init(project=CONFIG["project_name"], config=CONFIG)
    
    # 3. Load Data
    print("Loading data...")
    train_path = "data/processed/train_merged.parquet"
    train_loader, val_loader = create_dataloaders(
        train_path, 
        batch_size=CONFIG["batch_size"], 
        val_split=CONFIG["val_split"]
    )
    
    # 4. Initialize Model
    # We need to peek at the dataset to get cardinalities
    from src.data.dataset import FraudDetectionDataset
    temp_dataset = FraudDetectionDataset(train_path, is_train=True)
    cat_cardinalities = [len(temp_dataset.cat_mappings[col]) + 1 for col in temp_dataset.categorical_cols]
    
    model = TabularMLP(
        num_numerical_features=temp_dataset.numerical_data.shape[1],
        cat_cardinalities=cat_cardinalities,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dims=CONFIG["hidden_dims"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 5. Loss, Optimizer, Scheduler, and Scaler
    # pos_weight helps the model pay more attention to the minority class (fraud)
    pos_weight = torch.tensor([9.0]).to(device) # Approx 96.5% / 3.5%
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 6. Training Loop
    best_roc_auc = 0.0
    output_dir = Path("outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(CONFIG["epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_roc_auc = validate_epoch(model, val_loader, criterion, device)
        
        # Update Learning Rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_roc_auc,
            "learning_rate": current_lr
        })
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_roc_auc:.4f} | LR: {current_lr:.2e}")
        
        # Save Best Model
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  ↳ 🌟 New best ROC-AUC! Model saved.")
            
    print("\n" + "="*60)
    print(f"✅ TRAINING COMPLETE! Best Val ROC-AUC: {best_roc_auc:.4f}")
    print("="*60)
    wandb.finish()

if __name__ == "__main__":
    main()