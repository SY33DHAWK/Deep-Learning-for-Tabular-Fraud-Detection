import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from pathlib import Path

from src.data.dataset import create_dataloaders, FraudDetectionDataset
from src.models.mlp import TabularMLP

# --- Configuration (Must match training config) ---
CONFIG = {
    "batch_size": 4096,
    "val_split": 0.1,
    "embedding_dim": 16,
    "hidden_dims": [256, 128, 64],
    "dropout": 0.3,
    "checkpoint_path": "outputs/checkpoints/best_model.pth"
}

def evaluate_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # 2. Load Data
    print("Loading validation data...")
    train_path = "data/processed/train_merged.parquet"
    
    # We need the dataset temporarily to get cardinalities
    temp_dataset = FraudDetectionDataset(train_path, is_train=True)
    cat_cardinalities = [len(temp_dataset.cat_mappings[col]) + 1 for col in temp_dataset.categorical_cols]
    
    # Create dataloaders (we only care about the val_loader here)
    _, val_loader = create_dataloaders(
        train_path, 
        batch_size=CONFIG["batch_size"], 
        val_split=CONFIG["val_split"]
    )
    
    # 3. Initialize and Load Model
    print("Loading best model weights...")
    model = TabularMLP(
        num_numerical_features=temp_dataset.numerical_data.shape[1],
        cat_cardinalities=cat_cardinalities,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dims=CONFIG["hidden_dims"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Model loaded from {CONFIG['checkpoint_path']}")
    
    # 4. Run Inference
    print("Running inference on validation set...")
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            numerical = batch['numerical'].to(device)
            categorical = batch['categorical'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            logits = model(numerical, categorical).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(target.squeeze().cpu().numpy())
            
    # 5. Calculate Metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    # Use 0.5 as default threshold for the report
    preds_binary = (torch.tensor(all_probs) > 0.5).numpy()
    print(classification_report(all_targets, preds_binary, target_names=["Not Fraud (0)", "Fraud (1)"]))
    
    # FIX: Use roc_auc_score instead of auc for y_true and y_score
    roc_auc = roc_auc_score(all_targets, all_probs)
    print(f"📊 Validation ROC-AUC: {roc_auc:.4f}")
    print("="*60)
    
    # 6. Generate Plots
    print("\nGenerating evaluation plots...")
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set professional plotting style
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['figure.dpi'] = 300
    
    # --- Plot 1: ROC Curve ---
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2E86C1', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()
    print("  ✓ Saved: outputs/evaluation/roc_curve.png")
    
    # --- Plot 2: Confusion Matrix ---
    # Find optimal threshold (Youden's J statistic)
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    preds_optimal = (torch.tensor(all_probs) > optimal_threshold).numpy()
    
    cm = confusion_matrix(all_targets, preds_optimal)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"],
                annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    print(f"  ✓ Saved: outputs/evaluation/confusion_matrix.png")
    
    print("\n✅ Evaluation Complete! Check the outputs/evaluation/ folder for your plots.")

if __name__ == "__main__":
    evaluate_model()