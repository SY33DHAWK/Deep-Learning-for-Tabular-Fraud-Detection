import polars as pl
from pathlib import Path

# Define paths
data_dir = Path("data/raw")
train_trans_path = data_dir / "train_transaction.csv"
train_id_path = data_dir / "train_identity.csv"

print("=" * 60)
print("LOADING TRAINING DATA...")
print("=" * 60)

# Load transaction data
print("\n1. Loading train_transaction.csv...")
train_trans = pl.scan_csv(train_trans_path).collect()
print(f"   ✓ Shape: {train_trans.shape}")
print(f"   ✓ Columns: {len(train_trans.columns)}")
print(f"   ✓ Memory usage: {train_trans.estimated_size('mb'):.2f} MB")

# Load identity data
print("\n2. Loading train_identity.csv...")
train_id = pl.scan_csv(train_id_path).collect()
print(f"   ✓ Shape: {train_id.shape}")
print(f"   ✓ Columns: {len(train_id.columns)}")
print(f"   ✓ Memory usage: {train_id.estimated_size('mb'):.2f} MB")

# Merge on TransactionID
print("\n3. Merging transaction and identity data...")
train_merged = train_trans.join(train_id, on="TransactionID", how="left")
print(f"   ✓ Merged shape: {train_merged.shape}")
print(f"   ✓ Total columns: {len(train_merged.columns)}")

# Show schema
print("\n" + "=" * 60)
print("DATA SCHEMA (First 20 columns):")
print("=" * 60)
for col in train_merged.columns[:20]:
    dtype = train_merged.schema[col]
    null_count = train_merged[col].null_count()
    null_pct = (null_count / len(train_merged)) * 100
    print(f"   {col:30s} | {str(dtype):15s} | Nulls: {null_pct:5.2f}%")

if len(train_merged.columns) > 20:
    print(f"   ... and {len(train_merged.columns) - 20} more columns")

# Target variable distribution
print("\n" + "=" * 60)
print("TARGET VARIABLE DISTRIBUTION:")
print("=" * 60)
if "isFraud" in train_merged.columns:
    fraud_counts = train_merged["isFraud"].value_counts().sort("isFraud")
    total = len(train_merged)
    for row in fraud_counts.iter_rows():
        label, count = row
        pct = (count / total) * 100
        print(f"   isFraud={label}: {count:,} ({pct:.2f}%)")
    print(f"\n   ⚠️  This is a HIGHLY IMBALANCED dataset!")
else:
    print("   ERROR: 'isFraud' column not found!")

# Save merged data to processed folder
print("\n" + "=" * 60)
print("SAVING PROCESSED DATA...")
print("=" * 60)
processed_dir = Path("data/processed")
processed_dir.mkdir(exist_ok=True)
output_path = processed_dir / "train_merged.parquet"

# Save as Parquet (faster and more efficient than CSV)
train_merged.write_parquet(output_path)
print(f"   ✓ Saved to: {output_path}")
print(f"   ✓ Parquet size: {output_path.stat().st_size / (1024*1024):.2f} MB")

print("\n" + "=" * 60)
print("✅ DATA INSPECTION COMPLETE!")
print("=" * 60)