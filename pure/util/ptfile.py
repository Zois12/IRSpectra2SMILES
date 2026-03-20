import torch
import pandas as pd

# 加载数据
loaded_data = torch.load('data/raw_processed_data.pt')

# 分别查看每个字段
print("=== Molecular Formula ===")
print(f"Length: {len(loaded_data['molecular_formula'])}")
print(f"First 5: {loaded_data['molecular_formula'][:5]}")

print("\n=== SMILES ===")
print(f"Length: {len(loaded_data['smiles'])}")
print(f"First 5: {loaded_data['smiles'][:5]}")

print("\n=== IR Spectra ===")
print(f"Length: {len(loaded_data['ir_spectra'])}")
print(f"Shape: {loaded_data['ir_spectra'].shape}")
print(f"Data type: {loaded_data['ir_spectra'].dtype}")
print(f"Min value: {loaded_data['ir_spectra'].min().item()}")
print(f"Max value: {loaded_data['ir_spectra'].max().item()}")
print(f"Mean: {loaded_data['ir_spectra'].mean().item()}")
print(f"Sample row (first 10 values): {loaded_data['ir_spectra'][0, :1]}")

# 如果你想看具体某个分子的光谱
print(f"\nFirst spectrum (first 20 values): {loaded_data['ir_spectra'][0, :20].tolist()}")