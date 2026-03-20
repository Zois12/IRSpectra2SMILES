import pandas as pd 
import torch 
import numpy as np
import pyarrow.parquet as pq
from typing import List
from scipy.interpolate import interp1d

def interpolate(spectra:List[float]):
    old_x = np.arange(400, 4000 if len(spectra) == 1800 else 3982, 2)
    new_x = np.arange(650, 3900, 2)
    interp = interp1d(old_x, spectra)
    return interp(new_x)

def fillEmpty(spectra:List[float]):
    spec_size_mask = [len(spectrum) if spectrum is not None else -1 for spectrum in spectra]
    max_spec_size = max(spec_size_mask) if max(spec_size_mask) != -1 else 500
    for i in range(len(spectra)):
        if spectra[i] is None:
            spectra[i] = [0] * 1625

def process_spectral_data(parquet_path):
    # 读取数据
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    
    print("读取数据成功")
    spectras = np.array(df['ir_spectra']).tolist()
    # 将不同长度的数据进行插值
    interpolate_spectra = []
    for i,spectra in enumerate(spectras):
        if spectra is None:
            spectra = [0.0] * 1625
        try:
            spectra = interpolate(spectra)
            interpolate_spectra.append(spectra)
        except:
            print(f"第{i}行数据处理失败")
            spectra = fillEmpty(spectra)
            interpolate_spectra.append(spectra)

    # spectra = spectra.apply(interpolate)
    print("插值完毕")
    interpolated_array = np.array(interpolate_spectra)
    # 计算均值和标准差
    flattened_values = interpolated_array.flatten()
    non_zero_values = flattened_values[flattened_values != 0]
    mean = non_zero_values.mean()
    std = non_zero_values.std()
    print(f"mean: {mean}, std: {std}")

    # # 填充空数据
    # spectra = spectra.apply(fillEmpty)
    # print("填充空数据完毕")

    # 标准化
    spectra_tensor = torch.Tensor(interpolated_array)
    standardised_spectra = (spectra_tensor - mean) / std
    print("标准化完毕")

    torch.save({
        'molecular_formula':df['molecular_formula'].tolist(),
        'smiles':df['smiles'].tolist(),
        'ir_spectra':standardised_spectra
    }, 'data/raw_processed_data.pt')
    print("已保存数据到data/raw_processed_data.pt")

datapath = 'data/pretrain_data.parquet'
process_spectral_data(datapath)