import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Custom pytorch dataset"""
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        label = row.label
        sample = row.sample
        if self.transform:
            sample = self.transform(sample)
        return sample, label