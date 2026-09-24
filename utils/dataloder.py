import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for filename in os.listdir(root_dir):
            if filename.endswith('.csv'):
                try:
                    label_part = filename.split('%')[0].split('-')[-1]
                    label = float(label_part) / 100.0
                except (ValueError, IndexError) as e:
                    print(f"Fail to extract label of{filename} ：{e},skipped")
                    continue

                file_path = os.path.join(root_dir, filename)
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                    df = df.iloc[:10000, :6]
                    data_array = df.values
                    self.data.append(data_array)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Fail to read file of {filename} ：{e},skipped")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       data = self.data[idx]
       x = data
       y = self.labels[idx]
       return  x , y
