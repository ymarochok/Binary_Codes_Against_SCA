import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

sequence_len = 10

def save_dataset(ds, filename):
    # data = sequence_len feature columns, label = 1
    df = pd.DataFrame(torch.cat([ds.data, ds.labels], dim=1).numpy())
    df.columns = [f"x{i}" for i in range(sequence_len)] + ["label"]
    df.to_csv(filename, index=False)

class SensorDataset(Dataset):
    def __init__(self, n_samples=5000):
        self.data = []
        self.labels = []
        for _ in range(n_samples):
            seq = self.generate_seq()
            label = self.is_anomaly(seq)
            self.data.append(seq)
            self.labels.append([label])

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def generate_seq(self):
        # smooth pattern
        x = np.linspace(0, 3*np.pi, sequence_len)
        seq = np.sin(x) + np.random.normal(0, 0.05, size=sequence_len)

        # inject anomaly randomly (50% cases)
        if np.random.rand() < 0.5:
            idx = np.random.randint(0, sequence_len)
            seq[idx] += np.random.uniform(-3, 3)  # spike
            self._anomaly = 1
        else:
            self._anomaly = 0

        return seq

    def is_anomaly(self, seq):
        return self._anomaly
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# create dataset
train_ds = SensorDataset(5000)
test_ds = SensorDataset(1000)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

save_dataset(train_ds, "train_sensor.csv")
save_dataset(test_ds,  "test_sensor.csv")
