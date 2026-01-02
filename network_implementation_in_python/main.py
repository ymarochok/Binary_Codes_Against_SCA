import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from save_params import *


# Class which will contain whole dataset from sensors with the aim of future 
# network training and evaluation
class SensorCSV(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename)
        data = df.iloc[:, :-1].values   # all feature columns
        labels = df.iloc[:, -1].values  # last column is label

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Loading datasets from the files
train_loader = DataLoader(SensorCSV("train_sensor.csv"), batch_size=32, shuffle=True)
test_loader  = DataLoader(SensorCSV("test_sensor.csv"),  batch_size=32, shuffle=False)


import torch.nn as nn
import torch.optim as optim

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc2(self.relu(self.fc1(x))))

model = TinyNet()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(15):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")


def test_accuracy(model):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = (model(x) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

print("Float accuracy:", test_accuracy(model))


import numpy as np

def quantize_4bit(tensor):
    qmin, qmax = -8, 7
    max_val = tensor.abs().max().item() + 1e-12
    scale = max_val / qmax
    q = torch.round(tensor / scale).clamp(qmin, qmax).to(torch.int8)
    return q, scale


def quantized_forward(x, model_params):
    # unpack params
    W1_q, b1_q, s1W, s1B, W2_q, b2_q, s2W, s2B = model_params

    # Layer 1
    h = (x @ (W1_q * s1W).float().t()) + (b1_q * s1B).float()
    h = torch.relu(h)

    # Layer 2
    y = (h @ (W2_q * s2W).float().t()) + (b2_q * s2B).float()
    y = torch.sigmoid(y)
    return y


# Quantize all weights & biases
W1_q, s1W = quantize_4bit(model.fc1.weight.data)
b1_q, s1B = quantize_4bit(model.fc1.bias.data)
W2_q, s2W = quantize_4bit(model.fc2.weight.data)
b2_q, s2B = quantize_4bit(model.fc2.bias.data)

params_q = (W1_q, b1_q, s1W, s1B, W2_q, b2_q, s2W, s2B)

# Test quantized model
def test_accuracy_quantized(params):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = (quantized_forward(x, params) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

print("Quantized 4-bit accuracy:", test_accuracy_quantized(params_q))

# Export weights & biases ===
save_txt("fc1_weights.txt", W1_q)
save_txt("fc1_bias.txt", b1_q)

save_txt("fc2_weights.txt", W2_q)
save_txt("fc2_bias.txt", b2_q)

#  Export scales ===
save_txt("fc1_weight_scale.txt", s1W)
save_txt("fc1_bias_scale.txt", s1B)
save_txt("fc2_weight_scale.txt", s2W)
save_txt("fc2_bias_scale.txt", s2B)

print("TXT export finished!")
