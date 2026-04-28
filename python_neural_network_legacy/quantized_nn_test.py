import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SensorCSV(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        self.data = torch.torch.tensor(data, dtype=torch.float32)
        self.labels = torch.torch.tensor(labels, dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

test_loader  = DataLoader(SensorCSV("test.csv"),  batch_size=32, shuffle=False)

# def load_torch.tensor_from_file(filename, dtype=torch.float32, shape=None):
#     """Load a torch.tensor from a text file where numbers are separated by whitespace."""
#     data = np.loadtxt(filename)
#     torch.tensor = torch.torch.tensor(data, dtype=dtype)
#     if shape is not None:
#         torch.tensor = torch.tensor.reshape(shape)
#     return torch.tensor

# Load all quantized parameters
# W1_q = load_torch.tensor_from_file("fc1_weights.txt", dtype=torch.int32, shape=(5,10))
W1_q = torch.tensor([[13,  3,  6, 15,  7, 14,  5,  9,  5,  0],
        [15,  0, 13, 10,  7,  8, 13,  4, 15,  3],
        [ 5, 15,  2, 11,  0, 15,  2, 13,  1, 11],
        [ 0, 14, 14,  5, 12,  2, 12, 11,  1, 15],
        [ 3,  6,  0,  2, 15,  5,  1,  4,  5,  3]], dtype=torch.int32)

# b1_q = load_torch.tensor_from_file("fc1_bias.txt", dtype=torch.int32, shape=(5,))
b1_q = torch.tensor([-3,  0, -5,  5,  1], dtype=torch.int32)

# W2_q = load_torch.tensor_from_file("fc2_weights.txt", dtype=torch.int32, shape=(1,5))
W2_q = torch.tensor([[ 0, 14, 13, 15, 14]], dtype=torch.int32)

# b2_q = load_torch.tensor_from_file("fc2_bias.txt", dtype=torch.int32, shape=(1,))
b2_q = torch.tensor([-7], dtype=torch.int32)

# S_w1 = load_torch.tensor_from_file("fc1_weight_scale.txt", dtype=torch.float32, shape=(5,))
S_w1 = torch.tensor([0.0360, 0.1141, 0.0979, 0.0851, 0.1075])

# Z_w1 = load_torch.tensor_from_file("fc1_weight_zero_point.txt", dtype=torch.int32, shape=(5,))
Z_w1 = torch.tensor([11,  9,  8,  9,  5], dtype=torch.int32)

# S_w2 = load_torch.tensor_from_file("fc2_weight_scale.txt", dtype=torch.float32, shape=(1,))
S_w2 = torch.tensor([0.3405])

# Z_w2 = load_torch.tensor_from_file("fc2_weight_zero_point.txt", dtype=torch.int32, shape=(1,))
Z_w2 = torch.tensor([1], dtype=torch.int32)

# S_x = load_torch.tensor_from_file("input_scale.txt", dtype=torch.float32).item()
S_x = 1.0 
Z_x = 0
# Z_x = load_torch.tensor_from_file("input_zero_point.txt", dtype=torch.int32).item()
# S_a1 = load_torch.tensor_from_file("activation1_scale.txt", dtype=torch.float32).item()
S_a1 = 1.1105035146077473
# Z_a1 = load_torch.tensor_from_file("activation1_zero_point.txt", dtype=torch.int32).item()
Z_a1 = 0

Q_MIN = 0
Q_MAX = 15

def quantized_forward(x):
    # input quantization (fixed scale S_x, Z_x)
    # x_q = torch.round(x / S_x + Z_x).clamp(Q_MIN, Q_MAX).to(torch.int32)
    x_q = torch.tensor(x, dtype=torch.int32)
    # x_q = int(x)
    # ----- layer 1 -----
    # compute (x_q - Z_x) and (W1_q - Z_w1) per channel
    x_centered = x_q - Z_x                     # shape (batch, in_features)
    w1_centered = W1_q - Z_w1.unsqueeze(1)     # shape (out_features, in_features)
    # integer matrix multiplication -> acc1 shape (batch, out_features)
    acc1 = torch.matmul(x_centered, w1_centered.t())   # int32
    # add per‑channel bias (also int32)
    acc1 += b1_q.unsqueeze(0)
    # dequantize: multiply by per‑channel scale product (S_w1[i] * S_x)
    h_float = acc1.float() * (S_w1 * S_x).unsqueeze(0)
    # activation
    h = torch.relu(h_float)

    # ----- activation quantization (fixed S_a1, Z_a1) -----
    h_q = torch.round(h / S_a1 + Z_a1).clamp(Q_MIN, Q_MAX).to(torch.int32)

    # ----- layer 2 -----
    h_centered = h_q - Z_a1
    w2_centered = W2_q - Z_w2.unsqueeze(1)
    acc2 = torch.matmul(h_centered, w2_centered.t())
    acc2 += b2_q.unsqueeze(0)
    y_float = acc2.float() * (S_w2 * S_a1).unsqueeze(0)

    # output activation
    y = torch.sigmoid(y_float)
    return y

def test_accuracy_quantized():
    correct,total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            pred = (quantized_forward(x) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

quant_acc = test_accuracy_quantized()
print("Quantized 4-bit accuracy (improved):", quant_acc)
