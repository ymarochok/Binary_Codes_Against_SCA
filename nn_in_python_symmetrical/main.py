import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

# dataset loading
class SensorCSV(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_loader = DataLoader(SensorCSV("train.csv"), batch_size=32, shuffle=True)
test_loader  = DataLoader(SensorCSV("test.csv"),  batch_size=32, shuffle=False)

# neural network model
# 10 inputs
# 6 neurons in hidden layer
# 1 neuron is ouptut

# neural network is predicting if there is anomaly in sequence of numbers
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6,1)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x

model = TinyNet()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

ALLOWED_VALS = torch.tensor([-7., -6., -5., -4., -3., -2., -1.,
                              1.,  2.,  3.,  4.,  5.,  6.,  7.])

# train for a few epochs
for epoch in range(15):
    for x,y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss={loss.item():.4f}")

def test_accuracy(model):
    correct,total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            pred = (model(x) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

float_acc = test_accuracy(model)
print("Float accuracy:", float_acc)

# Symmetric 4‑bit quantization parameters
QMIN = -7
QMAX = 7
ACT_QMAX = 7   # after ReLU, range [0,7]

def symmetric_scale(tensor):
    """Return scale = max(|tensor|) / 7 for symmetric int4 range [-7,7]."""
    max_abs = tensor.abs().max().item()
    return max_abs / 7.0

def activation_scale_after_relu(tensor):
    """For ReLU output, range [0, max_val], scale = max_val / 7."""
    max_val = tensor.max().item()
    return max_val / 7.0

# collect activation ranges
model.eval()
act1_list = []
with torch.no_grad():
    for x, _ in train_loader:
        h1 = model.relu(model.fc1(x))
        act1_list.append(h1)
all_act1 = torch.cat(act1_list)

# input scale (symmetric)
in_list = []
with torch.no_grad():
    for x, _ in train_loader:
        in_list.append(x)
all_in = torch.cat(in_list)
S_x = symmetric_scale(all_in)          # input scale
Z_x = 0                                # symmetric zero point

# activation1 scale
S_a1 = activation_scale_after_relu(all_act1)
Z_a1 = 0

# per‑channel symmetric weight quantization
ALLOWED_VALS = torch.tensor([-7., -6., -5., -4., -3., -2., -1.,
                              1.,  2.,  3.,  4.,  5.,  6.,  7.])

def quantize_weight_symmetric_per_channel_no_zero(w):
    out_c, in_c = w.shape
    w_q = torch.empty_like(w, dtype=torch.int8)
    scales = torch.empty(out_c)
    device = w.device
    allowed = ALLOWED_VALS.to(device)

    for i in range(out_c):
        scale = w[i].abs().max().item() / 7.0
        # normalised weights
        w_norm = w[i] / scale
        # find nearest allowed value for each weight element
        diff = w_norm.unsqueeze(-1) - allowed
        idx = torch.abs(diff).argmin(dim=-1)
        w_q[i] = allowed[idx].to(torch.int8)
        scales[i] = scale

    return w_q, scales

W1_q, S_w1 = quantize_weight_symmetric_per_channel_no_zero(model.fc1.weight.data)
W2_q, S_w2 = quantize_weight_symmetric_per_channel_no_zero(model.fc2.weight.data)

# bias quantization
bias1_scale = S_w1 * S_x
b1_q = torch.round(model.fc1.bias.data / bias1_scale).to(torch.int32)

bias2_scale = S_w2 * S_a1
b2_q = torch.round(model.fc2.bias.data / bias2_scale).to(torch.int32)

# compute integer multipliers M0 and shift n for each channel
def compute_multiplier_shift(S_prev, S_w, S_next):
    """
    Returns (M0_int, n) where:
        M = S_prev * S_w / S_next = 2^{-n} * (M0_int / 2^{31})
    and M0_int is in [2^30, 2^31 - 1].
    """
    M = S_prev * S_w / S_next   # may be > 1 or < 1
    # find n such that 0.5 <= M * 2^n < 1.0
    n = 0
    while M < 0.5:
        M *= 2
        n += 1
    while M >= 1.0:
        M /= 2
        n -= 1
    # now m in [0.5, 1.0)
    M0_int = int(round(M * (1 << 31)))
    # ensure it fits in int32
    M0_int = max(2**30, min(2**31 - 1, M0_int))
    return M0_int, n

# layer 1 multiplier
M0_1_list = []
n_1_list = []
for i in range(W1_q.shape[0]):  # iterate over output channels
    M0_i, n_i = compute_multiplier_shift(S_x, S_w1[i].item(), S_a1)
    M0_1_list.append(M0_i)
    n_1_list.append(n_i)

# layer 2 multiplier
with torch.no_grad():
    all_outputs = []
    for x, _ in train_loader:
        out = model(x)
        all_outputs.append(out)
    all_outputs = torch.cat(all_outputs)
    S_out = all_outputs.max().item() / 7.0 

S_out = 1.0 / 256.0

M0_2_list = []
n_2_list = []
for i in range(W2_q.shape[0]):  # only 1 output channel, but loop anyway
    M0_i, n_i = compute_multiplier_shift(S_a1, S_w2[i].item(), S_out)
    M0_2_list.append(M0_i)
    n_2_list.append(n_i)

# quantized forward pass
def quantized_forward_int_sim(x_float):
    # input quantization (inpust are already quantized and we can remove it but i leave it for future work)
    x_q = torch.round(x_float / S_x).clamp(QMIN, QMAX).to(torch.int32)  # Z=0

    # layer 1- > matrix multiplication
    acc1 = torch.matmul(x_q.to(torch.int32), W1_q.t().to(torch.int32))
    acc1 += b1_q.unsqueeze(0)
    
    # apply per-channel multiplier and shift
    h_q = torch.empty_like(acc1, dtype=torch.int32)
    for i in range(acc1.shape[1]):  # iterate over 6 channels
        temp = acc1[:, i].to(torch.int64) * M0_1_list[i]
        
        # rounding: add half before shift
        temp += (1 << (31 + n_1_list[i] - 1))
        h_q[:, i] = (temp >> (31 + n_1_list[i])).to(torch.int32)
    
    # ReLU
    h_q = torch.clamp(h_q, min=0, max=ACT_QMAX)

    # layer 2
    acc2 = torch.matmul(h_q.to(torch.int32), W2_q.t().to(torch.int32))
    acc2 += b2_q.unsqueeze(0)
    
    out_q = torch.empty_like(acc2, dtype=torch.int32)
    for i in range(acc2.shape[1]):  # iterate over 1 output channel
        temp = acc2[:, i].to(torch.int64) * M0_2_list[i]
        temp += (1 << (31 + n_2_list[i] - 1))
        out_q[:, i] = (temp >> (31 + n_2_list[i])).to(torch.int32)

    y_float = out_q.float() * S_out
    return torch.sigmoid(y_float)

def test_accuracy_quantized():
    correct,total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            pred = (quantized_forward_int_sim(x) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

quant_acc = test_accuracy_quantized()
print("Quantized 4-bit accuracy (integer sim):", quant_acc)

# generate C header with all integer parameters
def format_1d(lst):
    return ", ".join(str(x) for x in lst)

def format_2d(lst, dtype):
    rows = []
    for row in lst:
        rows.append("    {{ {} }}".format(", ".join(str(int(x)) for x in row)))
    return ",\n".join(rows)

with open("network_config.h", "w") as f:
    f.write(f"""#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 6
#define NET_L2 1

// ===== INPUT QUANTIZATION =====
static const float INPUT_SCALE = {S_x}f;   // for reference only (not used in integer code)
static const int INPUT_ZERO_POINT = {Z_x}; // 0 for symmetric

// ===== LAYER 1 =====
static const int8_t L1_W[NET_L1][NET_INPUTS] = {{
{format_2d(W1_q.tolist(), "int8_t")}
}};

static const int32_t L1_B[NET_L1] = {{ {format_1d(b1_q.tolist())} }};

// Multiplier (M0) and shift (n) per output channel
static const int32_t L1_M0[NET_L1] = {{ {format_1d(M0_1_list)} }};
static const int32_t L1_N[NET_L1] = {{ {format_1d(n_1_list)} }};

static const int ACT1_SCALE_INV = {int(round(1.0/S_a1))}; // optional, if needed

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {{
{format_2d(W2_q.tolist(), "int8_t")}
}};

static const int32_t L2_B[NET_L2] = {{ {format_1d(b2_q.tolist())} }};

static const int32_t L2_M0[NET_L2] = {{ {format_1d(M0_2_list)} }};
static const int32_t L2_N[NET_L2] = {{ {format_1d(n_2_list)} }};

// Output dequantization scale (for converting logit to float probability)
static const float OUTPUT_SCALE = {S_out}f;

#endif
""")

print("network_config.h generated successfully!")