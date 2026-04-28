import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from save_params import *   # your export utility (keep if needed)

# ==========================================================
# Dataset (unchanged)
# ==========================================================
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


# ==========================================================
# Model (original, but you can increase capacity if needed)
# ==========================================================
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5,1)
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

# ==========================================================
# Training (float)
# ==========================================================
for epoch in range(15):
    for x,y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss={loss.item():.4f}")

# ==========================================================
# Float accuracy
# ==========================================================
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

# ==========================================================
# Quantization parameters (4-bit affine)
# ==========================================================
QMIN = 0
QMAX = 15

def calculate_scale_zero_point(rmin, rmax):
    """Standard affine scale & zero‑point from min/max."""
    scale = (rmax - rmin) / (QMAX - QMIN)
    zp = QMIN - rmin / scale
    zp = round(zp)
    zp = max(QMIN, min(QMAX, zp))
    return scale, int(zp)

def percentile_range(tensor, percentile=99.9):
    """Return (low, high) covering `percentile` of the data."""
    low = torch.quantile(tensor, (100-percentile)/100.0).item()
    high = torch.quantile(tensor, percentile/100.0).item()
    return low, high

# ==========================================================
# Calibration: collect activation ranges
# ==========================================================
model.eval()
act1_list = []
with torch.no_grad():
    for x, _ in train_loader:          # use training set for calibration
        h1 = model.relu(model.fc1(x))
        act1_list.append(h1)
all_act1 = torch.cat(act1_list)
a1_min, a1_max = percentile_range(all_act1, 99.9)   # robust min/max
S_a1, Z_a1 = calculate_scale_zero_point(a1_min, a1_max)

# Input scale (also from calibration)
in_list = []
with torch.no_grad():
    for x, _ in train_loader:
        in_list.append(x)
all_in = torch.cat(in_list)
in_min, in_max = percentile_range(all_in, 99.9)
S_x, Z_x = calculate_scale_zero_point(in_min, in_max)

# ==========================================================
# Per‑channel weight quantization
# ==========================================================
def quantize_weight_per_channel(w):
    """Quantize weight matrix (out_features × in_features) per output channel.
       Returns quantized tensor (int32), scales (1D), zero_points (1D)."""
    out_c, in_c = w.shape
    w_q = torch.empty_like(w, dtype=torch.int32)
    scales = torch.empty(out_c)
    zps = torch.empty(out_c, dtype=torch.int32)
    for i in range(out_c):
        rmin, rmax = percentile_range(w[i], 99.9)   # per‑channel range
        scale, zp = calculate_scale_zero_point(rmin, rmax)
        w_q[i] = torch.round(w[i] / scale + zp).clamp(QMIN, QMAX)
        scales[i] = scale
        zps[i] = zp
    return w_q, scales, zps

W1_q, S_w1, Z_w1 = quantize_weight_per_channel(model.fc1.weight.data)
W2_q, S_w2, Z_w2 = quantize_weight_per_channel(model.fc2.weight.data)

# ==========================================================
# Bias quantization (per‑channel, scale = weight_scale * input_scale)
# ==========================================================
# For first layer bias: scale = S_w1[i] * S_x
bias1_scale = S_w1 * S_x                     # element‑wise product (per channel)
b1_q = torch.round(model.fc1.bias.data / bias1_scale).to(torch.int32)

# For second layer bias: scale = S_w2[i] * S_a1
bias2_scale = S_w2 * S_a1
b2_q = torch.round(model.fc2.bias.data / bias2_scale).to(torch.int32)

# ==========================================================
# Quantized forward pass (using fixed activation scale)
# ==========================================================
def quantized_forward(x):
    # input quantization (fixed scale S_x, Z_x)
    # x_q = torch.round(x / S_x + Z_x).clamp(QMIN, QMAX).to(torch.int32)
    x_q = torch.tensor(x, dtype=torch.int32)

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
    h_q = torch.round(h / S_a1 + Z_a1).clamp(QMIN, QMAX).to(torch.int32)

    # ----- layer 2 -----
    h_centered = h_q - Z_a1
    w2_centered = W2_q - Z_w2.unsqueeze(1)
    acc2 = torch.matmul(h_centered, w2_centered.t())
    acc2 += b2_q.unsqueeze(0)
    y_float = acc2.float() * (S_w2 * S_a1).unsqueeze(0)

    # output activation
    y = torch.sigmoid(y_float)
    return y

# ==========================================================
# Quantized accuracy
# ==========================================================
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

# # ==========================================================
# # Optional: Quantization‑Aware Fine‑Tuning (5 epochs)
# # ==========================================================
# DO_FINETUNE = False   # set to True to enable fine‑tuning

# if DO_FINETUNE:
#     # ... (fine‑tuning code unchanged) ...
#     pass

# ==========================================================
# Export parameters to TXT (original, keep if needed)
# ==========================================================
save_txt("fc1_weights.txt", W1_q)                     # shape (5,10)
save_txt("fc1_bias.txt", b1_q)                        # shape (5,)
save_txt("fc2_weights.txt", W2_q)                     # shape (1,5)
save_txt("fc2_bias.txt", b2_q)                        # shape (1,)

save_txt("fc1_weight_scale.txt", S_w1)                # per‑channel (5,)
save_txt("fc1_weight_zero_point.txt", Z_w1)           # per‑channel (5,)
save_txt("fc2_weight_scale.txt", S_w2)                # (1,)
save_txt("fc2_weight_zero_point.txt", Z_w2)           # (1,)

save_txt("input_scale.txt", S_x)
save_txt("input_zero_point.txt", Z_x)

save_txt("activation1_scale.txt", S_a1)
save_txt("activation1_zero_point.txt", Z_a1)

print("TXT export finished (improved 4‑bit quantization)!")

# ==========================================================
# Generate network_config.h C header
# ==========================================================
def generate_c_header(filename, S_x, Z_x,
                      W1_q, S_w1, Z_w1, b1_q,
                      S_a1, Z_a1,
                      W2_q, S_w2, Z_w2, b2_q):
    """Write all quantized parameters to a C header file."""
    
    # Helper function to safely convert to Python scalar
    def to_py_scalar(val):
        if hasattr(val, 'item'):  # It's a tensor
            return val.item()
        return val  # Already a Python scalar
    
    # Helper function to convert tensor to list
    def to_list(val):
        if hasattr(val, 'tolist'):  # It's a tensor
            return val.tolist()
        return val  # Already a list
    
    with open(filename, 'w') as f:
        f.write("""#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <stdint.h>

#define NET_INPUTS 10
#define NET_L1 5
#define NET_L2 1

// ===== INPUT QUANTIZATION =====
static const float INPUT_SCALE = {}f;
static const int INPUT_ZERO_POINT = {};

// ===== LAYER 1 =====
static const int8_t L1_W[NET_L1][NET_INPUTS] = {{
{}
}};

static const int32_t L1_B[NET_L1] = {{ {}
 }};

static const float L1_W_SCALE[NET_L1] = {{ {}
 }};
static const int L1_W_ZP[NET_L1] = {{ {}
 }};

static const float ACT1_SCALE = {}f;
static const int ACT1_ZP = {};

// ===== LAYER 2 =====
static const int8_t L2_W[NET_L2][NET_L1] = {{
{}
}};

static const int32_t L2_B[NET_L2] = {{ {}
 }};

static const float L2_W_SCALE[NET_L2] = {{ {}
 }};
static const int L2_W_ZP[NET_L2] = {{ {}
 }};

#endif
""".format(
            repr(to_py_scalar(S_x)), int(to_py_scalar(Z_x)),
            format_2d_array(to_list(W1_q), "int8_t"),
            format_1d_array(to_list(b1_q)),
            format_1d_array(to_list(S_w1)),
            format_1d_array(to_list(Z_w1)),
            repr(to_py_scalar(S_a1)), int(to_py_scalar(Z_a1)),
            format_2d_array(to_list(W2_q), "int8_t"),
            format_1d_array(to_list(b2_q)),
            format_1d_array(to_list(S_w2)),
            format_1d_array(to_list(Z_w2))
        ))

def format_1d_array(lst):
    """Convert a 1D list to a C initializer string: num, num, ..."""
    return ", ".join(str(x) for x in lst)

def format_2d_array(lst, dtype):
    """Convert a 2D list to a C initializer string with nested braces."""
    rows = []
    for row in lst:
        rows.append("    {{ {} }}".format(", ".join(str(int(x)) for x in row)))
    return ",\n".join(rows)

# Call the generator with all tensors (converted safely)
generate_c_header("network_config.h",
                  S_x, Z_x,                    # These are already scalars
                  W1_q, S_w1, Z_w1, b1_q,
                  S_a1, Z_a1,                  # These are scalars too
                  W2_q, S_w2, Z_w2, b2_q)

print("network_config.h generated successfully!")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import copy

# # ==========================================================
# # Dataset
# # ==========================================================
# class SensorCSV(Dataset):
#     def __init__(self, filename):
#         df = pd.read_csv(filename)
#         data = df.iloc[:, :-1].values.astype(np.float32)
#         labels = df.iloc[:, -1].values.astype(np.float32).reshape(-1,1)

#         # Normalize [0,15] → [0,1]
#         data = data / 15.0

#         self.data = torch.tensor(data)
#         self.labels = torch.tensor(labels)

#     def __len__(self): return len(self.data)
#     def __getitem__(self, idx): return self.data[idx], self.labels[idx]


# train_loader = DataLoader(SensorCSV("train.csv"), batch_size=32, shuffle=True)
# test_loader  = DataLoader(SensorCSV("test.csv"),  batch_size=32, shuffle=False)

# # ==========================================================
# # Model (STRONGER)
# # ==========================================================
# class TinyNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# model = TinyNet()

# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.005)

# # ==========================================================
# # Accuracy
# # ==========================================================
# def test_accuracy(model, loader):
#     model.eval()
#     correct, total = 0, 0

#     with torch.no_grad():
#         for x, y in loader:
#             probs = torch.sigmoid(model(x))
#             pred = (probs > 0.5).float()
#             correct += (pred == y).sum().item()
#             total += y.size(0)

#     return correct / total


# # ==========================================================
# # Training
# # ==========================================================
# print("Training float model...")
# best_acc = 0
# for epoch in range(20):
#     model.train()
#     total_loss = 0

#     for x, y in train_loader:
#         optimizer.zero_grad()
#         logits = model(x)
#         loss = loss_fn(logits, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     # if (epoch+1) % 50 == 0:
#     print(f"Epoch {epoch+1} loss={total_loss/len(train_loader):.4f}")
#     acc = test_accuracy(model, test_loader)

#     if acc > best_acc:
#         best_acc = acc
#         best_model = copy.deepcopy(model)

# model = best_model

# float_acc = test_accuracy(model, test_loader)
# train_acc = test_accuracy(model, train_loader)

# print("\nFloat accuracy (train):", train_acc)
# print("Float accuracy (test):", float_acc)

# # ==========================================================
# # Quantization (Affine 4-bit)
# # ==========================================================
# QMIN, QMAX = 0, 15

# def calc_qparams(rmin, rmax):
#     scale = (rmax - rmin) / (QMAX - QMIN)
#     if scale < 1e-8:
#         scale = 1.0
#     zp = QMIN - rmin / scale
#     zp = int(round(zp))
#     zp = max(QMIN, min(QMAX, zp))
#     return scale, zp

# def quantize(t, S, Z):
#     return torch.round(t / S + Z).clamp(QMIN, QMAX).to(torch.int32)

# # ==========================================================
# # Calibration
# # ==========================================================
# model.eval()

# # Input
# all_x = torch.cat([x for x,_ in train_loader])
# xmin, xmax = all_x.min().item(), all_x.max().item()
# S_x, Z_x = calc_qparams(xmin, xmax)

# # Activation after fc1
# act_list = []
# with torch.no_grad():
#     for x,_ in train_loader:
#         act_list.append(torch.relu(model.fc1(x)))
# act = torch.cat(act_list)
# a1min, a1max = act.min().item(), act.max().item()
# S_a1, Z_a1 = calc_qparams(a1min, a1max)

# # Activation after fc2
# act_list = []
# with torch.no_grad():
#     for x,_ in train_loader:
#         act_list.append(torch.relu(model.fc2(torch.relu(model.fc1(x)))))
# act = torch.cat(act_list)
# a2min, a2max = act.min().item(), act.max().item()
# S_a2, Z_a2 = calc_qparams(a2min, a2max)

# # ==========================================================
# # Weight quantization (per-channel)
# # ==========================================================
# def quantize_weight_per_channel(w):
#     out_c = w.shape[0]
#     w_q = torch.empty_like(w, dtype=torch.int32)
#     S = torch.empty(out_c)
#     Z = torch.empty(out_c, dtype=torch.int32)

#     for i in range(out_c):
#         rmin, rmax = w[i].min().item(), w[i].max().item()
#         s, z = calc_qparams(rmin, rmax)
#         w_q[i] = quantize(w[i], s, z)
#         S[i], Z[i] = s, z

#     return w_q, S, Z

# W1_q, S_w1, Z_w1 = quantize_weight_per_channel(model.fc1.weight.data)
# W2_q, S_w2, Z_w2 = quantize_weight_per_channel(model.fc2.weight.data)
# # W3_q, S_w3, Z_w3 = quantize_weight_per_channel(model.fc3.weight.data)

# # Bias
# b1_q = torch.round(model.fc1.bias.data / (S_w1 * S_x)).to(torch.int32)
# b2_q = torch.round(model.fc2.bias.data / (S_w2 * S_a1)).to(torch.int32)
# # b3_q = torch.round(model.fc3.bias.data / (S_w3 * S_a2)).to(torch.int32)

# # ==========================================================
# # Quantized forward
# # ==========================================================
# # def q_forward(x):
# #     x_q = quantize(x, S_x, Z_x)

# #     # L1
# #     acc1 = torch.matmul(x_q - Z_x, (W1_q - Z_w1.unsqueeze(1)).t()) + b1_q
# #     h1 = torch.relu(acc1.float() * (S_w1 * S_x))
# #     h1_q = quantize(h1, S_a1, Z_a1)

# #     # L2
# #     acc2 = torch.matmul(h1_q - Z_a1, (W2_q - Z_w2.unsqueeze(1)).t()) + b2_q
# #     y = acc2.float() * (S_w2 * S_a1)

# #     return torch.sigmoid(y)
# def quantized_forward(x):
#     # input quantization (fixed scale S_x, Z_x)
#     x_q = torch.round(x / S_x + Z_x).clamp(QMIN, QMAX).to(torch.int32)

#     # ----- layer 1 -----
#     # compute (x_q - Z_x) and (W1_q - Z_w1) per channel
#     x_centered = x_q - Z_x                     # shape (batch, in_features)
#     w1_centered = W1_q - Z_w1.unsqueeze(1)     # shape (out_features, in_features)
#     # integer matrix multiplication -> acc1 shape (batch, out_features)
#     acc1 = torch.matmul(x_centered, w1_centered.t())   # int32
#     # add per‑channel bias (also int32)
#     acc1 += b1_q.unsqueeze(0)
#     # dequantize: multiply by per‑channel scale product (S_w1[i] * S_x)
#     h_float = acc1.float() * (S_w1 * S_x).unsqueeze(0)
#     # activation
#     h = torch.relu(h_float)

#     # ----- activation quantization (fixed S_a1, Z_a1) -----
#     h_q = torch.round(h / S_a1 + Z_a1).clamp(QMIN, QMAX).to(torch.int32)

#     # ----- layer 2 -----
#     h_centered = h_q - Z_a1
#     w2_centered = W2_q - Z_w2.unsqueeze(1)
#     acc2 = torch.matmul(h_centered, w2_centered.t())
#     acc2 += b2_q.unsqueeze(0)
#     y_float = acc2.float() * (S_w2 * S_a1).unsqueeze(0)

#     # output activation
#     y = torch.sigmoid(y_float)
#     return y

# # ==========================================================
# # Quant accuracy
# # ==========================================================
# def test_q():
#     correct, total = 0, 0
#     with torch.no_grad():
#         for x, y in test_loader:
#             pred = (quantized_forward(x) > 0.5).float()
#             correct += (pred == y).sum().item()
#             total += y.size(0)
#     return correct / total

# quant_acc = test_q()

# print("\nQuantized accuracy:", quant_acc)