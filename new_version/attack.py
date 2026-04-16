import matplotlib.pyplot as plt
import numpy as np
import struct
import time

# ------------------------------------------------------------
# 1. Configuration – adjust these to your device/traces
# ------------------------------------------------------------
ALLOWED_WEIGHTS = np.array([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
INPUT_SCALE = 1.0          # ← your actual input quantisation scale
NUM_CANDIDATES = len(ALLOWED_WEIGHTS)

# Time window – you MUST adjust these after looking at the averaged trace
START_SAMPLE = 5100         # ← change after visual inspection
END_SAMPLE   = 7500        # ← change after visual inspection

# Attack parameters
NUM_TRACES = 2000
WEIGHTS_PER_LAYER = 10      # layer 1: 10 input weights

# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------
def float_to_binary_str(f):
    [packed] = struct.unpack('!I', struct.pack('!f', f))
    return f"{packed:032b}"

def HW_float32(f):
    return float_to_binary_str(f).count('1')

def HW_int32(x):
    """Hamming weight of a 32‑bit signed integer (full 32 bits)."""
    return (x & 0xFFFFFFFF).bit_count()

def fast_corr(x, y):
    """Stable, fast Pearson correlation (replaces np.corrcoef)."""
    x = x - np.mean(x)
    y = y - np.mean(y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return np.dot(x, y) / (norm_x * norm_y)

# ------------------------------------------------------------
# 3. Leakage model – Hamming distance of accumulator transition
# ------------------------------------------------------------
def get_hypothetical_leakages_accumulator(num_traces, inputs_q, weight_position,
                                          recovered_weights_prefix):
    """
    Hamming distance model:
        leakage = HW( prev_acc XOR (prev_acc + w * x) )
    where prev_acc is the accumulator value before this MAC, computed from
    previously recovered weights and the inputs.

    For the first weight (position 0), prev_acc = 0.
    """
    leakages = np.zeros((NUM_CANDIDATES, num_traces))

    # For each candidate weight
    for cand_idx, w in enumerate(ALLOWED_WEIGHTS):
        for trace_idx in range(num_traces):
            # Compute previous accumulator value (sum over already attacked weights)
            prev_acc = 0
            for prev_pos, prev_w in enumerate(recovered_weights_prefix):
                prev_acc += int(prev_w) * int(inputs_q[trace_idx, prev_pos])

            # Current multiplication result (scaled if needed)
            mult = int(w) * int(inputs_q[trace_idx, weight_position]) * INPUT_SCALE
            new_acc = prev_acc + mult

            # Hamming distance between old and new accumulator
            leakage = HW_int32(prev_acc ^ new_acc)
            leakages[cand_idx, trace_idx] = leakage

    return leakages

# ------------------------------------------------------------
# 4. CPA attack using the fast correlation
# ------------------------------------------------------------
def cpa_attack(num_traces, inputs_q, start, end, trace_arr, weight_position,
               recovered_prefix):
    """
    Perform CPA for one weight position using the accumulator HD model.
    """
    total_samples = end - start
    r_abs = np.zeros((NUM_CANDIDATES, total_samples))

    # Hypothetical leakages for this weight position
    hypo = get_hypothetical_leakages_accumulator(num_traces, inputs_q,
                                                 weight_position, recovered_prefix)

    for t in range(total_samples):
        if t % 100 == 0:
            print(f"  CPA at sample {start + t}")
        trace_col = trace_arr[:, start + t]
        for cand_idx in range(NUM_CANDIDATES):
            corr = fast_corr(hypo[cand_idx], trace_col)
            r_abs[cand_idx, t] = abs(corr)

    return r_abs

# ------------------------------------------------------------
# 5. Trace & input loading (unchanged, but keep it)
# ------------------------------------------------------------
def load_traces(num_of_traces, folder_name):
    trace_waves_arr = []
    inputs_arr = []

    for i in range(num_of_traces):
        with open(folder_name + '/trace_' + str(i) + '.txt') as f:
            line = f.read().strip()
            samples = [float(x) for x in line.split()]
            trace_waves_arr.append(samples)

    with open(folder_name + '/inputs.txt') as f:
        lines = f.read().splitlines()
        for line in lines:
            inputs = [int(x) for x in line.strip().split()]
            inputs_arr.append(inputs)

    trace_waves_arr = np.array(trace_waves_arr, dtype='f')
    inputs_arr = np.array(inputs_arr, dtype=np.int8)
    print("Loaded traces shape:", trace_waves_arr.shape)
    print("Loaded inputs shape:", inputs_arr.shape)
    return trace_waves_arr, inputs_arr

# ------------------------------------------------------------
# 6. Visualise average trace – to help you pick the correct window
# ------------------------------------------------------------
def plot_average_trace(traces, start, end):
    avg = np.mean(traces, axis=0)
    plt.figure(figsize=(12, 4))
    plt.plot(avg, color='blue', alpha=0.7)
    plt.axvspan(start, end, color='red', alpha=0.3, label='Current window')
    plt.xlabel('Sample point')
    plt.ylabel('Amplitude')
    plt.title('Average trace – adjust START_SAMPLE / END_SAMPLE')
    plt.legend()
    plt.show()

# ------------------------------------------------------------
# 7. Main attack – sequential recovery of all weights in the layer
# ------------------------------------------------------------
if __name__ == "__main__":
    # Load traces
    traces, inputs_q = load_traces(NUM_TRACES, folder_name="unprotected")

    # ---- Optional: visualise to choose correct window ----
    plot_average_trace(traces, START_SAMPLE, END_SAMPLE)
    input("Press Enter after adjusting START_SAMPLE and END_SAMPLE in the code...")

    total_samples = END_SAMPLE - START_SAMPLE

    # Recover weights one by one (sequential attack)
    recovered_weights = []   # will store recovered int8 values
    for pos in range(WEIGHTS_PER_LAYER):
        print(f"\n--- Attacking weight position {pos} ---")
        # Pass already recovered weights as prefix
        r = cpa_attack(NUM_TRACES, inputs_q, START_SAMPLE, END_SAMPLE,
                       traces, pos, recovered_weights)

        # Best candidate
        best_idx = np.unravel_index(np.argmax(r), r.shape)[0]
        recovered_w = ALLOWED_WEIGHTS[best_idx]
        recovered_weights.append(recovered_w)
        print(f"Recovered weight {pos} = {recovered_w}")

    print("\n✅ All weights recovered:")
    print(recovered_weights)