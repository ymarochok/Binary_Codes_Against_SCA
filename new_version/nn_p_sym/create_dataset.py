import numpy as np
import pandas as pd

# ============================================
# Parameters
# ============================================
MIN_VAL = -7
MAX_VAL = 7
ALLOWED_VALUES = list(range(MIN_VAL, MAX_VAL + 1))  # [-7, -6, ..., 7]

def is_valid(value):
    """Check if value is within the allowed range."""
    return MIN_VAL <= value <= MAX_VAL

def abs_diff(a, b):
    """Absolute difference between two integers."""
    return abs(a - b)

# ============================================
# Normal Sequence Generator
# ============================================
def generate_normal_sequence(seq_len=10, max_step=4):
    """
    Generate a random walk where each consecutive difference <= max_step.
    All values stay within [MIN_VAL, MAX_VAL].
    """
    seq = []
    # Start from a random allowed value
    current = np.random.choice(ALLOWED_VALUES)
    seq.append(current)
    
    for _ in range(seq_len - 1):
        # Possible next values: within max_step and within bounds
        candidates = [v for v in ALLOWED_VALUES 
                      if abs_diff(v, current) <= max_step]
        # If no candidate (should not happen for max_step>=1), fallback to current
        if not candidates:
            candidates = [current]
        current = np.random.choice(candidates)
        seq.append(current)
    return seq

# ============================================
# Anomalous Sequence Generator
# ============================================
def generate_anomalous_sequence(seq_len=10, max_step=4, 
                                min_anomaly_diff=7, 
                                num_anomalies=(2, 4)):
    """
    Start with a normal sequence, then replace `k` positions (1 to 3)
    with values that create a large jump (>= min_anomaly_diff) from the
    previous element. The jump is forced to be at least min_anomaly_diff.
    """
    # Generate a normal base sequence
    seq = generate_normal_sequence(seq_len, max_step)
    
    # How many anomalies to inject
    k = np.random.randint(num_anomalies[0], num_anomalies[1] + 1)
    
    # To avoid modifying the same index twice
    positions = set()
    while len(positions) < k:
        pos = np.random.randint(1, seq_len)  # cannot be first (no predecessor)
        positions.add(pos)
    
    for pos in positions:
        prev_val = seq[pos - 1]
        # Choose a new value such that |new - prev| >= min_anomaly_diff
        # and new is within allowed range.
        candidates = [v for v in ALLOWED_VALUES 
                      if abs_diff(v, prev_val) >= min_anomaly_diff]
        # If no candidate (possible only if range is too small), skip
        if not candidates:
            continue
        new_val = np.random.choice(candidates)
        seq[pos] = new_val
    
    return seq

# ============================================
# Dataset Creation and Saving
# ============================================
def create_and_save_dataset(total_samples=10000, train_ratio=0.8, 
                            seq_len=10, max_step=4, min_anomaly_diff=7,
                            random_seed=42):
    """
    Creates balanced dataset (50% normal, 50% anomalous) and saves train/test CSV.
    """
    np.random.seed(random_seed)
    
    half = total_samples // 2
    
    # Generate normal and anomalous sequences
    X_norm = [generate_normal_sequence(seq_len, max_step) for _ in range(half)]
    X_anom = [generate_anomalous_sequence(seq_len, max_step, min_anomaly_diff) 
              for _ in range(half)]
    
    X = np.vstack([X_norm, X_anom])
    y = np.array([0] * half + [1] * half)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Train/test split
    split = int(total_samples * train_ratio)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # Save to CSV
    for name, (data_X, data_y) in [('train.csv', (X_train, y_train)),
                                   ('test.csv', (X_test, y_test))]:
        cols = [f'input_{i}' for i in range(seq_len)] + ['label']
        df = pd.DataFrame(np.column_stack([data_X, data_y]), columns=cols)
        df.to_csv(name, index=False)
        print(f"Saved {name} | Shape: {df.shape} | Labels: 0={sum(data_y==0)}, 1={sum(data_y==1)}")
    
    # Quick sanity check: compute average max diff in normal vs anomalous
    def avg_max_diff(seqs):
        diffs = []
        for seq in seqs:
            max_diff = max(abs(seq[i] - seq[i-1]) for i in range(1, len(seq)))
            diffs.append(max_diff)
        return np.mean(diffs)
    
    print("\nSanity check (average maximum consecutive difference):")
    print(f"  Normal sequences: {avg_max_diff(X_norm):.2f}")
    print(f"  Anomalous sequences: {avg_max_diff(X_anom):.2f}")
    print(f"  (Anomaly threshold = {min_anomaly_diff})")

if __name__ == "__main__":
    create_and_save_dataset()