import numpy as np
import pandas as pd

# ============================================
# 1. FIXED: Strict -7 to 7 Logic
# ============================================
def gray_code_strict_range():
    """
    Generates Gray code values and filters them to ONLY 
    allow the 15 values between -7 and 7.
    """
    valid_values = []
    for i in range(16):
        gray_val = (i ^ (i >> 1)) - 7
        # Strict filter: remove 8 or anything outside our target range
        if -7 <= gray_val <= 7:
            valid_values.append(gray_val)
    return valid_values

def hamming_distance(a, b):
    """Calculates bit-diff using the original 4-bit unsigned space"""
    u_a, u_b = int(a + 7), int(b + 7)
    return bin(u_a ^ u_b).count('1')

# ============================================
# 2. Sequence Generation
# ============================================
def generate_normal_sequence(sequence_length=10, valid_list=None):
    if valid_list is None: valid_list = gray_code_strict_range()
    
    start_idx = np.random.randint(0, len(valid_list))
    sequence = []
    for i in range(sequence_length):
        # Use modulo len(valid_list) so it only loops through the 15 allowed values
        idx = (start_idx + i) % len(valid_list)
        sequence.append(valid_list[idx])
    return sequence

def generate_anomalous_sequence(sequence_length=10, valid_list=None):
    if valid_list is None: valid_list = gray_code_strict_range()
    sequence = generate_normal_sequence(sequence_length, valid_list)
    
    # Inject faults using ONLY the allowed range
    for _ in range(np.random.randint(1, 3)):
        idx = np.random.randint(0, sequence_length)
        # Random choice from our strict list ensures no '8' can ever be picked
        faulty_val = np.random.choice(valid_list)
        
        # Ensure it's actually an anomaly
        if idx > 0:
            while hamming_distance(sequence[idx-1], faulty_val) == 1:
                faulty_val = np.random.choice(valid_list)
        
        sequence[idx] = faulty_val
    return sequence

# ============================================
# 3. Create and Save
# ============================================
def create_and_save_dataset(total_samples=10000, train_ratio=0.8, seq_len=10):
    valid_list = gray_code_strict_range()
    half = total_samples // 2
    
    X_norm = [generate_normal_sequence(seq_len, valid_list) for _ in range(half)]
    X_anom = [generate_anomalous_sequence(seq_len, valid_list) for _ in range(half)]
    
    X = np.vstack([X_norm, X_anom])
    y = np.array([0]*half + [1]*half)
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    split = int(total_samples * train_ratio)
    
    # Save Files
    for name, (data_X, data_y) in [('train.csv', (X[:split], y[:split])), 
                                   ('test.csv', (X[split:], y[split:]))]:
        cols = [f'input_{i}' for i in range(seq_len)] + ['label']
        df = pd.DataFrame(np.column_stack([data_X, data_y]), columns=cols)
        df.to_csv(name, index=False)
        print(f"Saved {name} | Range: {df.iloc[:, :-1].min().min()} to {df.iloc[:, :-1].max().max()}")

if __name__ == "__main__":
    create_and_save_dataset()