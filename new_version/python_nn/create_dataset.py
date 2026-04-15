import numpy as np
import pandas as pd
from itertools import product

# ============================================
# 1. Generate all 4-bit Gray code sequence (0 to 15)
# ============================================
def gray_code_4bit():
    """Generate 4-bit Gray code for numbers 0..15"""
    gray = []
    for i in range(16):
        # Standard Gray code formula: i ^ (i >> 1)
        gray_code_value = i ^ (i >> 1)
        gray.append(gray_code_value)
    return gray

# ============================================
# 2. Hamming distance function
# ============================================
def hamming_distance(a, b):
    """
    Calculate Hamming distance between two 4-bit numbers.
    Definition from textbook: number of positions where bits differ.
    """
    xor_result = a ^ b
    # Count number of 1 bits in XOR result
    return bin(xor_result).count('1')

# ============================================
# 3. Check if a sequence is normal (Gray code property)
# ============================================
def is_normal_sequence(sequence):
    """
    Returns True if every consecutive pair has Hamming distance = 1.
    Otherwise returns False (anomaly detected).
    """
    for i in range(len(sequence) - 1):
        if hamming_distance(sequence[i], sequence[i + 1]) != 1:
            return False
    return True

# ============================================
# 4. Generate a normal (non-anomalous) sequence
# ============================================
def generate_normal_sequence(sequence_length=10, gray_code_list=None):
    """
    Generate a normal sequence by taking consecutive numbers from Gray code.
    Gray code wraps around (after 15 comes 0).
    """
    if gray_code_list is None:
        gray_code_list = gray_code_4bit()
    
    # Random starting position in Gray code (0 to 15)
    start_idx = np.random.randint(0, len(gray_code_list))
    
    sequence = []
    for i in range(sequence_length):
        idx = (start_idx + i) % len(gray_code_list)
        sequence.append(gray_code_list[idx])
    
    return sequence

# ============================================
# 5. Generate an anomalous sequence (with one fault)
# ============================================
def generate_anomalous_sequence(sequence_length=10, gray_code_list=None):
    """
    Generate a normal sequence, then inject ONE fault at a random position.
    The fault replaces a number with a random value (0-15) that is NOT
    the correct next value according to Gray code.
    """
    if gray_code_list is None:
        gray_code_list = gray_code_4bit()
    
    # First generate a normal sequence
    sequence = generate_normal_sequence(sequence_length, gray_code_list)
    
    for i in range(5):
        # Choose random position to inject fault (0 to 9)
        fault_position = np.random.randint(0, sequence_length)
        
        # Choose random faulty value (0-15)
        faulty_value = np.random.randint(0, 16)
        
        # Ensure the faulty value is actually wrong:
        # For position 0: no previous check, just ensure it breaks at least one rule
        # For other positions: ensure Hamming distance with previous number is NOT 1
        if fault_position > 0:
            while hamming_distance(sequence[fault_position - 1], faulty_value) == 1:
                faulty_value = np.random.randint(0, 16)
        
        # Also check with next number if it's not the last position
        if fault_position < sequence_length - 1:
            while hamming_distance(faulty_value, sequence[fault_position + 1]) == 1:
                faulty_value = np.random.randint(0, 16)
        
        # Inject the fault
        sequence[fault_position] = faulty_value
    
    # Verify that sequence is now anomalous (safety check)
    # If by some chance it's still normal, regenerate
    if is_normal_sequence(sequence):
        # Recursive call (should be rare)
        return generate_anomalous_sequence(sequence_length, gray_code_list)
    
    return sequence

# ============================================
# 6. Create the full dataset
# ============================================
def create_dataset(num_samples=20000, sequence_length=10, balance=True):
    """
    Create balanced dataset with normal and anomalous sequences.
    
    Parameters:
    - num_samples: total number of sequences to generate
    - sequence_length: length of each sequence (default 10)
    - balance: if True, exactly half normal, half anomalous
    
    Returns:
    - X: numpy array of shape (num_samples, sequence_length) with values 0-15
    - y: numpy array of shape (num_samples,) with 0=normal, 1=anomalous
    """
    gray = gray_code_4bit()
    
    if balance:
        # Half normal, half anomalous
        samples_per_class = num_samples // 2
        
        # Generate normal sequences
        normal_sequences = []
        for _ in range(samples_per_class):
            seq = generate_normal_sequence(sequence_length, gray)
            normal_sequences.append(seq)
        
        # Generate anomalous sequences
        anomalous_sequences = []
        for _ in range(samples_per_class):
            seq = generate_anomalous_sequence(sequence_length, gray)
            anomalous_sequences.append(seq)
        
        # Combine
        X = np.vstack([normal_sequences, anomalous_sequences])
        y = np.array([0] * samples_per_class + [1] * samples_per_class)
        
    else:
        # Random mix (50% normal, 50% anomalous by probability)
        X = []
        y = []
        for _ in range(num_samples):
            if np.random.rand() < 0.5:
                seq = generate_normal_sequence(sequence_length, gray)
                X.append(seq)
                y.append(0)
            else:
                seq = generate_anomalous_sequence(sequence_length, gray)
                X.append(seq)
                y.append(1)
        X = np.array(X)
        y = np.array(y)
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

# ============================================
# 7. Split into train and test sets
# ============================================
def split_train_test(X, y, train_ratio=0.8):
    """
    Split dataset into training and testing sets.
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, y_train, X_test, y_test

# ============================================
# 8. Save to CSV files
# ============================================
def save_to_csv(X, y, filename):
    """
    Save dataset to CSV file.
    Columns: input_0, input_1, ..., input_9, label
    """
    num_samples = X.shape[0]
    sequence_length = X.shape[1]
    
    # Create column names
    columns = [f'input_{i}' for i in range(sequence_length)] + ['label']
    
    # Combine X and y
    data = np.column_stack([X, y])
    
    # Create DataFrame and save
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    
    print(f"Saved {num_samples} samples to {filename}")
    return df

# ============================================
# 9. Main execution
# ============================================
if __name__ == "__main__":
    # Parameters
    TOTAL_SAMPLES = 10000
    SEQUENCE_LENGTH = 10
    TRAIN_RATIO = 0.8
    
    print("=" * 60)
    print("GENERATING DATASET FOR ANOMALY DETECTION")
    print("=" * 60)
    print(f"Total samples: {TOTAL_SAMPLES}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Train ratio: {TRAIN_RATIO}")
    print()
    
    # Create dataset
    print("Creating dataset...")
    X, y = create_dataset(num_samples=TOTAL_SAMPLES, 
                          sequence_length=SEQUENCE_LENGTH, 
                          balance=True)
    
    # Verify balance
    normal_count = np.sum(y == 0)
    anomaly_count = np.sum(y == 1)
    print(f"Normal sequences: {normal_count} ({normal_count/TOTAL_SAMPLES*100:.1f}%)")
    print(f"Anomalous sequences: {anomaly_count} ({anomaly_count/TOTAL_SAMPLES*100:.1f}%)")
    print()
    
    # Split into train and test
    print("Splitting into train/test...")
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_ratio=TRAIN_RATIO)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Save to CSV
    print("Saving to CSV files...")
    train_df = save_to_csv(X_train, y_train, 'train.csv')
    test_df = save_to_csv(X_test, y_test, 'test.csv')
    
    # Display sample from each class
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    # Find one normal and one anomalous sequence in training set
    normal_idx = np.where(y_train == 0)[0][0]
    anomaly_idx = np.where(y_train == 1)[0][0]
    
    print("\n✅ NORMAL SEQUENCE (label = 0):")
    print(f"   {X_train[normal_idx].tolist()}")
    
    print("\n⚠️ ANOMALOUS SEQUENCE (label = 1):")
    print(f"   {X_train[anomaly_idx].tolist()}")
    
    # Verify the anomaly is real
    seq = X_train[anomaly_idx]
    print("\n   Verifying anomaly:")
    for i in range(len(seq)-1):
        dist = hamming_distance(seq[i], seq[i+1])
        status = "✗ ANOMALY" if dist != 1 else "✓"
        print(f"   Pair ({seq[i]} → {seq[i+1]}): Hamming distance = {dist} {status}")
    
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE!")
    print("Files created: train.csv, test.csv")
    print("=" * 60)