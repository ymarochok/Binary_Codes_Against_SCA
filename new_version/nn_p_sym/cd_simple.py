import numpy as np
import pandas as pd

# =========================
# 1. Sequence Generators
# =========================
def generate_normal_sequence(length):
    """Generates a normal sequence within a tight safe zone."""
    # choose center in safe zone
    center = np.random.randint(-3, 4)  # avoid edges

    seq = []
    for _ in range(length):
        val = center + np.random.randint(-2, 3)  # small variation
        val = int(np.clip(val, -7, 7))
        seq.append(val)

    return seq

def generate_anomalous_sequence(length):
    """Generates a normal sequence and injects extreme spikes."""
    seq = generate_normal_sequence(length)

    # inject 1–2 spikes
    num_spikes = np.random.randint(1, 3)

    for _ in range(num_spikes):
        pos = np.random.randint(0, length)

        # strong spike
        spike = np.random.choice([-7, -6, 6, 7])
        seq[pos] = spike

    return seq

def is_anomalous(seq):
    """Helper to verify if a sequence spread breaks the normal limit."""
    return (max(seq) - min(seq)) > 4

# =========================
# 2. Dataset Generator
# =========================
def create_dataset(num_samples=10000, sequence_length=10):
    """Creates a balanced dataset of normal and anomalous sequences."""
    X = []
    y = []

    # Using a while loop ensures we don't fall short of num_samples 
    # if the 'continue' statements are triggered.
    while len(X) < num_samples:
        if np.random.rand() < 0.5:
            # NORMAL
            seq = generate_normal_sequence(sequence_length)

            # safety check: if it accidentally generated an anomaly, throw it out
            if is_anomalous(seq):
                continue

            X.append(seq)
            y.append(0)

        else:
            # ANOMALY
            seq = generate_anomalous_sequence(sequence_length)

            # ensure it's really anomalous, otherwise throw it out
            if not is_anomalous(seq):
                continue

            X.append(seq)
            y.append(1)

    X = np.array(X)
    y = np.array(y)

    # shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ============================================
# 3. Split and Save logic
# ============================================
def split_train_test(X, y, train_ratio=0.8):
    """Splits data into training and testing sets."""
    split_idx = int(len(X) * train_ratio)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def save_to_csv(X, y, filename):
    """Saves dataset to a CSV file using pandas."""
    seq_len = X.shape[1]
    columns = [f'input_{i}' for i in range(seq_len)] + ['label']
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Saved {len(X)} samples to {filename}")
    return df

# ============================================
# 4. Main Execution
# ============================================
if __name__ == "__main__":
    TOTAL_SAMPLES = 10000
    SEQUENCE_LENGTH = 10
    TRAIN_RATIO = 0.8

    print("=" * 60)
    print("GENERATING SPIKE-ANOMALY DATASET (-7..7)")
    print("=" * 60)
    print(f"Total samples: {TOTAL_SAMPLES}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Train ratio: {TRAIN_RATIO}\n")

    print("Creating dataset...")
    X, y = create_dataset(num_samples=TOTAL_SAMPLES,
                          sequence_length=SEQUENCE_LENGTH)

    normal_count = np.sum(y == 0)
    anomaly_count = np.sum(y == 1)
    print(f"Normal sequences: {normal_count} ({normal_count/TOTAL_SAMPLES*100:.1f}%)")
    print(f"Anomalous sequences: {anomaly_count} ({anomaly_count/TOTAL_SAMPLES*100:.1f}%)\n")

    print("Splitting into train/test...")
    X_train, y_train, X_test, y_test = split_train_test(X, y, TRAIN_RATIO)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")

    print("Saving to CSV files...")
    train_df = save_to_csv(X_train, y_train, 'train.csv')
    test_df = save_to_csv(X_test, y_test, 'test.csv')

    # Display samples
    print("\n" + "=" * 60)
    print("SAMPLE DATA VERIFICATION")
    print("=" * 60)

    normal_idx = np.where(y_train == 0)[0][0]
    anomaly_idx = np.where(y_train == 1)[0][0]

    print("\n✅ NORMAL SEQUENCE (label = 0):")
    print(f"   {X_train[normal_idx].tolist()}")
    print(f"   Spread (Max - Min): {max(X_train[normal_idx]) - min(X_train[normal_idx])} (must be <= 4)")

    print("\n⚠️ ANOMALOUS SEQUENCE (label = 1):")
    print(f"   {X_train[anomaly_idx].tolist()}")
    print(f"   Spread (Max - Min): {max(X_train[anomaly_idx]) - min(X_train[anomaly_idx])} (must be > 4)")

    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE!")
    print("Files created: train.csv, test.csv")
    print("=" * 60)