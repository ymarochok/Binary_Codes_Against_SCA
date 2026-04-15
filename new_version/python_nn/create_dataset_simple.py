import numpy as np
import pandas as pd

def generate_normal_sequence(length=10):
    # All numbers 0-7
    return np.random.randint(0, 8, size=length)

def generate_anomalous_sequence(length=10):
    # Start normal, then inject one number >=8 at random position
    seq = generate_normal_sequence(length)
    pos = np.random.randint(0, length)
    seq[pos] = np.random.randint(8, 16)
    return seq

def create_dataset(num_samples=10000, length=10):
    X = []
    y = []
    for _ in range(num_samples // 2):
        X.append(generate_normal_sequence(length))
        y.append(0)
    for _ in range(num_samples // 2):
        X.append(generate_anomalous_sequence(length))
        y.append(1)
    X = np.array(X)
    y = np.array(y)
    idx = np.random.permutation(num_samples)
    return X[idx], y[idx]

X, y = create_dataset(10000, 10)
split = 8000
train_df = pd.DataFrame(np.column_stack([X[:split], y[:split]]),
                        columns=[f'input_{i}' for i in range(10)] + ['label'])
test_df = pd.DataFrame(np.column_stack([X[split:], y[split:]]),
                       columns=[f'input_{i}' for i in range(10)] + ['label'])
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
print("Datasets created: train.csv, test.csv")