import numpy as np

with open('train_handcraft_based.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)

print(a, b)