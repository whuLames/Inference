import numpy as np

shape = (5,15)

data = np.random.rand(*shape).astype(np.float32)
print(data.dtype)
data.tofile("test.bin")

data = data.flatten()

data1 = np.fromfile("test.bin", dtype=np.float32)
print(data1)
mask = data != data1

print(np.sum(mask))
