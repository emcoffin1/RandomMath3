import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])

print(np.shape(x))


for i, _ in enumerate(x):
    print(f"i={i}, x={_}")