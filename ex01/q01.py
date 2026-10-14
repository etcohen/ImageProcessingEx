import numpy as np

a = np.array([[10, 100, 200],
              [50, 150, 250]], dtype=np.uint8)
print("",a.dtype)
print(a.shape)
print(a.ndim)
print(a.size)
print(a.nbytes)
print(a[1, 2])
print(a[0, 1])
print(a + 10)
print(a - 20)
print(a * 2)
print(a.max())
print(a.max(axis=0))
print(a.max(axis=1))