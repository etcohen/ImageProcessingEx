import numpy as np

def add_safe(x, b):
    # המרה ל-int16 כדי שלא יהיה גלישה, חיבור, חיתוך לטווח 0..255, והמרה חזרה ל-uint8
    return np.clip(x.astype(np.int16) + b, 0, 255).astype(np.uint8)

x = np.array([0, 50, 100, 200, 255], dtype=np.uint8)

print(add_safe(x, 100))   # [100 150 200 255 255]
print(add_safe(x, -150))  # [  0   0   0  50 105]