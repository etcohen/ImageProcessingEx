import numpy as np
x = np.array([100, 200, 250], dtype=np.uint8)
print(x + 10)
print(x + 100)
print(x - 150)
print(x * 2)
print(x.astype(np.int16) + 100)
print(x * 1.5)
print((x * 1.5).dtype)
print(x + 300) #גלישה הוא יכול להתאפס פעם אחת כשהגענו כבר ל550 זה יותר מידי