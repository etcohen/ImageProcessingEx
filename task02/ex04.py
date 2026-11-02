import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "flower.png"

# Part 1 - read a color image and print its properties
img = cv2.imread(PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {PATH}")
print("type:", type(img))
print("dtype:", img.dtype)
print("shape:", img.shape)
print("top-left pixel (B, G, R):", img[0, 0])

# Part 2 - what does cv2.imread return for a wrong file name?
bad = cv2.imread("no_such_file.png")
print("imread of a wrong name returns:", bad)   # None - no exception is raised
print("bad is None:", bad is None)
# The check that was added to the code above:  if img is None: raise ...

# Part 3 - imshow without and with BGR -> RGB conversion
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title("Without conversion (BGR read as RGB)")
axes[1].imshow(rgb)
axes[1].set_title("After cv2.cvtColor BGR -> RGB")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("ex04_part3.png", dpi=100, bbox_inches="tight", pad_inches=0.3)
plt.close()

# Part 4 - original + the three channels as grayscale, side by side
b, g, r = cv2.split(img)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(rgb)
axes[0].set_title("Original")
for ax, ch, name in zip(axes[1:], (r, g, b), ("R channel", "G channel", "B channel")):
    ax.imshow(ch, cmap="gray", vmin=0, vmax=255)
    ax.set_title(name)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("ex04.png", dpi=100, bbox_inches="tight", pad_inches=0.3)
plt.close()

# Part 5 - read the same image as grayscale
gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
print("grayscale shape:", gray.shape)
