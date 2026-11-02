import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "flower.png"
img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not read image: {PATH}")


def to_uint8(arr):
    """Clip to [0, 255] and convert back to a valid uint8 image."""
    return np.clip(arr, 0, 255).astype(np.uint8)


# Part 1 - img * 1.5
r15 = img * 1.5
print("img * 1.5 -> dtype:", r15.dtype, "| max:", r15.max(), "| min:", r15.min())

# Part 2 - back to a valid uint8 image, shown next to the original
r15_u8 = to_uint8(r15)
print("after clip + astype(uint8): dtype", r15_u8.dtype, "max", r15_u8.max())

# Part 3 - compare with img * 2 (integer multiplication)
r2 = img * 2
print("img * 2 -> dtype:", r2.dtype, "| max:", r2.max())
print("pixels that overflowed in img*2 (orig > 127):", int((img > 127).sum()))
r2_clip = to_uint8(img.astype(np.int16) * 2)   # the safe version of *2

# Part 4 - k = 0.5
r05 = img * 0.5
r05_u8 = to_uint8(r05)
print("img * 0.5 -> dtype:", r05.dtype, "| max:", r05.max(), "| min:", r05.min())

# Figure: original next to the results of parts 2, 3 and 4
fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))
items = [(img, "Original"),
         (r15_u8, "Part 2: img*1.5 (clipped, uint8)"),
         (r2, "Part 3: img*2 (uint8, wraps around)"),
         (r2_clip, "Part 3: img*2 (clipped, safe)"),
         (r05_u8, "Part 4: img*0.5")]
for ax, (im, title) in zip(axes, items):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("ex06.png", dpi=100, bbox_inches="tight", pad_inches=0.3)
plt.close()

# Part 5 (bonus) - stretch around the mean: (img - m) * k + m
m = img.mean()
print("\nmean m:", round(float(m), 2), "| dtype of m:", type(m))
diff = img - m
print("img - m -> dtype:", diff.dtype, "| min:", round(float(diff.min()), 1),
      "| max:", round(float(diff.max()), 1))

ks = [0.5, 1.0, 1.5, 2.0, 3.0]
fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 5.5))
for ax, k in zip(axes, ks):
    result = (img - m) * k + m                 # float64 - no overflow
    out_of_range = int(((result < 0) | (result > 255)).sum())
    print(f"k={k}: min {result.min():.1f}, max {result.max():.1f}, "
          f"pixels out of [0,255] before clipping: {out_of_range}")
    ax.imshow(to_uint8(result), cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"contrast around the mean, k={k}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("ex06_bonus.png", dpi=100, bbox_inches="tight", pad_inches=0.3)
plt.close()
