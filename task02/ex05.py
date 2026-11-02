import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "flower.png"
img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not read image: {PATH}")


def add_safe(img, b):
    """Add b to a uint8 image with clipping to [0, 255] (the add_safe of exercise 3)."""
    result = img.astype(np.int16) + b
    return np.clip(result, 0, 255).astype(np.uint8)


# Part 1 - brighten(img, b, method)
def brighten(img, b, method):
    if method == "np":
        return img + b                      # uint8 arithmetic: wraps around (mod 256)
    elif method == "cv2":
        return cv2.add(img, b)              # saturated arithmetic
    elif method == "clip":
        return add_safe(img, b)             # our own safe function
    raise ValueError(f"Unknown method: {method}")


def show(images, titles, filename, use_vmin_vmax):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for ax, im, title in zip(axes, images, titles):
        if use_vmin_vmax:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.3)
    plt.close()


# Part 2 - b = 100 with the three methods, four images side by side
b = 100
# NumPy 2: img + 100 with a Python int keeps uint8 and wraps around silently
results = {m: brighten(img, b, m) for m in ("np", "cv2", "clip")}
images = [img, results["np"], results["cv2"], results["clip"]]
titles = ["Original", 'method="np"', 'method="cv2"', 'method="clip"']
show(images, titles, "ex05.png", use_vmin_vmax=True)

# Part 3 - why are there dark areas in the "np" result?
bright_mask = img > 155            # pixels that overflow when adding 100
print("pixels with value > 155:", int(bright_mask.sum()))
print("original values at those pixels (sample):", img[bright_mask][:5])
print("'np' values at the same pixels (sample):  ", results["np"][bright_mask][:5])
print("'cv2' values at the same pixels (sample): ", results["cv2"][bright_mask][:5])
print("max of np result:", results["np"].max(), "| max of cv2 result:", results["cv2"].max())

# Part 4 - b = -100 with every method
b = -100
print("\n--- b = -100 ---")
try:
    r_np = brighten(img, b, "np")
    print("np: dtype", r_np.dtype, "min", r_np.min(), "max", r_np.max())
except Exception as e:
    r_np = None
    print("np raised:", type(e).__name__, e)

# cv2.add with a negative scalar: the scalar is saturated into the uint8 range
r_cv2 = brighten(img, b, "cv2")
print("cv2.add(img, -100): min", r_cv2.min(), "max", r_cv2.max(),
      "| identical to original?", np.array_equal(r_cv2, img))

r_clip = brighten(img, b, "clip")
print("clip: min", r_clip.min(), "max", r_clip.max())

# cv2.subtract is the right tool for darkening: saturates at 0
r_sub = cv2.subtract(img, 100)
print("cv2.subtract(img, 100): min", r_sub.min(), "max", r_sub.max(),
      "| equals clip result?", np.array_equal(r_sub, r_clip))

# Part 5 - the same figure without vmin / vmax
show(images, titles, "ex05_no_vmin.png", use_vmin_vmax=False)

# extra figure for the b = -100 discussion (not required)
imgs_neg = [img,
            r_np if r_np is not None else img,
            r_cv2, r_clip]
show(imgs_neg, ["Original", 'np, b=-100', 'cv2.add, b=-100', 'clip, b=-100'],
     "ex05_negative.png", use_vmin_vmax=True)
