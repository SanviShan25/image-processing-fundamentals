# %% Setup
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "photo.png"

# %% Task 1: original photograph
bgr = cv2.imread(PATH)                       # OpenCV loads as BGR
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # convert to RGB for display
h, w, c = rgb.shape
size_kb = os.path.getsize(PATH) / 1024

print("Image shape (H, W, C):", rgb.shape)
print(f"Dimensions (width x height): {w} x {h}")
print("Number of colour channels:", c)
print(f"File size: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")

plt.figure(figsize=(7, 5.25))
plt.imshow(rgb); plt.title("Original photograph (RGB)"); plt.axis("off")
plt.show()

# %% Task 2: grayscale
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print("Gray shape:", gray.shape, "| mean: %.1f" % gray.mean())

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].imshow(rgb);                                 ax[0].set_title("Colour (RGB)")
ax[1].imshow(gray, cmap="gray", vmin=0, vmax=255); ax[1].set_title("Grayscale")
for a in ax: a.axis("off")
plt.show()

# %% Task 3: negative and gamma
negative = 255 - gray

def gamma_correct(img, gamma):
    r = img.astype(np.float64) / 255.0      # scale to [0, 1]
    s = np.power(r, gamma)                  # s = r ^ gamma
    return np.uint8(np.clip(s * 255.0, 0, 255))

g05 = gamma_correct(gray, 0.5)
g10 = gamma_correct(gray, 1.0)
g20 = gamma_correct(gray, 2.0)

for name, im in [("gray", gray), ("negative", negative),
                 ("gamma 0.5", g05), ("gamma 1.0", g10), ("gamma 2.0", g20)]:
    print(f"{name:10s} mean intensity = {im.mean():6.1f}")

imgs   = [gray, negative, g05, g10, g20]
titles = ["Grayscale (input)", "Negative", "Gamma = 0.5", "Gamma = 1.0", "Gamma = 2.0"]
fig, ax = plt.subplots(2, 3, figsize=(12, 6.6))
for a, im, t in zip(ax.ravel(), imgs, titles):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
ax[1, 2].axis("off")
plt.show()

# %% Task 4: contrast stretching and histogram equalization
low, high = np.percentile(gray, (1, 99))
stretch = np.clip((gray.astype(np.float64) - low) / (high - low), 0, 1)
stretch = np.uint8(stretch * 255)
print(f"Stretch limits: low = {low:.0f}, high = {high:.0f}")

equal = cv2.equalizeHist(gray)

imgs   = [gray, stretch, equal]
titles = ["Input (gray)", "Contrast stretching", "Histogram equalization"]
fig, ax = plt.subplots(2, 3, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 2]})
hist_max = 0
for i, (im, t) in enumerate(zip(imgs, titles)):
    ax[0, i].imshow(im, cmap="gray", vmin=0, vmax=255)
    ax[0, i].set_title(t); ax[0, i].axis("off")
    cnt, _, _ = ax[1, i].hist(im.ravel(), bins=64, range=(0, 256), color="gray")
    hist_max = max(hist_max, cnt.max())
    ax[1, i].set_xlim(0, 255)
    ax[1, i].set_xlabel("Intensity"); ax[1, i].set_ylabel("Pixel count")
    ax[1, i].set_title("Histogram: " + t, fontsize=9)
    print(f"{t:24s} mean = {im.mean():6.1f}, std = {im.std():6.1f}")
for i in range(3): ax[1, i].set_ylim(0, hist_max * 1.05)   # same y-scale
plt.tight_layout(); plt.show()

# %% Task 5: Gaussian smoothing
blur3 = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.8)
blur7 = cv2.GaussianBlur(gray, (7, 7), sigmaX=1.4)

imgs   = [gray, blur3, blur7]
titles = ["Input (gray)", "Gaussian 3x3 (sigma=0.8)", "Gaussian 7x7 (sigma=1.4)"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
for a, im, t in zip(ax, imgs, titles):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()

# zoomed crop to compare fine detail (hedge leaves)
y0, y1, x0, x1 = 560, 760, 560, 860
fig, ax = plt.subplots(1, 3, figsize=(14, 3.7))
for a, im, t in zip(ax, imgs, titles):
    a.imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    a.set_title("Zoom: " + t, fontsize=9); a.axis("off")
plt.tight_layout(); plt.show()

# %% Task 6: Sobel edge detection
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # horizontal gradient
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # vertical gradient
mag = np.sqrt(gx**2 + gy**2)                      # gradient magnitude

m = mag.max()                                     # same divisor for all three
gx_disp  = np.uint8(np.clip(np.abs(gx) / m * 255, 0, 255))
gy_disp  = np.uint8(np.clip(np.abs(gy) / m * 255, 0, 255))
mag_disp = np.uint8(np.clip(mag / m * 255, 0, 255))

imgs   = [gray, gx_disp, gy_disp, mag_disp]
titles = ["Input (gray)", "Sobel Gx", "Sobel Gy", "Gradient magnitude"]
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
for a, im, t in zip(ax.ravel(), imgs, titles):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
# %%
