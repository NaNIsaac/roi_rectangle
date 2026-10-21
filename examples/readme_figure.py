"""Regenerate docs/roi_example.png, the figure at the top of the README.

Needs matplotlib, which roi_rectangle itself does not depend on:

    pip install matplotlib
    python examples/readme_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from roi_rectangle import RoiRectangle

rng = np.random.default_rng(0)
y, x = np.indices((100, 120))
image = 100 * np.exp(-((x - 70) ** 2 + (y - 40) ** 2) / 50) + rng.normal(10, 2, (100, 120))

roi = RoiRectangle(y1=20, y2=50, x1=40, x2=80)
crop = roi.slice(image)
peak_y, peak_x = np.unravel_index(crop.argmax(), crop.shape)
moved = roi.recenter((roi.x1 + int(peak_x), roi.y1 + int(peak_y))).resize(20, 20)

lines = [
    f"roi        = {roi}",
    f"roi.shape  = {roi.shape}",
    f"roi.center = {roi.center}   # (x, y)",
    f"crop.sum() = {crop.sum():.1f}",
    f"crop.max() = {crop.max():.1f} at (x, y) = ({roi.x1 + peak_x}, {roi.y1 + peak_y})",
    "",
    "roi.recenter(peak).resize(20, 20)",
    f"  -> {moved}",
]
print("\n".join(lines))

fig = plt.figure(figsize=(10, 3.2))
ax_img = fig.add_axes((0.01, 0.04, 0.3, 0.88))
ax_crop = fig.add_axes((0.33, 0.04, 0.2, 0.88))
ax_txt = fig.add_axes((0.55, 0.04, 0.44, 0.88))

ax_img.imshow(image, cmap="gray")
for r, color in ((roi, "tab:orange"), (moved, "tab:blue")):
    ax_img.add_patch(
        Rectangle((r.x1 - 0.5, r.y1 - 0.5), r.width, r.height, fill=False, ec=color, lw=1.5)
    )
ax_img.set_title("image (100, 120)", fontsize=10)
ax_crop.imshow(crop, cmap="gray")
ax_crop.set_title(f"roi.slice(image) {crop.shape}", fontsize=10)
ax_txt.text(0, 0.95, "\n".join(lines), family="monospace", fontsize=8.5, va="top")
for ax in (ax_img, ax_crop, ax_txt):
    ax.set_axis_off()

out = Path(__file__).resolve().parents[1] / "docs" / "roi_example.png"
fig.savefig(out, dpi=100, facecolor="white")
print(f"wrote {out}")
