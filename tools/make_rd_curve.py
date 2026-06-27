"""
Render the NVENC codec rate-distortion curve (bitrate vs VMAF) for the README,
from the measured AV1-efficiency data (Big Buck Bunny 1080p, NVENC preset p6).
Output: docs/codec_rd_curve.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

br   = [0.996, 1.986, 3.967, 8.049]          # Mbps (measured)
h264 = [65.6, 81.4, 89.9, 94.2]
hevc = [74.9, 85.6, 91.9, 95.3]
av1  = [78.1, 87.8, 92.9, 95.7]

plt.rcParams.update({"font.size": 12, "figure.dpi": 130})
fig, ax = plt.subplots(figsize=(7.6, 4.6))

ax.plot(br, h264, "o-", color="#9aa0a6", lw=2.2, ms=7, label="H.264  (h264_nvenc)")
ax.plot(br, hevc, "s-", color="#4a86e8", lw=2.2, ms=7, label="HEVC  (hevc_nvenc)")
ax.plot(br, av1,  "^-", color="#34a853", lw=2.6, ms=8, label="AV1   (av1_nvenc)")

# Equal-quality guide line at VMAF 90.
ax.axhline(90, color="#cc3333", ls="--", lw=1, alpha=0.7)
ax.annotate("VMAF 90 — AV1 needs ~34% less\nbitrate than H.264 for the same quality",
            xy=(4.0, 90), xytext=(3.4, 70), fontsize=10, color="#cc3333",
            ha="left")

ax.set_xlabel("Bitrate (Mbps)")
ax.set_ylabel("Quality (VMAF)")
ax.set_title("NVENC codec efficiency — 1080p, RTX 4090 (higher / left = better)")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 8.5)
ax.set_ylim(60, 100)
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "codec_rd_curve.png")
fig.savefig(path, bbox_inches="tight")
print("wrote", os.path.normpath(path))
