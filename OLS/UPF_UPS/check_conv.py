#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import oaconvolve


def load(fn):
    return np.fromfile(fn, dtype=np.float32)


x = load("input.bin")
h = load("fir.bin")
gpu = load("gpu.bin")

ref = oaconvolve(x, h, mode="full").astype(np.float32)


for tag, vec in (("gpu", gpu),):
    m = min(len(vec), len(ref))
    diff = vec[:m] - ref[:m]
    print(
        f"{tag:3s} :  max|err| {np.max(np.abs(diff)):.3e}   "
        f"RMS = {np.sqrt(np.mean(diff*diff)):.3e}"
    )


t = np.arange(len(ref))
plt.plot(t, ref, label="SciPy", lw=1)
plt.plot(t[: len(gpu)], gpu, "--", label="GPU")

plt.xlim(0, 4096)
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.title("UPF-UPS convolution comparison")
plt.legend(loc="upper right", fontsize="small")
plt.tight_layout()
plt.savefig("correctness.png")
plt.show()
