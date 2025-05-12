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


# t = np.arange(len(ref))
# plt.plot(t, ref, label="SciPy", lw=1)
# plt.plot(t[: len(gpu)], gpu, "--", label="GPU")

# plt.xlim(0, 4096)
# plt.xlabel("Sample index")
# plt.ylabel("Amplitude")
# plt.title("UPF-UPS convolution comparison")
# plt.legend(loc="upper right", fontsize="small")
# plt.tight_layout()
# plt.savefig("correctness.png")
# plt.show()

m = min(len(gpu), len(ref))
diff = gpu[:m] - ref[:m]                 
t    = np.arange(len(ref))

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=False)

ax[0].plot(t, ref,          lw=1, label="SciPy")
ax[0].plot(t[:len(gpu)], gpu, "--", label="GPU")
ax[0].set_xlim(0, 4096)
ax[0].set_xlabel("Sample index")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("UPF-UPS convolution")
ax[0].legend(fontsize="small")

ax[1].plot(t[:m], diff,  lw=1, label="error = GPU – SciPy")
ax[1].axhline(0.0, ls="--", lw=0.8, label="ideal (0)")
ax[1].set_xlim(0, 4096)
ax[1].set_xlabel("Sample index")
ax[1].set_ylabel("Amplitude error")
ax[1].set_title("Point-wise difference")
ax[1].legend(fontsize="small")

fig.tight_layout()
fig.savefig("correctness_with_error.png", dpi=150)
plt.show()
