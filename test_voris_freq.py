import numpy as np
import matplotlib.pyplot as plt
import math

class DummyGrayCode:
    def __init__(self, resolution=(1920, 1080), n_bits=5, px_f=64):
        self.resolution = resolution
        self.n_bits = n_bits
        self.px_f = px_f

    def generate_gc(self):
        w = self.resolution[0]
        # Number of gray code periods?
        # Let's check how Voris generates Gray Code.
        # Voris GrayCode.py says:
        # col_idx = np.arange(width)
        # For classic Gray Code, bit i has period w / (2**i).
        pass

# I have GrayCode.py in the scratch directory.
# Let's import it directly!
import sys
sys.path.append('scratch/fringe_projection/include')
from GrayCode import GrayCode
from FringePattern import FringePattern

class TestFringeProcess(GrayCode, FringePattern):
    def __init__(self):
        FringePattern.__init__(self, resolution=(1920, 1080), px_f=64, steps=4)
        GrayCode.__init__(self, resolution=(1920, 1080), n_bits=6, px_f=64)

test = TestFringeProcess()
gray_imgs = test.generate_gc()
sine_imgs = test.generate_fringe()

# Get a row
row_gray = [img[0, :] for img in gray_imgs]
row_sine = [img[0, :] for img in sine_imgs]

# Calculate QSI (k)
white = row_gray[0]
bits = np.array(row_gray[2:])
binary = (bits > white/2).astype(int)
powers = 2 ** np.arange(binary.shape[0])[::-1]
qsi = np.sum(binary * powers[:, None], axis=0)

# Calculate phase
angle = 2.0 * math.pi * np.arange(1, 5) / 4.0
sines = np.array(row_sine)
sin_c = np.sum(sines * np.sin(angle)[:, None], axis=0)
cos_c = np.sum(sines * np.cos(angle)[:, None], axis=0)
phi = np.arctan2(-sin_c, cos_c)

plt.figure(figsize=(15, 5))
plt.plot(phi, label='Phase')
plt.plot(qsi % 4 * np.pi/2, label='QSI (mod 4, scaled)')
plt.xlim(0, 300)
plt.legend()
plt.title("Voris Patterns: Phase vs QSI")
plt.savefig('out/debug_voris_freq.png')
print("Saved debug_voris_freq.png")
