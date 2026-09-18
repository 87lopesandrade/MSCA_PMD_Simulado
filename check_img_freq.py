import cv2
import numpy as np

img_gray = cv2.imread('/Users/igorlopesdeandrade/Desktop/sim_deflectometria/PMD/Gray_Code/cam0_gray_bit3.png', cv2.IMREAD_GRAYSCALE)
img_phase = cv2.imread('/Users/igorlopesdeandrade/Desktop/sim_deflectometria/PMD/Phase/cam0_phase_step0.png', cv2.IMREAD_GRAYSCALE)

row_gray = img_gray[1024, :]
row_phase = img_phase[1024, :]

print("Gray code bit 3 transitions:")
diffs_gc = np.diff(row_gray.astype(int))
trans_gc = np.where(np.abs(diffs_gc) > 50)[0]
print(trans_gc[:10])

print("Sine wave peaks:")
# roughly find peaks in sine wave
diffs_ph = np.diff(row_phase.astype(int))
peaks = []
for i in range(1, len(diffs_ph)-1):
    if diffs_ph[i-1] > 0 and diffs_ph[i] < 0:
        if row_phase[i] > 200:
            peaks.append(i)
print(peaks[:10])
