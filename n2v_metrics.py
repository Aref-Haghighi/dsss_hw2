import numpy as np
from czifile import imread
from n2v.models import N2V
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# --------- PARAMETERS ---------
czi_path = 'WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi'   # Change to your .czi image
basedir = './models'
model_name = 'N2V_MODEL'

# --------- LOAD IMAGE ---------
raw = imread(czi_path)
img = np.squeeze(raw)[1]  # Use green channel; change [0] for red, [2] for blue if needed
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# --------- LOAD TRAINED MODEL ---------
n2v = N2V(config=None, name=model_name, basedir=basedir)

# --------- DENOISE ---------
print("Denoising image with Noise2Void ...")
denoised = n2v.predict(img, axes='YX')

# --------- CALCULATE METRICS ---------
psnr_value = psnr(img, denoised, data_range=1.0)
ssim_value = ssim(img, denoised, data_range=1.0)

# --------- SHOW BEFORE/AFTER WITH METRICS ---------
plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(denoised, cmap='gray')
plt.title(f'N2V Denoised\nPSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.3f}')
plt.axis('off')

plt.tight_layout()
plt.show()
