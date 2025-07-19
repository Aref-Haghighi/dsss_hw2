import numpy as np
from czifile import imread
from n2v.models import N2V, N2VConfig
import os

# --- Patch extraction function ---
def extract_patches(img, shape=(64,64), n_patches=512, random_seed=42):
    """
    Randomly extract n_patches from a 2D image.
    """
    np.random.seed(random_seed)
    h, w = img.shape
    ph, pw = shape
    patches = []
    for _ in range(n_patches):
        top = np.random.randint(0, h - ph)
        left = np.random.randint(0, w - pw)
        patch = img[top:top+ph, left:left+pw]
        patches.append(patch)
    return np.stack(patches, axis=0)

# ---------- PARAMETERS ----------
czi_path = 'WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi'   # <-- change to your .czi file if needed
basedir = './models'
model_name = 'N2V_MODEL'
patch_size = (64, 64)
n_patches = 512

# ---------- READ IMAGE ----------
print(f"Reading {czi_path} ...")
raw = imread(czi_path)
# Use green channel (edit index as needed: 0=red, 1=green, 2=blue)
img = np.squeeze(raw)[1]
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# ---------- EXTRACT PATCHES ----------
print(f"Extracting {n_patches} patches of size {patch_size} ...")
patches = extract_patches(img, shape=patch_size, n_patches=n_patches)
patches = patches[..., np.newaxis]  # Add channel axis

# ---------- SPLIT TRAIN/VALIDATION ----------
val_split = 0.1  # 10% validation
n_val = int(len(patches) * val_split)
validation_X = patches[:n_val]
train_X = patches[n_val:]

# ---------- CREATE CONFIG ----------
print("Creating N2V config ...")
config = N2VConfig(
    X=patches,
    unet_kern_size=3,
    train_steps_per_epoch=100,
    train_epochs=20,
    train_loss='mse',
    batch_norm=True,
    train_batch_size=16,
    n2v_perc_pix=0.198,
    n2v_patch_shape=patch_size,
    train_learning_rate=1e-3,
    train_reduce_lr={'patience': 3, 'factor': 0.5},
    train_epochs_per_step=1
)

# ---------- TRAIN MODEL ----------
print("Training Noise2Void model ...")
if not os.path.exists(os.path.join(basedir, model_name)):
    os.makedirs(os.path.join(basedir, model_name))

n2v = N2V(config=config, name=model_name, basedir=basedir)
n2v.train(train_X, validation_X, epochs=20, steps_per_epoch=100)
print(" Training complete! Model saved at ./models/N2V_MODEL/")
