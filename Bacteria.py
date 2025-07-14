import os
import numpy as np
import matplotlib.pyplot as plt
from czifile import imread
from read_roi import read_roi_zip
from skimage.draw import polygon2mask
from skimage.util import img_as_ubyte
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from cellpose import models
import pandas as pd

# === Input files ===
image_roi_pairs = [
    ("2D_WT_NADA_RADA_HADA_THY_40min_ROI1_SIM.czi", "RoiSet_2D_WT_NADA_THY1.zip"),
    ("2D_WT_NADA_RADA_HADA_THY_40min_ROI2_SIM.czi", "RoiSet_2D_WT_NADA_THY2.zip"),
    ("2D_WT_NADA_RADA_HADA_THY_40min_ROI3_SIM.czi", "RoiSet_2D_WT_NADA_THY3.zip"),

]

# === Helper functions ===
def normalize(channel):
    return (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

def read_valid_rois(zip_path, shape):
    try:
        rois = read_roi_zip(zip_path)
    except Exception as e:
        print(f"\u274c Error reading ROI zip {zip_path}: {e}")
        return np.zeros(shape, dtype=np.uint16)

    mask = np.zeros(shape, dtype=np.uint16)
    for i, roi in enumerate(rois.values(), 1):
        if 'x' not in roi or 'y' not in roi or len(roi['x']) < 3:
            continue
        try:
            y, x = roi['y'], roi['x']
            poly_mask = polygon2mask(shape, np.column_stack((y, x)))
            mask[poly_mask] = i
        except Exception:
            continue
    return remove_small_objects(mask, min_size=10)

def dice_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    return 2.0 * intersection / (gt.sum() + pred.sum() + 1e-8)

def calc_metrics(gt, pred, region):
    gt_eval = gt[region]
    pred_eval = pred[region]
    return {
        "IoU": jaccard_score(gt_eval.flatten(), pred_eval.flatten()),
        "F1": f1_score(gt_eval.flatten(), pred_eval.flatten()),
        "Dice": dice_score(gt_eval, pred_eval),
        "Accuracy": accuracy_score(gt_eval.flatten(), pred_eval.flatten())
    }

# === Evaluate using pre-trained "cyto" model ===
model = models.CellposeModel(gpu=True, pretrained_model='cyto')

all_metrics = {}

for i, (czi_file, roi_file) in enumerate(image_roi_pairs):
    print(f"\n📂 Processing: {czi_file}")
    raw = imread(czi_file)
    img = np.squeeze(raw)[0:3, :, :]
    rgb = np.stack([normalize(img[2]), normalize(img[1]), normalize(img[0])], axis=-1)
    rgb_uint8 = img_as_ubyte(rgb)

    mask_gt = read_valid_rois(roi_file, rgb.shape[:2])
    gt_bin = (mask_gt > 0).astype(np.uint8)
    eval_region = gt_bin > 0

    pred_mask, flows, styles = model.eval(rgb_uint8, diameter=None, channels=[0, 1])
    pred_bin = (pred_mask > 0).astype(np.uint8)

    metrics = calc_metrics(gt_bin, pred_bin, eval_region)
    all_metrics[czi_file] = metrics

    # === Visualization ===
    overlay = np.zeros((*gt_bin.shape, 3), dtype=np.float32)
    overlay[(gt_bin == 1) & (pred_bin == 0)] = [0, 1, 0]
    overlay[(gt_bin == 0) & (pred_bin == 1)] = [1, 0, 0]
    overlay[(gt_bin == 1) & (pred_bin == 1)] = [1, 1, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"{czi_file}\nDice={metrics['Dice']:.3f}, IoU={metrics['IoU']:.3f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Summary ===
print("\n=== Summary Metrics ===")
print(pd.DataFrame(all_metrics).T.round(3))
