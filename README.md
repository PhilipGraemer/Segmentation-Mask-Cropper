# Segmentation-Mask-Cropper

> Convert MicroSAM/Napari segmentation masks into **per-class HDF5 datasets** of cropped cells, ready for downstream ML (classification, self-supervised pretraining, etc.). Includes a lightweight **HDF5 crop viewer** for quick QA.

<p align="center">
MicroSAM (Napari) ➜ <b>Segmentation-Mask-Cropper</b> ➜ Classifier training
</p>

---

## Features

- **Mask modes:** integer-label masks, RGB colorized masks (each color = instance), or binary masks (connected components).
- **Touching instances supported:** RGB masks with different colours are split into distinct labels.
- **Per-class HDF5s** with:
  - `images` — variable-length flattened RGB `uint8` crops  
  - `shapes` — `(H, W, 3)` for each crop  
  - `filenames`, `instance_ids`, `bboxes` (traceability & debugging)
- **Hardcoded defaults** in a `CONFIG` block (CLI can still override).
- **Inspector app** (`h5_crop_inspector.py`) to scroll through crops, view metadata, and export PNGs.

---

## Installation

```bash
# minimal runtime deps
pip install numpy h5py tifffile pillow scikit-image matplotlib
