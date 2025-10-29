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
```

---

## Repository Layout

```
Segmentation-Mask-Cropper/
├─ microsam_crops_to_hdf5.py      # main converter
├─ h5_crop_inspector.py           # viewer
└─ README.md
```

---

## Input Layout (example)

Each **class** has its own folder with image/mask pairs sharing a numeric suffix:

```
/data/Cells/
  ├─ Cell1/
  │    ├─ Image_01.tif
  │    ├─ Mask_01.tif
  │    ├─ Image_02.tif
  │    └─ Mask_02.tif
  ├─ Cell2/
  └─ Cell3/
```

**Defaults (hardcoded)** expect `Image_XX.tif` ↔ `Mask_XX.tif`. Prefixes and class names are configurable.

---

## Usage — Converter

### Quick start (uses the hardcoded defaults)
```bash
python microsam_crops_to_hdf5.py
```

### Defaults (CONFIG block)
```python
ROOT_DIR     = "/data/Cells"
CLASSES      = ["Cell1", "Cell2", "Cell3"]
IMAGE_PREFIX = "Image_"
MASK_PREFIX  = "Mask_"
APPLY_MASK   = True      # zero background outside the instance within each crop
PAD          = 6         # pixels of padding around each bbox
MIN_AREA     = 36        # ignore tiny specks
MASK_MODE    = "auto"    # auto|label|rgb|binary
BG_COLOR_HEX = None      # e.g. "#000000" if your RGB background is known
OUT_DIR      = None      # default: writes HDF5s next to ROOT_DIR
```

### CLI (overrides any defaults)
```bash
python microsam_crops_to_hdf5.py   --root_dir /data/Cells   --classes Cell1 Cell2 Cell3   --image_prefix Image_   --mask_prefix Mask_   --apply_mask   --pad 6   --min_area 36   --mask_mode auto
```

**Outputs (per class):**
```
/data/Cells/
  ├─ Cell1.h5
  ├─ Cell2.h5
  └─ Cell3.h5
```

Each HDF5 contains:
- `images` (vlen `uint8` flattened RGB)
- `shapes` `(H, W, 3)`
- `filenames` (UTF-8)
- `instance_ids` (int)
- `bboxes` `(y0, x0, y1, x1)`

HDF5 attributes record the configuration used (prefixes, padding, mask mode, etc.).

---

## Usage — HDF5 Crop Inspector

```bash
python h5_crop_inspector.py /path/to/Cell1.h5
```

**Controls**
- `← / →` previous / next
- `Home / End` jump to first / last
- `[ / ]` step −10 / +10
- `i` toggle metadata overlay
- `g` go to index
- `s` save current crop as PNG → `./exports/<h5_name>/index_XXXXXX.png`
- `q` or `Esc` quit

The overlay shows: index, crop shape, source filename, instance ID, and bbox.

---

## Notes & Tips

- **RGB masks with touching cells:** each unique color is treated as a **separate instance** (background is the most frequent color by default; override with `--bg_color_hex` if needed).
- **Binary masks:** we run connected components to split instances.
- **16-bit images:** downscaled linearly to 8-bit (no auto-contrast) to keep training deterministic.
- **Multi-channel inputs:** first 3 channels are used; single-channel is replicated to RGB.

---

## Troubleshooting

- **“Found 0 pairs”**  
  Check `ROOT_DIR`, `CLASSES`, and prefix spelling. The matcher is case-insensitive for extensions (`.tif/.tiff/.TIF/.TIFF`), but prefixes must match.
- **Shape mismatch (image vs mask):**  
  Ensure your `Image_XX` and `Mask_XX` have identical height/width.
- **Background isn’t the majority color in RGB masks:**  
  Use `--mask_mode rgb --bg_color_hex "#000000"` (or your background colour) to pin the background.


---

## Citation

Archit, A., Freckmann, L., Nair, S., Khalid, N., Hilt, P., Rajashekar, V., ... & Pape, C. (2025). Segment anything for microscopy. Nature Methods, 22(3), 579-591.

If you use this tool in academic work, please cite the repository and the tools you build on (e.g., napari, MicroSAM, etc.).
