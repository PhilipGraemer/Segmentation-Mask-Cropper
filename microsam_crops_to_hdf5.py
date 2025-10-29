#!/usr/bin/env python3
"""
A tool for downstream use for classification. This sits between segmentation mask creation and classifier training.
(MicroSAM > This > Training algorithms).

MicroSAM (Napari) masks (label, RGB-colorized, or binary) > per-class HDF5s of cropped cells.

Hardcoded defaults live in the CONFIG section below. CLI flags inherit these as defaults,
but you can still override them if needed.

Dependencies: numpy, h5py, tifffile, Pillow, scikit-image
To Install: pip install numpy h5py tifffile pillow scikit-image
"""

from pathlib import Path
from typing import Tuple, List, Optional
import argparse
import os
import h5py
import numpy as np
from tifffile import imread
from PIL import Image
from skimage.measure import label as cc_label  # connected components

# ============================
# CONFIG 
# ============================
#ROOT_DIR       = "/Users/phil/AI/imgs/Cells"
ROOT_DIR       = str(Path(__file__).parent.resolve())  # default: folder of this script
CLASSES        = ["ASC", "MG63", "C2C12"] # Change to your classes/cell types
IMAGE_PREFIX   = "Image_"   # case-insensitive match
MASK_PREFIX    = "Mask_"    # case-insensitive match
APPLY_MASK     = True
PAD            = 6
MIN_AREA       = 36
MASK_MODE      = "auto"
BG_COLOR_HEX   = None
OUT_DIR        = None  # If None => save next to ROOT_DIR


# ---------- image utils ----------
def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Convert 2D/3D image of various dtypes to RGB uint8 (H,W,3), no auto-contrast."""
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 257.0).astype(np.uint8)  # linear 16->8
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)

def bbox_from_label(mask: np.ndarray, label_id: int, pad: int, H: int, W: int) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask == label_id)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(H - 1, y1 + pad); x1 = min(W - 1, x1 + pad)
    return int(y0), int(x0), int(y1), int(x1)

# ---------- mask parsing (auto/label/rgb/binary) ----------
def rgb_to_labels(mask_rgb: np.ndarray, bg_color: Optional[Tuple[int,int,int]] = None) -> np.ndarray:
    """
    Map each unique RGB color to a distinct integer label. Background color is 0.
    If bg_color is None, use the most frequent color as background.
    """
    H, W, _ = mask_rgb.shape
    flat = mask_rgb.reshape(-1, 3)
    colors, inv = np.unique(flat, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    if bg_color is None:
        bg_idx = np.argmax(counts)
        bg = tuple(colors[bg_idx])
    else:
        bg = tuple(bg_color)
    label_map = {}
    next_label = 1
    for c in colors:
        ct = tuple(c)
        if ct == bg:
            label_map[ct] = 0
        else:
            label_map[ct] = next_label
            next_label += 1
    out = np.empty((H*W,), dtype=np.int32)
    for idx, c in enumerate(colors):
        out[inv == idx] = label_map[tuple(c)]
    return out.reshape(H, W)

def bin_to_labels(mask_bin: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Connected-component labeling on binary mask (nonzero -> foreground)."""
    bin01 = (mask_bin != 0).astype(np.uint8)
    lbl = cc_label(bin01, connectivity=connectivity)
    return lbl.astype(np.int32)

def parse_mask(mask: np.ndarray, mode: str = "auto", bg_color_hex: Optional[str] = None) -> np.ndarray:
    """
    Return integer label image with 0=background, 1..K instances.
    mode: auto|label|rgb|binary
    """
    if mode not in {"auto", "label", "rgb", "binary"}:
        raise ValueError(f"Unknown mask mode: {mode}")

    if mask.ndim == 3 and mask.shape[-1] == 4:
        mask = mask[..., :3]  # drop alpha

    bg_color = None
    if bg_color_hex is not None:
        s = bg_color_hex.lstrip("#")
        if len(s) != 6:
            raise ValueError("bg_color_hex must look like '#000000' or '000000'")
        bg_color = (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

    if mode == "label":
        if mask.ndim != 2:
            raise ValueError("label mode expects a 2D integer mask.")
        return mask.astype(np.int32)

    if mode == "rgb":
        if mask.ndim != 3 or mask.shape[-1] != 3:
            raise ValueError("rgb mode expects an HxWx3 RGB mask.")
        return rgb_to_labels(mask.astype(np.uint8), bg_color=bg_color)

    if mode == "binary":
        if mask.ndim == 3:
            mask = mask[..., 0]
        return bin_to_labels(mask)

    # auto
    if mask.ndim == 2:
        uniq = np.unique(mask)
        return mask.astype(np.int32) if uniq.size > 2 else bin_to_labels(mask)
    elif mask.ndim == 3:
        return rgb_to_labels(mask.astype(np.uint8), bg_color=bg_color)
    else:
        raise ValueError("Unsupported mask dimensionality.")

# ---------- pairing & processing ----------
def find_pairs(
    cls_dir: Path,
    image_prefix: str,
    mask_prefix: str,
    image_exts=(".tif", ".tiff"),
    mask_exts=(".tif", ".tiff"),
):
    """
    Case-insensitive pairing:
      <IMAGE_PREFIX><suffix>.<tif|tiff>  ↔  <MASK_PREFIX><same suffix>.<tif|tiff>
    Works with .tif/.tiff in any case (e.g., .TIF).
    """
    # list all files once
    files = [p for p in cls_dir.iterdir() if p.is_file()]
    # index masks by lowercased name for O(1) lookup
    mask_names = {p.name.lower(): p for p in files}

    def is_image(p):
        n = p.name
        return n.lower().startswith(image_prefix.lower()) and any(n.lower().endswith(e) for e in image_exts)

    pairs = []
    for img_path in sorted([p for p in files if is_image(p)], key=lambda x: x.name.lower()):
        name = img_path.name
        # suffix after the image prefix (preserve original casing of suffix)
        suffix = name[len(image_prefix):]
        # try all mask extensions, insensitive
        found = None
        for me in mask_exts:
            candidate = (mask_prefix + suffix)  # keep same suffix incl. original extension
            # swap extension to me, regardless of current case
            base, _ext = os.path.splitext(candidate)
            cand_name = (base + me).lower()
            if cand_name in mask_names:
                found = mask_names[cand_name]
                break
            # also try the untouched candidate (if suffix already includes correct ext)
            if candidate.lower() in mask_names:
                found = mask_names[candidate.lower()]
                break
        if found is not None:
            pairs.append((img_path, found))
    return pairs

def process_class(
    cls_dir: Path,
    class_name: str,
    out_dir: Path,
    image_prefix: str,
    mask_prefix: str,
    pad: int,
    apply_mask: bool,
    min_area: int,
    mask_mode: str,
    bg_color_hex: Optional[str],
):
    pairs = find_pairs(cls_dir, image_prefix=image_prefix, mask_prefix=mask_prefix)
    if not pairs:
        print(f"[WARN] No image/mask pairs found in {cls_dir}")
        return

    out_path = out_dir / f"{class_name}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as hf:
        vlen_u8 = h5py.vlen_dtype(np.uint8)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        images_flat, shapes, filenames, instance_ids, bboxes = [], [], [], [], []

        for img_path, msk_path in pairs:
            img = imread(str(img_path))      # (H,W[,C])
            raw_msk = imread(str(msk_path))  # could be 2D labels, RGB-colorized, or binary

            # If paletted, convert to RGB so 'auto' can detect color labels
            if raw_msk.ndim == 2 and (str(msk_path).lower().endswith((".png", ".tif", ".tiff"))):
                try:
                    with Image.open(str(msk_path)) as pm:
                        if pm.mode == "P":
                            raw_msk = np.array(pm.convert("RGB"))
                except Exception:
                    pass

            if img.ndim == 3 and img.shape[-1] not in (1, 3):
                img = img[..., :3]

            H, W = img.shape[0], img.shape[1]
            if raw_msk.shape[:2] != (H, W):
                raise ValueError(f"Shape mismatch: {img_path.name} {img.shape} vs {msk_path.name} {raw_msk.shape}")

            msk = parse_mask(raw_msk, mode=mask_mode, bg_color_hex=bg_color_hex)  # -> int labels 0..K
            labels = np.unique(msk)
            labels = labels[labels != 0]

            for lab in labels:
                area = np.count_nonzero(msk == lab)
                if area < min_area:
                    continue

                bbox = bbox_from_label(msk, int(lab), pad=pad, H=H, W=W)
                if bbox is None:
                    continue
                y0, x0, y1, x1 = bbox

                img_crop = img[y0:y1+1, x0:x1+1, ...]
                if apply_mask:
                    m_crop = (msk[y0:y1+1, x0:x1+1] == lab)
                    if img_crop.ndim == 2:
                        img_crop = img_crop * m_crop.astype(img_crop.dtype)
                    else:
                        img_crop = img_crop * m_crop[..., None].astype(img_crop.dtype)

                rgb_u8 = to_uint8_rgb(img_crop)
                images_flat.append(rgb_u8.reshape(-1))
                shapes.append(rgb_u8.shape)
                filenames.append(img_path.name)
                instance_ids.append(int(lab))
                bboxes.append((y0, x0, y1, x1))

        n = len(images_flat)
        d_images = hf.create_dataset("images", shape=(n,), dtype=vlen_u8, compression="gzip", compression_opts=4)
        d_shapes = hf.create_dataset("shapes", shape=(n, 3), dtype=np.int32)
        d_files  = hf.create_dataset("filenames", shape=(n,), dtype=str_dtype)
        d_inst   = hf.create_dataset("instance_ids", shape=(n,), dtype=np.int32)
        d_bboxes = hf.create_dataset("bboxes", shape=(n, 4), dtype=np.int32)

        for i in range(n):
            d_images[i] = images_flat[i]
            h, w, c = shapes[i]
            d_shapes[i, :] = (h, w, c)
            d_files[i] = filenames[i]
            d_inst[i] = instance_ids[i]
            d_bboxes[i, :] = bboxes[i]

        # attrs
        hf.attrs["creator"] = "microsam_crops_to_hdf5"
        hf.attrs["class_name"] = class_name
        hf.attrs["channels"] = 3
        hf.attrs["dtype_written"] = "uint8"
        hf.attrs["color_space"] = "RGB"
        hf.attrs["image_prefix"] = image_prefix
        hf.attrs["mask_prefix"] = mask_prefix
        hf.attrs["padding_px"] = pad
        hf.attrs["apply_mask"] = int(apply_mask)
        hf.attrs["min_area_px"] = int(min_area)
        hf.attrs["mask_mode"] = mask_mode
        if bg_color_hex:
            hf.attrs["bg_color_hex"] = bg_color_hex

    print(f"[OK] {class_name}: wrote {out_path} (N={len(images_flat)})")

def main():
    p = argparse.ArgumentParser(
        description="Crop cells from MicroSAM masks (label/rgb/binary) into per-class HDF5s."
    )
    # Defaults come from your CONFIG block
    p.add_argument("--root_dir", type=str, default=ROOT_DIR)
    p.add_argument("--classes", type=str, nargs="+", default=CLASSES)
    p.add_argument("--image_prefix", type=str, default=IMAGE_PREFIX)
    p.add_argument("--mask_prefix", type=str, default=MASK_PREFIX)
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--pad", type=int, default=PAD)
    p.add_argument("--min_area", type=int, default=MIN_AREA)
    p.add_argument("--mask_mode", type=str, default=MASK_MODE, choices=["auto", "label", "rgb", "binary"])
    p.add_argument("--bg_color_hex", type=str, default=BG_COLOR_HEX)

    # Boolean toggle; default comes from CONFIG.APPLY_MASK
    p.add_argument("--apply_mask", dest="apply_mask", action="store_true")
    p.add_argument("--no-apply_mask", dest="apply_mask", action="store_false")
    p.set_defaults(apply_mask=APPLY_MASK)

    args = p.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root

    print(f"[INFO] ROOT_DIR = {root}")
    print(f"[INFO] CLASSES  = {args.classes}")
    print(f"[INFO] image_prefix = '{args.image_prefix}', mask_prefix = '{args.mask_prefix}'")

    for cls in args.classes:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            print(f"[WARN] Missing class directory: {cls_dir}")
            # Show a few entries under root to sanity-check
            try:
                entries = sorted([p.name for p in root.iterdir()])
                print(f"[HINT] Entries under {root}:\n  " + "\n  ".join(entries[:20]))
            except Exception:
                pass
            continue

        pairs = find_pairs(
            cls_dir=cls_dir,
            image_prefix=args.image_prefix,
            mask_prefix=args.mask_prefix,
        )
        print(f"[INFO] {cls}: found {len(pairs)} image/mask pairs")

        # Process and write HDF5 for this class
        process_class(
            cls_dir=cls_dir,
            class_name=cls,
            out_dir=out_dir,
            image_prefix=args.image_prefix,
            mask_prefix=args.mask_prefix,
            pad=args.pad,
            apply_mask=args.apply_mask,
            min_area=args.min_area,
            mask_mode=args.mask_mode,
            bg_color_hex=args.bg_color_hex,
        )

if __name__ == "__main__":
    main()

