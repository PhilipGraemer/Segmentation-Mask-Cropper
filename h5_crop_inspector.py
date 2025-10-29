#!/usr/bin/env python3
# H5 Crop Inspector — browse per-cell crops stored in HDF5.
#
# Controls:
#   ← / → : previous / next crop
#   Home/End : jump to first / last crop
#   [ / ] : step -10 / +10
#   s : save current crop as PNG (to ./exports/<h5_basename>/index_<i>.png)
#   i : toggle info overlay (filename, instance_id, bbox, shape)
#   g : go to index (type a number then press Enter in the terminal)
#   q or Esc : quit
#
# Usage:
#   python h5_crop_inspector.py /path/to/Class.h5
#
# Notes:
#   - Images are reconstructed from datasets:
#       images[i] (vlen uint8 flattened), shapes[i] = (H,W,3)
#     The script renders as RGB uint8.

import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")  # use default backend
import matplotlib.pyplot as plt
from PIL import Image

def load_crop(hf: h5py.File, idx: int) -> np.ndarray:
    flat = hf["images"][idx]
    h, w, c = hf["shapes"][idx]
    arr = np.asarray(flat, dtype=np.uint8).reshape((h, w, c))
    return arr

def _read_str(ds, idx):
    """Robustly read a string from an HDF5 dataset that may store bytes or fixed-length strings."""
    try:
        return ds.asstr()[idx]
    except Exception:
        val = ds[idx]
        if isinstance(val, (bytes, np.bytes_)):
            try:
                return val.decode("utf-8")
            except Exception:
                return val.decode(errors="ignore")
        return str(val)

def get_meta(hf: h5py.File, idx: int):
    if "filenames" in hf:
        try:
            fn = _read_str(hf["filenames"], idx)
        except Exception:
            fn = ""
    else:
        fn = ""

    inst = int(hf["instance_ids"][idx]) if "instance_ids" in hf else -1
    bbox = tuple(int(x) for x in hf["bboxes"][idx]) if "bboxes" in hf else ()
    return fn, inst, bbox

class Inspector:
    def __init__(self, path: Path):
        self.path = path
        self.hf = h5py.File(str(path), "r")
        self.n = len(self.hf["images"])
        self.i = 0
        self.show_info = True

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    def render(self):
        arr = load_crop(self.hf, self.i)
        self.ax.clear()
        self.ax.imshow(arr)
        self.ax.axis("off")

        if self.show_info:
            fn, inst, bbox = get_meta(self.hf, self.i)
            h, w, c = self.hf["shapes"][self.i]
            title = f"[{self.i+1}/{self.n}] shape=({h}x{w}x{c})"
            subtitle = []
            if fn: subtitle.append(f"file: {fn}")
            if inst != -1: subtitle.append(f"inst: {inst}")
            if bbox: subtitle.append(f"bbox: {bbox}")
            self.ax.set_title(title + ("  |  " + "  •  ".join(subtitle) if subtitle else ""), fontsize=10)
        else:
            self.ax.set_title(f"[{self.i+1}/{self.n}]", fontsize=10)

        self.fig.canvas.draw_idle()

    def clamp(self):
        self.i = max(0, min(self.n - 1, self.i))

    def save_current(self):
        arr = load_crop(self.hf, self.i)
        out_root = Path.cwd() / "exports" / self.path.stem
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / f"index_{self.i:06d}.png"
        Image.fromarray(arr).save(out_path)
        print(f"[saved] {out_path}")

    def goto_prompt(self):
        try:
            idx = input(f"Go to index [1..{self.n}]: ").strip()
            if idx:
                self.i = int(idx) - 1
                self.clamp()
                self.render()
        except Exception as e:
            print(f"[warn] goto failed: {e}")

    def on_key(self, event):
        key = event.key
        if key in ("right",):
            self.i += 1
        elif key in ("left",):
            self.i -= 1
        elif key in ("home",):
            self.i = 0
        elif key in ("end",):
            self.i = self.n - 1
        elif key == "[":
            self.i -= 10
        elif key == "]":
            self.i += 10
        elif key in ("q", "escape"):
            plt.close(self.fig)
            self.hf.close()
            return
        elif key == "s":
            self.save_current()
        elif key == "i":
            self.show_info = not self.show_info
        elif key == "g":
            self.goto_prompt()

        self.clamp()
        self.render()

def main():
    ap = argparse.ArgumentParser(description="HDF5 Crop Inspector")
    ap.add_argument("h5_path", type=str, help="Path to a per-class .h5 file of crops")
    args = ap.parse_args()

    h5_path = Path(args.h5_path).expanduser().resolve()
    if not h5_path.exists():
        print(f"[error] file not found: {h5_path}")
        sys.exit(1)

    print("Controls: ←/→ prev/next | Home/End | [/] -10/+10 | s save | i info | g goto | q quit")
    insp = Inspector(h5_path)
    plt.show()

if __name__ == "__main__":
    main()
