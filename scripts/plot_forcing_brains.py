"""
Cortical forcing terms of every subtype on inflated surfaces, one shared colour scale.

    python scripts/plot_forcing_brains.py <forcing_dir> [--out forcing_brains.png] [--hemi both]
                                          [--views lateral medial] [--cmap afmhot] [--show]

<forcing_dir> holds comind_cortical_forcing_subtype<k>_brainutils.csv (columns region = L_/R_<aparc
name>, FX_size = f), written by the "Forcing terms by subtype" cell of the analysis notebook.
One row per subtype, one column per hemisphere x view, a single colorbar; the colour limits span all
subtypes, so colours are comparable across rows. Needs visbrain (not in the comind env) and an
OpenGL display; --show opens the interactive window instead of only saving the PNG.
"""
import argparse
import glob
import os
import re

import pandas as pd
from visbrain.io import download_file
from visbrain.objects import BrainObj, ColorbarObj, SceneObj

GRAY = (123 / 255, 123 / 255, 123 / 255)
# visbrain camera directions: looking from the left shows the left hemisphere's lateral side
ROTATE = {("left", "lateral"): "left", ("left", "medial"): "right",
          ("right", "lateral"): "right", ("right", "medial"): "left"}


def load(csv_path):
    """{'L': (rois, values), 'R': (rois, values)} from a brain_utils CSV."""
    df = pd.read_csv(csv_path)
    out = {"L": ([], []), "R": ([], [])}
    for region, val in zip(df["region"], df["FX_size"]):
        hemi, roi = region.split("_", 1)
        out[hemi][0].append(roi)
        out[hemi][1].append(float(val))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forcing_dir")
    ap.add_argument("--out", help="PNG path (default: <forcing_dir>/forcing_brains.png)")
    ap.add_argument("--hemi", choices=("both", "left", "right"), default="both")
    ap.add_argument("--views", nargs="+", choices=("lateral", "medial"), default=["lateral", "medial"])
    ap.add_argument("--cmap", default="afmhot")
    ap.add_argument("--clim", nargs=2, type=float, help="colour limits (default: min / max over all subtypes)")
    ap.add_argument("--show", action="store_true", help="open the interactive window as well")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.forcing_dir, "comind_cortical_forcing_subtype*_brainutils.csv"))
    files = sorted(files, key=lambda f: int(re.search(r"subtype(\d+)_", f).group(1)))
    if not files:
        raise SystemExit(f"no comind_cortical_forcing_subtype*_brainutils.csv in {args.forcing_dir}")
    data = [load(f) for f in files]
    hemis = ["left", "right"] if args.hemi == "both" else [args.hemi]
    if args.clim:
        clim = tuple(args.clim)
    else:
        vals = [v for d in data for h in hemis for v in d[h[0].upper()][1]]
        clim = (min(vals), max(vals))
    print(f"{len(files)} subtypes, colour limits {clim[0]:.4g} .. {clim[1]:.4g}")

    annot = {h: download_file(f"{h[0]}h.aparc.annot", astype="example_data") for h in hemis}
    panels = [(h, v) for h in hemis for v in args.views]
    sc = SceneObj(bgcolor=GRAY, size=(380 * len(panels) + 200, 330 * len(files)))
    b_obj = None
    for row, d in enumerate(data):
        for col, (h, view) in enumerate(panels):
            rois, vals = d[h[0].upper()]
            b_obj = BrainObj("inflated", hemisphere=h, translucent=False)
            b_obj.parcellize(annot[h], hemisphere=h, select=rois, data=vals, clim=clim, cmap=args.cmap)
            sc.add_to_subplot(b_obj, row=row, col=col, rotate=ROTATE[(h, view)],
                              title=f"subtype {row + 1}: {h} {view}", title_size=10)
    cb = ColorbarObj(b_obj, cblabel="forcing f", cbtxtsz=12, txtsz=10, clim=clim, cmap=args.cmap)
    sc.add_to_subplot(cb, row=0, col=len(panels), row_span=len(files), width_max=200)

    out = args.out or os.path.join(args.forcing_dir, "forcing_brains.png")
    sc.screenshot(out, autocrop=True)
    print(f"saved {out}")
    if args.show:
        sc.preview()


if __name__ == "__main__":
    main()
