"""
Brain figures of a fitted COMIND model: a still of each subtype's forcing terms, and a GIF / MP4 of
each subtype's regional trajectories x(t) from t = 0 to t_max. Plain matplotlib (no OpenGL), so it runs
on a laptop or a headless node.

    python plot_forcing_brains.py still  <data_dir> [--out forcing.png]
    python plot_forcing_brains.py movie  <data_dir> [--out trajectories.mp4] [--seconds 8] [--fps 20]
                                          [--hold 1 1.5] [--ease sigmoid] [--scale x|z] [--relative] [--gif]
    python plot_forcing_brains.py views  [--mesh ~/visbrain_data/templates/inflated.npz]   # one-off

<data_dir> holds forcing_all.csv, trajectories_x.csv (and trajectories_z.csv for --scale z) and
model_info.json, as written by scripts/export_brain_data.py. The surfaces come from brain_views.npz (pre-rendered region
maps of the fsaverage inflated surface + Desikan-Killiany atlas; looked for next to this script, then in
<data_dir>); `views` rebuilds it from visbrain's template data (needs nibabel).

Layout: one card per subtype (1 x Z up to 3 subtypes, 2 x 2 for 4, ...). Each card shows the cortex in the
views given by --views (default lateral and medial, both hemispheres) and, on the right, a strip with the
subcortical regions (not on the cortical surface). All cards share one colour scale.

Needs numpy, pandas and matplotlib; MP4 (default) needs ffmpeg, either on the PATH or from
`pip install imageio-ffmpeg`; GIF (--gif) needs pillow.
"""
import argparse
import json
import math
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = {  # camera direction (forward) and up vector; x = right, y = anterior, z = superior
    "left-lateral": ((1, 0, 0), (0, 0, 1), ("L",)),
    "left-medial": ((-1, 0, 0), (0, 0, 1), ("L",)),
    "right-lateral": ((-1, 0, 0), (0, 0, 1), ("R",)),
    "right-medial": ((1, 0, 0), (0, 0, 1), ("R",)),
    "dorsal": ((0, 0, -1), (0, 1, 0), ("L", "R")),
    "ventral": ((0, 0, 1), (0, 1, 0), ("L", "R")),
}
DEFAULT_VIEWS = ["left-lateral", "right-lateral", "left-medial", "right-medial"]
SUBCORTICAL = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens"]
SUB_SHORT = ["Thal", "Caud", "Put", "Pall", "Hipp", "Amyg", "Acc"]
HEMI_OFFSET = 64          # code = hemi * 64 + atlas index; -1 = background
SUFFIXES = ("_wscore", "_Z_predict", "_etivnorm_z", "_thickness", "_volume", "_z")
# "comind_hot": near-black deep red -> red -> orange -> yellow -> white (brighter = further progressed)
HOT = [(0.0, "#140000"), (0.22, "#5e0000"), (0.42, "#b80000"), (0.58, "#f52a00"), (0.72, "#ff8200"),
       (0.86, "#ffd52a"), (1.0, "#ffffff")]


def get_cmap(name):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if name == "comind_hot":
        return LinearSegmentedColormap.from_list("comind_hot", HOT, N=1024)
    return plt.get_cmap(name)


def ffmpeg_exe():
    """ffmpeg on the PATH, else the binary bundled with `pip install imageio-ffmpeg`."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit("MP4 needs ffmpeg: install it, or `pip install imageio-ffmpeg`; or pass --gif")


# ---------------------------------------------------------------- names
def parse_name(name):
    """Biomarker column -> ('ctx' | 'sub', 'L' | 'R', region) or None (clinical / unknown)."""
    n = name
    for suf in SUFFIXES:
        n = n.removesuffix(suf)
    for side, pre in (("L", "ctx-lh-"), ("R", "ctx-rh-"), ("L", "lh_"), ("R", "rh_"), ("L", "L_"), ("R", "R_")):
        if n.startswith(pre):
            return "ctx", side, n[len(pre):]
    for side, pre in (("L", "Left-"), ("R", "Right-")):
        if n.startswith(pre):
            region = n[len(pre):].removesuffix("-Proper").removesuffix("-area")
            return ("sub", side, region) if region in SUBCORTICAL else None
    return None


# ---------------------------------------------------------------- views (one-off)
def build_views(mesh_path, annot_dir, out, px=420):
    """Rasterise the inflated surface once per view into a region-code map and a shading map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from matplotlib.collections import PolyCollection

    m = np.load(mesh_path)
    verts, faces, normals, is_left = m["vertices"].astype(float), m["faces"], m["normals"].astype(float), m["lr_index"]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    code = np.full(len(verts), -1, int)
    names = {}
    for hemi, side, mask in ((0, "L", is_left), (1, "R", ~is_left)):
        labels, _, nm = nib.freesurfer.read_annot(os.path.join(annot_dir, f"{side.lower()}h.aparc.annot"))
        names[side] = [x.decode() if isinstance(x, bytes) else x for x in nm]
        if labels.shape[0] != mask.sum():
            raise ValueError(f"{side}h annot has {labels.shape[0]} vertices, mesh hemisphere {mask.sum()}")
        code[np.flatnonzero(mask)] = np.where(labels >= 0, hemi * HEMI_OFFSET + labels, -1)
    face_hemi = np.where(is_left[faces[:, 0]], "L", "R")
    saved = {"names_L": np.array(names["L"]), "names_R": np.array(names["R"])}
    for view, (fwd, up, hemis) in VIEWS.items():
        f, u = np.asarray(fwd, float), np.asarray(up, float)
        r = np.cross(f, u)
        keep = np.isin(face_hemi, hemis)
        fc = faces[keep]
        fn = normals[fc].mean(axis=1)
        front = fn @ f < 0
        fc, fn = fc[front], fn[front]
        depth = (verts[fc].mean(axis=1) @ f)
        order = np.argsort(-depth)                      # far first (painter's algorithm)
        fc, fn = fc[order], fn[order]
        xy = np.stack([verts @ r, verts @ u], axis=1)[fc]  # (F, 3, 2)
        lam = np.clip(-(fn @ f), 0, 1)
        face_code = code[fc[:, 0]]
        # R channel = code + 1 (0 = medial wall), G = shading; background is white (255)
        rgb = np.stack([(face_code + 1) / 255.0, lam, np.zeros_like(lam)], axis=1)
        lo, hi = xy.reshape(-1, 2).min(0), xy.reshape(-1, 2).max(0)
        pad = 0.03 * (hi - lo).max()
        w, h = hi - lo + 2 * pad
        W, H = px, int(round(px * h / w))
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.axis("off")
        ax.add_collection(PolyCollection(xy, facecolors=rgb, edgecolors=rgb, linewidths=0.4, antialiaseds=False))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
        plt.close(fig)
        bg = (buf == 255).all(axis=2)
        saved[f"{view}__code"] = np.where(bg, -2, buf[:, :, 0] - 1).astype(np.int16)  # -2 bg, -1 medial wall
        saved[f"{view}__shade"] = np.where(bg, 0, buf[:, :, 1]).astype(np.uint8)
        print(f"{view}: {W}x{H}px, {len(fc)} faces")
    np.savez_compressed(out, **saved)
    print(f"saved {out}")


def load_views(path):
    v = np.load(path)
    names = {"L": list(v["names_L"]), "R": list(v["names_R"])}
    return {k.split("__")[0]: (v[f"{k.split('__')[0]}__code"], v[f"{k.split('__')[0]}__shade"])
            for k in v.files if k.endswith("__code")}, names


# ---------------------------------------------------------------- data
def load_data(data_dir, scale):
    info = json.load(open(os.path.join(data_dir, "model_info.json")))
    forcing = pd.read_csv(os.path.join(data_dir, "forcing_all.csv"), index_col=0)
    traj = pd.read_csv(os.path.join(data_dir, f"trajectories_{scale}.csv"))
    return info, forcing, traj


def code_values(values, biomarkers, atlas_names):
    """Region values (one per biomarker) -> lookup table over surface codes, and subcortical (7, 2) grid."""
    lut = np.full(2 * HEMI_OFFSET, np.nan)
    sub = np.full((len(SUBCORTICAL), 2), np.nan)
    for name, val in zip(biomarkers, values):
        p = parse_name(name)
        if p is None:
            continue
        kind, side, region = p
        if kind == "ctx" and region in atlas_names[side]:
            lut[int(side == "R") * HEMI_OFFSET + atlas_names[side].index(region)] = val
        elif kind == "sub":
            sub[SUBCORTICAL.index(region), int(side == "R")] = val
    return lut, sub


# ---------------------------------------------------------------- figure
class BrainFigure:
    def __init__(self, views, view_names, n_subtypes, titles, cmap, clim, cbar_label, card_cols=2,
                 subcortical=True, width=12.0, bg="0.42", medial="0.56"):
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        self.plt = plt
        self.views, self.view_names = views, view_names
        self.cmap = get_cmap(cmap)
        self.bg, self.medial = mcolors.to_rgb(bg), mcolors.to_rgb(medial)
        fg = "white" if np.dot(self.bg, (0.299, 0.587, 0.114)) < 0.5 else "black"
        self.norm = mcolors.Normalize(*clim)
        grid_cols = n_subtypes if n_subtypes <= 3 else math.ceil(math.sqrt(n_subtypes))
        grid_rows = math.ceil(n_subtypes / grid_cols)
        card_cols = min(card_cols, len(view_names))
        card_rows = math.ceil(len(view_names) / card_cols)
        aspect = np.mean([views[v][0].shape[0] / views[v][0].shape[1] for v in view_names])
        card_w = card_cols + (0.6 if subcortical else 0)
        card_h = card_rows * aspect + 0.25
        height = width * (grid_rows * card_h) / (grid_cols * card_w) + 0.9
        self.fig = plt.figure(figsize=(width, height), facecolor=self.bg)
        outer = self.fig.add_gridspec(grid_rows, grid_cols, left=0.01, right=0.99, top=1 - 0.55 / height,
                                      bottom=0.5 / height, wspace=0.06, hspace=0.12)
        self.images, self.sub_images = [], []
        for k in range(n_subtypes):
            spec = outer[k // grid_cols, k % grid_cols]
            ratios = [1] * card_cols + ([0.32, 0.28] if subcortical else [])
            card = GridSpecFromSubplotSpec(card_rows, len(ratios), subplot_spec=spec, width_ratios=ratios,
                                           wspace=0.02, hspace=0.02)
            ims = []
            for i, v in enumerate(view_names):
                ax = self.fig.add_subplot(card[i // card_cols, i % card_cols])
                ims.append(ax.imshow(np.ones(views[v][0].shape + (3,)), interpolation="antialiased"))
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(titles[k], fontsize=11, loc="left", fontweight="bold", color=fg)
                if k == 0:
                    ax.text(0.5, 0.0, v.replace("-", " "), transform=ax.transAxes, ha="center", va="top",
                            fontsize=7, color=fg, alpha=0.7)
            self.images.append(ims)
            if subcortical:
                axs = self.fig.add_subplot(card[:, card_cols])
                self.sub_images.append(axs.imshow(np.zeros((len(SUBCORTICAL), 2)), cmap=self.cmap, norm=self.norm,
                                                  aspect="auto"))
                axs.set_xticks([0, 1], ["L", "R"], fontsize=7)
                axs.set_yticks(range(len(SUBCORTICAL)), SUB_SHORT, fontsize=8)
                axs.yaxis.tick_right()
                axs.tick_params(length=0, colors=fg)
                for s_ in axs.spines.values():
                    s_.set_visible(False)
        cax = self.fig.add_axes([0.3, 0.22 / height, 0.4, 0.13 / height])
        sm = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        cb = self.fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label, color=fg)
        cb.ax.tick_params(colors=fg)
        cb.outline.set_edgecolor(fg)
        self.suptitle = self.fig.suptitle("", fontsize=13, y=1 - 0.12 / height, color=fg)

    def render(self, view, lut):
        code, shade = self.views[view]
        val = np.full(code.shape, np.nan)
        ok = code >= 0
        val[ok] = lut[code[ok]]
        rgb = self.cmap(self.norm(val))[:, :, :3]
        rgb[~np.isfinite(val)] = self.medial              # medial wall / regions without a value
        light = 0.35 + 0.65 * (shade / 255.0)
        rgb = rgb * light[:, :, None]
        rgb[code == -2] = self.bg                        # background
        return np.clip(rgb, 0, 1)

    def update(self, luts, subs, title):
        for k, (lut, sub) in enumerate(zip(luts, subs)):
            for im, v in zip(self.images[k], self.view_names):
                im.set_data(self.render(v, lut))
            if self.sub_images:
                self.sub_images[k].set_data(sub)
        self.suptitle.set_text(title)


def ease(u, kind, steepness=8.0):
    if kind == "linear":
        return u
    s = lambda v: 1 / (1 + np.exp(-steepness * (v - 0.5)))
    return (s(u) - s(0)) / (s(1) - s(0))


def card_titles(info, n):
    counts = info.get("n_train_per_subtype") or [None] * n
    return [f"Subtype {k + 1}" + (f" (n train = {c})" if c is not None else "") for k, c in enumerate(counts)]


def find_views(arg, data_dir):
    for p in [arg, os.path.join(HERE, "brain_views.npz"), os.path.join(data_dir, "brain_views.npz")]:
        if p and os.path.exists(p):
            return p
    raise SystemExit("brain_views.npz not found: pass --views-file, or build it with `views`")


# ---------------------------------------------------------------- commands
def cmd_still(a):
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    info, forcing, _ = load_data(a.data_dir, "x")
    views, atlas = load_views(find_views(a.views_file, a.data_dir))
    Z = forcing.shape[1]
    luts, subs = zip(*[code_values(forcing.iloc[:, k].to_numpy(float), forcing.index, atlas) for k in range(Z)])
    shown = np.concatenate([np.ravel(l[np.isfinite(l)]) for l in luts] + [np.ravel(s[np.isfinite(s)]) for s in subs])
    clim = tuple(a.clim) if a.clim else (0.0, float(shown.max()))
    bf = BrainFigure(views, a.views, Z, card_titles(info, Z), a.cmap, clim, "forcing f", a.card_cols,
                     not a.no_subcortical, a.width, a.bg)
    bf.update(luts, subs, a.title or f"Forcing terms by subtype ({info.get('experiment', '')}, Z = {Z}, "
                                     f"trial {info.get('trial', '?')})")
    out = a.out or os.path.join(a.data_dir, "forcing_brains.png")
    bf.fig.savefig(out, dpi=a.dpi or 100, facecolor=bf.fig.get_facecolor())
    print(f"saved {out}  (colour limits {clim[0]:.4g} .. {clim[1]:.4g})")
    if a.show:
        bf.plt.show()


def cmd_movie(a):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import animation
    info, _, traj = load_data(a.data_dir, a.scale)
    views, atlas = load_views(find_views(a.views_file, a.data_dir))
    subtypes = sorted(traj.subtype.unique())
    biomarkers = [c for c in traj.columns if c not in ("subtype", "time")]
    curves = {k: traj[traj.subtype == k].sort_values("time") for k in subtypes}
    t_grid = curves[subtypes[0]].time.to_numpy(float)
    t0 = a.t_start if a.t_start is not None else float(t_grid[0])
    t1 = a.t_end if a.t_end is not None else float(info.get("t_max", t_grid[-1]))
    n_move = max(2, int(round(a.seconds * a.fps)))
    n_hold0, n_hold1 = (int(round(h * a.fps)) for h in a.hold)
    times = np.concatenate([np.full(n_hold0, t0), t0 + (t1 - t0) * ease(np.linspace(0, 1, n_move), a.ease),
                            np.full(n_hold1, t1)])
    if a.scale == "x":
        clim, label = (0.0, 1.0), "x(t)  (ODE state: 0 = start, 1 = plateau)"
    else:
        v = traj[biomarkers].to_numpy(float)
        lim = float(np.nanmax(np.abs(v)))
        clim, label = (-lim, lim), "predicted z(t) = s (2x - 1)  (flipped: positive = abnormal)"
    arrays = {k: curves[k][biomarkers].to_numpy(float) for k in subtypes}
    if a.relative:
        mean = np.mean([arrays[k] for k in subtypes], axis=0)
        arrays = {k: arrays[k] - mean for k in subtypes}
        lim = float(np.nanmax(np.abs([arrays[k] for k in subtypes])))
        clim = (-lim, lim)
        label = f"{'x' if a.scale == 'x' else 'z'}(t) of the subtype minus the mean over subtypes"
    if a.clim:
        clim = tuple(a.clim)
    cmap = a.cmap or ("comind_hot" if a.scale == "x" and not a.relative else "RdBu_r")
    Z = len(subtypes)
    bf = BrainFigure(views, a.views, Z, card_titles(info, Z), cmap, clim, label, a.card_cols,
                     not a.no_subcortical, a.width, a.bg)

    def frame(i):
        t = times[i]
        luts, subs = [], []
        for k in subtypes:
            vals = np.array([np.interp(t, t_grid, arrays[k][:, b]) for b in range(len(biomarkers))])
            lut, sub = code_values(vals, biomarkers, atlas)
            luts.append(lut)
            subs.append(sub)
        bf.update(luts, subs, f"disease time t = {t:5.1f} years")
        return []

    out = a.out or os.path.join(a.data_dir, f"trajectories_{a.scale}{'_relative' if a.relative else ''}."
                                            f"{'gif' if a.gif else 'mp4'}")
    if out.endswith(".gif"):
        writer = animation.PillowWriter(fps=a.fps)
    else:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe()
        # H.264 at constant quality; yuv420p plays everywhere (QuickTime, PowerPoint, browsers)
        writer = animation.FFMpegWriter(fps=a.fps, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p", "-crf", str(a.crf), "-preset", "slow"])
    anim = animation.FuncAnimation(bf.fig, frame, frames=len(times), blit=False)
    anim.save(out, writer=writer, dpi=a.dpi or 150, savefig_kwargs={"facecolor": bf.fig.get_facecolor()})
    print(f"saved {out}: {len(times)} frames at {a.fps} fps ({len(times) / a.fps:.1f} s), t {t0:g} -> {t1:g}, "
          f"{a.ease}, colour limits {clim[0]:.3g} .. {clim[1]:.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("data_dir")
    common.add_argument("--out")
    common.add_argument("--views-file", help="brain_views.npz (default: next to this script, then <data_dir>)")
    common.add_argument("--views", nargs="+", choices=list(VIEWS), default=DEFAULT_VIEWS)
    common.add_argument("--card-cols", type=int, default=2, help="views per row inside a subtype card")
    common.add_argument("--no-subcortical", action="store_true", help="drop the subcortical strip")
    common.add_argument("--cmap")
    common.add_argument("--clim", nargs=2, type=float)
    common.add_argument("--width", type=float, default=12.0, help="figure width, inches")
    common.add_argument("--dpi", type=int, help="default 100 for the still, 150 for movies")
    common.add_argument("--bg", default="0.42", help="background colour (default mid grey, so both the "
                                                    "near-black and the white ends stand out)")
    common.add_argument("--title")

    s = sp.add_parser("still", parents=[common], help="forcing terms, one card per subtype")
    s.add_argument("--show", action="store_true", help="also open a window")
    m = sp.add_parser("movie", parents=[common], help="x(t) (or z(t)) over disease time, GIF or MP4")
    m.add_argument("--scale", choices=("x", "z"), default="x")
    m.add_argument("--relative", action="store_true",
                   help="colour each subtype by its difference from the mean over subtypes at the same t")
    m.add_argument("--seconds", type=float, default=8.0, help="length of the t0 -> t1 sweep")
    m.add_argument("--fps", type=int, default=20)
    m.add_argument("--hold", nargs=2, type=float, default=(1.0, 1.5), metavar=("START", "END"),
                   help="seconds paused on the first / last frame")
    m.add_argument("--ease", choices=("sigmoid", "linear"), default="sigmoid")
    m.add_argument("--t-start", type=float)
    m.add_argument("--t-end", type=float, help="default: t_max from model_info.json")
    m.add_argument("--gif", action="store_true", help="write a GIF instead of MP4 (colour banding)")
    m.add_argument("--crf", type=int, default=16, help="H.264 quality, lower = better / bigger (18 ~ visually lossless)")
    v = sp.add_parser("views", help="rebuild brain_views.npz from visbrain's inflated template (one-off)")
    home = os.path.expanduser("~/visbrain_data")
    v.add_argument("--mesh", default=os.path.join(home, "templates", "inflated.npz"))
    v.add_argument("--annot-dir", default=os.path.join(home, "example_data"))
    v.add_argument("--px", type=int, default=420, help="width of each view, pixels")
    v.add_argument("--out", default=os.path.join(HERE, "brain_views.npz"))
    a = ap.parse_args()

    if a.cmd == "views":
        build_views(a.mesh, a.annot_dir, a.out, a.px)
    elif a.cmd == "still":
        a.cmap = a.cmap or "comind_hot"
        cmd_still(a)
    else:
        cmd_movie(a)


if __name__ == "__main__":
    main()
