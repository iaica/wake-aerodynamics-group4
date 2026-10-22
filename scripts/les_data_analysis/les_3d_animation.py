"""3D animation of the streamwise LES wind speed U.

The hub-height x-y plane is shown intersected by one or more y-z planes, and
the result is saved as a GIF. Same as the 3D view in
notebooks/les_flow_analysis.ipynb.

Usage (defaults use the test data in data/test_les_data):
    python scripts/les_data_analysis/les_3d_animation.py
    python scripts/les_data_analysis/les_3d_animation.py --hh path/to/Flowpos_hubheightplane.nc \
        --yz path/to/Flowpos_05D.nc path/to/Flowpos_10D.nc \
        --out plots/les_U_3d_animation.gif --t-start 1000 --t-end 1060

Run `python scripts/les_data_analysis/les_3d_animation.py --help` for all options.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, only the GIF is written
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "test_les_data"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--hh", type=Path,
                   default=DATA / "Flowpos_hubheightplane.nc",
                   help="hub-height x-y plane file")
    p.add_argument("--yz", type=Path, nargs="+",
                   default=[DATA / "Flowpos_n2D.nc",
                            DATA / "Flowpos_05D.nc",
                            DATA / "Flowpos_10D.nc"],
                   help="y-z plane files")
    p.add_argument("--out", type=Path,
                   default=REPO / "plots" / "les_U_3d_animation.gif",
                   help="output GIF")
    p.add_argument("--t-start", type=float, default=1000.0,
                   help="start of the time window [s]")
    p.add_argument("--t-end", type=float, default=1060.0,
                   help="end of the time window [s]")
    p.add_argument("--stride", type=int, default=1,
                   help="use every n-th time step")
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    p.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    p.add_argument("--coarsen-hh", type=int, default=3,
                   help="spatial coarsening of the x-y plane (speed)")
    p.add_argument("--coarsen-yz", type=int, default=2,
                   help="spatial coarsening of the y-z planes (speed)")
    p.add_argument("--z-stretch", type=float, default=2.0,
                   help="vertical exaggeration of the plot box")
    p.add_argument("--elev", type=float, default=30.0,
                   help="viewing elevation [deg]")
    p.add_argument("--azim", type=float, default=-55.0,
                   help="viewing azimuth [deg]")
    return p.parse_args()


def nodes_in_range(c, lo, hi, step, extra=()):
    """Every `step`-th grid point within [lo, hi], plus lo, hi and `extra`."""
    c = c[(c >= lo) & (c <= hi)][::step]
    return np.union1d(c, [lo, hi, *extra])


def quads(X, Y, Z):
    """Corner coordinates of every grid cell, shape (n_faces, 4, 3)."""
    P = np.stack([X, Y, Z], axis=-1)
    return np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]],
                    axis=2).reshape(-1, 4, 3)


def face_values(F):
    """Mean of the 4 corner values of every cell, for each time step."""
    return 0.25 * (F[..., :-1, :-1] + F[..., :-1, 1:]
                   + F[..., 1:, 1:] + F[..., 1:, :-1]).reshape(F.shape[0], -1)


def main():
    args = parse_args()

    ds_les_hh = xr.open_dataset(args.hh)
    ds_yz_list = [xr.open_dataset(fn) for fn in args.yz]
    R = ds_les_hh["R"].item()
    D = 2 * R

    # Time steps available in all planes
    t_common = ds_les_hh.t.sel(
        t=slice(args.t_start, args.t_end)).values[::args.stride]
    for ds in ds_yz_list:
        t_common = np.intersect1d(t_common, ds.t.values)
    if len(t_common) == 0:
        raise SystemExit("No time steps common to all planes in the window")
    x_cuts = [ds.x.item() for ds in ds_yz_list]

    # Common extent: the y-z planes are cut to the y and z range covered by
    # all of them, and the x-y plane to the same y range
    y_lo = max(ds.y.min().item() for ds in ds_yz_list)
    y_hi = min(ds.y.max().item() for ds in ds_yz_list)
    z_lo = max(ds.z.min().item() for ds in ds_yz_list)
    z_hi = min(ds.z.max().item() for ds in ds_yz_list)

    # Grid nodes: coarsened grid plus the intersection lines (x = x_cut in
    # the x-y plane, z = 0 in the y-z planes), so no face crosses another
    # plane. This lets matplotlib's per-face depth sorting draw the
    # intersection correctly.
    U_hh = ds_les_hh["U"].sel(t=t_common).squeeze("z")
    x_nodes = np.union1d(U_hh.x.values[::args.coarsen_hh], x_cuts)
    y_nodes_hh = nodes_in_range(U_hh.y.values, y_lo, y_hi, args.coarsen_hh)
    U_hh = (U_hh.interp(x=x_nodes, y=y_nodes_hh)
            .transpose("t", "y", "x").values)

    U_yz, yz_nodes = [], []
    for ds in ds_yz_list:
        U = ds["U"].sel(t=t_common).squeeze("x")
        y_n = nodes_in_range(U.y.values, y_lo, y_hi, args.coarsen_yz)
        z_n = nodes_in_range(U.z.values, z_lo, z_hi, args.coarsen_yz,
                             extra=[0.0])
        U_yz.append(U.interp(y=y_n, z=z_n).transpose("t", "z", "y").values)
        yz_nodes.append((y_n, z_n))

    # Geometry of all planes in a single collection
    Xh, Yh = np.meshgrid(x_nodes, y_nodes_hh)
    verts = [quads(Xh, Yh, np.zeros_like(Xh))]
    vals = [face_values(U_hh)]
    for x_cut, (y_n, z_n), U in zip(x_cuts, yz_nodes, U_yz):
        Yv, Zv = np.meshgrid(y_n, z_n)
        verts.append(quads(np.full_like(Yv, x_cut), Yv, Zv))
        vals.append(face_values(U))
    verts = np.concatenate(verts)
    vals = np.concatenate(vals, axis=1)   # (n_t, n_faces)

    # Fixed colour limits so frames are comparable
    vmin, vmax = np.nanpercentile(vals, [1, 99])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin, vmax)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=0, right=1, bottom=0.12, top=1)
    coll = Poly3DCollection(verts, facecolors=cmap(norm(vals[0])),
                            edgecolors="none", linewidths=0,
                            antialiased=False)
    ax.add_collection3d(coll)
    # The zoomed 3D box extends beyond the axes rectangle; without this the
    # planes get clipped by that (screen-aligned) rectangle
    coll.set_clip_on(False)

    # Rotor disk at x = 0 (assumes the rotor centre is at x = 0, y = 0, z = 0)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.zeros_like(theta), R * np.cos(theta), R * np.sin(theta),
            color="red", lw=1.5, zorder=10)
    for x_cut in x_cuts:
        ax.text(x_cut, y_hi, z_hi, f"{x_cut / D:.0f}D", color="black",
                zorder=10)

    ax.set_xlim(x_nodes.min(), x_nodes.max())
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()),
                       args.z_stretch * np.ptp(ax.get_zlim())), zoom=1.6)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.xaxis.labelpad = 15
    ax.set_zticks([-100, 0, 100, 200, 300, 400])
    ax.set_yticks([-400, -200, 0, 200, 400])

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="U [m/s]", orientation="horizontal",
                 shrink=0.4, pad=0.0, fraction=0.04)
    title = ax.set_title("")

    def update(i):
        coll.set_facecolor(cmap(norm(vals[i])))
        title.set_text(f"Instantaneous U, t = {t_common[i]:.1f} s")
        return coll, title

    anim = FuncAnimation(fig, update, frames=len(t_common), blit=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.out} ({len(t_common)} frames)")


if __name__ == "__main__":
    main()
