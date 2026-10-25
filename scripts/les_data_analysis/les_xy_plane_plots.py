"""Plots of the streamwise LES wind speed U in the hub-height x-y plane.

Saves an animation of the instantaneous U (GIF) and a plot of the
time-averaged U (PNG). Same as the hub-height x-y plane plots in
notebooks/les_flow_analysis.ipynb.

Usage (defaults use the test data in data/test_les_data):
    python scripts/les_data_analysis/les_xy_plane_plots.py
    python scripts/les_data_analysis/les_xy_plane_plots.py --hh path/to/Flowpos_hubheightplane.nc \
        --out-dir plots --anim-t-start 1000 --anim-t-end 1060 \
        --mean-t-start 1000 --mean-t-end 4600

Run `python scripts/les_data_analysis/les_xy_plane_plots.py --help` for all options.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, only the files are written
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/work3/emlh/IgnacioJoey/flowdata/")

ROTOR_THICKNESS = 20.0  # [m], visual thickness of the rotor seen from above


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--hh", type=Path,
                   default=DATA / "Flowpos_hubheightplane.nc",
                   help="hub-height x-y plane file")
    p.add_argument("--out-dir", type=Path, default=REPO / "plots",
                   help="output folder for the GIF and the PNG")
    p.add_argument("--anim-t-start", type=float, default=None,
                   help="start of the animation window [s] "
                        "(default: first step)")
    p.add_argument("--anim-t-end", type=float, default=None,
                   help="end of the animation window [s] "
                        "(default: 600 s after its start)")
    p.add_argument("--mean-t-start", type=float, default=None,
                   help="start of the averaging window [s] "
                        "(default: first step)")
    p.add_argument("--mean-t-end", type=float, default=None,
                   help="end of the averaging window [s] (default: last step)")
    p.add_argument("--stride", type=int, default=1,
                   help="use every n-th time step in the animation")
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    p.add_argument("--dpi", type=int, default=80, help="GIF resolution")
    p.add_argument("--dpi-png", type=int, default=150,
                   help="resolution of the mean plot")
    p.add_argument("--no-anim", action="store_true",
                   help="skip the animation, only save the mean plot")
    return p.parse_args()


def add_rotor(ax, R):
    """Rotor seen from above (assumes the rotor is at x = 0, y = 0)."""
    ax.add_patch(Rectangle((-ROTOR_THICKNESS / 2, -R), ROTOR_THICKNESS, 2 * R,
                           fill=False, edgecolor="red", linestyle="-",
                           linewidth=1))


def save_animation(U, R, fn_gif, fps, dpi):
    """Animation of the instantaneous U, U with dims (t, y, x)."""
    U = U.load()

    # Fixed colour limits so frames are comparable
    vmin, vmax = np.nanpercentile(U.values, [1, 99])

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(U.x, U.y, U.isel(t=0).values,
                         cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax, label="U [m/s]")
    add_rotor(ax, R)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title("")

    def update(i):
        mesh.set_array(U.isel(t=i).values.ravel())
        title.set_text(f"Instantaneous U at hub-height, "
                       f"t = {U.t[i].item():.1f} s")
        return mesh, title

    anim = FuncAnimation(fig, update, frames=U.sizes["t"], blit=False)
    anim.save(fn_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved {fn_gif} ({U.sizes['t']} frames)")


def save_mean_plot(U, R, fn_png, dpi):
    """Plot of the time-averaged U, U with dims (t, y, x)."""
    U_mean = U.mean(dim="t")

    fig, ax = plt.subplots(figsize=(10, 5))
    U_mean.plot.pcolormesh(x="x", y="y", ax=ax, cmap="viridis",
                           cbar_kwargs={"label": r"$\overline{U}$ [m/s]"})
    add_rotor(ax, R)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Time-averaged U at hub-height, "
                 f"t = {U.t[0].item():.0f}–{U.t[-1].item():.0f} s")
    fig.savefig(fn_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fn_png}")


def main():
    args = parse_args()

    ds_les_hh = xr.open_dataset(args.hh)
    R = ds_les_hh["R"].item()

    U = ds_les_hh["U"].squeeze("z").transpose("t", "y", "x")

    # Default animation window: the first ten minutes of the data
    anim_t_start = (U.t[0].item() if args.anim_t_start is None
                    else args.anim_t_start)
    anim_t_end = (anim_t_start + 600.0 if args.anim_t_end is None
                  else args.anim_t_end)
    U_anim = U.sel(t=slice(anim_t_start, anim_t_end))
    U_mean = U.sel(t=slice(args.mean_t_start, args.mean_t_end))
    for name, U_win in [("animation", U_anim), ("averaging", U_mean)]:
        if U_win.sizes["t"] == 0:
            raise SystemExit(f"No time steps in the {name} window")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_anim:
        save_animation(U_anim.isel(t=slice(None, None, args.stride)), R,
                       args.out_dir / "les_U_hubheight_animation.gif",
                       args.fps, args.dpi)
    save_mean_plot(U_mean, R, args.out_dir / "les_U_hubheight_mean.png",
                   args.dpi_png)


if __name__ == "__main__":
    main()
