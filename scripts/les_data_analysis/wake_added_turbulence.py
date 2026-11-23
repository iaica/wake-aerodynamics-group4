"""Wake-added turbulence of the LES wake in the fixed and moving frame of reference.

For each velocity component (u, v, w) the standard deviation in time (about the
mean over the selected window, no detrending) is computed for the waked and the
free-stream (upstream) planes, and the wake-added standard deviation

    sigma_add = sqrt(sigma_wake^2 - sigma_free^2)

is formed; points where sigma_wake < sigma_free are set to NaN. This is done

- in the fixed frame, on the wake grid (free stream interpolated onto it), and
- in the moving (wake-centred) frame: every frame of both planes is shifted by
  the wake centre of that time step (found from U as in sigma_meandering.py)
  and interpolated onto the common grid, as in sigma_moving_fit.py.

Maps of sigma_free, sigma_wake and sigma_add are plotted for both frames. For
each frame, sigma_add and (in a separate figure) sigma_wake and sigma_free are
plotted along the horizontal and vertical lines through the rotor centre
(fixed) and through the wake centre (moving).

Usage:
    python scripts/les_data_analysis/wake_added_turbulence.py
    python scripts/les_data_analysis/wake_added_turbulence.py \
        --wake path/to/Flowpos_10D.nc --free path/to/Flowpos_n2D.nc \
        --t-start 1000 --t-end 4600 --stride 2 --show

Run `python scripts/les_data_analysis/wake_added_turbulence.py --help` for all options.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sigma_fix_fit import load_aligned, plot_plane
from sigma_moving_fit import (common_grid, load_frames, moving_frame_std,
                              relabel_moving, wake_centres)

REPO = Path(__file__).resolve().parents[2]
DATA = Path(r"C:/Users/iaica/OneDrive - Vestas Wind Systems A S/PhD/courses/Wake aerodynamics_special course/wake-aerodynamics-group4/data/test_les_data")
#DATA = Path("/work3/emlh/IgnacioJoey/flowdata/")

COMPONENTS = {"u": "U", "v": "V", "w": "W"}  # component label -> variable in the files

# Fixed colormap ranges, shared by all components, frames and downstream distances
SIGMA_LIM = (0.0, 2.0)       # sigma_free and sigma_wake [m/s]
SIGMA_ADD_LIM = (0.0, 2.0)   # sigma_add [m/s]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wake", type=Path, default=DATA / "Flowpos_05D.nc",
                   help="waked y-z plane file")
    p.add_argument("--free", type=Path, default=DATA / "Flowpos_n2D.nc",
                   help="free-stream (upstream) y-z plane file")
    p.add_argument("--t-start", type=float, default=1000, help="first time [s]")
    p.add_argument("--t-end", type=float, default=4600, help="last time [s]")
    p.add_argument("--stride", type=int, default=1, help="use every n-th time step")
    p.add_argument("--out-dir", type=Path, default=REPO / "plots" / "wake_added_turbulence",
                   help="directory for the PNG figures")
    p.add_argument("--dpi", type=int, default=150, help="figure resolution")
    p.add_argument("--show", action="store_true",
                   help="also show the figures interactively")
    return p.parse_args()


def added_sigma(sigma_wake, sigma_free):
    """sqrt(sigma_wake^2 - sigma_free^2); NaN where the wake is less turbulent."""
    d_var = sigma_wake ** 2 - sigma_free ** 2
    return np.sqrt(d_var.where(d_var >= 0))


def plot_add_profiles(add, c, R, x_D, frame):
    """sigma_add along the horizontal and vertical lines of one frame.

    fixed: through the rotor centre (z = 0 / y = 0); moving: through the wake
    centre (z' = 0 / y' = 0)."""
    moving = frame == "moving"
    p = "'" if moving else ""
    centre = "wake centre" if moving else "rotor centre"
    figs = {}
    for name, dim, other in [("horizontal", "y", "z"), ("vertical", "z", "y")]:
        prof = add.interp({other: 0})
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(prof[dim], prof, "o-", markersize=3,
                label=rf"Wake-added $\sigma_{{{c}}}$")
        if not moving:
            ax.axvspan(-R, R, color="grey", alpha=0.15, label="Rotor")
        ax.set_ylim(SIGMA_ADD_LIM)
        ax.set_xlabel(f"{dim}{p} [m]")
        ax.set_ylabel(rf"$\sigma_{{{c},add}}$ [m/s]")
        ax.set_title(f"{frame.capitalize()}-frame {name} profile of wake-added $\\sigma_{c}$ at "
                     f"x = {x_D:.0f}D, {other}{p} = 0 ({centre})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid()
        figs[f"sigma_add_{c}_profile_{name}_{frame}"] = fig
    return figs


def plot_sigma_profiles(fields, c, R, x_D, frame):
    """sigma_wake and sigma_free along the horizontal and vertical lines of one frame.

    fixed: through the rotor centre (z = 0 / y = 0); moving: through the wake
    centre (z' = 0 / y' = 0)."""
    moving = frame == "moving"
    p = "'" if moving else ""
    centre = "wake centre" if moving else "rotor centre"
    figs = {}
    for name, dim, other in [("horizontal", "y", "z"), ("vertical", "z", "y")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for qty, marker, label in [("wake", "o-", "Wake plane"),
                                   ("free", "s-", "Free-stream plane")]:
            prof = fields[qty].interp({other: 0})
            ax.plot(prof[dim], prof, marker, markersize=3,
                    label=rf"{label} $\sigma_{{{c}}}$")
        if not moving:
            ax.axvspan(-R, R, color="grey", alpha=0.15, label="Rotor")
        ax.set_ylim(SIGMA_LIM)
        ax.set_xlabel(f"{dim}{p} [m]")
        ax.set_ylabel(rf"$\sigma_{{{c}}}$ [m/s]")
        ax.set_title(f"{frame.capitalize()}-frame {name} profile of $\\sigma_{c}$ at "
                     f"x = {x_D:.0f}D, {other}{p} = 0 ({centre})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid()
        figs[f"sigma_{c}_profile_{name}_{frame}"] = fig
    return figs


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")  # no display needed, only the PNGs are written
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds_wake, ds_free = load_aligned(args.wake, args.free)
    R = ds_wake["R"].item()
    x_D = ds_wake.x.item() / (2 * R)

    # Wake centre of every frame from U; drop the frames where no centre was found
    U_wake, U_free = load_frames(ds_wake, ds_free, args.t_start, args.t_end, args.stride)
    centres = wake_centres(U_wake, U_free, R)
    valid = np.all(np.isfinite(centres), axis=1)
    if not valid.all():
        print(f"Dropping {np.sum(~valid)} of {valid.size} frames without a wake centre")
    t_valid = U_wake.t.values[valid]
    yc, zc = centres[valid, 0] * R, centres[valid, 1] * R  # [m]

    y_new, z_new = common_grid(U_wake.y.values, U_wake.z.values, yc, zc)
    print(f"Moving-frame grid: y' = {y_new.min():.1f} to {y_new.max():.1f} m, "
          f"z' = {z_new.min():.1f} to {z_new.max():.1f} m ({y_new.size} x {z_new.size} points)")

    ds_sel = ds_wake.sel(t=t_valid)  # for the plot titles (x position, time range)
    for c, var in COMPONENTS.items():
        figs = {}
        if var == "U":
            f_wake, f_free = U_wake.isel(t=valid), U_free.isel(t=valid)
        else:
            f_wake, f_free = load_frames(ds_wake, ds_free, args.t_start, args.t_end,
                                         args.stride, var=var)
            f_wake, f_free = f_wake.sel(t=t_valid), f_free.sel(t=t_valid)

        sigma = {
            "fixed": {"wake": f_wake.std("t"), "free": f_free.std("t")},
            "moving": {"wake": moving_frame_std(f_wake, yc, zc, y_new, z_new),
                       "free": moving_frame_std(f_free, yc, zc, y_new, z_new)},
        }
        for frame in sigma.values():
            frame["add"] = added_sigma(frame["wake"], frame["free"])

        titles = {"free": f"free-stream $\\sigma_{c}$", "wake": f"$\\sigma_{c}$",
                  "add": f"wake-added $\\sigma_{c}$"}
        for frame, fields in sigma.items():
            moving = frame == "moving"
            for qty, field in fields.items():
                label = (rf"$\sigma_{{{c},add}}$ [m/s]" if qty == "add"
                         else rf"$\sigma_{{{c}}}$ [m/s]")
                fig = plot_plane(field, ds_sel, R, label, "viridis",
                                 f"{frame.capitalize()}-frame {titles[qty]}",
                                 SIGMA_ADD_LIM if qty == "add" else SIGMA_LIM,
                                 rotor=not moving)
                figs[f"sigma_{qty}_{c}_{frame}"] = relabel_moving(fig) if moving else fig

        for frame, fields in sigma.items():
            figs.update(plot_add_profiles(fields["add"], c, R, x_D, frame))
            figs.update(plot_sigma_profiles(fields, c, R, x_D, frame))

        # Save per component, so at most 14 figures are open without --show
        for name, fig in figs.items():
            out = args.out_dir / f"{args.wake.stem}_{name}.png"
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {out}")
            if not args.show:
                plt.close(fig)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
