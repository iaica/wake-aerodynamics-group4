"""Gaussian fit of the LES wake deficit in the moving (wake-centred) frame.

The wake centre of every frame is found as in sigma_meandering.py. Each frame
of the wake and free-stream planes is then shifted by the wake centre of that
time step and linearly interpolated onto a common grid: the original grid
points covered by all shifted frames. The time-averaged deficit in this moving
frame of reference is plotted and Gaussian profiles are fitted along the
horizontal (z' = 0) and vertical (y' = 0) lines through the wake centre, as in
sigma_fix_fit.py.

Usage:
    python scripts/les_data_analysis/sigma_moving_fit.py
    python scripts/les_data_analysis/sigma_moving_fit.py \
        --wake path/to/Flowpos_10D.nc --free path/to/Flowpos_n2D.nc \
        --t-start 1000 --t-end 4600 --stride 2 --show --normalize

Run `python scripts/les_data_analysis/sigma_moving_fit.py --help` for all options.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sigma_fix_fit import (DATA, DEFICIT_LIM, DEFICIT_LIM_NORM, REPO, U_LIM,
                           U_LIM_NORM, load_aligned, plot_gaussian_fits,
                           plot_plane)
from sigma_meandering import (plot_histogram_with_gaussian_fit_from_wake_centres,
                              wake_centre)


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
    p.add_argument("--out-dir", type=Path, default=REPO / "plots" / "sigma_moving_fit",
                   help="directory for the PNG figures")
    p.add_argument("--dpi", type=int, default=150, help="figure resolution")
    p.add_argument("--show", action="store_true",
                   help="also show the figures interactively")
    p.add_argument("--normalize", action="store_true",
                   help="also make the plots with y', z' divided by R and U by the "
                        "local free-stream U (saved with a '_norm' suffix)")
    return p.parse_args()


def load_frames(ds_wake, ds_free, t_start, t_end, stride):
    """Load the selected frames of both planes as (t, z, y) arrays."""
    U_wake = (ds_wake["U"].sel(t=slice(t_start, t_end))
              .isel(t=slice(None, None, stride))
              .squeeze("x")
              .transpose("t", "z", "y")
              .load())
    U_free = (ds_free["U"].sel(t=U_wake.t.values)
              .squeeze("x")
              .transpose("t", "z", "y")
              .load())
    return U_wake, U_free


def wake_centres(U_wake, U_free, R):
    """Wake centre (Yc, Zc) of every frame in units of R, as in sigma_meandering.py."""
    # Time-averaged, laterally averaged free-stream profile U0(z), shape (nz, 1)
    U0_prof = U_free.mean(dim=("t", "y")).values[:, None]
    yg, zg = np.meshgrid(U_wake.y.values, U_wake.z.values)
    return np.array([wake_centre(yg, zg, U_wake.isel(t=i).values, U0_prof, R=R)
                     for i in range(U_wake.sizes["t"])])


def common_grid(y, z, yc, zc):
    """Original grid points covered by every frame after shifting by (yc, zc) [m].

    A point y' of the moving frame samples the original field at y' + yc, so it
    must lie in [y_min - min(yc), y_max - max(yc)] for all frames (same for z)."""
    eps = 1e-6 * min(np.diff(y).min(), np.diff(z).min())  # avoid NaN from round-off at the edges
    y_new = y[(y >= y.min() - yc.min() + eps) & (y <= y.max() - yc.max() - eps)]
    z_new = z[(z >= z.min() - zc.min() + eps) & (z <= z.max() - zc.max() - eps)]
    return y_new, z_new


def moving_frame_mean(U, yc, zc, y_new, z_new):
    """Time average of U in the frame moving with the wake centre (yc, zc) [m]."""
    U_sum = 0.0
    for i in range(U.sizes["t"]):
        frame = U.isel(t=i).interp(y=y_new + yc[i], z=z_new + zc[i])  # linear
        U_sum = U_sum + frame.values
    return xr.DataArray(U_sum / U.sizes["t"], coords={"z": z_new, "y": y_new},
                        dims=("z", "y"), name=f"{U.name}_moving_mean")


def relabel_moving(fig, norm=False, profile_dim=None):
    """Mark the coordinates as relative to the wake centre (y', z')."""
    ax = fig.axes[0]
    if profile_dim is None:
        ax.set_xlabel("y'/R [-]" if norm else "y' [m]")
        ax.set_ylabel("z'/R [-]" if norm else "z' [m]")
    else:
        ax.set_xlabel(f"{profile_dim}'/R [-]" if norm else f"{profile_dim}' [m]")
        ax.set_title(ax.get_title().replace(", z = ", ", z' = ").replace(", y = ", ", y' = "),
                     fontsize=10)
    return fig


def plot_wake_centre_series(t, yc, zc):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, yc, label="$Y_C$ (wake centre in y)")
    ax.plot(t, zc, label="$Z_C$ (wake centre in z)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position normalized by rotor radius R [-]")
    ax.legend()
    ax.grid()
    return fig


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")  # no display needed, only the PNGs are written
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds_wake, ds_free = load_aligned(args.wake, args.free)
    R = ds_wake["R"].item()
    U_wake, U_free = load_frames(ds_wake, ds_free, args.t_start, args.t_end, args.stride)

    # 1) Wake centre of every frame; drop the frames where no centre was found
    centres = wake_centres(U_wake, U_free, R)
    valid = np.all(np.isfinite(centres), axis=1)
    if not valid.all():
        print(f"Dropping {np.sum(~valid)} of {valid.size} frames without a wake centre")
    U_wake, U_free, centres = U_wake.isel(t=valid), U_free.isel(t=valid), centres[valid]
    yc, zc = centres[:, 0] * R, centres[:, 1] * R  # [m]

    # 2)-3) Common moving-frame grid with the original resolution
    y_new, z_new = common_grid(U_wake.y.values, U_wake.z.values, yc, zc)
    print(f"Moving-frame grid: y' = {y_new.min():.1f} to {y_new.max():.1f} m, "
          f"z' = {z_new.min():.1f} to {z_new.max():.1f} m ({y_new.size} x {z_new.size} points)")

    # 4) Time-averaged fields and deficit in the moving frame
    U_mean_wake = moving_frame_mean(U_wake, yc, zc, y_new, z_new)
    U_mean_free = moving_frame_mean(U_free, yc, zc, y_new, z_new)
    U_mean_diff = U_mean_free - U_mean_wake

    ds_sel = ds_wake.sel(t=U_wake.t.values)  # for the plot titles (x position, time range)
    U_label = r"$\overline{U}$ [m/s]"
    figs = {
        "U_mean_wake": plot_plane(U_mean_wake, ds_sel, R, U_label, "viridis",
                                  "Moving-frame time-averaged U", U_LIM, rotor=False),
        "U_mean_free": plot_plane(U_mean_free, ds_sel, R, U_label, "viridis",
                                  "Moving-frame time-averaged free-stream U", U_LIM, rotor=False),
        "U_mean_deficit": plot_plane(
            U_mean_diff, ds_sel, R,
            r"$\overline{U}_{free}$ - $\overline{U}_{wake}$ [m/s]",
            "bwr", "Moving-frame time-averaged U deficit", DEFICIT_LIM, rotor=False),
    }
    for fig in figs.values():
        relabel_moving(fig)

    # 5)-6) Profiles through the wake centre (y' = 0, z' = 0) and Gaussian fits
    x_D = ds_wake.x.item() / (2 * R)
    fit_figs, fits = plot_gaussian_fits(U_mean_diff, R, x_D, rotor=False)
    for name, fig in fit_figs.items():
        relabel_moving(fig, profile_dim="y" if "horizontal" in name else "z")
    figs.update(fit_figs)

    for dim, (A, mu, sigma, r2) in fits.items():
        print(f"{dim}': A = {A:.3f} m/s, mu = {mu:.2f} m, sigma = {sigma:.2f} m = {sigma/(2*R):.3f}D, R2 = {r2:.3f}")

    if args.normalize:
        # Normalized by the local free stream, so the free-flow plane itself (= 1) is not plotted
        figs["U_mean_wake_norm"] = relabel_moving(plot_plane(
            U_mean_wake / U_mean_free, ds_sel, R,
            r"$\overline{U}_{wake} / \overline{U}_{free}$ [-]",
            "viridis", "Moving-frame time-averaged normalized U", U_LIM_NORM,
            norm=True, rotor=False), norm=True)
        figs["U_mean_deficit_norm"] = relabel_moving(plot_plane(
            U_mean_diff / U_mean_free, ds_sel, R,
            r"$(\overline{U}_{free} - \overline{U}_{wake}) / \overline{U}_{free}$ [-]",
            "bwr", "Moving-frame time-averaged normalized U deficit", DEFICIT_LIM_NORM,
            norm=True, rotor=False), norm=True)
        fit_figs_norm, fits_norm = plot_gaussian_fits(U_mean_diff / U_mean_free, R, x_D,
                                                      norm=True, rotor=False)
        for name, fig in fit_figs_norm.items():
            relabel_moving(fig, norm=True, profile_dim="y" if "horizontal" in name else "z")
        figs.update(fit_figs_norm)

        for dim, (A, mu, sigma, r2) in fits_norm.items():
            print(f"{dim}' (normalized): A = {A:.3f}, mu = {mu:.3f}R, sigma = {sigma:.3f}R, R2 = {r2:.3f}")

    # Wake-centre time series and its distribution (the histogram is saved by the function itself)
    figs["wake_centre_coordinates"] = plot_wake_centre_series(U_wake.t.values, centres[:, 0], centres[:, 1])
    plot_histogram_with_gaussian_fit_from_wake_centres(
        centres[:, 0], centres[:, 1], "Wake Centres", args.out_dir, args.wake.stem)

    for name, fig in figs.items():
        out = args.out_dir / f"{args.wake.stem}_{name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
