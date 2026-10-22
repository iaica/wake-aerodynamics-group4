"""Gaussian fit of the time-averaged LES wake deficit in a y-z plane.

The free-stream (upstream) plane is cut to the extent of the wake plane and
interpolated onto its grid. The time-averaged U fields and their difference
(the deficit) are plotted, and Gaussian profiles are fitted to the deficit
along the horizontal (z = 0) and vertical (y = 0) lines through the rotor
centre. Same as notebooks/sigma_fix_fit.ipynb.

Usage (defaults use the test data in data/test_les_data):
    python scripts/les_data_analysis/sigma_fix_fit.py
    python scripts/les_data_analysis/sigma_fix_fit.py \
        --wake path/to/Flowpos_10D.nc --free path/to/Flowpos_n2D.nc \
        --out-dir plots/sigma_fit --show

Run `python scripts/les_data_analysis/sigma_fix_fit.py --help` for all options.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Circle
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/work3/emlh/IgnacioJoey/flowdata/")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wake", type=Path, default=DATA / "Flowpos_10D.nc",
                   help="waked y-z plane file")
    p.add_argument("--free", type=Path, default=DATA / "Flowpos_n2D.nc",
                   help="free-stream (upstream) y-z plane file")
    p.add_argument("--out-dir", type=Path, default=REPO / "plots" / "sigma_fix_fit",
                   help="directory for the PNG figures")
    p.add_argument("--dpi", type=int, default=150, help="figure resolution")
    p.add_argument("--show", action="store_true",
                   help="also show the figures interactively")
    return p.parse_args()


def gaussian(s, A, mu, sigma):
    return A * np.exp(-0.5 * ((s - mu) / sigma) ** 2)  # Force mu to be 0? wake center at rotor center


def load_aligned(fn_wake, fn_free):
    """Open both planes, cut them to a common y-z extent and put the free
    stream on the wake grid."""
    ds_wake = xr.open_dataset(fn_wake)
    ds_free = xr.open_dataset(fn_free)

    y_min = max(ds_wake.y.min().item(), ds_free.y.min().item())
    y_max = min(ds_wake.y.max().item(), ds_free.y.max().item())
    z_min = max(ds_wake.z.min().item(), ds_free.z.min().item())
    z_max = min(ds_wake.z.max().item(), ds_free.z.max().item())

    ds_wake = ds_wake.sel(y=slice(y_min, y_max), z=slice(z_min, z_max))
    ds_free = ds_free.sel(y=slice(y_min, y_max), z=slice(z_min, z_max))

    # The y spacing differs slightly between the files, so interpolate the free-stream data onto the wake grid
    ds_free = ds_free.interp(y=ds_wake.y, z=ds_wake.z)
    return ds_wake, ds_free


def plot_plane(field, ds, R, cbar_label, cmap, title_prefix):
    fig, ax = plt.subplots(figsize=(10, 5))
    field.plot.pcolormesh(x="y", y="z", ax=ax, cmap=cmap,
                          cbar_kwargs={"label": cbar_label})

    # Rotor disk outline (assumes y, z are relative to the rotor centre)
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor="white",
                        linestyle="--", linewidth=1))

    ax.set_aspect("equal")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"{title_prefix} at x = {(ds.x.item()/(2*R)):.0f}D, "
                 f"t = {ds.t[0].item():.0f}–{ds.t[-1].item():.0f} s")
    return fig


def plot_gaussian_fits(U_mean_diff, R):
    # Centerline profiles through the rotor centre, linearly interpolated to exactly z = 0 and y = 0
    U_diff_y = U_mean_diff.interp(z=0)
    U_diff_z = U_mean_diff.interp(y=0)

    fits = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    profiles = [("y", U_diff_y, f"Horizontal profile at z = {U_diff_y.z.item():.1f} m"),
                ("z", U_diff_z, f"Vertical profile at y = {U_diff_z.y.item():.1f} m")]
    for ax, (dim, profile, location) in zip(axes, profiles):
        s = profile[dim].values
        popt, pcov = curve_fit(gaussian, s, profile.values, p0=[profile.max().item(), 0, R])
        fits[dim] = popt
        A, mu, sigma = popt

        s_fine = np.linspace(s.min(), s.max(), 500)
        ax.plot(s, profile.values, "o", markersize=3, label="LES")
        ax.plot(s_fine, gaussian(s_fine, *popt), "-", label="Gaussian fit")
        ax.axvspan(-R, R, color="grey", alpha=0.15, label="Rotor")
        ax.set_xlabel(f"{dim} [m]")
        ax.set_title(f"{location}\n"
                     rf"$A$ = {A:.2f} m/s, $\mu$ = {mu:.1f} m, $\sigma$ = {sigma:.1f} m",
                     fontsize=10)

    axes[0].set_ylabel(r"$\overline{U}_{free} - \overline{U}_{wake}$ [m/s]")
    # Single legend for both panels, below the axes so it does not cover the data
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, fits


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")  # no display needed, only the PNGs are written
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds_wake, ds_free = load_aligned(args.wake, args.free)
    R = ds_wake["R"].item()

    U_mean_wake = ds_wake["U"].mean(dim="t").squeeze("x")
    U_mean_free = ds_free["U"].mean(dim="t").squeeze("x")
    U_mean_diff = U_mean_free - U_mean_wake

    U_label = r"$\overline{U}$ [m/s]"
    figs = {
        "U_mean_wake": plot_plane(U_mean_wake, ds_wake, R, U_label, "viridis", "Time-averaged U"),
        "U_mean_free": plot_plane(U_mean_free, ds_free, ds_free["R"].item(), U_label, "viridis", "Time-averaged U"),
        "U_mean_deficit": plot_plane(
            U_mean_diff, ds_wake, R,
            r"$\overline{U}_{free}$ - $\overline{U}_{wake}$ [m/s]",
            "bwr",
            "Time-averaged U deficit"),
    }
    figs["gaussian_fit"], fits = plot_gaussian_fits(U_mean_diff, R)

    for dim, (A, mu, sigma) in fits.items():
        print(f"{dim}: A = {A:.3f} m/s, mu = {mu:.2f} m, sigma = {sigma:.2f} m = {sigma/(2*R):.3f}D")

    for name, fig in figs.items():
        out = args.out_dir / f"{args.wake.stem}_{name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
