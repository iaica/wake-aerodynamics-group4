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
        --out-dir plots/sigma_fit --show --normalize

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

# Fixed plot ranges, so figures at different downstream distances are comparable
U_LIM = (4.0, 10.0)            # colormap of U [m/s]
DEFICIT_LIM = (-4.0, 4.0)      # colormap of U_free - U_wake [m/s]
FIT_YLIM = (-0.5, 4.5)         # y axis of the Gaussian fit plots [m/s]
# Normalized ranges, derived from the ones above with a reference wind speed
U_REF = 10.0                   # [m/s]
DEFICIT_LIM_NORM = tuple(v / U_REF for v in DEFICIT_LIM)
U_LIM_NORM = (1 - DEFICIT_LIM_NORM[1], 1 - DEFICIT_LIM_NORM[0])  # U_wake/U_free = 1 - normalized deficit
FIT_YLIM_NORM = tuple(v / U_REF for v in FIT_YLIM)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wake", type=Path, default=DATA / "Flowpos_05D.nc",
                   help="waked y-z plane file")
    p.add_argument("--free", type=Path, default=DATA / "Flowpos_n2D.nc",
                   help="free-stream (upstream) y-z plane file")
    p.add_argument("--out-dir", type=Path, default=REPO / "plots" / "sigma_fix_fit",
                   help="directory for the PNG figures")
    p.add_argument("--dpi", type=int, default=150, help="figure resolution")
    p.add_argument("--show", action="store_true",
                   help="also show the figures interactively")
    p.add_argument("--normalize", action="store_true",
                   help="also make the plots with y, z divided by R and U by the "
                        "local free-stream U (saved with a '_norm' suffix)")
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


def to_radii(field, R):
    """Express the y and z coordinates in rotor radii."""
    return field.assign_coords(y=field.y / R, z=field.z / R)


def plot_plane(field, ds, R, cbar_label, cmap, title_prefix, clim, norm=False, rotor=True):
    if norm:
        field = to_radii(field, R)
    r_rotor, ylabel, zlabel = (1.0, "y/R [-]", "z/R [-]") if norm else (R, "y [m]", "z [m]")

    fig, ax = plt.subplots(figsize=(10, 5))
    field.plot.pcolormesh(x="y", y="z", ax=ax, cmap=cmap, vmin=clim[0], vmax=clim[1],
                          cbar_kwargs={"label": cbar_label})

    # Rotor disk outline (assumes y, z are relative to the rotor centre)
    if rotor:
        ax.add_patch(Circle((0, 0), r_rotor, fill=False, edgecolor="white",
                            linestyle="--", linewidth=1))

    ax.set_aspect("equal")
    ax.set_xlabel(ylabel)
    ax.set_ylabel(zlabel)
    ax.set_title(f"{title_prefix} at x = {(ds.x.item()/(2*R)):.0f}D, "
                 f"t = {ds.t[0].item():.0f}–{ds.t[-1].item():.0f} s")
    return fig


def plot_gaussian_fits(U_mean_diff, R, x_D, norm=False, rotor=True):
    """One figure per centerline profile (horizontal and vertical).

    With norm=True, U_mean_diff is expected to be already divided by U_free;
    the coordinates are converted to rotor radii here."""
    if norm:
        U_mean_diff = to_radii(U_mean_diff, R)
        r_rotor, fmt, unit, A_unit, suffix = 1.0, ".2f", "R", "", "_norm"
        ylabel = r"$(\overline{U}_{free} - \overline{U}_{wake}) / \overline{U}_{free}$ [-]"
        ylim = FIT_YLIM_NORM
    else:
        r_rotor, fmt, unit, A_unit, suffix = R, ".1f", " m", " m/s", ""
        ylabel = r"$\overline{U}_{free} - \overline{U}_{wake}$ [m/s]"
        ylim = FIT_YLIM

    # Centerline profiles through the rotor centre, linearly interpolated to exactly z = 0 and y = 0
    U_diff_y = U_mean_diff.interp(z=0)
    U_diff_z = U_mean_diff.interp(y=0)

    figs, fits = {}, {}
    profiles = [("horizontal", "y", U_diff_y, f"Horizontal profile at x = {x_D:.0f}D, z = {U_diff_y.z.item():{fmt}}{unit}"),
                ("vertical", "z", U_diff_z, f"Vertical profile at x = {x_D:.0f}D, y = {U_diff_z.y.item():{fmt}}{unit}")]
    for name, dim, profile, location in profiles:
        s = profile[dim].values
        popt, pcov = curve_fit(gaussian, s, profile.values, p0=[profile.max().item(), 0, r_rotor])
        A, mu, sigma = popt
        # Coefficient of determination over all fitted points
        residuals = profile.values - gaussian(s, *popt)
        r2 = 1 - np.sum(residuals ** 2) / np.sum((profile.values - profile.values.mean()) ** 2)
        fits[dim] = (A, mu, sigma, r2)

        fig, ax = plt.subplots(figsize=(7, 4))
        s_fine = np.linspace(s.min(), s.max(), 500)
        ax.plot(s, profile.values, "o", markersize=3, label="LES")
        ax.plot(s_fine, gaussian(s_fine, *popt), "-", label="Gaussian fit")
        if rotor:
            ax.axvspan(-r_rotor, r_rotor, color="grey", alpha=0.15, label="Rotor")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"{dim}/R [-]" if norm else f"{dim} [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{location}\n"
                     rf"$A$ = {A:.2f}{A_unit}, $\mu$ = {mu:{fmt}}{unit}, $\sigma$ = {sigma:{fmt}}{unit} ($R^2$ = {r2:.2f})",
                     fontsize=10)
        # Legend below the axes so it does not cover the data
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        fig.tight_layout()
        figs[f"gaussian_fit_{name}{suffix}"] = fig
    return figs, fits


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
        "U_mean_wake": plot_plane(U_mean_wake, ds_wake, R, U_label, "viridis", "Time-averaged U", U_LIM),
        "U_mean_free": plot_plane(U_mean_free, ds_free, ds_free["R"].item(), U_label, "viridis", "Time-averaged U", U_LIM),
        "U_mean_deficit": plot_plane(
            U_mean_diff, ds_wake, R,
            r"$\overline{U}_{free}$ - $\overline{U}_{wake}$ [m/s]",
            "bwr",
            "Time-averaged U deficit", DEFICIT_LIM),
    }
    x_D = ds_wake.x.item() / (2 * R)
    fit_figs, fits = plot_gaussian_fits(U_mean_diff, R, x_D)
    figs.update(fit_figs)

    for dim, (A, mu, sigma, r2) in fits.items():
        print(f"{dim}: A = {A:.3f} m/s, mu = {mu:.2f} m, sigma = {sigma:.2f} m = {sigma/(2*R):.3f}D, R2 = {r2:.3f}")

    if args.normalize:
        # Normalized by the local free stream, so the free-flow plane itself (= 1) is not plotted
        figs["U_mean_wake_norm"] = plot_plane(
            U_mean_wake / U_mean_free, ds_wake, R,
            r"$\overline{U}_{wake} / \overline{U}_{free}$ [-]",
            "viridis", "Time-averaged normalized U", U_LIM_NORM, norm=True)
        figs["U_mean_deficit_norm"] = plot_plane(
            U_mean_diff / U_mean_free, ds_wake, R,
            r"$(\overline{U}_{free} - \overline{U}_{wake}) / \overline{U}_{free}$ [-]",
            "bwr", "Time-averaged normalized U deficit", DEFICIT_LIM_NORM, norm=True)
        fit_figs_norm, fits_norm = plot_gaussian_fits(U_mean_diff / U_mean_free, R, x_D, norm=True)
        figs.update(fit_figs_norm)

        for dim, (A, mu, sigma, r2) in fits_norm.items():
            print(f"{dim} (normalized): A = {A:.3f}, mu = {mu:.3f}R, sigma = {sigma:.3f}R, R2 = {r2:.3f}")

    for name, fig in figs.items():
        out = args.out_dir / f"{args.wake.stem}_{name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
