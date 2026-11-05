"""Fixed-frame wake deficit predicted by convolving the moving-frame deficit with meandering.

The Gaussian fit of the moving-frame deficit (sigma_moving_fit.py) is convolved
with the Gaussian fit of the wake-centre distribution (sigma_meandering.py). The
convolution is 1D, along the direction s of each profile, and is again a Gaussian:

    mu_s      = mu_m,s + mu_c,s
    sigma_s^2 = sigma_m,s^2 + sigma_c,s^2
    A_conv    = A_s * sigma_m,s / sigma_s

where m denotes the moving-frame deficit fit and c the wake-centre distribution.
The result is added to the gaussian_fit_horizontal and gaussian_fit_vertical
plots of sigma_fix_fit.py. All quantities are computed from the same frames.

Usage:
    python scripts/les_data_analysis/convolution_gaussian.py
    python scripts/les_data_analysis/convolution_gaussian.py \
        --wake path/to/Flowpos_10D.nc --free path/to/Flowpos_n2D.nc \
        --t-start 1000 --t-end 4600 --stride 2 --show

Run `python scripts/les_data_analysis/convolution_gaussian.py --help` for all options.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sigma_fix_fit import gaussian, load_aligned, plot_gaussian_fits
from sigma_meandering import plot_histogram_with_gaussian_fit_from_wake_centres
from sigma_moving_fit import common_grid, load_frames, moving_frame_mean, wake_centres


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
    p.add_argument("--t-start", type=float, default=1000, help="first time [s]")
    p.add_argument("--t-end", type=float, default=4600, help="last time [s]")
    p.add_argument("--stride", type=int, default=1, help="use every n-th time step")
    p.add_argument("--out-dir", type=Path, default=REPO / "plots" / "gaussian_convolution",
                   help="directory for the PNG figures")
    p.add_argument("--dpi", type=int, default=150, help="figure resolution")
    p.add_argument("--show", action="store_true",
                   help="also show the figures interactively")
    return p.parse_args()


def convolve_fits(fits_moving, meandering):
    """1D convolution of the moving-frame deficit fits with the meandering PDF.

    fits_moving: {dim: (A, mu, sigma, r2)} from plot_gaussian_fits [m/s, m].
    meandering: {dim: (mu, sigma)} of the wake centre [m].
    Returns {dim: (A, mu, sigma)} of the profile along each dim."""
    conv = {}
    for dim in "yz":
        A_m, mu_m, sigma_m, _ = fits_moving[dim]
        mu_c, sigma_c = meandering[dim]
        sigma = np.hypot(sigma_m, sigma_c)
        conv[dim] = (A_m * sigma_m / sigma, mu_m + mu_c, sigma)
    return conv


def add_convolution_curve(fig, A, mu, sigma):
    """Add the convolved Gaussian to a figure made by plot_gaussian_fits."""
    ax = fig.axes[0]
    s_fine = ax.lines[1].get_xdata()  # same points as the fixed-frame fit curve
    ax.plot(s_fine, gaussian(s_fine, A, mu, sigma), "-", label="Convolution")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    ax.set_title(f"{ax.get_title()}\n"
                 rf"Convolution: $A$ = {A:.2f} m/s, $\mu$ = {mu:.1f} m, $\sigma$ = {sigma:.1f} m",
                 fontsize=10)
    fig.tight_layout()


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")  # no display needed, only the PNGs are written
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds_wake, ds_free = load_aligned(args.wake, args.free)
    R = ds_wake["R"].item()
    x_D = ds_wake.x.item() / (2 * R)
    U_wake, U_free = load_frames(ds_wake, ds_free, args.t_start, args.t_end, args.stride)

    # Wake centre of every frame; the frames without one are dropped everywhere
    centres = wake_centres(U_wake, U_free, R)
    valid = np.all(np.isfinite(centres), axis=1)
    if not valid.all():
        print(f"Dropping {np.sum(~valid)} of {valid.size} frames without a wake centre")
    U_wake, U_free, centres = U_wake.isel(t=valid), U_free.isel(t=valid), centres[valid]
    yc, zc = centres[:, 0] * R, centres[:, 1] * R  # [m]

    # Gaussian fits of the deficit in the moving frame (only the values are needed)
    y_new, z_new = common_grid(U_wake.y.values, U_wake.z.values, yc, zc)
    U_diff_moving = (moving_frame_mean(U_free, yc, zc, y_new, z_new)
                     - moving_frame_mean(U_wake, yc, zc, y_new, z_new))
    figs_moving, fits_moving = plot_gaussian_fits(U_diff_moving, R, x_D, rotor=False)
    for fig in figs_moving.values():
        plt.close(fig)

    # Gaussian fit of the meandering distribution (in R), which also saves the histogram
    mu_y, sigma_y, mu_z, sigma_z = plot_histogram_with_gaussian_fit_from_wake_centres(
        centres[:, 0], centres[:, 1], "Wake Centres", args.out_dir, args.wake.stem)
    meandering = {"y": (mu_y * R, sigma_y * R), "z": (mu_z * R, sigma_z * R)}

    # Fixed-frame LES profiles and fits, with the convolution on top
    U_diff_fixed = U_free.mean(dim="t") - U_wake.mean(dim="t")
    figs, fits_fixed = plot_gaussian_fits(U_diff_fixed, R, x_D)
    conv = convolve_fits(fits_moving, meandering)
    add_convolution_curve(figs["gaussian_fit_horizontal"], *conv["y"])
    add_convolution_curve(figs["gaussian_fit_vertical"], *conv["z"])

    for dim in "yz":
        A_fix, mu_fix, sigma_fix, r2 = fits_fixed[dim]
        A_mov, mu_mov, sigma_mov, _ = fits_moving[dim]
        A_con, mu_con, sigma_con = conv[dim]
        print(f"{dim} fixed fit:   A = {A_fix:.3f} m/s, mu = {mu_fix:.2f} m, sigma = {sigma_fix:.2f} m, R2 = {r2:.3f}")
        print(f"{dim} moving fit:  A = {A_mov:.3f} m/s, mu = {mu_mov:.2f} m, sigma = {sigma_mov:.2f} m")
        print(f"{dim} meandering:  mu = {meandering[dim][0]:.2f} m, sigma = {meandering[dim][1]:.2f} m")
        print(f"{dim} convolution: A = {A_con:.3f} m/s, mu = {mu_con:.2f} m, sigma = {sigma_con:.2f} m")

    for name, fig in figs.items():
        out = args.out_dir / f"{args.wake.stem}_{name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
