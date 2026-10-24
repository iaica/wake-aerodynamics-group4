import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats as stats

def main():

    # Load LES data
    fn = "./data/test_les_data/Flowpos_05D.nc"
    ds_les = xr.open_dataset(fn)

    # Define time interval for analysis
    t_start = 1000
    t_end = 1060

    # Define reference velocity and rotor radius
    U0 = 8.0  # [m/s] reference (freestream)
    R = ds_les["R"].item()  # [m] rotor radius

    stride = 1  # use every n-th time step

    # Load only the frames needed into memory; fix dim order so values are (t, z, y)
    U = (ds_les["U"].sel(t=slice(t_start, t_end))
          .isel(t=slice(None, None, stride))
          .squeeze("x")
          .transpose("t", "z", "y")
          .load())

    # Create meshgrid for y and z coordinates
    yg, zg = np.meshgrid(U.y.values, U.z.values)
    centers = np.array([wake_centre(yg, zg, U.isel(t=i).values, U0, R=R) for i in range(U.sizes["t"])])

    # Plot the wake centres
    plt.figure(figsize=(8, 6))
    plt.plot(U.t.values, centers[:, 0], label='Yc (wake centre in y)')
    plt.plot(U.t.values, centers[:, 1], label='Zc (wake centre in z)')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.grid()
    plt.show()

    # yc, zc: wake-centre time series (e.g. in units of R), NaN where no centre was found
    yc = centers[:, 0]
    zc = centers[:, 1]

    mu, sigma = plot_histogram_with_gaussian_fit_from_wake_centres(yc, zc, "Wake Centres")


def plot_histogram_with_gaussian_fit_from_wake_centres(yc,zc, label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, s, label in zip(axes, [yc, zc], ["$Y_C/R$", "$Z_C/R$"]):
        s = s[~np.isnan(s)]                      # drop frames where the centre was outside the box
        mu, sigma = s.mean(), s.std(ddof=1)

        ax.hist(s, bins="auto", density=True, alpha=0.5, label="LES")
        x = np.linspace(s.min(), s.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-",
                label=fr"Gaussian, $\mu$={mu:.3f}, $\sigma$={sigma:.3f}")

        ax.set_xlabel(label)
        ax.legend(fontsize=8)

        print(f"{label}: mu = {mu:.3f}, sigma = {sigma:.3f}")

    axes[0].set_ylabel("Probability density [-]")
    plt.tight_layout()
    plt.show()

    return mu, sigma

def wake_centre(Y, Z, U, U0, y_hub=0.0, z_hub=0.0, R=1.0,
                m=3, half_width_y=3.0, half_width_z=1.7,
                tol=1e-3, max_iter=100,
                clip_negative=True, dA=None):
    """
    Find the centre of a wake in a 2D flow field.

    Parameters:
    -----------
    Y : array-like
        Array of y-coordinates.
    Z : array-like
        Array of z-coordinates.
    U : array-like
        Array of velocity values.
    U0 : float
        Reference (freestream) velocity.
    y_hub : float, optional
        y-coordinate of the rotor hub (default: 0.0).
    z_hub : float, optional
        z-coordinate of the rotor hub (default: 0.0).
    R : float, optional
        Rotor radius (default: 1.0).
    m : int, optional
        Power to which the deficit is raised (default: 3).
    half_width_y : float, optional
        Half-width of the search window in y-direction (default: 3.0).
    half_width_z : float, optional
        Half-width of the search window in z-direction (default: 1.7).
    tol : float, optional
        Tolerance for convergence (default: 1e-3).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    clip_negative : bool, optional
        Whether to clip negative deficits to zero (default: True).
    dA : array-like, optional
        Array of cell areas. If None, it will be calculated from the grid spacing (default: None).

    Returns:
    --------
    Yc : float
        y-coordinate of the wake centre.
    Zc : float
        z-coordinate of the wake centre.
    """

    # Input arrays are assumed to be 2D, with Y varying along axis 1 and Z along axis 0
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    U = np.asarray(U, dtype=float)

    if dA is None:
        # Check if Y and Z are 2D arrays; if so, calculate cell areas from the grid spacing
        if Y.ndim == 2:
            # Calculate cell areas from the grid spacing
            dy = np.abs(np.gradient(Y, axis=1)) # 2D array of cell widths in y-direction
            dz = np.abs(np.gradient(Z, axis=0)) # 2D array of cell heights in z-direction
            dA = dy * dz                        # 2D array of cell areas
        # If Y and Z are not 2D arrays, use a default value for dA
        else:
            dA = np.ones_like(U)                

    # Normalize coordinates relative to the hub and rotor radius
    y = (Y - y_hub) / R
    z = (Z - z_hub) / R

    # Compute the wake deficit
    deficit = 1.0 - U / U0
    # Clip negative deficits to zero if requested
    if clip_negative:
        deficit = np.clip(deficit, 0.0, None)

    # Compute the weighted sum
    w_all = deficit ** m * dA

    # Find the wake centre by iteratively updating its position
    Yc, Zc = 0.0, 0.0
    # Search within a fixed window around the initial guess
    for _ in range(max_iter):
        # Create a mask for points within the search box centered at (Yc, Zc)
        mask = (np.abs(y - Yc) <= half_width_y) & (np.abs(z - Zc) <= half_width_z)
        w = w_all[mask]
        # Compute the weighted sum of the mask
        wsum = w.sum()

        # Compute the jump in the wake centre position based on the weighted average of the masked points
        dY = np.sum((y[mask] - Yc) * w) / wsum
        dZ = np.sum((z[mask] - Zc) * w) / wsum

        # Update the wake centre position
        Yc += dY
        Zc += dZ

        # Check for convergence
        if abs(dY) < tol and abs(dZ) < tol:
            break

    return Yc, Zc


if __name__ == "__main__":
    main()