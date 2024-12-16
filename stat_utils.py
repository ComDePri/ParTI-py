import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.linalg import sqrtm


def confidence_ellipse_by_cov(cov, mean, ax, n_std=3.0, edgecolor='b', facecolor='none', **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor=edgecolor, facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipse_3d_by_cov(cov, mean, ax, n_std=3.0, facecolor='b', **kwargs):
    cov_3d = cov[:3,:3]
    r_cov = sqrtm(cov_3d)
    s = 100
    u = np.linspace(0, 2 * np.pi, s)
    v = np.linspace(0, np.pi, s)

    x = np.outer(np.cos(u), np.sin(v)).flatten()
    y = np.outer(np.sin(u), np.sin(v)).flatten()
    z = np.outer(np.ones_like(v), np.cos(v)).flatten()

    sph = np.stack([x,y,z]) * n_std # [3, 100]
    sph_cov = r_cov @ sph + mean[:3, None]
    sph_cov = sph_cov.reshape((3,s,s))
    ax.plot_surface(*sph_cov, rstride=4, cstride=4, color=facecolor, alpha=0.5)


def confidence_ellipse(x, y, ax, n_std=3.0, edgecolor='b', facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    return confidence_ellipse_by_cov(cov,
                                     [np.mean(x), np.mean(y)],
                                     ax=ax,
                                     n_std=n_std,
                                     edgecolor=edgecolor,
                                     facecolor=facecolor,
                                     **kwargs)


def fdr_bh(p_vals: np.ndarray, alpha: float):
    """
    False discovery rate correction for multiple hypothesis testing using the Benjamini-Hochberg correction
    as described here: https://en.wikipedia.org/wiki/False_discovery_rate
    :param p_vals: array of N p-values for N independent or positively correlated tests
    :param alpha: threshold for false discovery rate
    :return: boolean array of size N indicating for which tests H0 is still rejected after correction.
    """
    p_vals = p_vals.flatten()
    n = p_vals.size
    p_sorted_indices = np.argsort(p_vals)
    pk = p_vals[p_sorted_indices]
    crit = (np.arange(1, n + 1) / n) * alpha
    valid_k = np.nonzero(pk <= crit)[0]
    if valid_k.size == 0:
        return np.zeros((n,), dtype=bool)
    max_k = valid_k[-1]
    rejected_indices = p_sorted_indices[:max_k+1]
    rejected = np.zeros((n,))
    rejected[rejected_indices] = 1.
    return rejected > 0

