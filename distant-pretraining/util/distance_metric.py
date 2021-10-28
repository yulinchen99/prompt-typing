import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

def js_div(p, q):
    kldiv = nn.KLDivLoss(reduction='none')
    log_mean = ((p+q) / 2).log()
    sim = (kldiv(log_mean, p)+kldiv(log_mean, q)) / 2
    sim = sim.sum(1)
    return sim

def ws_dis(u_values, v_values):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:
    .. math::
        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.
    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    Returns
    -------
    distance : float
        The computed distance between the distributions.
    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.
    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.
    """
    sorted_u_values, u_sorter = torch.sort(u_values, dim=1)
    sorted_v_values, v_sorter = torch.sort(v_values, dim=1)

    all_values = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values, dim=1)
    # all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values, dim=1)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = sorted_u_values.searchsorted(all_values[:-1], right=True)
    v_cdf_indices = sorted_v_values.searchsorted(all_values[:-1], right=True)

    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices / u_values.size(-1)
    v_cdf = v_cdf_indices / v_values.size(-1)

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    return torch.mean(torch.sum(torch.mul(torch.abs(u_cdf - v_cdf), deltas), dim=1))