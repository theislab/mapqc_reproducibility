import numpy as np


def calculate_neighborhood_weights(neighbor_idc_of_sampled_cells, n_obs):
    """calculate the weight for each index cell's neighborhood, depending on
    the number of times the cells in the Nhood are also present in other
    Nhoods. Specifically, a cells is weighed as 1/(k-1)/(present_in_n_Nhoods).
    Thus, if all cells in the Nhood are only present in that single Nhood,
    the total weight of the Nhood will be 1.
    Arguments:
    neighbor_idc_of_sampled_cells - np.array of shape n_cells_sampled * (k-1)
    specifying the cell indices of the cells in a given index cell's Nhood
    n_obs - integer, total number of cells in the data
    Returns:
    np.array of size n_cells_sampled specifying the weight per neighborhood
    """
    n_cells_sampled = neighbor_idc_of_sampled_cells.shape[0]
    k = neighbor_idc_of_sampled_cells.shape[1] + 1
    # check in how many Nhoods each cell in the data is present:
    # first generate a matrix with only zeros of size (n_cells_sampled) x (n_cells_in_total)
    neighbor_matrix = np.zeros((n_cells_sampled, n_obs))
    # now for each sampled cell, set cells in its neighborhood to 1:
    neighbor_matrix[
        np.arange(n_cells_sampled)[:, np.newaxis], neighbor_idc_of_sampled_cells
    ] = 1
    # count the number of neighborhoods each cells occurs in:
    n_Nhoods_per_cell = neighbor_matrix.sum(axis=0)
    # now we want to calculate how much weight we will give to each Nhood.
    # The more cells the Nhood has that are also present in other Nhoods, the less
    # heavily we want to weigh that Nhood.
    # If the cells from an Nhood only occur in one Nhood, they will weigh
    # 1/(Nhood size), so that the overall Nhood weight sums up to 1:
    normal_cell_weight = 1 / (k - 1)
    # If a cell occurs in multiple N hoods, we want to spread its weight out over
    # all those neighborhoods, i.e. set the weight to normal_cell_weight/(n_Nhoods)
    # let's first set the standard weights
    cell_weights_standard = np.full(shape=(n_obs), fill_value=normal_cell_weight)
    # now we decrease the weights as described above:
    cell_weights = np.divide(
        cell_weights_standard, n_Nhoods_per_cell, where=n_Nhoods_per_cell > 0
    )
    # note that cells that weren't in any neighborhood now have weird weights,
    # (close to 0, or np.nan). We'll set them to 0 to prevent errors or nans
    # further down.
    cells_not_in_any_Nhood = np.where(n_Nhoods_per_cell == 0)
    cell_weights[cells_not_in_any_Nhood] = 0
    # Now let's calculate the full weight per neighborhood; this is the sum over all
    # cells in the neighborhood (standard matrix multiplication):
    nhood_weights = neighbor_matrix @ cell_weights
    # note that if all cells in a neighborhood are unique, the neighborhood's
    # weight should be 1!
    return nhood_weights


def weighted_fdr_corr_pvals(pvals_unsorted, weights_unsorted=None):
    """Calculate FDR-corrected p-values (a.k.a. q-values) based on a list of
    uncorrected p-values. If wanted, weights can be incorporated that
    indicate how heavily each observation sthould be weighed.
    This is based on the origina publication
    Bejamini & Hochberg, Scandinavian Journal of Statistics 1997
    and inspired by the Milo paper:
    Dann et al., Nature Biotechnology 2021
    which builds on the cydar paper:
    Lun, Richard & Marioni, Nature Methods 2017

    Arguments:
    pvals_unsorted - 1D np.array of p-values in any order
    weights - np.array of the same shape and order as pvals_unsorted,
     specifying the weight per observation/p-value. Default: None (i.e. weigh
     each observation once)

    Returns:
    np.array of same shape as pvals_unsorted with FDR-corrected p-values
    """
    # sort pvals from small to large:
    pvals_sortind = np.argsort(pvals_unsorted)
    pvals = pvals_unsorted[pvals_sortind]
    weights = weights_unsorted[pvals_sortind]
    # calculate the expected p-value if H0 were true, for observations 1-k
    # sorted by p-value. This is the same as the cumulative density function
    # of a uniform distribution from 0 to 1 summing up to 1.
    # this will look like k/n, i.e. 1/n, 2/n, 3/n etc.
    # in the weighted version, this will look like weight_1 / sum_weights,
    # (weight 1 + 2) / sum_weights, (weight 1 + 2 + 3) / sum_weights etc.
    if weights is None:
        weights = np.full(len(pvals), 1)
    n_obs = sum(weights)
    ecdffactor = np.cumsum(weights) / float(n_obs)
    # now correct the p-values. Here, we calculate what the actual probability
    # of seeing a p-value <= uncorr_p_value is at a given rank k in the
    # list of sorted p-values. If we would only do 1 test, this would be equal
    # to the original p-value. If we do multiple (n) tests, the probability
    # becomes n times larger, so original_p * n. However, we go through the
    # list of p-values from low to high, so at position k we do not ask what
    # is the probability of observing 1 p-value <= original_kth_p_value, but
    # rather of observing k p-values <= original_kth_p_value. Therefore,
    # the probability is divided by k. So we end up at
    # p_corr = orig_pval * n / k, which is the same as orig_pval / (k/n)
    pvals_corr_raw = pvals / ecdffactor
    # It is possible that the probabity for the kth p-value is smaller than
    # that for k-1 (or k-2, k-3 etc.). However, note that the corr. p-values
    # represent the probability of observing *k* p-values of
    # p-value <= orig. p-value given n tests. We can therefore consider the
    # corr_p-value valid for all corr_p_vals of k < k_current (they are at the
    # highest the corr_pvalue calculated for k). If the corr
    # p-value is lower for a later k than an earlier k, we set the higher earlier
    # k's p-value to the lower later p-value.
    # In practice, we can do this by reversing the order of the corr_pvalues
    # (now ranging from the highest original p-value to the lowest), and iteratively
    # setting each value (starting at position 2) to the minimum of itself and the
    # previous value. This ensures that corr p-values are always set to the
    # lowest corr p-value observed at its own position or at higher (=originally
    # higher p-values) positions (but not at lower). (and then reversing
    # the order back again)
    pvals_corr = np.minimum.accumulate(pvals_corr_raw[::-1])[::-1]
    # now replace all pvals > 1 with 1
    pvals_corr[pvals_corr > 1] = 1
    # and undo the sorting done in the beginning, so that we return the p-values
    # in the same order as they were given to us
    pvals_corr_unsorted = np.empty_like(pvals_corr)
    pvals_corr_unsorted[pvals_sortind] = pvals_corr

    # I don't understand the purpose of line 119.
    # Wouldn't `pvals_corr_unsorted = np.empty_like(pvals_corr)[pvals_sortind]`
    # achieve the same result, more concisely?

    # return corrected pvalues (also named q-values)
    return pvals_corr_unsorted
