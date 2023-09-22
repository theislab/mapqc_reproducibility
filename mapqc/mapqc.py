import nhoods
import numpy as np
import pandas as pd
from scipy.stats import kstest
import pval_corr


class mapQC_results:
    def __init__(
        self,
        adata,
        center_cells,
        k,
        emb_key="X",
        ref_q_key="reference_or_query",
        ref_cat="reference",
        query_cat="query",
        sample_key="sample",
    ):
        # TO DO: build in sanity checks here
        self.adata = adata
        self.center_cells = center_cells
        self.k = k
        self.emb_key = emb_key
        self.ref_q_key = ref_q_key
        self.query_cat = query_cat
        self.ref_cat = ref_cat
        self.sample_key = sample_key
        # self.intersample_distances = np.nan
        # set parameters specifying if specific functions were run or not
        # (TO DO: complete this further)
        self._adata_prepped_for_plotting = False
        self._results_prepped_for_boxplot = False
        self._nhood_stats_calculated = False
        self._sample_dists_to_ref_calculated = False

    def _calculate_intersample_distances(self):
        # generate empty results dataframe
        self.ref_samples_all = np.array(
            sorted(
                self.adata.obs.loc[
                    self.adata.obs[self.ref_q_key] == self.ref_cat, self.sample_key
                ].unique()
            )
        )
        self.ref_q_samples_all = np.concatenate(
            (
                self.ref_samples_all,
                np.array(
                    sorted(
                        self.adata.obs.loc[
                            self.adata.obs[self.ref_q_key] == self.query_cat, "sample"
                        ].unique()
                    )
                ),
            )
        )
        mean_s2s_dist_per_nhood = np.empty(
            (
                len(self.ref_samples_all),
                len(self.ref_q_samples_all),
                len(self.center_cells),
            )
        )
        mean_s2s_dist_per_nhood[:] = np.nan
        # generate empty array in which to store each nhoods cells (i.e.
        # the k nearest neighbors of each index cell, including itself)
        self.nhood_cells = np.zeros((len(self.center_cells), self.k), dtype=int)
        # now loop through neighborhoods and calculate mean inter-sample
        # distances per nhood
        for nhood_count, center_cell in enumerate(self.center_cells):
            nhood = nhoods.Neighborhood(
                adata=self.adata,
                index_cell=center_cell,
                k=self.k,
                emb_key=self.emb_key,
                sample_key=self.sample_key,
                ref_q_key=self.ref_q_key,
                ref_cat=self.ref_cat,
                query_cat=self.query_cat,
            )
            nhood_results = nhood.get_pairwise_sample_dists()
            self.nhood_cells[nhood_count] = nhood.cell_idc
            # add results to overall tensor
            if not isinstance(nhood_results, type(np.nan)):
                row_idc_for_nhood_ref_samples = np.where(
                    np.isin(self.ref_samples_all, nhood_results.index)
                )[0]
                col_idc_for_nhood_all_samples = np.where(
                    np.isin(self.ref_q_samples_all, nhood_results.columns)
                )[0]
                mean_s2s_dist_per_nhood[
                    row_idc_for_nhood_ref_samples[:, np.newaxis, np.newaxis],
                    col_idc_for_nhood_all_samples[np.newaxis, :, np.newaxis],
                    np.array([nhood_count])[np.newaxis, np.newaxis, :],
                ] = nhood_results.values[:, :, np.newaxis]
        self.intersample_distances = mean_s2s_dist_per_nhood

    def calculate_sample_dists_to_ref_samples(self):
        """Based on intersample distances, calculate the mean distance to
        reference samples for each sample in each neighborhood.

        Result:
        np.array of size (n_samples x n_nhoods)
        """
        self._calculate_intersample_distances()
        self.mean_sample_dists_to_ref = np.nanmean(self.intersample_distances, axis=0)
        self._sample_dists_to_ref_calculated = True

#    def _convert_nhood_dists_to_plottable_dataframe(self):
#        """Creates a DataFrame of with ... and ... as multi-level index, and
#        ... as columns."""
#        # convert to pandas dataframe
#        mean_s2ref_dist_per_nhood_df_for_boxplot = (
#            pd.DataFrame(
#                self.mean_sample_dists_to_ref,
#                index=self.ref_q_samples_all,
#                columns=self.center_cells,
#            )
#            .unstack()
#            .reset_index()
#            .rename(
#                columns={"level_0": "nhood", "level_1": "sample", 0: "mean_dist_to_ref"}
#            )
#        )
#        # remove rows with nans
#        mean_s2ref_dist_per_nhood_df_for_boxplot.dropna(how="any", axis=0, inplace=True)
#        # add info on whether sample is from ref or query
#        sample_from_r_or_q_df = self.adata.obs.groupby(self.sample_key).agg(
#            {self.ref_q_key: "first"}
#        )
#        sample_from_r_or_q_dict = {
#            s: rvsq
#            for s, rvsq in zip(
#                sample_from_r_or_q_df.index, sample_from_r_or_q_df[ref_q_key]
#            )
#        }
#        mean_s2ref_dist_per_nhood_df_for_boxplot[
#            "sample_r_or_q"
#        ] = mean_s2ref_dist_per_nhood_df_for_boxplot["sample"].map(
#            sample_from_r_or_q_dict
#        )
#        return mean_s2ref_dist_per_nhood_df_for_boxplot

    def calculate_nhood_statistics(self, alpha=0.05, verbose=True):
        self.nhood_stats = pd.DataFrame(
            index=self.center_cells, columns=["ks_stat", "pval"]
        )
        self._calculate_nhood_ks_statistics()
        # keep only neighborhoods that passed filtering
        self.nhood_stats["nhood_number"] = range(0, len(self.center_cells))
        self.nhood_stats.dropna(axis=0, how="any", inplace=True)
        # store perc. of nhoods that passed filtering
        self.perc_of_nhoods_passed_filter = round(
            self.nhood_stats.shape[0] / len(self.center_cells) * 100
        )
        if verbose:
            print(
                f"{self.perc_of_nhoods_passed_filter}% of neighborhoods passed filtering."
            )
        # calculate weight per neighborhood based on its overlap with other
        # nhoods
        self.nhood_stats["nhood_weight"] = pval_corr.calculate_neighborhood_weights(
            self.nhood_cells[self.nhood_stats.nhood_number, :], self.adata.n_obs
        )
        # perform FDR-based p-value correction using nhood weights
        self.nhood_stats["pval_corr"] = pval_corr.weighted_fdr_corr_pvals(
            self.nhood_stats.pval, self.nhood_stats.nhood_weight
        )
        # add boolean specifying if neighborhood is below significance
        # threshold of alpha=0.05
        self.nhood_stats["sign"] = self.nhood_stats.pval_corr < alpha
        if verbose:
            perc_of_nhoods_sign = round(
                self.nhood_stats.sign.sum() / self.nhood_stats.shape[0] * 100
            )
            print(
                f"{perc_of_nhoods_sign}% of neighborhoods that passed filtering are below the significance threshold alpha={alpha}."
            )
        # remember that this function has been successfully run:
        self._nhood_stats_calculated = True

    def _calculate_nhood_ks_statistics(self):
        for nhood_number, nhood in enumerate(self.center_cells):
            # slice distances to ref samples for the current neighborhood, and
            # for the ref samples
            r2r_dists = self.mean_sample_dists_to_ref[
                : len(self.ref_samples_all), nhood_number
            ]
            # do the same for the distances to ref samples for the query samples
            r2q_dists = self.mean_sample_dists_to_ref[
                len(self.ref_samples_all) :, nhood_number
            ]
            # if no results for the given neighborhood (because it did not
            # pass filtering during distance calculation), either for ref
            # or for query, skip it
            if np.isnan(r2r_dists).all() or np.isnan(r2q_dists).all():
                continue
            # remove nans, i.e. drop samples that did not pass earlier filtering
            r2r_dists = r2r_dists[~np.isnan(r2r_dists)]
            r2q_dists = r2q_dists[~np.isnan(r2q_dists)]
            # perform KS test
            self.nhood_stats.loc[nhood] = kstest(
                rvs=r2r_dists, cdf=r2q_dists, alternative="greater"
            )[:2]

    # def plot_ref_vs_q_dists_per_nhood(self):
    #     if isinstance(self.mean_sample_dists_to_ref, type(np.nan)):
    #         return ValueError(
    #             "Plot can only be generated after running calculate_sample_dists_to_ref_samples()."
    #         )
    #     fig = plt.figure(figsize=(24, 8))
    #     ax = fig.add_subplot()
    #     nhood_order = self.nhood_stats.sort_values(by="pval_corr").index.tolist()
    #     df_for_boxplot = self._convert_nhood_dists_to_plottable_dataframe()
    #     y = "mean_dist_to_ref"
    #     hue = "sample_r_or_q"
    #     sns.boxplot(
    #         df_for_boxplot,
    #         x="nhood",
    #         y=y,
    #         order=nhood_order,
    #         hue=hue,
    #         ax=ax,
    #         saturation=0.2,
    #     )
    #     sns.stripplot(
    #         df_for_boxplot,
    #         x="nhood",
    #         y=y,
    #         order=nhood_order,
    #         hue=hue,
    #         dodge=True,
    #         ax=ax,
    #         size=1,
    #         palette="dark:black",
    #     )
    #     ax.tick_params(axis="x", rotation=90)

    # def _prepare_adata_for_plotting(self, scale=1):
    #     if not self._nhood_stats_calculated:
    #         raise ValueError(
    #             "Calculate result statistics first using the 'calculate_nhood_statistics' function."
    #         )
    #     for stat_to_copy in ["nhood_weight", "pval_corr", "sign"]:
    #         self.adata.obs[f"mapqc_{stat_to_copy}"] = np.nan
    #         self.adata.obs.loc[
    #             self.nhood_stats.index, f"mapqc_{stat_to_copy}"
    #         ] = self.nhood_stats[stat_to_copy]
    #     # set sizes for plotting:
    #     self.adata.obs["mapqc_dotsize"] = scale * 1
    #     self.adata.obs.loc[self.nhood_stats.index, "mapqc_dotsize"] = scale * 20
    #     # order cells so that index cells are plotted on top
    #     cells_ordered = (
    #         self.adata.obs.index[
    #             ~self.adata.obs.index.isin(self.nhood_stats.index)
    #         ].tolist()
    #         + self.nhood_stats.index.tolist()
    #     )
    #     # store ordered version of adata
    #     # TO DO: we're storing adata twice now, which is very memory inefficient
    #     # but I don't want to re-order the original AnnData (it will mess
    #     # with all the code above which relies on the adata row order). Think
    #     # of a better alternative here...
    #     self._adata_ordered = self.adata[cells_ordered, :].copy()
    #     # remember that adata has been prepped for plotting:
    #     self._adata_prepped_for_plotting = True

    # def plot_umap_significance(self, title="neighborhood significance"):
    #     if not self._adata_prepped_for_plotting:
    #         self._prepare_adata_for_plotting()
    #         sc.set_figure_params(figsize=(5, 5))
    #     sc.pl.umap(
    #         self._adata_ordered,
    #         color="mapqc_sign",
    #         size=self._adata_ordered.obs.mapqc_dotsize,
    #         cmap="Reds",
    #         sort_order=False,
    #         title=title,
    #         vmin=0,
    #         vmax=1,
    #     )
