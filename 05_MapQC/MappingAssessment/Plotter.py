from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from sklearn.neighbors import KNeighborsTransformer


class MapPlot:
    def __init__(self, workspace):
        self.workspace = workspace
        sc.settings.figdir = workspace
        sc.set_figure_params(figsize=(9, 9))

    def plot_baseline_comparison(
        self,
        adata: AnnData,
        color_query: str,
        title_stub: str,
        sample_filter: Union[str, None] = None,
        violin: bool = False,
        disease: bool = False,
        save: bool = False,
        score_aggregation_metric: Literal["mean", "variance"] = "mean",
        transform: KNeighborsTransformer = None,
    ):
        """
        Plots the query and the baseline scores distributions.

        Parameters
        ----------
        adata
            Query annotated dataset.
        color_query
            Color to be used for the query in the plot.
        title_stub
            Title of the plot.
        sample_filter
            String ID of the sample to apply a filter for.
        violin
            Whether to display a violin plot.
        disease
            Whether to plot diseased samples (?).
        save
            Whether to save the resulting plot.
        score_aggregation_metric
            Metric to compute for the mixing score. Could be either mean or variance.
        transform
            The transformation to apply to the query object.
        """
        group = (
            adata.obs.groupby(["cell_label", "sample"])
            .agg(
                {
                    "score": score_aggregation_metric,  # score_aggregation_metric
                    "query_or_reference": "first",
                    "disease_or_healthy": "first",
                    "dataset": "count",
                }
            )
            .rename(columns={"dataset": "count"})
        )

        concat = group.reset_index(drop=False)
        if transform is not None:
            concat = self.__transform(concat, transform)

        concat.dropna(inplace=True)
        concat["cell_label"] = concat["cell_label"].astype("category")
        concat["cell_label"] = concat["cell_label"].cat.remove_unused_categories()
        categories = concat["cell_label"].cat.categories.tolist()
        categories.sort()
        concat["cell_label"] = concat["cell_label"].cat.reorder_categories(categories)

        # Plot
        fig, ax = plt.subplots(figsize=(22, 8))

        if violin:
            sns.violinplot(
                x="cell_label",
                y="score",
                hue="query_or_reference",
                split=True,
                palette={"query": color_query, "reference": "silver"},
                data=concat,
            )
        else:
            sns.boxplot(
                x="cell_label",
                y="score",
                hue="query_or_reference",
                palette={"query": color_query, "reference": "silver"},
                data=concat,
            )

        concat_query = concat[concat.query_or_reference == "query"]
        concat_query["sample"].cat.remove_unused_categories()

        if sample_filter is None:
            print("Filtering is not performed")
            sns.stripplot(
                x="cell_label",
                y="score",
                hue="disease_or_healthy" if disease else None,
                hue_order=["healthy", "disease"] if disease else None,
                palette=["limegreen", "navy"] if disease else "darkgreen",
                jitter=0.1,
                data=concat_query,
            )
        else:
            print(f"Filtering according to {sample_filter}")
            data = pd.merge(
                concat, sample_filter, how="outer", on=["cell_label", "sample"]
            )
            data["cell_label"] = data["cell_label"].astype("category")
            categories = data["cell_label"].cat.categories.tolist()
            categories.sort()
            # FIXME: something is happening twice above here for no apparent reason
            data["cell_label"].cat.reorder_categories(categories, inplace=True)
            sns.stripplot(
                x="cell_label",
                y="score",
                hue="label"
                if disease
                else data["label"].map(
                    {
                        "healthy accepted": "query accepted",
                        "healthy rejected": "query rejected",
                    }
                ),
                hue_order=[
                    "healthy accepted",
                    "healthy rejected",
                    "disease accepted",
                    "disease rejected",
                ]
                if disease
                else [
                    "query accepted",
                    "query rejected",
                ],
                palette={
                    "healthy accepted": "dodgerblue",
                    "healthy rejected": "orangered",
                    "disease accepted": "midnightblue",
                    "disease rejected": "maroon",
                }
                if disease
                else {"query accepted": "dodgerblue", "query rejected": "orangered"},
                data=data,
                size=7,
            )

            if disease:
                plt.legend(
                    bbox_to_anchor=(1.05, 1),
                )

        sns.despine(offset=10, trim=True)

        plt.ylabel("weighted fraction of reference neighbors", fontsize=20)
        plt.xlabel("cell label", fontsize=20)

        plt.yticks(fontsize=15)
        plt.xticks(rotation=90, fontsize=20)

        title = f"{title_stub} sample {score_aggregation_metric} scores"
        filename = (
            f"{self.workspace}/{title_stub} sample {score_aggregation_metric} scores"
        )

        if sample_filter is None:
            title += f" vs. baseline sample {score_aggregation_metric}"
        else:
            title += " rejection"
            filename += " rejection"

        plt.title(title, fontsize=20)

        if save:
            plt.savefig(filename, dpi=100, bbox_inches="tight")

        plt.show()

    def plot_subject_scores(
        self,
        adata: AnnData,
        color: str,
        title_stub: str,
        aggregation_level=Literal["sample", "subject_ID"],
        sort: bool = False,
        save: bool = False,
    ):
        """
        Plots the distribution of subject scores.

        Parameters
        ----------
        adata
            Query annotated dataset.
        color
            Color to be used for the plot.
        title_stub
            Title of the plot.
        aggregation_level
            Whether to aggregate the score by samples or subject ID.
        sort
            Whether to sort scores by sample medians.
        save
            Whether to save the resulting plot.
        """
        # adata = adata.copy()
        cleaned = adata.obs.dropna(
            subset=["score"]
        ).copy()  # FIXME: change to sensible variable name
        cleaned["cell_label"] = cleaned["cell_label"].cat.remove_unused_categories()
        clean_q = cleaned[cleaned.query_or_reference == "query"].copy()
        clean_q["cell_label"] = clean_q["cell_label"].cat.remove_unused_categories()
        clean_q[aggregation_level] = clean_q[
            aggregation_level
        ].cat.remove_unused_categories()

        if sort:
            to_sort = clean_q.groupby([aggregation_level]).agg({"score": "median"})
            idx = to_sort.sort_values(by="score").index.tolist()
            clean_q[aggregation_level] = clean_q[
                aggregation_level
            ].cat.reorder_categories(idx)

        sns.set_theme(style="ticks", palette="pastel")
        fig = plt.subplots(figsize=(20, 8))  # FIXME: not used

        sns.boxplot(x=aggregation_level, y="score", data=clean_q, color=color)
        sns.despine(offset=10, trim=True)

        plt.ylabel("weighted fraction of reference neighbors", fontsize=20)
        plt.xlabel(aggregation_level, fontsize=20)

        plt.yticks(fontsize=15)
        plt.xticks(rotation=90, fontsize=15)

        plt.title(
            f"{title_stub} score distributions per {aggregation_level}", fontsize=20
        )

        if save:
            plt.savefig(
                f"{self.workspace}/{title_stub} score distributions per {aggregation_level}",
                dpi=100,
                bbox_inches="tight",
            )

        plt.show()

    # FIXME: in general, this function is a sliced verison of `plot_baseline_comparison`
    # Hence, a single function would be enough
    def plot_baseline_distribution(
        self, adata: AnnData, violin: bool = True, save: bool = False
    ):
        """
        Plots the distribution of baseline scores.

        Parameters
        ----------
        adata
            Query annotated dataset.
        violin
            Whether to display a violin plot.
        save
            Whether to save the resulting plot.
        """
        group = adata.obs.groupby(["cell_label", "sample"]).agg(
            {"score": "mean", "query_or_reference": "first"}
        )  # FIXME: Is the reference or the query considered here?

        concat = group.reset_index(drop=False)
        concat.dropna(inplace=True)
        concat["cell_label"] = concat["cell_label"].astype("category")
        concat["cell_label"] = concat["cell_label"].cat.remove_unused_categories()
        categories = concat["cell_label"].cat.categories.tolist()
        categories.sort()
        concat["cell_label"] = concat["cell_label"].cat.reorder_categories(categories)
        concat = concat[concat["query_or_reference"] == "reference"]

        # plot
        sns.set_theme(style="ticks", palette="pastel")
        fig = plt.subplots(figsize=(20, 7))

        plot_func = sns.violinplot if violin else sns.boxplot
        plot_func(x="cell_label", y="score", data=concat, color="lightgray")

        sns.stripplot(x="cell_label", y="score", data=concat, color="darkslategray")
        sns.despine(offset=10, trim=True)

        plt.ylabel("weighted fraction of reference neighbors", fontsize=20)
        plt.xlabel("cell label", fontsize=20)

        plt.yticks(fontsize=15)
        plt.xticks(rotation=90, fontsize=20)

        title = "Baseline score distributions per celltype"
        plt.title(title, fontsize=20)

        if save:
            plt.savefig(f"{self.workspace}/{title}", dpi=100, bbox_inches="tight")
        plt.show()

    def plot_umap(
        self,
        adata: AnnData,
        title_stub: str,
        threshold_cell: Optional[float] = None,
        threshold_sample: Optional[float] = None,
        restrict_to: Union[List[str], None] = None,
        mode: Literal[
            "reference", "query", "query_scores", "cell_labels", "rejection_status"
        ] = "reference",
        color_query: Optional[str] = "green",
        on_data: Optional[bool] = False,
        excluded_data: Optional[AnnData] = None,
        save: bool = False,
    ):
        """
        Plots the UMAP projection of the query dataset(s).

        Parameters
        ----------
        adata
            Query annotated dataset.
        title_stub
            Title of the plot.
        threshold_cell
            Threshold for a given cell type.
        threshold_sample
            Threshold for a given cell sample.
        restrict_to
            Whether to plot a subset of cell labels.
        mode
            Plotting mode: "reference", "query", "query_scores", "cell_labels", or "rejection_status".
        color_query
            Color to be used to display query cells.
        on_data
            Legend positioning in "cell_labels" mode.
        excluded_data
            Data to exclude from further analyses.
        save
            Whether to save the resulting plot.
        """
        if mode not in ["reference", "rejection_status"]:
            adata = adata.copy()
        title = f"{title_stub} umap {mode}"
        filename = f"{title_stub} {mode}.png"
        color_label, palette, color_map, legend_loc, legend_fontsize, na_color, vmax = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        def preprocess_query_mode(adata):
            datasets_query = adata[
                adata.obs.query_or_reference == "query"
            ].obs.dataset.cat.categories.tolist()
            adata.obs.loc[~adata.obs.dataset.isin(datasets_query), "dataset"] = np.nan
            adata.obs.dataset = adata.obs.dataset.cat.remove_unused_categories()

        def preprocess_query_scores_mode(adata):
            adata.obs.loc[adata.obs.query_or_reference == "reference", "score"] = np.nan

        def preprocess_cell_labels_mode(adata, excluded_data):
            if excluded_data is not None:
                cell_labels = adata.obs.cell_label.cat.categories.tolist()
                adata.obs.cell_label = adata.obs.cell_label.astype(str)
                for ct in cell_labels:
                    if (
                        ct
                        not in excluded_data.cell_label.astype(
                            "category"
                        ).cat.categories
                    ):
                        adata.obs.loc[adata.obs.cell_label == ct, "cell_label"] = (
                            ct + " (ex)"
                        )

        def get_legend_settings(on_data):
            return ("on data", "x-small") if on_data else ("right margin", "x-large")

        def get_rejection_status_settings(adata, threshold_cell, threshold_sample):
            rejection_status_key = (
                f"rejection_status_{threshold_cell}_{threshold_sample}"
            )
            if rejection_status_key not in adata.obs:
                raise Exception(
                    f"'{rejection_status_key}' not in .obs make sure to calculate beforehand"
                )
            return (
                rejection_status_key,
                {"rejected": "darkorange", "accepted": "turquoise"},
                "0.7",
            )

        if mode == "query":
            preprocess_query_mode(adata)
            color_label, palette = "dataset", "colorblind"
        elif mode == "query_scores":
            preprocess_query_scores_mode(adata)
            color_label, color_map = "score", "gist_heat"
        elif mode == "reference":
            color_label = "query_or_reference"
            # TODO: fix color assignment
            palette = {
                "reference": "0.7",
                "query": color_query,
            }  # type: ignore[assignment]
            # palette = "colorblind"
        elif mode == "cell_labels":
            preprocess_cell_labels_mode(adata, excluded_data)
            color_label, palette, na_color = "cell_label", "colorblind", "0.7"
            legend_loc, legend_fontsize = get_legend_settings(on_data)
        elif mode == "rejection_status":
            color_label, palette, na_color = get_rejection_status_settings(
                adata, threshold_cell, threshold_sample
            )
            filename += f"rejection status t_cell = {threshold_cell} t_sample = {threshold_sample}.png"
            title += f"rejection status at t_cell = {threshold_cell} and t_sample = {threshold_sample}"
        else:
            raise ValueError(
                "Please select a valid option ('reference', 'query', 'query_scores', 'cell_labels', 'rejection_status') for argument 'mode'."
            )

        if restrict_to is not None:
            adata = adata[adata.obs.cell_label.isin(restrict_to)].copy()

        sc.pl.umap(
            adata,
            color=color_label,
            palette=palette if mode != "query_scores" else None,
            color_map=color_map if mode == "query_scores" else None,
            legend_loc=legend_loc if legend_loc is not None else "right margin",
            legend_fontsize=legend_fontsize,
            title=title,
            na_in_legend=True if mode == "cell_labels" else False,
            na_color=na_color if na_color is not None else "lightgray",
            vmax=1.2 if mode == "query_scores" else None,
            size=0.5,
            save=filename if save else None,
        )

    def __transform(self, group, transform_path):
        transform = pd.read_csv(transform_path, index_col="cell_label")
        for ct in group.cell_label.cat.categories:
            if ct in transform.index.to_list():
                trans = transform.loc[ct]
                group.loc[group.cell_label == ct, "score"] = group.loc[
                    group.cell_label == ct, "score"
                ] - (
                    np.log(group.loc[group.cell_label == ct, "count"]) * trans.slope
                    + trans.intercept
                )
        return group
