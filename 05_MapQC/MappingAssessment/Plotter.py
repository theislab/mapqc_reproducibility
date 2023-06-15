import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import os
import scipy
import itertools


class MapPlot:

    def __init__(self,workspace):
        self.workspace = workspace
        sc.settings.figdir = workspace
        sc.set_figure_params(figsize=(9, 9))


    def plot_baseline_comparison(
            self,
            adata, 
            color_query, 
            title_stub, 
            sample_filter=None, 
            violin=False, 
            disease=False,
            save = False,
            transform=None
        ):
        '''
        # FIXME: Missing docstrings
        '''
        #adata = adata.copy()
        group = adata.obs.groupby(["cell_label", "sample"]).agg(
            {"score": "mean", "query_or_reference": "first", "disease_or_healthy": "first","dataset":"count"}
        ).rename(columns={"dataset":"count"})

        concat = group.reset_index(drop=False)
        if transform is not None:
            concat=self.__transform(concat,transform) 

        concat.dropna(inplace=True)
        concat["cell_label"] = concat["cell_label"].astype("category")
        concat["cell_label"] = concat["cell_label"].cat.remove_unused_categories()
        categories = concat["cell_label"].cat.categories.tolist()
        categories.sort()
        concat["cell_label"] = concat["cell_label"].cat.reorder_categories(categories)


        # plot

        # sns.set_theme(style="ticks", palette="pastel")
        fig, ax = plt.subplots(figsize=(22, 8))

        if violin:
            sns.violinplot(
                x="cell_label",
                y="score",
                hue="query_or_reference",
                split=True,
                palette={"query":color_query, "reference":"silver"},
                data=concat,
            )

        else:

            sns.boxplot(
                x="cell_label",
                y="score",
                hue="query_or_reference",
                palette={"query":color_query, "reference":"silver"},
                data=concat,
            )

        concat_query = concat[concat.query_or_reference == "query"]
        concat_query["sample"].cat.remove_unused_categories()

        if sample_filter is None:
            if disease:
                print("No failed filtering") # FIXME: once is enough. Semantically better is "No filtering needed"
                sns.stripplot(
                    x="cell_label",
                    y="score",
                    hue="disease_or_healthy",
                    hue_order=["healthy", "disease"],
                    palette=["limegreen", "navy"],
                    jitter=0.1,
                    data=concat_query,
                )
            else:
                print("No failed filtering")
                sns.stripplot(
                    x="cell_label",
                    y="score",
                    color="darkgreen",
                    jitter=0.1,
                    data=concat_query,
                )

        else:
            print("Filter samples")
            data = pd.merge(concat, sample_filter, how="outer", on=["cell_label", "sample"])
            data["cell_label"] = data["cell_label"].astype("category")
            categories = data["cell_label"].cat.categories.tolist()
            categories.sort()
            # FIXME: something is happening twice above here for no apparent reason
            data["cell_label"].cat.reorder_categories(categories, inplace=True)
            if disease:
                sns.stripplot(
                    x="cell_label",
                    y="score",
                    hue="label",
                    hue_order=[
                        "healthy accepted",
                        "healthy rejected",
                        "disease accepted",
                        "disease rejected",
                    ],
                    palette={
                        "healthy accepted":"dodgerblue",
                        "healthy rejected": "orangered", 
                        "disease accepted":"midnightblue",
                        "disease rejected": "maroon"
                    },
                    data=data,
                    size=7,
                )

                plt.legend(
                    bbox_to_anchor=(1.05, 1),
                )
            else:
                sns.stripplot(
                    x="cell_label",
                    y="score",
                    hue=data["label"].map(
                        {
                            "healthy accepted": "query accepted",
                            "healthy rejected": "query rejected",
                        }
                    ),
                    hue_order=[
                        "query accepted",
                        "query rejected",
                    ],
                    palette={
                        "query accepted":"dodgerblue", 
                        "query rejected":"orangered"
                    },
                    data=data,
                    size=7,
                )

        sns.despine(offset=10, trim=True)   

        plt.ylabel("weighted fraction of reference neighbors", fontsize=20)
        plt.xlabel("cell label", fontsize=20)
        
        plt.yticks(fontsize=15)
        plt.xticks(rotation=90,fontsize=20)
        

        if sample_filter is None:
            plt.title(
                f"{title_stub} sample mean scores vs. baseline sample means", fontsize=20
            )
        else:
            plt.title(
                f"{title_stub} sample mean scores rejection", fontsize=20
            )

        # FIXME: these condition could be merged with the ones immediately above without opening other clause bifurcations
        if save:
            if sample_filter is  None:
                plt.savefig(f"{self.workspace}/{title_stub} sample mean scores vs baseline sample means",dpi=100,bbox_inches ="tight")
            else:
                plt.savefig(f"{self.workspace}/{title_stub} sample mean scores rejection",dpi=100,bbox_inches ="tight")

        plt.show()

    # FIXME: redundant function with `plot_baseline_comparison` - flag "score" aggregation
    def plot_baseline_comparison_variance(
            self,
            adata, 
            color_query, 
            title_stub, 
            sample_filter=None, 
            violin=False, 
            disease=False,
            save = False,
            # score_aggregation_metric: Literal["mean", "var"], - needs import from `typing`
            transform=None
        ):
        #adata = adata.copy()
        group = adata.obs.groupby(["cell_label", "sample"]).agg(
            {
                "score": "var", # score_aggregation_metric
                "query_or_reference": "first", 
                "disease_or_healthy": "first",
                "dataset":"count"
            }
        ).rename(columns={"dataset":"count"})

        concat = group.reset_index(drop=False)
        if transform is not None:
            concat=self.__transform(concat,transform)
        
        concat.dropna(inplace=True)
        concat["cell_label"] = concat["cell_label"].astype("category")
        concat["cell_label"] = concat["cell_label"].cat.remove_unused_categories()
        categories = concat["cell_label"].cat.categories.tolist()
        categories.sort()
        concat["cell_label"] = concat["cell_label"].cat.reorder_categories(categories)


        # plot

        # sns.set_theme(style="ticks", palette="pastel")
        fig, ax = plt.subplots(figsize=(22, 8))

        if violin:
            sns.violinplot(
                x="cell_label",
                y="score",
                hue="query_or_reference",
                split=True,
                palette={"query":color_query, "reference":"silver"},
                data=concat,
            )

        else:

            sns.boxplot(
                x="cell_label",
                y="score",
                hue="query_or_reference",
                palette={"query":color_query, "reference":"silver"},
                data=concat,
            )

        concat_query = concat[concat.query_or_reference == "query"]
        concat_query["sample"].cat.remove_unused_categories(inplace=True)

        if sample_filter is None:
            if disease:
                print("No failed filtering")
                sns.stripplot(
                    x="cell_label",
                    y="score",
                    hue="disease_or_healthy",
                    hue_order=["healthy", "disease"],
                    palette=["limegreen", "navy"],
                    jitter=0.1,
                    data=concat_query,
                )
            else:
                print("No failed filtering")
                sns.stripplot(
                    x="cell_label",
                    y="score",
                    color="darkgreen",
                    jitter=0.1,
                    data=concat_query,
                )

        else:
            print("Filter samples")
            data = pd.merge(concat, sample_filter, how="outer", on=["cell_label", "sample"])
            data["cell_label"] = data["cell_label"].astype("category")
            categories = data["cell_label"].cat.categories.tolist()
            categories.sort()
            data["cell_label"].cat.reorder_categories(categories, inplace=True)
            if disease:
                sns.stripplot(
                    x="cell_label",
                    y="score",
                    hue="label",
                    hue_order=[
                        "healthy accepted",
                        "healthy rejected",
                        "disease accepted",
                        "disease rejected",
                    ],
                    palette={"healthy accepted":"dodgerblue","healthy rejected": "orangered", "disease accepted":"midnightblue","disease rejected": "maroon"},
                    data=data,
                    size=7,
                )

                plt.legend(
                    bbox_to_anchor=(1.05, 1),
                )
            else:

                sns.stripplot(
                    x="cell_label",
                    y="score",
                    hue=data["label"].map(
                        {
                            "healthy accepted": "query accepted",
                            "healthy rejected": "query rejected",
                        }
                    ),
                    hue_order=[
                        "query accepted",
                        "query rejected",
                    ],
                    palette={"query accepted":"dodgerblue", "query rejected":"orangered"},
                    data=data,
                    size=7,
                )

        sns.despine(offset=10, trim=True)   

        plt.ylabel("weighted fraction of reference neighbors", fontsize=20)
        plt.xlabel("cell label", fontsize=20)
        
        plt.yticks(fontsize=15)
        plt.xticks(rotation=90,fontsize=20)
        

        if sample_filter is None:
            plt.title(
                f"{title_stub} sample variance scores vs. baseline sample variances", fontsize=20
            )
        else:
            plt.title(
                f"{title_stub} sample variance scores rejection", fontsize=20
            )

        if save:
            if sample_filter is  None:
                plt.savefig(f"{self.workspace}/{title_stub} sample variance scores vs baseline sample variances",dpi=100,bbox_inches ="tight")
            else:
                plt.savefig(f"{self.workspace}/{title_stub} sample variances scores rejection",dpi=100,bbox_inches ="tight")

        plt.show()
    
    def plot_subject_scores(
            self,
            adata, 
            color, 
            title_stub,
            sort =False,
            save = False
        ):
        #adata = adata.copy()
        cleaned = adata.obs.dropna(subset=["score"]).copy() # FIXME: change to sensible variable name
        cleaned["cell_label"] = cleaned["cell_label"].cat.remove_unused_categories()
        clean_q = cleaned[cleaned.query_or_reference == "query"].copy()
        clean_q["cell_label"] = clean_q["cell_label"].cat.remove_unused_categories()
        clean_q["subject_ID"] = clean_q["subject_ID"].cat.remove_unused_categories()
        
        if sort:
            to_sort = clean_q.groupby(["subject_ID"]).agg({"score":"median"})
            idx = to_sort.sort_values(by="score").index.tolist()
            clean_q["subject_ID"] = clean_q["subject_ID"].cat.reorder_categories(idx)
            

        sns.set_theme(style="ticks", palette="pastel")
        fig = plt.subplots(figsize=(20, 8)) # FIXME: not used

        sns.boxplot(x="subject_ID", y="score", data=clean_q, color=color)
        sns.despine(offset=10, trim=True) 
        
        plt.ylabel("weighted fraction of reference neighbors", fontsize=20)
        plt.xlabel("subject_ID", fontsize=20)
        
        plt.yticks(fontsize=15)
        plt.xticks(rotation=90,fontsize = 15)

        plt.title(f"{title_stub} score distributions per subject",fontsize=20)

        if save:
             plt.savefig(f"{self.workspace}/{title_stub} score distributions per subject",dpi=100,bbox_inches ="tight")

        plt.show()

    # FIXME: redundant function with `plot_subject_scores` - flag usage of `aggregation_level` and summarize under `plot scores`
    def plot_sample_scores(
            self, 
            adata, 
            color, 
            title_stub,
            #aggregation_level = Literal["sample", "subject_ID"],
            sort = False, 
            save=False
        ):
        #adata = adata.copy()
        cleaned = adata.obs.dropna(subset=["score"]).copy()
        cleaned["cell_label"] = cleaned["cell_label"].cat.remove_unused_categories()
        clean_q = cleaned[cleaned.query_or_reference == "query"].copy()
        clean_q["cell_label"] = clean_q["cell_label"].cat.remove_unused_categories()
        clean_q["sample"] = clean_q["sample"].cat.remove_unused_categories()


        if sort:
            to_sort = clean_q.groupby(["sample"]).agg({"score":"median"})
            idx = to_sort.sort_values(by="score").index.tolist()
            clean_q["sample"] = clean_q["sample"].cat.reorder_categories(idx)


        sns.set_theme(style="ticks", palette="pastel")
        fig = plt.subplots(figsize=(20, 8))
        

        sns.boxplot(x="sample", y="score", data=clean_q, color=color)
        sns.despine(offset=10, trim=True)
        
        plt.ylabel("weighted fraction of reference neighbors", fontsize=20)
        plt.xlabel("sample",fontsize=20)

        plt.yticks(fontsize=15)
        plt.xticks(rotation=90,fontsize = 13)

        plt.title(f"{title_stub} score distributions per sample",fontsize=20)
        if save:
            plt.savefig(f"{self.workspace}/{title_stub} score distributions per sample",dpi=100,bbox_inches ="tight")
        plt.show()


    # FIXME: in general, this function is a sliced verison of `plot_baseline_comparison(_variance)`
    # Hence, a single function is enough here.
    def plot_baseline_distribution(self, adata,violin=True,save = False):
        #adata = adata.copy()
        group = adata.obs.groupby(["cell_label", "sample"]).agg(
            {"score": "mean", "query_or_reference": "first"}
        ) # FIXME: Is the reference or the query considered here? 
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

        if violin:
            sns.violinplot(x="cell_label", y="score", data=concat, color="lightgray")
        else:
            sns.boxplot(x="cell_label", y="score", data=concat, color="lightgray")

        sns.stripplot(x="cell_label", y="score", data=concat, color="darkslategray")
        sns.despine(offset=10, trim=True)

        plt.ylabel("weighted fraction of reference neighbors",fontsize=20)
        plt.xlabel("cell label", fontsize = 20)
        
        plt.yticks(fontsize=15)
        plt.xticks(rotation=90,fontsize = 20)

        plt.title(f"Baseline score distributions per celltype",fontsize=20)

        if save:
             plt.savefig(f"{self.workspace}/Baseline score distributions per celltype",dpi=100,bbox_inches ="tight")
        plt.show()


    def umap_query_datasets(
            self,
            adata,
            title_stub, 
            palette = "tab20", # FIXME: It would be advisable to use "colorblind" here
            restrict_to = None,
            save = False
        ):
        adata = adata.copy()
        datasets_query = adata[adata.obs.query_or_reference == "query"].obs.dataset.cat.categories.tolist()
        adata.obs.loc[
        ~adata.obs.dataset.isin(datasets_query), "dataset"] = np.nan
        adata.obs.dataset = adata.obs.dataset.cat.remove_unused_categories()
        if restrict_to is not None:
            adata = adata[adata.obs.cell_label.isin(restrict_to)].copy()

        if save:
            save = f" {title_stub} query datasetwise.png"

        sc.pl.umap( 
            adata,
            color="dataset",
            palette=palette,
            title=f"{title_stub} umap query datasets",
            na_in_legend=False,
            size=0.5,
            save = save
        )

    # FIXME: redundant function with `umap_query_datasets`
    def umap_query_reference(
            self,
            adata,
            title_stub, 
            #palette,
            color_query="green", # FIXME: can be flagged
            restrict_to = None,
            save = False
        ):
        #adata = adata.copy()
        if restrict_to is not None:
            adata = adata[adata.obs.cell_label.isin(restrict_to)].copy()

        if save:
            save = f" {title_stub} query reference.png"

        sc.pl.umap( 
            adata,
            color="query_or_reference",
            palette={"reference": "0.7", "query": color_query},
            title=f"{title_stub} umap query reference", # FIXME: can be flagged
            na_in_legend=False,
            size=0.5,
            save = save
        )

    # FIXME: same as above
    def umap_mixing_score_query(self,adata,title_stub, palette = "gist_heat" ,restrict_to = None,save = False):
        adata = adata.copy()
        adata.obs.loc[adata.obs.query_or_reference == "reference", "score"] = np.nan
        if restrict_to is not None:
            adata = adata[adata.obs.cell_label.isin(restrict_to)].copy()

        if save:
            save = f" {title_stub} query mixing scores.png"

        sc.pl.umap( 
            adata,
            color="score",
            color_map=palette,
            title=f"{title_stub} umap mixing score",
            na_color="lightsteelblue",
            na_in_legend=False,
            vmax = 1.2,
            size=0.5,
            save = save
        )

    # FIXME: same as above
    def umap_cell_labels(self,adata,title_stub,palette ="tab20", on_data= False, excluded_data = None, restrict_to=None,save = False):
        adata = adata.copy()
        if excluded_data is not None:
            cell_labels = adata.obs.cell_label.cat.categories.tolist()
            adata.obs.cell_label = adata.obs.cell_label.astype(str)
            for ct in cell_labels:
                if ct not in excluded_data.cell_label.astype("category").cat.categories:
                    adata.obs.loc[adata.obs.cell_label == ct, "cell_label"] = ct + " (ex)"
        
        if restrict_to is not None:
            adata = adata[adata.obs.cell_label.isin(restrict_to)].copy()

        if save:
            save = f" {title_stub} cell label.png"



        if on_data:
            legend_loc = "on data"
            legend_fontsize = "x-small"
        else:
            legend_loc = "right margin"
            legend_fontsize = "x-large"


        sc.pl.umap( 
            adata,
            color="cell_label",
            palette=palette,
            legend_loc = legend_loc,
            legend_fontsize = legend_fontsize,
            title=f"{title_stub} umap cell label",
            na_color="0.7",
            na_in_legend=True,
            size=0.5,
            save = save
        )

    # FIXME: same as above
    def umap_rejection_status(self,adata,title_stub,excluded_data, threshold_cell, threshold_sample, palette ={"rejected":"darkorange","accepted":"turquoise"}, restrict_to=None,save = False):
        #adata = adata.copy()
        if f"rejection_status_{threshold_cell}_{threshold_sample}" not in adata.obs:
            raise Exception("'rejection_status_{threshold_cell}_{threshold_sample}' not in .obs make sure to calculate beforehand")
        if restrict_to is not None:
            adata = adata[adata.obs.cell_label.isin(restrict_to)].copy()

        if save:
            save = f" {title_stub} rejection status t_cell = {threshold_cell} t_sample = {threshold_sample}.png"
      
        sc.pl.umap( 
            adata,
            color=f"rejection_status_{threshold_cell}_{threshold_sample}",
            palette=palette,
            title=f"{title_stub} umap rejection status at t_cell = {threshold_cell} and t_sample = {threshold_sample}",
            na_color="0.7",
            na_in_legend=False,
            size=0.5,
            save = save
        )


    def __transform(self,group,transform_path):
        transform = pd.read_csv(transform_path,index_col="cell_label")
        #print(transform)
        for ct in group.cell_label.cat.categories:
            if ct in transform.index.to_list():
                trans = transform.loc[ct]
                group.loc[group.cell_label ==ct,"score"] = group.loc[group.cell_label ==ct,"score"] -(np.log(group.loc[group.cell_label ==ct,"count"])*trans.slope+trans.intercept)
        return group
    