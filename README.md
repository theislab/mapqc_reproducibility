# MapQC: an automated piepeline for validating query-to-reference mapping onto a reference-atlas

The goal of MapQC is to automatically determine whether or not a query-to-reference mapping was successful or not. There are two ways in which a mapping can fail: either the mapping removed too much variation from the query (biological variance removed —> “over-integration”), or it removed too little variation (batch effects retained —> “under-integration”). 

Several factors need to be taken into account when assessing the success or failure of a mapping:

- mapping is done per batch (e.g. sample or dataset), and should therefore also be assessed at the batch level.
- Once we have a metric that quantifies some version of failure/uncertainty: how do we assess what the boundary is between “normal” levels of uncertainty etc., versus abnormal levels?

MapQC aims to leverage the presence of a curated reference atlas to determine the boundaries between success and failure, thereby enabling a data-driven automation of query-to-reference mapping success assessment.


### Structure of Repository

#### env
Enviorments needed to run the package and analysises

#### mapqc
Main package folder containing the MapQC components:
- MapAnalyzer: Main module containing functions for preprocessing, cell type selection, calculation of mixing scores and quantification  of success and failure
- Plotter: Functions to visualize MapAnalyzer produced results 
- label_transfer: script to transfer cell_type labels from reference to query

#### notebooks
Notebooks created for and with MapQC.

01_decision_making_notebooks: Notebooks that visualize and explore concepts later intergrated in the main package
02_example_notebooks_mapqc_pipeline: Notebooks showing the MapQC pipeline in action on different examples + template notebook to try out yourself

#### res
Ressources that may be helpful for the analysis of examples such as an overview over datasets subjected to analysis or lists of genes related to certain conditions 