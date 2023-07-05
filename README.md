# MapQC: an automated piepeline for validating query-to-reference mapping onto a reference-atlas

The goal of MapQC is to automatically determine whether or not a query-to-reference mapping was successful or not. There are two ways in which a mapping can fail: either the mapping removed too much variation from the query (biological variance removed —> “over-integration”), or it removed too little variation (batch effects retained —> “under-integration”).

Several factors need to be taken into account when assessing the success or failure of a mapping:

- mapping is done per batch (e.g. sample or dataset), and should therefore also be assessed at the batch level.
- Once we have a metric that quantifies some version of failure/uncertainty: how do we assess what the boundary is between “normal” levels of uncertainty etc., versus abnormal levels?

MapQC aims to leverage the presence of a curated reference atlas to determine the boundaries between success and failure, thereby enabling a data-driven automation of query-to-reference mapping success assessment.


### Structure of Repository

#### mapqc
Main package folder containing the MapQC components:
- `MapAnalyzer.py`: Main module containing functions for preprocessing, cell type selection, calculation of mixing scores and quantification of success and failure
- `Plotter.py`: Visualizes MapAnalyzer results
- `label_transfer.py`: Transfers cell_type labels from reference to query

#### notebooks
Notebooks implementing the MapQC pipeline.

`dataset_examples`: visualize and explore concepts later intergrated in the main package
`mapQC_design`: show the MapQC pipeline in action on different example datasets + template notebook displaying the general pipeline


`HLCA_dataset_overview`: Overview over datasets in HLCA for which analyses have been performed (not yet - edit this description later).
