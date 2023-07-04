# Opportunities and Limitations of Transfer Learning in the Context of the Human Lung Cell Atlas 
Repository to explore what is the best way to iteratively extend and grow atlases, comparing a de-novo method (scANVI) with a query-to-reference mapping method (scArches) for three different use-cases.

Abstract of the Master Thesis:
Recent efforts led to the generation of the first comprehensive large-scale cell atlases by integration of multiple datasets.
Those atlases open up the possibility of employing latest query-to-reference mapping methods to analyze and interpret new datasets with respect to the reference atlas. Furthermore, those methods could also be used to extend existing atlases efficiently.
The impact of query-to-reference mapping on the integration quality compared to the de-novo methods used for the atlas creation has not yet been studied in detail.
This thesis compares two state-of-the-art models regarding their differences in integration for three different use-cases comprising the extension of a healthy single-cell lung atlas with healthy single-cell, disease single-cell, and healthy single-nucleus data.
Substantial differences between the integrations of the two models are found, providing researchers a foundation to make informed decisions regarding the model choice for the extension of atlases.


### Structure of Repository
#### 01_Preparation
Collect different data sources, apply annotation harmonization and set relevant labels to use later on.
- 00_prepare_data_for_harmonization: get clean metadata for all datasets
- 01_Annotation_Harmonization
  - ipf_label_harmonization: 5-level harmonization ipf cell type annotations
  - nucleus_label_harmonization: 5-level harmonization nucleus cell type annotations
- 02_set_labels_IPF: set labels used for integration and normalize counts
- 02_set_labels_nucleus: set labels used for integration and normalize counts
- prepare_healthy_data: set labels used for integration and normalize counts

#### 02_Integration
Code to run de-novo integration (scANVI) and transfer-learning approach (scArches), as well as reference atlas creation.
- 00_reference_scANVI_model: create reference atlas
- 01_healthy_integration_pipeline: learn scANVI, control scANVI, scArches, control scArches for healthy single-cell data
- 02_IPF_integration_pipeline: learn scANVI, control scANVI, scArches, control scArches for disease single-cell data
- 03_nucleus_integration_pipeline: learn scANVI, control scANVI, scArches, control scArches for healthy single-nucleus data

#### 03_Milo
Run differential abundance testing with Milo. Calculate differential embedding scores and perform further analysis and comparison of the different approaches, including, but not limited to, differential expression (DE) testing and gene set enrichment analysis (GSEA). Create main figures for thesis.
- Milopy_calculation: perform Milo calculation pipelin for all trained models on origin (reference/query) and on disease status for IPF data. Compare significance of reference/query integration vs. disease/healthy integration for the disease extension
- Milopy_Healthy_Plots_DE: analyse output of Milo, create relevant plots for thesis, perform Differential Expression analysis for healthy extension
- Milopy_IPF_Plots_DE: analyse output of Milo, create relevant plots for thesis, perform Differential Expression analysis for disease extension
- Milopy_Nucleus_Plots_DE: analyse output of Milo, create relevant plots for thesis, perform Differential Expression analysis for nucleus extension
- Get_plots_for_thesis: create plots of the atlases showing cell type annotations

#### 04_Analysis
Notebooks to test out preliminary changes to the main code:
 - 01_a_Celltype_selection: Notebook to test out the cell type level selection approach with merging PAGA-graph connected cell types to their higher levels 
 - 01_b_Cell_type_level_validation: First experimental analysis of cell label levels (too coarse? too fine?)
 - 02_Table_rejection: Notebook that tests out sample rejection based on cell types present in the sample. First a prelimiary lookup table is calculated that can be saved.
 - 03_Overintegration_analysis: Some prelimiary analysis of overintegration of healthy samples 

#### 05_MapQC
- MappingAssessment: Main package for the pipeline
- Template_dataset_analysis: Template for analyisis for reference + query adata

#### 06_Results
- MapAnalyzer_Results: very first results of the pipeline on selected datasets. No correction for sample size implemented. Global rejection threshold of samples based on 5th percentile of reference samples translated to a cell type count rejection threshold.
- MapAnalyzer_Results_2.0: New pipeline but old cell type selection
- MapAnalyzer_Results_3.0: New pipeline + new cell types -> preliminary results
- Dataset overview: Overview of query datasets and their rejection rates in healthy samples