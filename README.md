# MapQC reproducibility repo

This repository includes all code to reproduce the results shown in our mapQC preprint. For the mapQC package, please go to the [mapQC GitHub](https://github.com/theislab/mapqc) page.

## Figures and tables

All figures and tables from the preprint can be reproduced with the code in this repository. Below, I list where each figure/table is generated. Note that these results were generated with mapQC before I turned it into a package, and so the code is slightly different (i.e more complicated and less efficient) than it would be if you would generate it with the pip-installable package.

To reproduce the figures and tables, check out the following notebooks:

**Main figures**
1. Concept figure, no code.
2.  a - e: [analyses/HLCA/Sheppard_analysis/3_analyze_results_poor_mapping.ipynb](analyses/HLCA/Sheppard_analysis/3_analyze_results_poor_mapping.ipynb)<br>
    f - h: [analyses/HECA/3_analyze_mapQC_output.ipynb](analyses/HECA/3_analyze_mapQC_output.ipynb)
3. [analyses/HLCA/Sheppard_analysis/4a_analyze_results_good_mapping.ipynb](analyses/HLCA/Sheppard_analysis/4a_analyze_results_good_mapping.ipynb)
4. [analyses/HLCA/Sheppard_analysis/5a_alternative_metrics_worst_mapping.ipynb](analyses/HLCA/Sheppard_analysis/5a_alternative_metrics_worst_mapping.ipynb) and [analyses/HLCA/Sheppard_analysis/5b_alternative_metrics_best_mapping.ipynb](analyses/HLCA/Sheppard_analysis/5b_alternative_metrics_best_mapping.ipynb)

**Extended data figures**:
1. [analyses/HLCA/Sheppard_analysis/2a_run_mapQC_cellxgene_counts_dataset_kl1.0.ipynb](analyses/HLCA/Sheppard_analysis/2a_run_mapQC_cellxgene_counts_dataset_kl1.0.ipynb)
2. [analyses/HLCA/Sheppard_analysis/3_analyze_results_poor_mapping.ipynb](analyses/HLCA/Sheppard_analysis/3_analyze_results_poor_mapping.ipynb)
3.  a - d, f - h:   [analyses/HLCA/Sheppard_analysis/4a_analyze_results_good_mapping.ipynb](analyses/HLCA/Sheppard_analysis/4a_analyze_results_good_mapping.ipynb)<br>
    e:              [analyses/HLCA/Sheppard_analysis/4b_IPF_SM_in_other_HLCA.ipynb](analyses/HLCA/Sheppard_analysis/4b_IPF_SM_in_other_HLCA.ipynb) 
4. [analyses/HLCA/Sheppard_analysis/5c_metrics_on_downsampled_data_all_cells.ipynb](analyses/HLCA/Sheppard_analysis/5c_metrics_on_downsampled_data_all_cells.ipynb)

**Supplementary tables**
1. [analyses/HLCA/Sheppard_analysis/3_analyze_results_poor_mapping.ipynb](analyses/HLCA/Sheppard_analysis/3_analyze_results_poor_mapping.ipynb)
2. [analyses/HLCA/Sheppard_analysis/4a_analyze_results_good_mapping.ipynb](analyses/HLCA/Sheppard_analysis/4a_analyze_results_good_mapping.ipynb)
3 - 4: [analyses/HECA/3_analyze_mapQC_output.ipynb](analyses/HECA/3_analyze_mapQC_output.ipynb)

## Contact
In case of questions or comments regarding mapQC reproducibility, please submit an issue to this GitHub page. With questions about the package itself, please do so on the mapQC GitHub.

## Citation
[Sikkema et al., bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.05.23.655749v1)
