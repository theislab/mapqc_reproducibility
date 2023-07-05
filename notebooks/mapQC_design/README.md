# mapQC design notebooks

These are notebooks that explore, test and visualize concepts that are later implemented into the package.

**01_celltype_selection**: This notebook explores the concept of selecting cell type levels based on a PAGA graph.
For quantifying mapping success and failure, a unified cell type label column needs to be created that is used throughout the MapQC analysis. In a multi-levelled cell type annotation, the levels appropriate for analysis are decided on by merging cell labels with strong PAGA-graph-connection and their sister types into their parental cell type.

**02_table_rejection**: This notebook introduces the concept of sample rejection based on reference samples. The table `samples x cell_types` documemts whether a the mixing score of cell type in a sample is within the norm (1), below the 5th percentile (0) or the cell type is not represented in that sample (-1). The table can be subsampled to cell types represented in the query sample and serves as a basis for a sample-specific rejection threshold.
