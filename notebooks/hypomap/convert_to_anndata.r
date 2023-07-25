library(Seurat)
library(SeuratDisk)

input_file <- "FILL_IN"
output_file <- "FILL_IN"

seurat_obj <- readRDS(file = input_file)

SaveH5Seurat(seurat_obj, filename = output_file, verbose = TRUE)
Convert(output_file, dest = "h5ad", verbose = TRUE)
