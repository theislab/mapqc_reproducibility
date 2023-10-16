library(Seurat)
library(ggplot2)

process_data <- function(data, is_reference = TRUE) {
  data <- NormalizeData(data)
  # FIXME: for `query`, `FindVariableFeatures` used to merge the final object
  # by highly variable feature ids (lines 81-83) in `query_mapping.r`
  data <- FindVariableFeatures(data)
  if (is_reference) {
    data <- ScaleData(data)
    data <- FindNeighbors(data, reduction = "scvi", graph.name = "wsnn", verbose = TRUE)
    data <- RunSPCA(data, graph = "wsnn", verbose = TRUE)
  }
  return(data)
}

save_plot <- function(data, filename, reduction = "umap") {
  p <- DimPlot(data, group.by = "id", reduction = reduction, order = c("query", "ref"), raster = FALSE)
  print(p)
  ggsave(filename = filename, plot = p, dpi = 500)
}
