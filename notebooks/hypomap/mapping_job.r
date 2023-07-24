library(Seurat)
library(ggplot2)

process_data <- function(data) {
  data <- NormalizeData(data)
  data <- FindVariableFeatures(data)
  data <- FindNeighbors(data, reduction = 'scvi', graph.name = 'wsnn', verbose = TRUE)
  data <- ScaleData(data)
  data <- RunSPCA(data, graph = 'wsnn', verbose = TRUE)
  return(data)
}

save_plot <- function(data, filename, reduction = "umap") {
  p <- DimPlot(data, group.by = 'id', shuffle = TRUE, reduction = reduction)
  print(p)
  ggsave(filename = filename, plot = p, dpi = 300)
}

hmp <- readRDS(file="/home/icb/jan.engelmann/atlas_extension/data/hypomap/hypoMap.rds")

hmp.ref <- hmp[, hmp@meta.data$Dataset != "Rupp10x" & hmp@meta.data$Diet == "Normal chow"]
hmp.query <- hmp[, hmp@meta.data$Dataset == "Rupp10x"]

hmp.ref <- process_data(hmp.ref)
hmp.query <- process_data(hmp.query)

for(filter in c("no", "yes")) {
  print(paste0("Running with filter: ", filter))

  anchors <- FindTransferAnchors(
    reference = hmp.ref,
    query = hmp.query,
    normalization.method = "LogNormalize",
    reference.reduction = "spca",
    dims = 1:50,
    verbose = TRUE,
    k.filter = ifelse(filter == "no", NA, 100)
  )
  saveRDS(anchors, file = paste0(filter, "_filter_anchors.rds"))

  hmp.query <- MapQuery(
    anchorset = anchors,
    query = hmp.query,
    reference = hmp.ref,
    reference.reduction = "spca", 
    reduction.model = "umap_scvi",
    refdata = list(
            C7_named = "C7_named"
        )
  )
  p1 <- DimPlot(hmp.query, group.by = 'predicted.C7_named', shuffle = TRUE) + NoLegend() + theme(aspect.ratio = 0.8)
  p2 <- DimPlot(hmp.query, group.by = 'C7_named', shuffle = TRUE) + NoLegend() + theme(aspect.ratio = 0.8)
  jpd <- p1 + p2
  print(jpd)
  ggsave(filename = paste0(filter, "_filter_cell_type_transfer.png"), plot = jpd, dpi = 400)

  hmp.ref$id <- 'ref'
  hmp.query$id <- 'query'
  hmp.refquery <- merge(hmp.ref, hmp.query)
  hmp.refquery[["spca"]] <- merge(hmp.ref[["spca"]], hmp.query[["ref.spca"]])
  hmp.refquery[["umap"]] <- merge(hmp.ref[["umap_scvi"]], hmp.query[["ref.umap"]])

  save_plot(hmp.refquery, paste0(filter, "_filter_mapped_umap.png"))

  hmp.refquery <- RunUMAP(hmp.refquery, reduction = 'spca', dims = 1:50, reduction.name = "redone_umap")
  save_plot(hmp.refquery, paste0(filter, "_filter_recalculated_umap.png"), reduction = "redone_umap")

  saveRDS(hmp, file = paste0(filter, "_filter_hypomap.rds"))
}
