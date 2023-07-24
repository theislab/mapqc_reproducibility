library(Seurat)
library(ggplot2)
library(caret)

process_data <- function(data) {
  data <- NormalizeData(data)
  data <- FindVariableFeatures(data)
  data <- FindNeighbors(data, reduction = 'scvi', graph.name = 'wsnn', verbose = TRUE)
  data <- ScaleData(data)
  data <- RunSPCA(data, graph = 'wsnn', verbose = TRUE)
  return(data)
}

save_plot <- function(data, filename, reduction = "umap") {
  p <- DimPlot(data, group.by = 'id', shuffle = TRUE, reduction = reduction, raster.dpi = c(1024, 1024), order = c("ref", "query"))
  print(p)
  ggsave(filename = filename, plot = p, dpi = 500)
}

evaluate_label <- function(hmp.query, label, config, run_dir) {
  predicted_label <- paste0("predicted.", label)

  # Create the plot
  p1 <- DimPlot(hmp.query, group.by = predicted_label, shuffle = TRUE, raster.dpi = c(1024, 1024)) + NoLegend() + theme(aspect.ratio = 0.8)
  p2 <- DimPlot(hmp.query, group.by = label, shuffle = TRUE, raster.dpi = c(1024, 1024)) + NoLegend() + theme(aspect.ratio = 0.8)
  jpd <- p1 + p2
  print(jpd)
  ggsave(filename = file.path(run_dir, paste0(config, "_", label, "_predicted_plot.png")), plot = jpd, dpi = 400)
  
  common_levels <- sort(unique(c(hmp.query[[label]][[label]], hmp.query[[predicted_label]][[predicted_label]])))
  actual <- factor(hmp.query[[label]][[label]], levels = common_levels)
  predicted <- factor(hmp.query[[predicted_label]][[predicted_label]], levels = common_levels)
  cm <- confusionMatrix(table(actual, predicted))

  write.csv(as.data.frame(t(cm$overall)), file = file.path(run_dir, paste0(config, "_", label, "_overall_stats.csv")))
  write.csv(cm$byClass, file = file.path(run_dir, paste0(config, "_", label, "byClass_stats.csv")))
}

#####################################################################################
# Script starts here

start_time <- Sys.time()

job_id <- Sys.getenv("SLURM_JOB_ID")
print(paste0("SLURM Job ID: ", job_id))

args <- commandArgs(trailingOnly = TRUE)
config <- if(length(args) > 0) args[1] else "default"
print(paste0("Running with config: ", config))

# Create a new directory for this run
runs_dir <- "/lustre/groups/ml01/datasets/projects/2023_jan_mapqc/runs"  # Replace with your actual directory
run_dir <- file.path(runs_dir, job_id)
dir.create(run_dir)

hmp <- readRDS(file="/home/icb/jan.engelmann/atlas_extension/data/hypomap/hypoMap.rds")

hmp.ref <- hmp[, hmp@meta.data$Dataset != "Rupp10x" & hmp@meta.data$Diet == "Normal chow"]
hmp.query <- hmp[, hmp@meta.data$Dataset == "Rupp10x"]

hmp.ref <- process_data(hmp.ref)
hmp.query <- process_data(hmp.query)

# Default parameters
anchors_params <- list(
    reference = hmp.ref,
    query = hmp.query,
    normalization.method = "LogNormalize",
    reference.reduction = "spca",
    dims = 1:50,
    verbose = TRUE
  )

if(config == "no_filter") {
  # No filtering, should be ok according to:
  # https://github.com/satijalab/seurat/issues/5206
    anchors_params$k.filter <- NA
  } else if(config == "speed_filter") {
    anchors_params$k.filter <- 25
    anchors_params$k.score <- 15
    anchors_params$max.features <- 100
    anchors_params$n.trees <- 25
}

f_start_time <- Sys.time()
anchors <- do.call(FindTransferAnchors, anchors_params)
print("FindTransferAnchors time:")
print(round(Sys.time() - f_start_time, 0))


saveRDS(anchors, file = file.path(run_dir, paste0(config, "_anchors.rds")))

hmp.query <- MapQuery(
  anchorset = anchors,
  query = hmp.query,
  reference = hmp.ref,
  reference.reduction = "spca", 
  reduction.model = "umap_scvi",
  refdata = list(
          C7_named = "C7_named",
          C25_named = "C25_named",
          C66_named = "C66_named"
      )
)

labels <- c("C7_named", "C25_named", "C66_named")
for (label in labels) {
    tryCatch({
    evaluate_label(hmp.query, label, config, run_dir)
  }, error = function(e) {
    print(paste("Caught an error: ", e))
  })
}

hmp.ref$id <- 'ref'
hmp.query$id <- 'query'
hmp.refquery <- merge(hmp.ref, hmp.query)
hmp.refquery[["spca"]] <- merge(hmp.ref[["spca"]], hmp.query[["ref.spca"]])
hmp.refquery[["umap"]] <- merge(hmp.ref[["umap_scvi"]], hmp.query[["ref.umap"]])

save_plot(hmp.refquery, file.path(run_dir, paste0(config, "_mapped_umap.png")))

hmp.refquery <- RunUMAP(hmp.refquery, reduction = 'spca', dims = 1:50, reduction.name = "redone_umap")
save_plot(hmp.refquery, file.path(run_dir, paste0(config, "_recalculated_mapped_umap.png")), reduction = "redone_umap")

saveRDS(hmp.refquery, file = file.path(run_dir, paste0(config, "_refquery.rds")))
saveRDS(hmp.ref, file = file.path(run_dir, paste0(config, "_ref.rds")))
saveRDS(hmp.query, file = file.path(run_dir, paste0(config, "__query.rds")))

print("Total time:")
print(round(Sys.time() - start_time, 0))
