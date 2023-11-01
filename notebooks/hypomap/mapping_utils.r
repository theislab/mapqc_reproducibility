# Script Name: mapping_utils.r
# Description: This script contains functions for processing, integrating, 
# and visualizing an integrated object.

# Load required libraries
library(Seurat)
library(ggplot2)

# ------------------------------------------------------------------------------
# Function: process_data
# Description: Processes the given data. If the data is a reference, it performs
#              additional processing steps.
# Parameters: data (Seurat object) - The data to be processed.
#             is_reference (logical) - TRUE if the data is a reference, FALSE otherwise.
# Returns: data (Seurat object) - The processed data.
# ------------------------------------------------------------------------------
process_data <- function(data, is_reference = TRUE) {
  data <- NormalizeData(data)
  if (is_reference) {
    data <- FindVariableFeatures(data)
    data <- ScaleData(data)
    data <- FindNeighbors(data, reduction = "scvi", graph.name = "wsnn", verbose = TRUE)
  }
  return(data)
}

# Function: set_default_anchors_params
# Description: Sets default parameters for the FindTransferAnchors function if 
#              none are provided.
# Parameters: anchors_params (list) - The parameters for FindTransferAnchors.
#             ref (Seurat object) - The reference dataset.
#             query (Seurat object) - The query dataset.
#             dims (numeric vector) - The dimensions to use.
# Returns: anchors_params (list) - The updated list of parameters.
# ------------------------------------------------------------------------------
set_default_anchors_params <- function(anchors_params, config, ref, query, dims) {
  if (length(anchors_params) == 0) {
    anchors_params <- list(
      reference = ref,
      query = query,
      normalization.method = "LogNormalize",
      reference.reduction = "spca",
      dims = dims,
      verbose = TRUE
    )

    if (config == "no_filter") {
      # No filtering, should be ok according to:
      # https://github.com/satijalab/seurat/issues/5206
      anchors_params$k.filter <- NA
    } else if (config == "speed_filter") {
      anchors_params$k.filter <- 25
      anchors_params$k.score <- 15
      anchors_params$max.features <- 100
      anchors_params$n.trees <- 25
    }
  }
  return(anchors_params)
}

# ------------------------------------------------------------------------------
# Function: find_anchors
# Description: Finds transfer anchors between the reference and query datasets.
# Parameters: anchors_params (list) - The parameters for FindTransferAnchors.
# Returns: anchors (list) - The list of transfer anchors.
# ------------------------------------------------------------------------------
find_anchors <- function(anchors_params) {
  f_start_time <- Sys.time()
  anchors <- do.call(FindTransferAnchors, anchors_params)
  cat("FindTransferAnchors time:", round(Sys.time() - f_start_time, 0), "\n")
  return(anchors)
}

# ------------------------------------------------------------------------------
# Function: save_anchors
# Description: Saves the transfer anchors to a file.
# Parameters: anchors (list) - The list of transfer anchors.
#             run_dir (character) - The directory to save the anchors to.
#             config (character) - The configuration parameter.
# Returns: None
# ------------------------------------------------------------------------------
save_anchors <- function(anchors, run_dir, config) {
  saveRDS(anchors, file = file.path(run_dir, paste0(config, "_anchors.rds")))
}

# ------------------------------------------------------------------------------
# Function: map_query_data
# Description: Maps the query data to the reference data using the transfer anchors.
# Parameters: anchors (list) - The list of transfer anchors.
#             query (Seurat object) - The query dataset.
#             ref (Seurat object) - The reference dataset.
# Returns: query (Seurat object) - The mapped query dataset.
# ------------------------------------------------------------------------------
map_query_data <- function(anchors, query, ref, save_full_object = FALSE) {
  query <- MapQuery(
    anchorset = anchors,
    query = query,
    reference = ref,
    reference.reduction = "spca",
    reduction.model = "umap_spca"
  )

  if (save_full_object) {
    cat("Removing raw counts in query...", "\n")
    size_init <- object.size(query)

    # `DietSeurat` reduces the memory footprint 
    # of `ref` by removing the row count matrix
    query <- DietSeurat(
      query,
      counts = FALSE,
      data = TRUE,
      scale.data = TRUE,
      features = NULL,
      assays = 'RNA',
      dimreducs = 'ref.spca',
      graphs = NULL,
      misc = TRUE
    )
    
    # Print downsizing ratio
    size_final <- object.size(query)
    cat("Object size reduced by ", (size_final / size_init) * 100, "(%)", "\n")

    return(query)
  } else {
    ref_spca <- ref[["spca"]]@cell.embeddings
    query_spca <- query[["ref.spca"]]@cell.embeddings
  
    rownames(ref_spca) <- paste0("ref_", rownames(ref_spca))
    rownames(query_spca) <- paste0("query_", rownames(query_spca))
    
    # Merge the embeddings
    merged_spca <- rbind(ref_spca, query_spca)
    
    return(merged_spca)
  }
}

# ------------------------------------------------------------------------------
# Function: create_integrated_obj
# Description: Creates an integrated object from the reference and query datasets.
# Parameters: ref (Seurat object) - The reference dataset.
#             query (Seurat object) - The query dataset.
# Returns: refquery (Seurat object) - The integrated object.
# ------------------------------------------------------------------------------
create_integrated_obj <- function(ref, query, subset_hvg = FALSE) {
  ref$id <- "ref"
  query$id <- "query"

  if (subset_hvg) {
    ref <- ref[VariableFeatures(ref), ]
    query <- query[VariableFeatures(query), ]
    refquery <- base::merge(ref, query)
  }
  else {
    refquery <- merge(x = ref, y = query)
  }
  refquery[["spca"]] <- merge(ref[["spca"]], query[["ref.spca"]])
  return(refquery)
}

# ------------------------------------------------------------------------------
# Function: recalculate_umap
# Description: Recalculates the UMAP coordinates and optionally saves a UMAP plot.
# Parameters: refquery (Seurat object) - The integrated object.
#             dims (numeric vector) - The dimensions to use.
#             save_umap (logical) - Whether to save the UMAP plot.
#             run_dir (character) - The directory to save the plot to.
#             config (character) - The configuration parameter.
# Returns: refquery (Seurat object) - The integrated object with updated UMAP coordinates.
# ------------------------------------------------------------------------------
recalculate_umap <- function(refquery, dims, save_umap, run_dir, config) {
  refquery <- RunUMAP(refquery, reduction = "spca", dims = dims, reduction.name = "umap_spca")
  if (save_umap) {
    save_umap_plot(refquery, file.path(run_dir, paste0(config, "_recalculated_umap.png")), reduction = "umap_spca")
  }
  return(refquery)
}

# ------------------------------------------------------------------------------
# Function: save_umap_plot
# Description: Saves a UMAP plot on disk.
# Parameters: data (Seurat object) - The data to be plotted.
#             filename (character) - The name of the file to save the plot to.
#             reduction (character) - The reduction to use for the plot.
# Returns: None
# ------------------------------------------------------------------------------
save_umap_plot <- function(data, filename, reduction = "umap") {
  p <- DimPlot(data, group.by = "id", reduction = reduction, order = c("query", "ref"), raster = FALSE)
  ggsave(filename = filename, plot = p, dpi = 500)
}

# ------------------------------------------------------------------------------
# Function: get_embedding_diff
# Description: Compares embeddings up to a tolerance margin. 
#              Implementation is based on numpy.allclose().
#              https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
# Parameters: emb1 (Matrix) - The first embedding.
#             emb2 (Matrix) - The second embedding.
#             reduction (character) - The reduction to use for the plot.
# Returns: close_check (logical) - TRUE if the check is passed. FALSE otherwise.
# ------------------------------------------------------------------------------
get_embedding_diff <- function(emb1, emb2, rtol = 1e-05, atol = 1e-08) {
  close_check <- abs(emb1 - emb2) <= (atol + rtol * abs(emb2))
  return(all(close_check))
}
