# Script Name: reference_preprocessing.r
# Description: Processes and computes reductions on the reference.

# Load required libraries
library(Seurat)
library(ggplot2)

# Load external utility functions
source(paste0(config$scripts_dir, "mapping_utils.r"))

# ------------------------------------------------------------------------------
# Function: get_reference_reductions
# Description: Loads a Seurat object from a file, processes it, runs SPCA and UMAP
#              reductions, and either returns the full object or a new object with 
#              only the reductions.
# Parameters: path (character) - The path to the Seurat object file.
#             metadata_field (character) - The metadata field used for subsetting [default: "field"].
#             subset_key (character) - The key used for subsetting [default: "key"].
# Returns: ref (Seurat object) - The full or reduced Seurat object.
# ------------------------------------------------------------------------------
get_reference_reductions <- function(ref, metadata_field = "field", subset_key = "key", save = TRUE) {
  # Subset the Seurat object based on metadata_field and subset_key, if provided
  if (!is.null(subset_key) && metadata_field %in% colnames(ref@meta.data)) {
    ref <- ref[, ref@meta.data[[metadata_field]] == subset_key]
  }

  # Process reference
  ref <- process_data(ref)
  
  # Run SPCA
  ref <- RunSPCA(ref, graph = "wsnn", verbose = TRUE)
  ref <- RunUMAP(ref, reduction = "spca", dims = 1:50, reduction.name = "umap_spca", return.model = TRUE)

  cat("Reducing reference object size...", "\n")
  size_init <- object.size(ref)

  # Streamline object
  ref <- DietSeurat(
    ref,
    counts = FALSE,
    data = TRUE,
    scale.data = TRUE,
    features = NULL,
    assays = 'RNA',
    dimreducs = list('spca', 'umap_spca'),
    graphs = "wsnn",
    misc = TRUE
  )
  
  # Print downsizing ratio
  size_final <- object.size(ref)
  cat("Object size reduced by ", (size_final / size_init) * 100, "(%)", "\n")

  if (save) {
    saveRDS(
      ref, file = file.path(
        config$reference_path, 
        paste0(config$anchor_filter, "_ref_pp.rds")
        )
      )
  }
  
  return(ref)
}
