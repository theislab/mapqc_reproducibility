# This script performs the mapping of a query to HypoMap

# Load libraries
library(Seurat)
library(ggplot2)

# Load functions
source("mapping_utils.r", chdir = TRUE)

start_time <- Sys.time()

# Get SLURM Job ID
job_id <- Sys.getenv("SLURM_JOB_ID")
print(paste0("SLURM Job ID: ", job_id))

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
config <- if (length(args) > 0) args[1] else "default"
print(paste0("Running with config: ", config))

# Create a new directory for this run
runs_dir <- "/path/to/runs/dir" # Replace with your actual directory
run_dir <- file.path(runs_dir, paste0(config, "_", job_id))
dir.create(run_dir)

# Load data
hmp <- readRDS(file = "/path/to/data/hypoMap.rds") # reference
hmp.ref <- hmp[, hmp@meta.data$Diet == "Normal chow"] # subset reference
hmp.query <- readRDS(file = "/path/to/preprocessed/query/preprocessed_query.rds")

# Process data
hmp.ref <- process_data(hmp.ref)
hmp.query <- process_data(hmp.query, is_reference = FALSE)

# Default parameters for mapping
anchors_params <- list(
  reference = hmp.ref,
  query = hmp.query,
  normalization.method = "LogNormalize",
  reference.reduction = "spca",
  dims = 1:50,
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

# Find anchors
f_start_time <- Sys.time()
anchors <- do.call(FindTransferAnchors, anchors_params)
cat("FindTransferAnchors time:", round(Sys.time() - f_start_time, 0), "\n")

# Save anchors
saveRDS(anchors, file = file.path(run_dir, paste0(config, "_anchors.rds")))

# Map query
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

# Subset to variable features
hmp.ref$id <- "ref"
hmp.query$id <- "query"
hmp.ref <- hmp.ref[VariableFeatures(hmp.ref), ]
hmp.query <- hmp.query[VariableFeatures(hmp.query), ]
# FIXME: I think this merging is necessary to handle the
# `hmp.refquery` object more easily downstream, but happy
# to discuss alternatives to avoid line 8 in `mapping_utils.r`
hmp.refquery <- base::merge(hmp.ref, hmp.query)

# Merge embedded features
hmp.refquery[["spca"]] <- base::merge(hmp.ref[["spca"]], hmp.query[["ref.spca"]])
# (*) FIXME: the following line could also be avoided, IMO
# since line 95 references "spca" to compute the UMAP 
hmp.refquery[["umap"]] <- base::merge(hmp.ref[["umap_scvi"]], hmp.query[["ref.umap"]])

# Run UMAP using SPCA as embedding
hmp.refquery <- RunUMAP(hmp.refquery, reduction = "spca", dims = 1:50, reduction.name = "redone_umap")

# Save plots
save_plot(hmp.refquery, file.path(run_dir, paste0(config, "_recalculated_mapped_umap.png")), reduction = "redone_umap")
# The following could be avoided; see (*)
save_plot(hmp.refquery, file.path(run_dir, paste0(config, "_mapped_umap.png")), reduction = "umap")

# Save data
saveRDS(hmp.refquery, file = file.path(run_dir, paste0(config, "_refquery.rds")))
# FIXME: the following could be extracted from `hmp.refquery`
saveRDS(hmp.query, file = file.path(run_dir, paste0(config, "_query.rds")))

# Print total execution time
cat("Total time:", round(Sys.time() - start_time, 0), "\n")
