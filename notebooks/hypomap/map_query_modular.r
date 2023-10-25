# Script Name: integration_pipeline.r
# Description: This script defines a function to integrate a query dataset with a reference dataset, 
#              and obtain unified embeddings.

# Load scripts
source(paste0(config$scripts_dir, "config.r"))
source(paste0(config$scripts_dir, "mapping_utils.r"))
source(paste0(config$scripts_dir, "reference_creation.r"))

# ------------------------------------------------------------------------------
# Function: get_integrated_embedding
# Description: Integrates a query dataset with a reference dataset to obtain 
#              unified embeddings.
# Parameters: reference_path (character) - The path to the reference Seurat object file.
#             query_path (character) - The path to the query Seurat object file.
#             runs_dir (character) - The directory where the results will be saved.
#             filter_config (character) - Configuration for the filtering of anchors.
#             subset_hvg (logical) - Whether to subset and merge respective HVG in the integrated object.
#             save_umap (logical) - Whether to save the UMAP plot [default: TRUE].
#             dims (numeric vector) - The dimensions to use [default: 1:50].
#             anchors_params (list) - The parameters for FindTransferAnchors [default: list()].
#             seed (int) - Random seed for reproducibility
# Returns: refquery (Seurat object) - The integrated Seurat object with unified embeddings.
# ------------------------------------------------------------------------------
get_integrated_embedding <- function(
  reference_path, 
  query_path, 
  runs_dir,
  filter_config,
  subset_hvg,
  save_umap = TRUE, 
  dims = 1:50, 
  anchors_params = list(),
  seed = NULL
) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  start_time <- Sys.time()
  
  # Obtain SLURM Job ID
  job_id <- Sys.getenv("SLURM_JOB_ID")
  
  # Create directory for this run
  run_dir <- create_run_dir(runs_dir, filter_config, job_id)
  
  # Load data
  ref <- readRDS(file = reference_path)
  query <- readRDS(file = query_path)

  # Process data
  ref <- get_reference_reductions(ref, metadata_field = "Diet", subset_key = "Normal chow")
  query <- process_data(query, is_reference = FALSE)

  # Set default parameters for FindTransferAnchors if not provided
  anchors_params <- set_default_anchors_params(anchors_params, filter_config, ref, query, dims)
  
  # Find anchors
  anchors <- find_anchors(anchors_params)
  
  # Save anchors
  save_anchors(anchors, run_dir, filter_config)
  
  # Map query
  query <- map_query_data(anchors, query, ref)
  
  # Create unified object
  refquery <- create_integrated_obj(ref, query, subset_hvg = subset_hvg)
  
  # Recalculate and save UMAP
  refquery <- recalculate_umap(refquery, dims, save_umap, run_dir, filter_config)

  # Save data
  saveRDS(refquery, file = file.path(run_dir, paste0(filter_config, "_refquery.rds")))
  
  # Print total execution time
  cat("Total time:", round(Sys.time() - start_time, 0), "\n")
  
  return(refquery)
}
