# Script Name: config.r
# Description: This script contains utility functions for configuring the mapping environment.

# ==============================================================================
# Global Configurations
# ==============================================================================

# This section contains global configurations and utility functions
# required for setting up the mapping environment.

# ------------------------------------------------------------------------------
# Function: parse_args
# Description: This function retrieves the configuration parameters from 
#              command line arguments.
# Parameters: None
# Returns: parsed_args (list) - The parsed command line arguments.
# ------------------------------------------------------------------------------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed_args <- list()
  
  for (i in seq(1, length(args), by = 2)) {
    key <- gsub("^--", "", args[i])
    value <- args[i + 1]
    parsed_args[[key]] <- value
  }
  
  # Handle logical values and set defaults
  parsed_args[["anchor_filter"]] <- ifelse(
    is.null(parsed_args[["anchor_filter"]]), "default", parsed_args[["anchor_filter"]]
  )
  parsed_args[["subset_hvg"]] <- ifelse(
    is.null(parsed_args[["subset_hvg"]]), FALSE, as.logical(parsed_args[["subset_hvg"]])
  )
  return(parsed_args)
}

# ------------------------------------------------------------------------------
# Function: get_config
# Description: This function retrieves the configuration parameter from 
#              command line arguments. If no argument is provided, it defaults to "default".
# Parameters: None
# Returns: config (character) - The configuration parameter.
# ------------------------------------------------------------------------------
get_config <- function(config_file = config) {
  # Parse arguments
  args <- parse_args()

  # Override config with command line arguments (if provided)
  
  # `anchor_filter`
  if (!is.null(args$anchor_filter)) {
    config$anchor_filter <- args$anchor_filter
  }
  # `subset_hvg`
  if (!is.null(args$subset_hvg)) {
    config$subset_hvg <- args$subset_hvg
  }
  return(config)
}

# ------------------------------------------------------------------------------
# Function: create_run_dir
# Description: This function creates a directory for the current run based on
#              the provided configurations and job ID. 
# Parameters: runs_dir (character) - The root directory for all runs.
#             config (character) - The configuration parameter.
#             job_id (character) - The job ID.
# Returns: run_dir (character) - The path to the newly created directory.
# Example: run_dir <- create_run_dir("/path/to/runs/dir", "config_name", "job_id")
# ------------------------------------------------------------------------------
create_run_dir <- function(runs_dir, config, job_id) {
  # Construct the path
  run_dir <- file.path(runs_dir, paste0(config, "_", job_id))
  
  # Check if directory exists; if not, create it
  if (!dir.exists(run_dir)) {
    dir.create(run_dir, recursive = TRUE)
  }
  
  return(run_dir)
}