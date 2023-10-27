library(yaml)

config <- yaml.load_file("integration_config.yml")
source(paste0(config$scripts_dir, "config.r"))
source(paste0(config$scripts_dir, "map_query_modular.r"))

config <- get_config(config)

integrated_embedding <- get_integrated_embedding(
  reference_path = config$reference_path,
  ref_filename = config$ref_filename,
  query_path = config$query_path,
  runs_dir = config$runs_dir,
  filter_config = config$anchor_filter,
  subset_hvg = config$subset_hvg, 
  save_full_object = config$save_full_object,
  seed = config$seed
)