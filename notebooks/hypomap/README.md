# HypoMap Atlas Conversion

## Environments
use 
- `envs/seurat_r_env.yml` if you only aim to use R
- `envs/seurat_r_python_env.yml` if you want to use separate python notebooks within the same jupyter session

## Reference Mapping
The conversion happens in the [mapping_job.r](https://github.com/theislab/atlas_extension/blob/main/notebooks/hypomap/mapping_job.r) script.

From the repo base directory, it can be run on slurm with:

```
sbatch notebooks/hypomap/run_mapping_job.sbatch <PICK CONFIG HERE>
```

The results get stored under:
```
/lustre/groups/ml01/datasets/projects/2023_jan_mapqc/runs/<SLURM JOB ID>
```

Fill `<PICK CONFIG HERE>` with either `no_filter` or `speed_filter`.

This setting adresses the main bottleneck of the computation, the filtering of the anchors. Due to the large dataset size and high similarity of the current proof of concept with the Rupp10x dataset, many anchors are found and thus filtering takes a long time. This [Seurat Github Issue](https://github.com/satijalab/seurat/issues/5206) hints at that the filtering can be skipped alltogether. Alternatively, the filtering parameters can be changed which `speed_filter` does. 

## Seurat object to Anndata
Use `convert_to_anndata.r`