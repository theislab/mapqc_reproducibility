File structure

Example file name: {wiq/cps}_{dataset_id}_{parameters}.png

Labels
- `wiq` = k-NN correlation within a query

- `cps` = cluster preservation score

- `dataset_id` = identifier of query dataset 

- `parameters`:
	For `cps`:
         - findnn = whether the nearest neighbours have been computed via `FindNeighbors` and louvain
                    clusters found with `FindClusters` in Seurat.
	                If not the case, then file is named cps_{dataset_id}.png and the neighbours that I used
                    are the same as `sc.pp.neighbors` (and louvain clusters are the same as `sc.tl.louvain`)

        For `wiq`:
         - `30pc` = 30 PC used for PCA (i.e. same as SCANVI embedding)
         - `50pc` = 50 PC used for PCA 
