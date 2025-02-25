# MapQC: a metric to assess the quality of a query-to-reference mapping

MapQC is a metric that can be used to determine whether a a mapping of a single-cell query dataset onto an existing, large scale reference, was successful or not. By quantifying each *query* cell's distance to the reference, as compared to within-reference distances to the reference (the "baseline"), mapQC offers an interpretable distance score per cell. Cell's that are more distant from the reference than expected based on the baseline either show remaining batch effects, or constitute an unseen cell type or state. 

All analyses as included in the manuscript (including figure generation) can be found in the [analyses](./analyses) directory. For the paper manuscript, version v0.1.0-manuscript of this repository was used.

Note that we are currently in the process of re-organizing and cleaning up the mapqc source code, an updated version will be pushed soon. This version will include installation instructions, instructions on how to use mapQC, and tutorial notebooks. Stay tuned!

In case of questions, submit an issue to this github repository.
