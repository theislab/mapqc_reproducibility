
from MappingAssessment import label_transfer
import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import datetime
import os
import scipy
import itertools
import statsmodels.stats.multitest as mtc
from collections import defaultdict

class AdataHandler:

    '''Class to prepare adata for MapQC pipeline:
    - setting standarized column names
    - selection of appropriate level labels
    - label transfer from reference to query '''
    def __init__(
        self,
        adata, #input adata
        ident = "test" ,#just a name to identify files/create folder
        workspace= f"Data/MapAnalyzer_Results/",
        log=True, #enable logging 
    ): 
        
        #set logging variables
        self.log = log
        if log:
            directory = f"{ident}"
            path = os.path.join(workspace, directory)
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                print("Logger directory already exists")
            self.logfile = open(f"{path}logfile.txt", "a")
        self.ident = ident
        self.workspace = workspace
        self.__adata = adata
    
    #function to standarize adata obs columns
    # TODO: set type hints
    def standarize_columns(
            self,
            query_dataset,
            condition_name=None,
            cell_type_col = "cell_label",
            dataset_col = "dataset",
            sample_col="sample", 
            condition_col = "condition",
            subject_col ="subject_ID", 
            disease_exclude = []
        ):

        ''' 
        Function to make column names compliant with the MapQC nomenclature.
        query_dataset: name of the query dataset to be analyzed
        condition_name: Name of the condition in query dataset. If all healthy default is None.
        cell_type_col: name of the column with working cell_type labels
                    will usually be calculated in the piepline but if user wants custom cell types they can provide a label      
        dataset_col: name of the column with dataset of n of the cells # what is n?
        sample_col: name of the column with sample name
        condition_col: name of the column denotiong disease/healthy. 
                    only healthy and disease in question should be present
        subject_col: name of the column denoting the subject/patient/donor/healthy_volunteer ID
        disease_exclude: in case of multiple disease datasets, the other diseases can be exluded

        '''
        self.write("Standarizing columns...")
        
        #remove unwanted datasets
        adata = self.__adata[~(self.__adata.obs[condition_col].isin(disease_exclude))].copy()
       
        # get distinction between query and reference
        adata.obs["query_or_reference"] = "reference"
        adata.obs.loc[
            adata.obs[dataset_col] == query_dataset, "query_or_reference"
        ] = "query"

        # get binary health status
        adata.obs["disease_or_healthy"] = "healthy"
        if condition_name != None:
            adata.obs.loc[
                adata.obs[condition_col] == condition_name, "disease_or_healthy"
            ] = "disease"
            
        #either rename provided cell_label column or don't and warn the user that 'set_cell_type_levels' needs to be run then
        if type(cell_type_col) == str:
            adata.obs.rename(mapper={cell_type_col:"cell_label"},inplace=True,axis=1)
        else:
            self.write("No cell_type column set. Please make sure to run 'set_cell_type_levels' to obtain a unified cell type column 'cell_label'")
        
        #rename the other columns accordingly
        adata.obs.rename(mapper={subject_col:"subject_ID"},inplace=True,axis=1)
        adata.obs.rename(mapper={dataset_col:"dataset"},inplace=True,axis=1)
        adata.obs.rename(mapper={sample_col:"sample"},inplace=True,axis=1)
        self.write("Finished column standarizing\n")

        #save adata object into the class as a variable - FIXME adata is saved as class attribute, not variable.
        self.__adata = adata
        del adata


    def check_columns(self,adata):
        ''' function that checks wether the columns in a 
        provided adata (e.g saved) object already match the standards'''
        # TODO: if not maybe rerun standardization pipeline again?

        self.write("Checking columns...")
        if "dataset" not in adata.obs:
            return False
        if "sample" not in adata.obs:
            return False
        if "query_or_reference" not in adata.obs:
            return False
        if "disease_or_healthy" not in adata.obs:
            return False
        if "subject_ID" not in adata.obs:
            return False
        if "cell_label" not in adata.obs:
            return False 
        return True
    
    # TODO: set type hints
    def transfer_labels(self,embedding_column="X",transfer_columns=["cell_label"],pickle=False):
        ''' Function that transfers labels from reference onto query via weightedKNN transfer in a separate script.
        Advisory to run 'set_cell_type_levels' first and transfer 'cell_label' column.
        embedding_column: name of the column with the embedding (e.g ScANVI) representation. 
                            If embedding is in main matrix X write 'X'.
        transfer_columns: columns from which to transfer labels onto query.
                            Can be multiple (e.g for multiple annotation levels) but using default 'cell_label' column is advisable
        pickle: If transfer_labels has been previously ran on e.g another column, 
                the process of transfer can be sped up by using a pickle of column independant precalculations (same query).'''

        if "query_or_reference" not in self.__adata.obs.columns :
            raise Exception("Column 'query_or_reference' not found in adata. Please make sure to run 'standarize_columns'" )
        else:
            
            self.write("Transferring labels...")
            
            ref_adata = self.__adata[self.__adata.obs.query_or_reference == "reference"]

            self.write("Train KNNModel")
            self.knn_model = label_transfer.weighted_knn_trainer(
                ref_adata,
                f"{embedding_column}",
                "cell_label",
                n_neighbors=50, # FIXME: could be avoided.
            )
            ref_obs = ref_adata.obs
            del ref_adata

            query_adata = self.__adata[self.__adata.obs.query_or_reference == "query"]
            self.write("Transfer Labels") # FIXME: why streamed twice? 
            pickle=pickle

            #transferring labels column by column. Using pickle after first column!
            for label in transfer_columns:
                self.write(f"transferring {label}") # FIXME: counts to three now.
                # FIXME: specify arg names for consistency
                labels, uncert = label_transfer.weighted_knn_transfer(
                    query_adata,
                    f"{embedding_column}",
                    ref_obs,
                    label,
                    self.knn_model,
                    numpy_pickle=pickle,
                )
                
                #setting new columns with transferred labels
                if label != "cell_label":
                    self.__adata.obs[ f"{label}_transf"] = np.nan
                    self.__adata.obs.loc[labels.index, f"{label}_transf"] = labels.to_numpy()
                else:
                    self.__adata.obs.loc[labels.index, f"{label}"] = labels.to_numpy()
                
                #setting new columns witn label transfer uncertainity:
                #how sure or unsure the label is. 
                #If cell surrounded by hetergenous cell types: high uncertainty   
                #homogenous: low uncertainty

                self.__adata.obs[f"{label}_uncert"] = np.nan
                self.__adata.obs.loc[uncert.index, f"{label}_uncert"] = uncert.to_numpy()

                pickle=True

            del query_adata

    # TODO: set type hints
    def set_cell_type_levels(
            self,
            connect_threshold,
            annotation_columns,
            finest_level_column, # FIXME: How's is this supposed to be provided at runtime?
            unlabeled_tag, 
            query=False,
            transfer_columns=None
        ):
        ''' Function that decided on annoation level for cell type annotations. 
            Takes in all columns for the different annotation levels and decides on the best levels for a cell type annotation branch based on the reference.
            Algorithm in LabelLeveling Class.
            Also can set the labels for the transfered query labels after deciding on appropriate levels based only on the reference.
            connect_threshold: levels of connection (based on PAGA graph) between two cell types to merge into parent cell type
                            the lower the coarser the final labels will be
            annotation_columns: columns of cell label annotation ordered by level (lowest to highest) e.g ["ann_level_1","ann_level_2",...]
            finest_level_column: column that annotates each cell by the finest label available
            unlabeled_tag: annotation of cells that have no annotation e.g "unlabeled" or "None"
            query: If True labels will be set in the query to levels decided on previously. # FIXME: are they really 'true' labels?
                Usually wouldn't be used as label levels are advisory set before label transfer. 
            transfer_columns: if labels have been transferred before deciding on appropriate levels, 
                            transferred columns need to be provided to set 'cell label columns'
        '''
        if "query_or_reference" not in self.__adata.obs.columns:
                raise Exception("Column 'query_or_reference' not found in adata. Please make sure to run 'standarize_columns'" )
        if hasattr(self,"logfile"):
            label_leveling = LabelLeveling(self.__adata,annotation_columns,finest_level_column,unlabeled_tag,self.logfile)
        else:
            label_leveling = LabelLeveling(self.__adata,annotation_columns,finest_level_column,unlabeled_tag,None)
        
        if query == False: #select levels based on reference
            self.__adata.obs["cell_label"] = np.nan
            self.__adata.obs.loc[self.__adata.obs.query_or_reference=="reference","cell_label"] = label_leveling.set_cell_labels(connect_threshold)
        
        elif query == True and transfer_columns!= None: #using transferred columns to set the levels
            self.__adata.obs.loc[self.__adata.obs.query_or_reference=="query","cell_label"] = label_leveling.set_transferred_cell_labels(transfer_columns)
        
        elif query ==True: #falling back on annotation columns in query assuming they match annotation columns in reference # FIXME: `else` would fit better here
            self.__adata.obs.loc[self.__adata.obs.query_or_reference=="query","cell_label"] = label_leveling.set_transferred_cell_labels(annotation_columns)
    
    def get_adata(self):
        return self.__adata
    
    # TODO: set type hints
    def write(self, text):
        #helperfunction to write output in logger
        if self.log:
            print(f"{datetime.datetime.now()}\t{text}")
            text = text.encode("utf-8")
            print(f"{datetime.datetime.now()}\t{text}", file=self.logfile)
            self.logfile.flush()
        else:
            print(f"{datetime.datetime.now()}\t{text}")

class LabelLeveling:
    '''Helper class containing the label levelling algorithm and necessary helper functions.
     Main purpose: set levels based on reference,
     but can also apply changes to query columns if label transfer happened on all levels'''
    
    # TODO: set type hints
    def __init__(
            self,
            adata,
            annotation_columns,
            finest_level_column,
            unlabeled_tag,
            logfile=None
        ): 
        ''' 
        annotation_columns: columns of cell label annotation ordered by level (lowest to highest) e.g ["ann_level_1","ann_level_2",...]
        finest_level_column: column that annotates each cell by the finest label available
        unlabeled_tag: annotation of cells that have no annotation e.g "unlabeled" or "None"
        # FIXME: missing `logfile` 
        '''
   
        if logfile == None: ## if logfile passed logging will be enabled
            self.log = False
        else:
            self.log = True
            self.logfile = logfile

        self.__adata=adata
        self.annotation_columns = annotation_columns
        self.finest_level_column=finest_level_column
        self.dict_parents = self.get_dict_parents()
        self.dict_children = self.get_dict_children()
        self.unlabeled_tag = unlabeled_tag

    def get_dict_parents(self):
        ''' based on the annotation columns a dictionary is calculated of the form dict_parents[child]= (immediate) parent'''
        dict_parents={}
        for i in range(len(self.annotation_columns)-1,1,-1):
            for cat in self.__adata.obs[self.annotation_columns[i]].cat.categories:
                parent = self.__adata.obs.loc[self.__adata.obs[self.annotation_columns[i]]==cat,self.annotation_columns[i-1]][0] # TODO: setting each cat to parent(cat)?
                dict_parents[cat]=parent
        del dict_parents["None"]
        return dict_parents

    def get_dict_children(self):
        ''' reverses the dict_parents dictionary creating a dictionary of the form dict_children[parent]=[immediate child 1, immediate child 2,...]'''
        dict_children = defaultdict(list)
        for key, val in sorted(self.dict_parents.items()): # FIXME: why the sorting?
            dict_children[val].append(key)
        return dict_children

    def get_level(self,cell_label):
        # TODO: What's the tree structure? Binary or more complex?
        '''returns the level of the annotation from 0 to n_columns-1'''
        for i in range(len(self.annotation_columns)-1,-1,-1):
            if cell_label in self.__adata.obs[self.annotation_columns[i]].cat.categories:
                return i

    def get_parent(self,ct1,ct2):
        '''function to determine if two cell types are related either:
            share same parent -> returns common parent
            one is ancestor of the other (parent, grandparent) -> return the cell type that is parental/ancestral'''
        
        #check if one cell type is None type -> defining ancestry not possible 
        if ct1 == self.unlabeled_tag or ct2 == self.unlabeled_tag:
            return None

        #get both cell types levels
        l1 = self.get_level(ct1)
        l2 = self.get_level(ct2)

        #if they are on the same level they can either share a parent or not
        if l1 == l2:
            if l1==0 and l2==0: #if they are both of lowest level (e.g immune and epthelial), ancestry not possible
                return None
            if self.dict_parents[ct1] == self.dict_parents[ct2]:
                return self.dict_parents[ct2]
        
        #if either one is of higher level, 
        #iteratively go though parent dictionary until 
        #either parent is reached or tmp becomes lowest level that has no parent 
        #or the cell_type with the higher level is found and returned as parent
        if l1<l2:
            tmp = ct2
            while tmp in self.dict_parents.keys():
                tmp = self.dict_parents[tmp]
                if tmp==ct1:
                    return ct1
            return None
        
        if l1>l2:
            tmp = ct1
            while tmp in self.dict_parents.keys():
                tmp = self.dict_parents[tmp]
                if tmp==ct2:
                    return ct2
            return None

        # TODO: l1 != l2 and sharing common parent that is not eihter of the ccell types 
        # Tentative pc:
        # 1. Wlog assume ct_x and ct_y attaining levels l_x, l_y resp. with l_x < l_y;
        # 2. Decrease l_y to l_y' (corresp. to ct_y') s.t. l_y' = l_x;
        # 3. Iteratively get p(ct_x) and p(ct_y') and until a common parent is found.

    def get_edges(self,graph,groups):
        '''function translates a binary adjacency graph between the groups 
        into a list of edges with names of the group elements (cell labels)

        graph: binary adjacency graph as numpy.matrix 
        # FIXME: shouldn't it be a more efficient way to store this? Furthermore, it may be removed in the future

        groups: list of cell label categories'''
        #remove below diagonal to avoid double edges
        graph = np.tril(graph) 
        idx = np.where(graph.flatten()==1)[0]
        edgelist=[]
        for i in idx:
            i = np.unravel_index(i,graph.shape)
            edgelist.append((groups[i[0]],groups[i[1]])) # FIXME: it appears that a pair of labels is appended here, not edges
        return edgelist
    

    def set_cell_labels(self,connect_threshold): 
        '''core algorithm of the class:
            iteratively connects edges with connectivity >= connect_threshold
            and levels node cell types up to their parent 
            if they are related until all such edges disappear
            Assumption that interconnected cell types are harder to 
            distinguish when mapping leading to higher uncertainty

            1. set cells to their finest annotation (i.e. highest level)
            2. calculate PAGA graph and get the edges
            3. merge edges of related nodes with connectivity >= connect_threshold 
               set all children of parental cell type in the edge to parental cell type
            4. repeat 2 on new cell labels until no changes occur in cell type composition
            '''

        #Step 1
        adata_sub = self.__adata[self.__adata.obs["query_or_reference"] == "reference"].copy() # FIXME: change to sensible variable name
        adata_sub.obs["cell_label"] = adata_sub.obs[self.finest_level_column]
        categories = adata_sub.obs["cell_label"].astype("category").cat.categories # FIXME: change to sensible variable name
        self.write(f"starting with cell labels:\n{list(categories)}\n")

        threshold = connect_threshold # FIXME: unnecessary renaming
        iteration = 1
        same = False # FIXME: change to sensible variable name

        sc.pp.neighbors(adata_sub,n_neighbors=15,n_pcs=30) # Needed to run PAGA

        while same == False:
            #Step 2
            self.write(f"Iteration {iteration}\n")
            adata_sub.obs["cell_label"] = adata_sub.obs["cell_label"].astype(str) # FIXME: **
            sc.tl.paga(adata_sub,groups="cell_label")
            graph = adata_sub.uns["paga"]["connectivities"].toarray() # FIXME: change to sensible variable name `paga_graph`
            
            graph[np.where(graph>=threshold)]=1
            graph[np.where(graph<threshold)]=0
            

            edgelist = self.get_edges(graph,adata_sub.obs[f"cell_label"].cat.categories) # FIXME: unnecessary f-string formatting 
            #Step 3
            for edge in edgelist:
                parent = self.get_parent(edge[0],edge[1]) 
                if parent != None:
                    self.write(f"merging related cell types {self.dict_children[parent]} to parental cell type {parent}\n")
                    adata_sub.obs["cell_label"] = adata_sub.obs["cell_label"].astype(str) # ** FIXME: why twice? 
                    adata_sub.obs.loc[adata_sub.obs["cell_label"].isin(self.dict_children[parent]),"cell_label"] = parent
                    adata_sub.obs["cell_label"] = adata_sub.obs["cell_label"].astype("category") # FIXME: why is this reconversion needed at each iteration?

            #cleanup for cell types that were not merged 

            # "or set to their parental cell type because more than one generation away:"
            # FIXME: why would it be the case that cts more than one generation away can't be assigned to a parent cell type?

            #they get identified and set to their ancestral cell type if it is present in the new cell label list 
            for i in itertools.combinations(adata_sub.obs["cell_label"].cat.categories,2):
                if self.get_level(i[0]) == self.get_level(i[1]): # FIXME: why?
                    pass
                else:
                    parent = self.get_parent(i[0],i[1])
                    if parent != None:
                        self.write(f"merging related cell types {i[0]} and {i[1]} to parental cell type {parent}\n")
                        if parent == i[0]:
                            adata_sub.obs["cell_label"] = adata_sub.obs["cell_label"].astype(str)
                            adata_sub.obs.loc[adata_sub.obs["cell_label"]==i[1],"cell_label"] = parent
                            adata_sub.obs["cell_label"] = adata_sub.obs["cell_label"].astype("category")     
                        else:
                            adata_sub.obs["cell_label"] = adata_sub.obs["cell_label"].astype(str)
                            adata_sub.obs.loc[adata_sub.obs["cell_label"]==i[0],"cell_label"] = parent
                            adata_sub.obs["cell_label"] = adata_sub.obs["cell_label"].astype("category")     
            
            #Step 4
            if len(adata_sub.obs["cell_label"].cat.categories)!=len(categories):
                same = False
                categories = adata_sub.obs["cell_label"].cat.categories
                current_categories = list(adata_sub.obs["cell_label"].cat.categories) # FIXME: redundant assignment
                self.write(f"current cell labels: {current_categories}\n")
            else:
                same = all(adata_sub.obs["cell_label"].cat.categories == categories)
                if same:
                    current_categories = list(adata_sub.obs["cell_label"].cat.categories) # FIXME: redundant assignment
                    self.write(f"final cell labels: {current_categories}\n")
            iteration+=1

        labels = adata_sub.obs["cell_label"]
        del adata_sub
        return labels

    def set_transferred_cell_labels(self,transfer_columns): 
        '''
        If label transfer is done before the 'cell_label' column is set by the appropriate cell type levels, 
        the query cells can be reannotated here based on previous calculation of the 'cell_label' column based on the reference.
        transfer_columns: list of columns sorted by level lowest to highest ["ann_level_1_transf","ann_level_2_transf",...]
        '''
        adata_sub = self.__adata[self.__adata.obs.query_or_reference =="query"].copy()
        if "cell_label" not in self.__adata.obs.columns:
            raise Exception("Column 'query_or_reference' not found in adata. Please make sure to run 'set_cell_labels'" )
        
        if len(transfer_columns) != len(self.annotation_columns):
            raise Exception( "Number of transfered columns must match annotation columns")

        #for each label present in calculated cell_label column, 
        #get level and set query cells that match this label on this level to the label   
        labels = list(self.__adata.obs.cell_label.astype("category").cat.categories)
        for label in labels:
            lvl = self.get_level(label) #0-len(columns-1)
            column = transfer_columns[lvl]
            adata_sub.obs.loc[adata_sub.obs[column]==label,"cell_label"]=label
        
        labels = adata_sub.obs["cell_label"]
        del adata_sub
        return labels

    def write(self, text):
        '''helperfunction to write output'''
        if self.log:
            print(f"{datetime.datetime.now()}\t{text}")
            text = text.encode("utf-8")
            print(f"{datetime.datetime.now()}\t{text}", file=self.logfile)
            self.logfile.flush()
        else:
            print(f"{datetime.datetime.now()}\t{text}")
        
        

class MappingScoring:

    '''Central class of the package: 
    - calculation of mixing scores
    - calculation of the null distribution
    - assessment of dataset mapping quality per sample and cell type'''
    
    def __init__(
        self,
        adata_handler, 
        log=True): 
        '''adata_handler: AdataHandler object together with the adata inside of it
        log: whether or not to log the activities in a file (recommended for troubleshooting!!)
        '''

        self.log = log
        self.ident = adata_handler.ident #naming that identifies the workspace and creates the folder
        self.workspace = adata_handler.workspace
        if log:
            today = datetime.date.today()
            date = today.strftime("%d_%m_%Y")
            directory = f"{self.ident}/"
            path = os.path.join(self.workspace, directory)
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                print("Logger directory already exists")
            self. path = path
            self.logfile = open(f"{path}logfile.txt", "a")
            self.baseline=False #wether or not a baseline has been calculated or added




        self.transformation = None #parameters for sample size correction
        adata = adata_handler.get_adata()
        #check columns of adata object: must be standarized
        if adata_handler.check_columns(adata):
            self.__adata = adata
        else:
            raise Exception("Not all columns are present in .obs. Please run AdataHandler.standarize_columns")
    
    
    def get_subsampled_adata(self,sub_adata, sample, n_neighbors, exclude_dataset=True,embedding="X"):
        '''Function to get a subsampled adata object
        sub_adata: subsample of adata (e.g only of certain cell type)
        sample: name of the sample to retain (e.g query sample)
        n_neighbors: number of neighbors in KNN graph
        exclude_dataset: whether to exclude the dataset where the sample stems from
        embedding: embedding column to use'''

        if  exclude_dataset is True:
            dataset = sub_adata.obs.loc[
                sub_adata.obs["sample"] == sample, "dataset"
            ].to_numpy()[0]

            sub = sub_adata[
                (sub_adata.obs["sample"] == sample) | ~(sub_adata.obs["dataset"] == dataset) # FIXME: why this logical statement?
            ].copy()

            sc.pp.neighbors(sub, n_neighbors=n_neighbors, n_pcs=30, use_rep=f"{embedding}")
        else:
            sub = sub_adata

        return sub
    
    def count_subjects(self,sub_adata):
        '''function that counts subjects with at least 15 cells per cell label
        sub_adata: subsampled adata (e.g per cell_label)'''
        sub = sub_adata[sub_adata.obs.query_or_reference == "reference"]
        table = (
            sub.obs.groupby("subject_ID")
            .agg({"subject_ID": "count"})
            .rename(columns={"subject_ID": "count"})
        )
        subjects = table.loc[table["count"] >= 15].index.tolist()
        return len(subjects)

    def get_adata(self):
        return self.__adata

    def set_adata(self,adata):
        self.__adata = adata

    def calculate_mixing_scores(
            self, 
            adata = None,
            n_neighbors = 15, 
            min_subjects=40, 
            samples = None, 
            reinitialize=True , 
            embedding = "X", 
            verbose = False , 
            overwrite=True ,
            save = False
        ):
        '''function to calculate mixing scores of query or reference
        adata: adata to calculate the mixing score on. Passed e.g when calculating the null distribution (baseline)
        n_neighbors: size of neighborhood graph considered for mixing score calculation
        min_subjects: number of subjects to be sufficently (>= n_neighbors) represented in a cell type
        samples: batch of samples to analyze, if None all query samples are analyzed
        reinitialize: whether to reset the score columns or not. Not resetting when baseline is calculated.
        embedding: embedding column name. use 'X' when embedding in main countmatrix
        verbose: If True, log is printed in output stream
        overwrite: whether or not to overwrite the self.__adata attribute
        save: whether or not to save to workspace 
        '''
        
        self.cell_types = []
        if adata is None:
            adata = self.__adata
       
        if reinitialize:
            adata.obs["score"] = np.nan

        stats_cell_type = []
        min_cells_sample = n_neighbors
        if samples is None:
            samples = adata[adata.obs.query_or_reference == "query"].obs["sample"].cat.categories


        for cell_type in adata.obs["cell_label"].astype("category").cat.categories:
            self.write(f"Analyzing {cell_type}",verbose)

            # subset anndata to cell_type
            sub_adata = adata[(adata.obs["cell_label"] == cell_type)].copy()

            count_sub = self.count_subjects(sub_adata)

            #check if enough subjects are in reference for that cell type 
            if count_sub < min_subjects:
                self.write(f"skipping {cell_type} with {count_sub} subjects",verbose)
                continue
            #add cell type to list of cell types that made the selection for later use
            self.cell_types.append(cell_type)


            for sample in samples:
                
                #check if sample has enough cells in cell type
                count_sample_cells = len(sub_adata[sub_adata.obs["sample"] == sample])

                if (
                    count_sample_cells < min_cells_sample
                ):  # check if sample has enough cells to analyze
                    self.write(f"       skipping {sample} with {count_sample_cells} cells",verbose)
                    continue
                else:
                    self.write(f"       Analysing {sample}",verbose)

               #get adata with query_sample + reference excluding dataset of sample origin
                sub = self.get_subsampled_adata(
                    sub_adata,
                    sample= sample,
                    n_neighbors = n_neighbors,
                    exclude_dataset = True,
                    embedding=embedding
                )

                #assign temporary query and reference labels to not intefere with original ones (especially for baseline calculation)
                sub.obs["tmp_qr"] = "reference"
                sub.obs.loc[sub.obs["sample"] == sample, "tmp_qr"] = "query"

                #get indices of reference and query cells
                pandas_table = sub.obs.tmp_qr.copy()
                pandas_table = pandas_table.reset_index() # FIXME: redundant method application
                query_idx = pandas_table.loc[pandas_table.tmp_qr == "query"].index.tolist()
                ref_idx = pandas_table.loc[
                    pandas_table.tmp_qr == "reference"
                ].index.tolist()

                # score calculation:
                query_connect = sub.obsp["connectivities"][query_idx] # What's its shape?
                reference_matrix = query_connect[:, ref_idx]
                #sum of reference connectivities per cell
                ref_sum = np.sum(reference_matrix, axis=1)
                #sum of all reference connectivities per cell / sum of all connectivities per cell
                scores = ref_sum / np.sum(query_connect, axis=1)


                #set value in adata
                adata.obs.loc[
                    sub.obs.loc[sub.obs.tmp_qr == "query"].index, "score"
                ] = np.ravel(scores)

        if save:
            adata.obs["score"].to_csv(f'{self.workspace}/{self.ident}/scores.csv' )

        if overwrite:
            self.__adata = adata
        else:
            return adata


     
    def calculate_baseline(
            self,
            n_neighbors = 15,
            min_subjects=40, 
            embedding = "X", 
            overwrite = True,
            verbose=False, 
            save=False
        ):
        ''' function to calculate the null distribution mixing scores (baseline)
        calculates the mixinng scores for the reference samples while excluding samples from the same dataset from calculation
        n_neighbors: size of neighborhood graph considered for mixing score calculation
        min_subjects: number of subjects to be sufficently (>= n_neighbors) represented in a cell type
        embedding: embedding column name. use 'X' when embedding in main count matrix
        overwrite: whether or not to overwrite the self.__adata attribute
        verbose: If True, log is printed in output stream
        save: whether or not to save the baseline scores to workspace '''

        #subsample to reference and copy to temporary adata
        adata = self.__adata[self.__adata.obs.query_or_reference == "reference"].copy()
        grouped_samples = adata.obs.groupby("dataset")["sample"].apply(set).apply(list) # What is the data type of `grouped_samples`?
        # Is a `sample` present in more than one dataset?
        sample_list = []
        for i in itertools.zip_longest(*grouped_samples.tolist()): # Why would you convert `grouped_samples` to a list again?
            x = np.asarray(i)
            sample_list.append(x[np.where(x != None)])
        samples = adata.obs["sample"].cat.categories
        
        for samples in sample_list:
            adata = self.calculate_mixing_scores(
                adata = adata,
                n_neighbors = n_neighbors,
                min_subjects= min_subjects,
                samples = samples,
                reinitialize=False,
                embedding = embedding,
                verbose = verbose,
                overwrite = False
            )
         
        #log that baseline has been calculated
        self.baseline = True


        if save: 
            adata.obs["score"].to_csv(f'{self.workspace}/baseline.csv' ) # FIXME: change `["score"]` to more sensible name
            group = self.group_adata(adata) 
            file = open(f'{self.workspace}/cell_types.txt',"w")
            for ct in group.cell_label.cat.categories:
                print(ct,file=file)
            file.close()

        
        if overwrite:
             self.__adata[adata.obs.index].obs = adata.obs
        else:
            return adata

    def calculate_transformation(self,save =True):
        '''Correction for mixing score - sample size relationship.
        The log-linear relationship between sample size and sample mixing score is modelled 
        and the residuals to the regression line are calculated -> Transformed mean mixing score
        The transformed score is then used for all further decisions.
        In this function only the parameters for the models are calculated. The transformation is applied later on.
        save: save transformation to workspace '''
        
        #baseline needs to be calculated
        if self.baseline is False:
            raise Exception("You need to add or calculate the baseline first")
        else:
            #subsample adata to reference
            baseline = self.__adata[self.__adata.obs['query_or_reference']=="reference"]
            #group adata by sample and cell label + log_count = log(sample size)
            baseline_g = self.group_adata(adata=baseline)
            transformation=[]
            for ct in self.cell_types:
                #linear regression for all cell types
                x = baseline_g[baseline_g.cell_label == ct]["log_count"]
                y = baseline_g[baseline_g.cell_label == ct]["score"]
                slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
                transformation.append((ct,slope,intercept,r,p,std_err))
            #saving transformation as pandas column     
            transformation = pd.DataFrame(transformation,columns=("cell_label","slope","intercept","r","p","std_err"))
            transformation.set_index("cell_label",inplace=True)
            self.transformation=transformation
            
            if save:
                transformation.to_csv(f'{self.workspace}/transformation.csv' )
            
   
    # To be performed whether `save=False` and `transformation_path` is other than `{self.workspace}/transformation.csv`?
    def add_transformation(self,transformation_path):
        ''' add transformation from disc'''
        self.transformation = pd.read_csv(transformation_path,index_col="cell_label")
   
   
   
    def add_baseline(self,baseline_path,cell_label_path):
        '''add baseline from disc
        baseline_path: path to baseline
        cell_label_path: path to list of cell labels with sufficent subject count'''
        adata = self.__adata
        baseline_data = pd.read_csv(baseline_path)
        baseline_data.set_index("Unnamed: 0", inplace=True) # FIXME: change to have more sensible index column name 
        baseline_data = baseline_data.loc[adata[adata.obs.query_or_reference == "reference"].obs.index.tolist()]
        self.__adata.obs.loc[adata.obs.query_or_reference == "reference",
        ["query_prop", "query_count", "deviation", "score"]] = baseline_data
        cell_type_file = open(cell_label_path,"r")
        cell_types = []
        self.write("Baseline consists of following cell types:\n")
        for ct in cell_type_file.readlines():
            self.write(f'{ct}\n')
            cell_types.append(ct.strip("\n"))
        self.cell_types = cell_types
        self.baseline =True

    def add_scores(self,score_path):
        '''add precalculated scores.
        score_path: path to scores'''
        score_data = pd.read_csv(score_path)
        score_data.set_index("Unnamed: 0", inplace=True) # FIXME: change to have more sensible index column name
        self.__adata.obs.loc[(self.__adata.obs.query_or_reference == "query"),"score"] = score_data

    def filter_sample_mean(self,adata, sample_name, cutoff=0.05, verbose=False): 
        '''
        This function takes one sample and iterates over the cell labels with two purposes:
        - for each cell type the score is compared to the 5th percentile of the cell type baseline distribution
        - documents the cell types that are present in the query sample for later steps

        # FIXME: as of the purposes of this function I would consider changing its name.

        adata: adata passed from wrapper function
        sample_name: name of the sample in question
        cutoff: percentile at which sample is "rejected" at cell type
        verbose: wether to print output to out-stream
        '''
        feedback = [] #collects the rejection/acceptance data for each cell type 
        aggr_data = self.__preprocess_adata(adata, only_reference=False,transform=True)#adata.obs grouped by cell label and sample with mean scores 
        cell_type_list=[]
        
        for ct in self.cell_types:
            #obtain reference sample scores for cell type
            aggr_data = self.__preprocess_adata(adata, only_reference=False,transform=False)
            baseline_scores = aggr_data[
                (aggr_data["cell_label"] == ct) & (aggr_data["query_or_reference"] == "reference")
            ].score.to_numpy() 

            #obtain sample score mean for query sample and cell type
            sample_mean = aggr_data[
                (aggr_data["cell_label"] == ct) & (aggr_data["sample"] == sample_name)
            ].score.to_numpy()

            #check if cell type present in sample
            if len(sample_mean) == 0:
                continue
            sample_mean = sample_mean[0] # FIXME: merge with `sample_mean` instantiation above

                        
            #obtain sample health status
            sample_status = aggr_data[ # FIXME: why referencing `aggr_data` again? Can't use `sample_mean`
                (aggr_data["cell_label"] == ct) & (aggr_data["sample"] == sample_name)
            ].disease_or_healthy.to_numpy()[0] # FIXME: why is this straightforwardly merged unlike above?
     
            # add cell type to list
            cell_type_list.append(ct)

            #obtain cutoff value for cell type amongus based on null distribution/baseline
            cutoff_val = np.percentile(baseline_scores, cutoff * 100) # FIXME: specify cutoff via percentile directly instead of fraction


            if verbose:
                self.write(f"   {ct}")
                self.write(f"   cutoff_val ={cutoff_val} rejected: {sample_mean < cutoff_val}")
                
                self.write(f"   sample_mean: {sample_mean}")
            
            
            #reject if sample mean below cutoff value
            if sample_mean < cutoff_val:
                rejected = True
                label = f"{sample_status} rejected",
                
            else:
                rejected = False
                label = f"{sample_status} accepted",

            
            feedback.append(
                    (
                        ct,
                        sample_name,
                        sample_mean,
                        sample_status,
                        rejected,
                        label,
                    )
                )

        return feedback,cell_type_list

    def sample_rejection(self, cutoff=0.05, verbose = False):
        '''
        Function that manages the sample rejection.
        First rejection status for each cell type is obtained via 'filter_sample_mean'.

        Then for each sample, a custom threshold based on the cell types present in the query sample are obtained via 'get_threshold'.
        # FIXME: move `get_threshold` further up. I would expect to find it more easily and know what it does.

        Based on that threshold the overall sample is either rejected or accepted.
        cutoff: percentile at which sample is "rejected" at cell type
        verbose: wether to print output to out-stream
        '''
        adata = self.__adata
        #obtain query sample list
        samples = (
            adata[adata.obs.query_or_reference == "query"].obs["sample"].cat.categories
        )
        

        feedback_data = [] #collect feedback from filter_sample_mean
        sample_thresholds={} #collecting sample custom thresholds 
        for s in samples: # FIXME: change to more sensible variable name - e.g. `sample_id`
            sample_feedback, cell_type_list = self.filter_sample_mean(adata, s, cutoff, verbose=verbose)
            self.write(f'{s}: {cell_type_list}')
            if len(cell_type_list) > 0:
                custom_threshold=self.get_threshold(cell_type_list)
                #collect feedback in feedbacklist for all samples and cell types
                feedback_data.extend(sample_feedback)
                sample_thresholds[s]=custom_threshold


        #create dataframe from feedback list
        rejection_report = pd.DataFrame(
            feedback_data,
            columns=["cell_label", "sample", "mean", "health_status","excluded", "label"],
        ) # FIXME: use "rejected" vs "excluded" for consistency

        #creating sample report
        sample_report = rejection_report.groupby("sample").agg({"health_status":"first", "excluded":["sum","count"]}) 
        # FIXME: no need to add the "excluded"" column here - "fraction" can be computed without saving anything else
        sample_report["fraction"] = sample_report["excluded"]["sum"]/sample_report["excluded"]["count"] 

        # add rejection to obs variable   
        samples = list(sample_report.index)
        adata.obs[f"rejection_status"] = np.nan # FIXME: unnecessary f-string formatting
        adata.obs.cell_label = adata.obs.cell_label.astype("category")
        sample_report["accepted"]=True

        #adding sample rejection status to sample report vbased on threshold
        for s in samples: # FIXME: change to more sensible variable name - e.g. `sample_id`
            if sample_report.loc[s,"fraction"].to_numpy() >= sample_thresholds[s]:
                adata.obs.loc[(adata.obs["sample"] == s),f"rejection_status"] = "rejected" # FIXME: unnecessary f-string formatting
                sample_report.loc[s,"accepted"]=False
                # Why do we need both a "rejection_status" and an "accepted" field?
            else:
                adata.obs.loc[(adata.obs["sample"] == s),f"rejection_status",] = "accepted"
        
        self.__adata = adata
        self.report = rejection_report
        self.sample_report = sample_report

        return rejection_report,sample_report
    
    # FIXME: definitely change the name of the following function - not descriptive
    def cell_type_reporting(self,cutoff_pval = 0.05):
        '''
        This function checks wether the distribution of healthy query samples in a cell type differs from the reference distribution.
        This is done via a one-sided Kolmogorov-smirnov-test that checks whether query_healthy < reference healthy.
        Multiple testing correction is done.
        cutoff_pval: pvalue at which a cell type is flagged # FIXME: why not simply `pval` - be more precise on cutoff meaning?
        '''
        adata = self.__adata
        try:
            value = self.rejected
        except AttributeError:
            print("calculate rejected samples before cell type level rejection!") 
            return


        cell_type_ex = [] #list of exluded cell types FIXME: change to sensible variable name
        for ct in self.cell_types:
            #subsample adata to cell type
            adata_ct = adata[
                (adata.obs.cell_label == ct) & ~(adata.obs["sample"].isin(self.rejected))
            ].copy()

            #group by sample and cell type # FIXME: technically the grouping is done by sample only here  
            grouped = self.__preprocess_adata(adata=adata_ct,only_reference=False) # FIXME: change to sensible variable name

            #obtain query (healthy) sample scores
            query_scores = grouped[
                (grouped["query_or_reference"] == "query")
                & (grouped["disease_or_healthy"] == "healthy") # (**) FIXME: flag "healthy" as `disease_status` to pass as argument
            ].score.to_numpy()

            #obtain reference sample scores
            reference_scores = grouped[
                (grouped["query_or_reference"] == "reference")
                & (grouped["disease_or_healthy"] == "healthy")
            ].score.to_numpy()
            
            #perform one-sided kolmogrov smirnov test
            if min(len(query_scores),len(reference_scores)) > 0:
                pval = scipy.stats.ks_2samp(
                    reference_scores, query_scores, alternative="less"
                ).pvalue
                
                cell_type_ex.append((ct, pval)) # FIXME: change to sensible variable name (not yet excluded cell types)

        #multiple testing correction  
        res = pd.DataFrame(cell_type_ex, columns=["cell_label", "pvalue"]).sort_values("pvalue")
        res["pvals_corr"] = mtc.multipletests(res["pvalue"], 0.05, "fdr_bh")[1]
        
        self.affected_cell_types = res[res["pvals_corr"] < cutoff_pval].cell_label
        self.cell_type_report = res

        return self.affected_cell_types, res

    # FIXME: why two identical functions for the same purpose? See (**) above for how to fix
    def disease_effect_reporting(self,cutoff_pval = 0.05):
        '''
        This function checks whether the distribution of disease query samples in a cell type differs from the reference distribution -> possible disease effect
        This is done via a one-sided Kolmogorov-smirnov-test that checks whether query_disease < reference healthy.
        
        Cell types that have been flagged in 'cell_type_reporting' are excluded as batch effect and disease effect could be confounded 
        # IDEA: could there be to assess this difference, maybe with downstream marker analyses?

        Multiple testing correction is done.
        cutoff_pval: pvalue at which a cell type is flagged
        Works analogous to 'cell_type_reporting'
        '''
        adata = self.__adata
        try:
            value = self.affected_cell_types
        except AttributeError:
            print("Assess residual batch effect in healthy samples before accessing disease effect!") 
            return
        try:
            value = self.rejected
        except AttributeError:
            print("Calculate rejected samples before cell type level rejection!") 
            return
        

        cell_type_d =[]

        for ct in self.cell_types:
            if ct in self.affected_cell_types.tolist():
                self.write(f"skipping cell type {ct} for disease effect evaluation: confounded healthy")
                continue
            
            adata_ct = adata[(adata.obs.cell_label == ct) & (~(adata.obs["sample"].isin(self.rejected))|(adata.obs['disease_or_healthy'] == "disease"))].copy()
            
            grouped = self.__preprocess_adata(adata=adata_ct, only_reference=False)
            
            #obtain disease query sample scores
            query_scores = grouped[
                (grouped["query_or_reference"] == "query")
                & (grouped["disease_or_healthy"] == "disease")
            ].score.to_numpy()
           
            #obtain reference sample scores
            reference_scores = grouped[
                (grouped["query_or_reference"] == "reference")
                & (grouped["disease_or_healthy"] == "healthy")
            ].score.to_numpy()
            
            #One-sided kolmogrov smirnov test
            if min(len(query_scores),len(reference_scores)) > 0:
                pval = scipy.stats.ks_2samp(
                    reference_scores, query_scores, alternative="less"
                ).pvalue
                cell_type_d.append((ct, pval)) 
        
        res_d = pd.DataFrame(cell_type_d, columns=["cell_label", "pvalue"]).sort_values("pvalue")
        if len(res_d) == 0:
            print("no cell_types for disease effect accessment")
            self.disease_cell_types = []
            self.disease_report = res_d
            return
        #multiple testing corrcetion
        res_d["pvals_corr"] = mtc.multipletests(res_d["pvalue"], 0.05, "fdr_bh")[1]

        self.disease_cell_types = res_d[res_d["pvals_corr"]<cutoff_pval].cell_label
        self.disease_report = res_d

    
    def calculate_threshold_table(self,cutoff_sample=0.05):
        '''
        This function calculates a table of the form n_sample x n_cell_label for the reference where:
        1: accepted
        0: rejetced:
        -1: cell type not present
        cutoff_sample: cutoff at which a sample is rejected in a cell_type
        '''
        baseline = self.__adata[self.__adata.obs["query_or_reference"]=="reference"]
        group = baseline.obs.groupby(["cell_label","sample"]).agg({"score":"mean","dataset":"count"}).reset_index(drop=False)
        if self.baseline is False:
            raise Exception("Add or calculate baseline first!")
        
        ##add check for transformation
        transform = self.transformation.reset_index(drop=False,inplace=False)
        perc_5={}
        #calculate the transformed scores for each cell type and derive threshold per cell type
        for ct in self.cell_types:
            slope = transform.loc[transform.cell_label==ct,"slope"].to_numpy()
            intercept = transform.loc[transform.cell_label==ct,"intercept"].to_numpy()

            # Why is the fitted score computed at the dataset level? Moreover, there is no trace of "score"-related kwords in the `slope` multiplier
            group.loc[group.cell_label == ct,"score"] = group.loc[group.cell_label == ct,"score"] - (group.loc[group.cell_label == ct,"dataset"]*slope+intercept)

            perc_5[ct]=np.nanpercentile(group.loc[group.cell_label == ct,"score"].to_numpy(),cutoff_sample*100)
        
        
        #create table 
        list_cat= self.cell_types.copy()
        list_cat.append("sample")
        table= pd.DataFrame(data=[],columns=list_cat)

        samp=list(group["sample"].cat.categories)
        table["sample"]=samp
        table.set_index("sample",verify_integrity=True,inplace=True)

        #set values in table. 0: rejected 1: accepted and  nan -> -1
        for ct in  self.cell_types:
            th = perc_5[ct]
            sub= group[group.cell_label==ct].score.to_numpy()
            sub_no_nan = (sub>=th).astype(int)
            idx= np.argwhere(np.isnan(sub))
            sub_no_nan[idx]=-1
            table[ct]=sub_no_nan
        
        table.reset_index(drop=False,inplace=True)
        return table
    

    def get_threshold(self,cell_type_list,weighted=True):
        '''
        function that obtains a custom threshold based on the cell types present in the query.
        collects the counts of rejected cell types for reference samples. 
        The 95th percentile of counts is obtained (samples in reference that have more rejected cell types than 95% of the reference)
        This is then the threshold for the query sample.
        cell_type_list: list of cell types present in query
        weighted: weight samples by how many cell types from cell_type_list are present in that reference sample
        '''
        cell_type_list.append("sample") # FIXME: change to more sensible variable name (e.g. `cell_type_list_query`)
        #subsample precalculated table
        table = self.lookup_table[cell_type_list].copy()
        table.set_index("sample",inplace=True)
        cell_type_list.remove("sample")
        table = table.T

        count_list=[]
        for sample in table.columns:
            #0: rejected 1: accpeted -1: undefined 
            counts=table[sample].value_counts().reindex([0,1,-1], fill_value=0)
            #weight: number of defined cell types (0 and 1)
            weight = len(cell_type_list)-counts[-1]
            if weight > 0:
                value= counts[0]/(counts[0]+counts[1])
                if weighted:
                    count_list.extend([value]*weight)
                else:
                    count_list.extend([value])

        return np.percentile(count_list,95)
  
    def binomial_rejection(self,n_rejected, n_healthy, cutoff_probability=0.05):
        '''
        This function decides whether or not a datatset is rejected based on the number of rejected cell types. # FIXME: this does not match the return type
        We calculate the probabilty that at least n_rejected samples are rejected given the probabilty that a sample is rejected randomly at p = 0.05.
        n_rejected: number of rejected samples
        n_healthy: number of all healthy samples
        cutoff_probability: resulting probablity at which the dataset is rejected
        '''
        #1-BINOM.DIST(O2-1,N2,0.042,TRUE)

        n_rej = max([n_rejected-1,0])
        return 1-scipy.stats.binom.cdf(n_rej,n_healthy,cutoff_probability)



    def analyze_mapping(
            self,
            cutoff_sample = 0.05, 
            cutoff_cell_type = 0.05,
            cutoff_disease = 0.05
        ):
        ''' core function: analysis pipeline of the dataset
        cutoff_sample: cutoff quantile at which a sample is rejected in the cell type
        cutoff_cell_type: cutoff p-value at which a cell type is flagged as batch effect affected
        cutoff_disease: cutoff p-value at which a cell type is flagged as possibly affected by disease'''


        adata = self.__adata # FIXME: not used
        failed = False # FIXME: not used
        report_file = open(f'{self.path}report.txt',"w")
        
        #if no score transformation is added, recalculate
        if self.transformation is None:
            self.calculate_transformation()

        print("Parameters:",file = report_file)
        print(f"Rejection of cell_type per sample:\t{cutoff_sample}",file=report_file)

        #calculate lookup table 
        self.lookup_table = self.calculate_threshold_table(cutoff_sample)

        #calculate sample rejection
        self.sample_rejection(cutoff=cutoff_sample) #rewrite this to work with table
        self.rejected = list(self.sample_report[self.sample_report.accepted==False].index) #all rejected samples

        #filter out only healthy/control rejected
        rej_healthy =[]
        for rej in self.rejected: # FIXME: sensible variable name 
            if "healthy" in self.sample_report.loc[rej,"health_status"].tolist():
                rej_healthy.append(rej)
        sample_report = self.sample_report
        
        
        #calculate probabilty that at least n_rejeted samples are rejected given the probablity 0.05
        n_healthy = sum((sample_report.health_status=="healthy").to_numpy())[0]
        n_rejected = len(rej_healthy)
        rej_probability = self.binomial_rejection(n_rejected,n_healthy,cutoff_probability=0.05)

        #if it is unlikely (<= 0.05) that at least number of healthy samples is rejected at random we reject the dataset
        if rej_probability <= 0.05: # (*)
            print(f"{n_rejected} of healthy({n_healthy}) samples rejected (p = {np.round(rej_probability,3)}):\n please remap with different parameters,a different method or reintegrate",file=report_file)
            for rej in rej_healthy: # FIXME: sensible variable name
                print(f"{rej}:\n{self.sample_report.loc[rej]}\n",file = report_file )
            report_file.flush()
            #return #removed experimentally

        # FIXME: wouldn't it be streamed anyways even when (*) evaluates to `True`?
        print(f"{n_rejected} of healthy({n_healthy}) samples rejected (p = {np.round(rej_probability,3)}):\ndataset is good to go",file=report_file)
        
        #otherwise we report the samples that got rejected and move on
        for rej in rej_healthy:
            print(f"{rej}:\n{self.sample_report.loc[rej]}",file = report_file )
        report_file.flush()

        print("\n\n",file = report_file)
        print(f"After exclusion of rejected samples cell_types\nthat show signs of batch effect are filtered at p = {cutoff_cell_type}", file=report_file)


        #assessment which cell types can be affected by residual batch effect overall
        self.cell_type_reporting(cutoff_pval = cutoff_cell_type)
       
        print("Following cell_types show signs of strong batch effect in healthy samples",file = report_file)
        print("Excluding from disease effect candidate cell_types:",file=report_file)
        print("\n",file=report_file)
        print(f"Cell_type\tbh - corrected p-value",file=report_file)
        
        for ct in self.affected_cell_types: # FIXME: `affected_cell_types` is not descriptive enough - referring to batch-affected cts here?
            pval = self.cell_type_report[self.cell_type_report.cell_label == ct].pvals_corr.to_numpy()[0]
            if pval < 0.001:
                pval = " < 0.001"
            else:
                pval = np.round(pval,3)
            print(f"{ct}\t{pval}", file=report_file)
            report_file.flush()
        
        print("\n\n",file = report_file)
        print(f"After exclusion of rejected samples and batch-effect affected cell_types\ndisease effect is assumed at at p = {cutoff_disease}", file=report_file)
        
        #check for cell types with possible disease effect excluding batch efefct affected cell types
        self.disease_effect_reporting(cutoff_pval = cutoff_disease)
        
        print("Following cell_types show signs possible disease effect:", file = report_file)
        print("\n",file=report_file)
        print(f"Cell_type\tbh - corrected p-value",file=report_file)
       
        for ct in self.disease_cell_types:
            pval = self.disease_report[self.disease_report.cell_label == ct].pvals_corr.to_numpy()[0]
            if pval < 0.001:
                pval = " < 0.001"
            else:
                pval = np.round(pval,3)
            print(f"{ct}\t{pval}", file=report_file)
        report_file.flush()


    # Bring up for enhanced interpretability of more detailed analyses using this function
    def group_adata(self,adata):
        ''' helperfunction to group adata'''

        group_adata = (
            adata.obs.groupby(["cell_label", "sample"])
            .agg({"score": "mean", "dataset": "first", "subject_ID": "count"})
            .dropna()
            .reset_index(drop=False)
        )
        group_adata["cell_label"] = group_adata["cell_label"].astype("category")
        group_adata["cell_label"] = group_adata["cell_label"].cat.remove_unused_categories()
        group_adata = group_adata.rename(columns={"subject_ID": "count"})
        group_adata["log_count"] = np.log(group_adata["count"])
        return group_adata


    # Bring up for enhanced interpretability of more detailed analyses using this function
    def __preprocess_adata(self,adata, only_reference=True,transform = True):
        '''helperfunction to group adata by cell type and sample and apply transformation if transform=True
        only_reference: exclude query from aggregated table
        transform: apply correction to sample means'''

        group = adata.obs.groupby(["cell_label", "sample"]).agg(
            {"score": "mean", "query_or_reference": "first","dataset":"first", "disease_or_healthy": "first","subject_ID":"count"}
        )
        group = group.rename(columns={"subject_ID":"count"})
        aggr_data = group.reset_index(drop=False)
        aggr_data.loc[~aggr_data.cell_label.isin(self.cell_types),"score"]=np.nan
        aggr_data.dropna(inplace=True)

        aggr_data["cell_label"] = aggr_data["cell_label"].astype("category")
        aggr_data["cell_label"] = aggr_data["cell_label"].cat.remove_unused_categories()
        categories = aggr_data["cell_label"].cat.categories.tolist()
        categories.sort()
        aggr_data["cell_label"] = aggr_data["cell_label"].cat.reorder_categories(categories)


        if transform:
            self.write("Correcting the for sample size...")
            aggr_data["score_untrans"] = aggr_data["score"]
            for ct in aggr_data.cell_label.cat.categories:
                trans = self.transformation.loc[ct]

                aggr_data.loc[aggr_data.cell_label == ct,"score"] = aggr_data.loc[aggr_data.cell_label == ct,"score"] - (np.log(aggr_data.loc[aggr_data.cell_label == ct,"count"])*trans.slope+trans.intercept )

     
        if only_reference:
            aggr_data = aggr_data[aggr_data["query_or_reference"] == "reference"]
        return aggr_data


    def write(self, text,verbose = True):
        '''helperfunction to write output'''
        if self.log:
            if verbose:
                print(f"{datetime.datetime.now()}\t{text}")
            text = text.encode("utf-8")
            print(f"{datetime.datetime.now()}\t{text}", file=self.logfile)
            self.logfile.flush()
        else:
            if verbose:
                print(f"{datetime.datetime.now()}\t{text}")



