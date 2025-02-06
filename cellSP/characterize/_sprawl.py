import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import scale
from ._bicluster import LargeAverageSubmatrices, _log_combs, _expand_bicluster_rows, _expand_bicluster, _get_sprawl_score
from ._utils import random_mean_pairs_angle, random_mean_pairs_dist
import timeit
from datetime import timedelta
import shapely.geometry
import multiprocessing as mp

class DotAccessibleDict:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return -1

def _get_slice_data(Z_transcripts, Z_boundary = None, generate_hull = False):
    if not generate_hull:
        vertices = Z_boundary[['vertex_x', 'vertex_y']].values
    else:
        vertices = _generate_convex_hull(Z_transcripts)
    spot_coords = Z_transcripts[['absX', 'absY']].values
    spot_genes = Z_transcripts.gene.values
    return vertices, spot_coords, spot_genes

def _generate_convex_hull(Z):
    hull = ConvexHull(Z[['absX', 'absY']].values)
    vertices = np.array([Z[['absX', 'absY']].values[vertex] for vertex in hull.vertices])
    return vertices


def _parallel_sprawl(args):
    cell_dict, method, count = args[0], args[1], args[2]
    if method == 'Peripheral':
        temp_df = _peripheral(DotAccessibleDict(cell_dict))
    elif method == 'Radial':
        temp_df = _radial(DotAccessibleDict(cell_dict), num_iterations = 1000, num_pairs = 4)
    elif method == 'Punctate':
        temp_df = _punctate(DotAccessibleDict(cell_dict), num_iterations = 1000, num_pairs = 4)
    elif method == 'Central':
        temp_df = _central(DotAccessibleDict(cell_dict))
    else:
        raise ValueError("Invalid method. Options are 'Peripheral', 'Radial', 'Punctate', 'Central'.")
    return method, count, temp_df

    
def run_sprawl(adata_st, methods = ['Peripheral', 'Radial', 'Punctate', 'Central'], threads = 1):
    '''
    Run sprawl on the spatial data to find spatial gene expression patterns.
    Arguments
    ----------
    adata_st : AnnData
        Anndata object containing spatial transcriptomics data.
    methods : list
        List of methods to run sprawl on. Options are 'Peripheral', 'Radial', 'Punctate', 'Central'.
    threads : int
        Number of threads to use.
    '''
    print("Running SPRAWL...")
    start = timeit.default_timer()
    df_transcripts = adata_st.uns['transcripts'].copy()
    n_cells, genes, n_genes = adata_st.to_df().shape[0], adata_st.uns['geneList'], len(adata_st.uns['geneList'])
    sprawl_scores = np.zeros((len(methods), n_cells, n_genes))
    count = 0
    arguments = []
    for n, cell in df_transcripts.reset_index().groupby('uID'):
        cell = cell.groupby('gene').filter(lambda x: len(x) >= 4)
        cell_dict = {}
        cell_dict['zslices'] = []
        cell_dict['boundaries'] = {}
        cell_dict['spot_coords'] = {}
        cell_dict['spot_genes'] = {}
        cell_dict['n'] = 0
        if 'cell_boundary' in adata_st.uns: #check whether cell_boundary has been provided in the data
            cell_boundaries = adata_st.uns['cell_boundary']
            cell_boundaries.index = cell_boundaries.index.astype(str)
            if 'absZ' in df_transcripts.columns: #check whether data is 3D
                for n2, Z in cell.groupby("absZ"):
                    if 'absZ' in cell_boundaries.columns: #check whether cell_boundary is 3D
                        slice_boundary = cell_boundaries.loc[Z.uID.unique()[0]]
                        slice_boundary = slice_boundary[slice_boundary.absZ == n2]
                    else:
                        slice_boundary = cell_boundaries.loc[str(Z.uID.unique()[0])]
                        # slice_boundary = cell_boundaries.loc[Z.uID.unique()[0]]
                    vertices, spot_coords, spot_genes = _get_slice_data(Z, slice_boundary)
                    cell_dict['boundaries'][n2], cell_dict['spot_coords'][n2], cell_dict['spot_genes'][n2] = vertices, spot_coords, spot_genes
                    cell_dict['zslices'].append(n2)
                    if 'gene_counts' not in cell_dict:
                        cell_dict['gene_counts'] = Z.gene.value_counts().to_dict()
                    else:
                        cell_dict['gene_counts'] = dict(Counter(Z.gene.value_counts().to_dict()) + Counter(cell_dict['gene_counts']))
                    cell_dict['n'] += Z[['absX', 'absY']].values.shape[0]
            else: #data is 2D
                vertices, spot_coords, spot_genes = _get_slice_data(cell, cell_boundaries.loc[cell.uID])
                cell_dict['boundaries'][0], cell_dict['spot_coords'][0], cell_dict['spot_genes'][0] = vertices, spot_coords, spot_genes
                cell_dict['zslices'].append(0)
                cell_dict['gene_counts'] = cell.gene.value_counts().to_dict()
                cell_dict['n'] += cell[['absX', 'absY']].values.shape[0]
        else: #generate comvex hull
            if 'absZ' in df_transcripts.columns: #check whether data is 3D
                for n2, Z in cell.groupby("absZ"):
                    try:
                        vertices, spot_coords, spot_genes = _get_slice_data(Z, Z_boundary=None, generate_hull = True)
                        cell_dict['boundaries'][n2], cell_dict['spot_coords'][n2], cell_dict['spot_genes'][n2] = vertices, spot_coords, spot_genes
                        cell_dict['zslices'].append(n2)
                        if 'gene_counts' not in cell_dict:
                            cell_dict['gene_counts'] = Z.gene.value_counts().to_dict()
                        else:
                            cell_dict['gene_counts'] = dict(Counter(Z.gene.value_counts().to_dict()) + Counter(cell_dict['gene_counts']))
                        cell_dict['n'] += Z[['absX', 'absY']].values.shape[0]
                    except:
                        continue
            else: #data is 2D
                vertices, spot_coords, spot_genes = _get_slice_data(cell, Z_boundary=None, generate_hull = True)
                cell_dict['boundaries'][0], cell_dict['spot_coords'][0], cell_dict['spot_genes'][0] = vertices, spot_coords, spot_genes
                cell_dict['zslices'].append(0)
                cell_dict['gene_counts'] = cell.gene.value_counts().to_dict()
                cell_dict['n'] += Z[['absX', 'absY']].values.shape[0]
        if "gene_counts" in cell_dict:
            cell_dict['genes'] = np.unique(list(cell_dict['gene_counts'].keys()))
            if len(cell_dict["zslices"]) > 0 and min(cell_dict['gene_counts'].values()) >= 2:
                for method in methods:
                    arguments.append([cell_dict, method, count])
        count += 1
    with mp.Pool(processes=threads) as pool:
        results = pool.map(_parallel_sprawl, arguments)
    for i in results:
        temp_df = i[2].set_index('gene').reindex(genes).fillna(0)
        sprawl_scores[methods.index(i[0]), i[1], :] = temp_df.score.values
    adata_st.uns['sprawl_scores'] = {}
    for num, method in enumerate(methods):
        adata_st.uns['sprawl_scores'][method] = pd.DataFrame(sprawl_scores[num], columns = genes)
        adata_st.uns['sprawl_scores'][method]['uID'] = [str(x) for x in np.unique(df_transcripts.reset_index().uID.values)]
        adata_st.uns['sprawl_scores'][method].set_index('uID', inplace=True)
    print("SPRAWL completed in:", timedelta(seconds=timeit.default_timer() - start))
    return adata_st



def bicluster_sprawl(adata_st, methods = ['Peripheral', 'Radial', 'Punctate', 'Central'], num_biclusters = 10, randomized_searches = 50000, scale_data = True, cell_threshold = 5, threads = 1, expand = True):# 'Punctate', 'Central', 50000
    '''
    Perform LAS biclustering on SPRAWL spatial pattern scores to find spatial gene expression patterns.
    Arguments
    ----------
    adata_st : AnnData
        Anndata object containing spatial transcriptomics data.
    methods : list
        List of methods to run sprawl on. Options are 'Peripheral', 'Radial', 'Punctate', 'Central'.
    num_biclusters : int
        Number of biclusters to find.
    randomized_searches : int
        Number of randomized searches to perform in LAS.
    scale_data : bool
        If True, scale the data before biclustering.
    cell_threshold : int
        Minimum number of cells in a bicluster.
    threads : int
        Number of threads to use.
    '''
    print("Bi-clustering SPRAWL results...")
    start = timeit.default_timer()
    df_sprawl = adata_st.uns['sprawl_scores'][methods[0]].copy()
    row_log_combs = _log_combs(df_sprawl.shape[0])[1:] # self._log_combs(num_rows)[1:] discards the case where the bicluster has 0 rows
    col_log_combs = _log_combs(df_sprawl.shape[1])[1:] # self._log_combs(num_cols)[1:] discards the case where the bicluster has 0 columns
    col_range = np.arange(1, df_sprawl.shape[1] + 1)
    rows = []
    for method in methods:
        df_sp = adata_st.uns['sprawl_scores'][method].copy()
        df_sp_scaled = df_sp.copy()
        df_sp_scaled[:] = scale(df_sp)
        genes = df_sp.columns.values
        uids = adata_st.obs_names.values
        model = LargeAverageSubmatrices(num_biclusters = num_biclusters, randomized_searches = randomized_searches, scale_data = scale_data, threads = threads)
        biclustering = model.run(df_sp.values)
        for bicluster in biclustering.biclusters:
            if len(bicluster.cols) > 2:
                bicluster.rows = _expand_bicluster_rows(df_sp.values, bicluster.rows, bicluster.cols)
                bicluster_genes = genes[bicluster.cols]
                bicluster_cells = uids[bicluster.rows]
                calculated_score = _get_sprawl_score(bicluster.rows, df_sp_scaled.values, col_range, col_log_combs, row_log_combs)
                rows.append([method, ','.join(map(str, bicluster_genes)), ','.join(map(str, bicluster_cells)), len(bicluster_cells), 0, calculated_score])
    df_results = pd.DataFrame(rows, columns=['method', 'genes', 'uIDs', '#cells', 'combined', "LAS score"])
    df_results['sprawl average'] = df_results['sprawl average'].astype('str')
    df_results['sprawl score'] = df_results['sprawl score'].astype('str')
    score_issues = []
    if expand:
        while True:
            original_length = len(df_results)
            df_results, score_issues = _expand_bicluster(df_results, df_sp.values, df_sp_scaled.values, list(uids), score_issues, col_range, col_log_combs, row_log_combs, df_sprawl = adata_st.uns['sprawl_scores'])
            if len(df_results) == original_length:
                break
    df_results = df_results[df_results.uIDs.apply(lambda x: len(x.split(',')) > cell_threshold)]
    adata_st.uns[f'sprawl_biclustering'] = df_results.reset_index(drop=True)
    print("SPRAWL bi-clustering completed time:", timedelta(seconds=timeit.default_timer() - start))
    return adata_st
    

def _peripheral(cell, ret_spot_ranks=False):
    """
    Helper peripheral function gets called by peripheral() for multiprocessing of each cell. Adapted from https://github.com/salzman-lab/SPRAWL/blob/main/package/src/sprawl/metrics.py.
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
    }

    #score the cell
    periph_dists = []
    spot_genes = []

    for zslice in cell.zslices:

        #Calculate dists of each spot to periphery
        z_boundary = cell.boundaries[zslice]
        z_spot_coords = cell.spot_coords[zslice]
        z_spot_genes = cell.spot_genes[zslice]

        poly = shapely.geometry.Polygon(z_boundary)
        for xy,gene in zip(z_spot_coords,z_spot_genes):
            dist = poly.boundary.distance(shapely.geometry.Point(xy))
            periph_dists.append(dist)
            spot_genes.append(gene)

    #Rank the spots
    spot_genes = np.array(spot_genes)
    spot_ranks = np.array(periph_dists).argsort().argsort()+1 #add one so ranks start at 1 rather than 0

    if ret_spot_ranks:
        return spot_genes,spot_ranks


    #score the genes
    exp_med = (cell.n+1)/2
    for gene in cell.genes:
        gene_ranks = spot_ranks[spot_genes == gene]
        obs_med = np.median(gene_ranks)
        score = (exp_med-obs_med)/(exp_med-1)

        data['metric'].append('periph')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)

    return pd.DataFrame(data)


def _radial(cell, num_iterations, num_pairs):
    """
    Helper radial function gets called by radial() for multiprocessing of each cell. Adapted from https://github.com/salzman-lab/SPRAWL/blob/main/package/src/sprawl/metrics.py.
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
        'variance':[],
    }

    cell_centroid = np.mean(np.vstack(list(cell.boundaries.values())),axis=0)

    # cell = cell.filter_genes_by_count(min_gene_spots=2)

    all_genes = np.array([g for z in cell.zslices for g in cell.spot_genes[z]])
    all_spots = np.array([xy for z in cell.zslices for xy in cell.spot_coords[z]])

    #theoretical variance depends on just the number of iterations
    #Let Y ~ Discrete Uniform from 0 to n, where n is the number of iterations
    #Y represents the number of null permutations that are less than the obs distance and ranges from 0 to 'all'
    #But our score is X = (Y-n/2)/(n/2) because we wanted to center it at 0 and scale it to have values between -1 and 1

    #E[Y] = n/2 from definition of discrete uniform that ranges from 0 to n
    #Var[Y] = n(n+2)/12 

    #E[X] = (2/n)(E[Y]-n/2) = (2/n)(n/2-n/2) = 0
    #Var[X] = (4/n^2)Var[Y] #since Var(aX+b) = (a^2)Var[X]
    #Var[X] = (4/n^2)(n(n+2)/12)
    #Var[X] = (1/n^2)(n(n+2)/3)
    #Var[X] = (1/n)((n+2)/3)
    #Var[X] = (n+2)/3n

    #Also as n --> inf, Var[X] --> 1/3
    #Var[X] = (1+2/n)/3 --> 1/3

    var = (num_iterations+2)/(3*num_iterations)

    pre_calc_nulls = {}

    for gene,count in cell.gene_counts.items():
        # Calculate the obs mean dist
        gene_spots = all_spots[all_genes == gene]
        obs_dist = random_mean_pairs_angle(gene_spots, cell_centroid, num_pairs)

        # Null distribution by gene-label swapping
        if count in pre_calc_nulls:
            null_dists = pre_calc_nulls[count]

        else:
            null_dists = []

            for i in range(num_iterations):
                try:
                    spot_inds = np.random.choice(cell.n,count,replace=False)
                    null = random_mean_pairs_angle(all_spots[spot_inds], cell_centroid, num_pairs)
                except:
                    raise("Some error")
                null_dists.append(null)

            null_dists = np.array(null_dists)
            pre_calc_nulls[count] = null_dists

        obs = sum(null_dists < obs_dist)
        exp = num_iterations/2
        score = (exp-obs)/exp

        data['metric'].append('radial')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)
        data['variance'].append(var)


    return pd.DataFrame(data)


def _punctate(cell, num_iterations, num_pairs):
    """
    Helper punctate function gets called by punctate() for multiprocessing of each cell. Adapted from https://github.com/salzman-lab/SPRAWL/blob/main/package/src/sprawl/metrics.py.
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
        'variance':[],
    }

    all_genes = np.array([g for z in cell.zslices for g in cell.spot_genes[z]])
    all_spots = np.array([xy for z in cell.zslices for xy in cell.spot_coords[z]])

    var = (num_iterations+2)/(3*num_iterations)

    pre_calc_nulls = {}
    for gene,count in cell.gene_counts.items():

        # Calculate the obs mean dist
        gene_spots = all_spots[all_genes == gene]
        obs_dist = random_mean_pairs_dist(gene_spots, num_pairs)

        # Null distribution by gene-label swapping
        if count in pre_calc_nulls:
            null_dists = pre_calc_nulls[count]
        else:
            null_dists = []
            for i in range(num_iterations):
                spot_inds = np.random.choice(cell.n,count,replace=False)
                null = random_mean_pairs_dist(all_spots[spot_inds], num_pairs)
                null_dists.append(null)

            null_dists = np.array(null_dists)
            pre_calc_nulls[count] = null_dists

        obs = sum(null_dists < obs_dist)
        exp = len(null_dists)/2
        score = (exp-obs)/exp


        data['metric'].append('puncta')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)
        data['variance'].append(var)

    return pd.DataFrame(data)


def _central(cell):
    """
    Helper central function gets called by central() for multiprocessing of each cell. Adapted from https://github.com/salzman-lab/SPRAWL/blob/main/package/src/sprawl/metrics.py.
    """
    data = {
        'metric':[],
        'cell_id':[],
        'annotation':[],
        'num_spots':[],
        'gene':[],
        'num_gene_spots':[],
        'score':[],
    }

    #calculate distance of spot coords to cell centroid
    spot_genes = []
    spot_dists = []
    for zslice in cell.zslices:
        z_spot_coords = cell.spot_coords[zslice]
        z_spot_genes = cell.spot_genes[zslice]

        #Euclidean distance to slice centroid
        slice_centroid = np.mean(cell.boundaries[zslice], axis=0)
        dists = np.sum((z_spot_coords-slice_centroid)**2, axis=1)

        spot_genes.extend(z_spot_genes)
        spot_dists.extend(dists)

    #Rank the spots
    spot_genes = np.array(spot_genes)
    spot_ranks = np.array(spot_dists).argsort().argsort()+1 #add one so ranks start at 1 rather than 0

    #score the genes
    exp_med = (cell.n+1)/2
    for gene in cell.genes:
        gene_ranks = spot_ranks[spot_genes == gene]
        obs_med = np.median(gene_ranks)
        score = (exp_med-obs_med)/(exp_med-1)
        data['metric'].append('central')
        data['cell_id'].append(cell.cell_id)
        data['annotation'].append(cell.annotation)
        data['num_spots'].append(cell.n)
        data['gene'].append(gene)
        data['num_gene_spots'].append(cell.gene_counts[gene])
        data['score'].append(score)
    return pd.DataFrame(data)
