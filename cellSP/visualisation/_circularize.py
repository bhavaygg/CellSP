import numpy as np
import matplotlib.pyplot as plt
import smallestenclosingcircle
from matplotlib.colors import Normalize
import matplotlib
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

def _calculate_density_sector(points, num_sectors=20):
    densities = np.zeros(num_sectors)
    angles = np.linspace(0, 2*np.pi, num_sectors+1)

    for i in range(num_sectors):
        sector_points = []
        for point in points:
            angle = np.arctan2(point[1], point[0])
            if angle < 0:
                angle += 2*np.pi
            if angles[i] <= angle < angles[i+1]:
                sector_points.append(point)
        densities[i] = len(sector_points)#(0.5 * radius * radius * (angles[i+1] - angles[i]))
    densities = np.array(densities) / len(points) if len(points) > 0 else np.zeros(num_sectors)
    return densities

def _plot_sector_density(densities, num_sectors, ax, rest):
    angles = np.linspace(0, 2*np.pi, num_sectors+1) + (2*np.pi) / (num_sectors * 2)
    colors = densities
    if rest:
        bars = ax.bar(angles[:-1], colors, width=(2*np.pi) / num_sectors, bottom=0.0, color='orange', alpha = 0.7, edgecolor='orange', linewidth=3)
    else:
        bars = ax.bar(angles[:-1], colors, width=(2*np.pi) / num_sectors, bottom=0.0, color='blueviolet', edgecolor='blueviolet', linewidth=3, alpha=0.7)
    if not rest:
        ax.grid(alpha=0.2, color='black')
        ax.tick_params(labelbottom=False)
        ax.set_yticks(np.linspace(0, np.max(densities), 5))
        ax.set_yticklabels([])
        for r in np.linspace(0, np.max(densities), 5):
            ax.text(3*np.pi/2, r, f'{r:.1f}', horizontalalignment='left', verticalalignment='center')

def _calculate_density_concentric_circles(points, num_circles, center = 0):
    # Calculate distances of points from the center
    distances = np.linalg.norm(points - center, axis=1)
    
    # Define radii for concentric circles
    min_radius = 0.0
    circle_radii = np.linspace(min_radius, 1, num_circles + 1)
    
    # Calculate the density for each circle
    densities = np.zeros(num_circles)
    for i in range(num_circles):
        points_in_circle = points[(distances >= circle_radii[i]) & (distances < circle_radii[i + 1])]
        densities[i] = len(points_in_circle)
    densities = np.array(densities) / len(densities)
    return densities

def _plot_density_concentric_circles(densities, num_circles, fig, ax):
    min_radius = 0.0
    circle_radii = np.linspace(min_radius, 1, num_circles + 1)
    norm = Normalize(vmin=0, vmax=1)
    colors = matplotlib.colormaps.get_cmap('Blues')(norm(densities))
    circles = []
    for n, i in enumerate(densities):
        im = circles.append(plt.Circle((1,1), circle_radii[n+1], facecolor=colors[n], edgecolor="black", linewidth=0.7, alpha=0.7))
    for i in circles[::-1]:
        ax.add_patch(i)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Blues'), ax=ax, label='Abundance', shrink = 0.3, aspect = 20*0.3)
    fig.set_tight_layout(True)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('none')
    ax.axis('off')
    ax.autoscale_view()

def _rearrange_max_at_zero(densities):
    max_index = np.argmax(densities)
    return np.roll(densities, -max_index)

def _shift_circle(points, center_x, center_y, radius):
    shift_x = 0 - center_x
    shift_y = 0 - center_y
    center_x = center_x + shift_x
    center_y = center_y + shift_y
    old_radius = radius
    radius = radius / old_radius
    return shift_x, shift_y, old_radius


def _calculate_density(cell, module_genes, pattern, num_sectors = None, num_ccircles = None):
    points = cell[['absX', 'absY']].values
    center_x, center_y, radius = smallestenclosingcircle.make_circle(points)
    shift_x, shift_y, radius = _shift_circle(points, center_x, center_y, radius)
    points_genes_x = (cell[cell.gene.isin(module_genes)][['absX']].values + shift_x)/radius
    points_genes_y = (cell[cell.gene.isin(module_genes)][['absY']].values + shift_y)/radius
    points_rest_x = (cell[~cell.gene.isin(module_genes)][['absX']].values + shift_x)/radius
    points_rest_y = (cell[~cell.gene.isin(module_genes)][['absY']].values + shift_y)/radius
    if pattern == "Radial" or pattern == "Punctate":
        return _rearrange_max_at_zero(_calculate_density_sector(points = np.concatenate((points_genes_x, points_genes_y), axis=1), num_sectors=num_sectors)) \
                , _rearrange_max_at_zero(_calculate_density_sector(points = np.concatenate((points_rest_x, points_rest_y), axis=1), num_sectors=num_sectors))
    else:
        return _calculate_density_concentric_circles(points = np.concatenate((points_genes_x, points_genes_y), axis=1), num_circles=num_ccircles) \
               , _calculate_density_concentric_circles(points = np.concatenate((points_rest_x, points_rest_y), axis=1), num_circles=num_ccircles)

def _plot_spatial_cells(median_positions, module_cells, ax):
    positive_cells = median_positions[median_positions.uID.isin([x for x in module_cells])]
    negative_cells = median_positions[~median_positions.uID.isin([x for x in module_cells])]
    ax.scatter(positive_cells.absX, positive_cells.absY, label="Module", color="red", alpha=0.6, s=1.5)
    ax.scatter(negative_cells.absX, negative_cells.absY, label="Background", color="grey", alpha=0.2, s=0.75)


def plot_module_tissue(adata_st, module_number, pattern, mode, show = True, filename = None):
    results = adata_st.uns[mode]
    module = results.iloc[module_number]
    module_genes = [x.lower() for x in module.genes.split(",")]
    module_cells = [x for x in module.uIDs.split(",")]
    median_positions = adata_st.uns['transcripts'].groupby('uID')[['absX', 'absY']].median()
    median_positions.reset_index(inplace=True)
    fig, ax = plt.subplots(figsize=(5,5))
    _plot_spatial_cells(median_positions, module_cells, ax)
    if filename != None:
        plt.savefig(filename, dpi=1000)
    if show:
        plt.show()

def _run_umap(gene_ex):
    reducer = umap.UMAP()
    X_umap = reducer.fit_transform(gene_ex)
    df_umap = pd.DataFrame(X_umap, columns=["UMAP-1", "UMAP-2"], index = gene_ex.index)
    return df_umap


def plot_module_umap(adata_st, module_number, pattern, mode, show = True, filename = None):
    results = adata_st.uns[mode]
    module = results.iloc[module_number]
    module_genes = [x.lower() for x in module.genes.split(",")]
    module_cells = [x for x in module.uIDs.split(",")]
    if 'X_UMAP' not in adata_st.obsm.keys():
        adata_st.obsm['X_UMAP'] = run_umap(adata_st)
    fig, ax = plt.subplots(figsize=(5,5))
    colors = ['red' if x in module_cells else 'grey' for x in adata_st.obs_names.values]
    sizes = [10 if x in module_cells else 1 for x in adata_st.obs_names.values]
    ax.scatter(adata_st.obsm['X_UMAP'].values[:, 0], adata_st.obsm['X_UMAP'].values[:, 1], c=colors, s=sizes)
    if show:
        plt.show()
    if filename != None:
        plt.savefig(filename, dpi=1000)

def _plot_circular_cell(density_module, density_background, median_positions, module_cells, num_sectors, num_ccircles, pattern, genes, pdiff = None, positions = True, filename = None, show = True):
    if pattern == "Radial" or pattern == "Punctate":
        if positions:
            fig = plt.figure(figsize=(12,5))
            ax = plt.subplot(121, projection='polar')
            ax2 = plt.subplot(122)
            ax2.axes.set_aspect('equal')
            _plot_spatial_cells(median_positions, module_cells, ax2)
        else:
            fig = plt.figure(figsize=(6,5))
            ax = plt.subplot(111, projection='polar')
        ax.axes.set_aspect('equal')
        _plot_sector_density(density_module, num_sectors, ax, False)
        _plot_sector_density(density_background, num_sectors, ax, True)
    else:
        if positions:
            fig = plt.figure(figsize=(15,5))
            ax = plt.subplot(131)
            ax1 = plt.subplot(132)
            ax2 = plt.subplot(133)
            ax2.axes.set_aspect('equal')
            _plot_spatial_cells(median_positions, module_cells, ax2)
        else:
            fig = plt.figure(figsize=(5,7))
            ax = plt.subplot(211)
            ax1 = plt.subplot(212)
        density_module = np.array(density_module) / np.max(density_module)
        density_background = np.array(density_background) / np.max(density_background)
        _plot_density_concentric_circles(density_module, num_ccircles, fig, ax)
        _plot_density_concentric_circles(density_background, num_ccircles, fig, ax1)
        ax.set_title("Module", y=-0.1)
        ax1.set_title("Background", y=-0.1)
        ax.axes.set_aspect('equal')
        ax1.axes.set_aspect('equal')
        if len(genes.split(",")) <= 10:
            fig.suptitle(f"Genes: {genes}, #Cells: {len(module_cells)}, Pattern: {pattern}", wrap = True)
        else:
            fig.suptitle(f"#Genes: {len(genes.split(","))}, #Cells: {len(module_cells)}, Pattern: {pattern}", wrap = True)
    plt.tight_layout()
    if filename != None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()

def _calculate_gene_cell_neighborhood_slice(cell, geneList, distance_threshold):
    slice_neighbors = []
    for n2, zslice in cell.groupby('absZ'):
        points = zslice[['absX', 'absY']].values
        # hull = ConvexHull(points)
        point_tree = cKDTree(points)
        pairs = point_tree.query_pairs(distance_threshold)
        genes = zslice.gene.values
        pairs = [(genes[i], genes[j]) for (i,j) in pairs]
        pairs = pd.DataFrame(pairs, columns=['gene1', 'gene2'])
        matrix = pd.pivot_table(pairs, index='gene1', columns='gene2', aggfunc='size', fill_value=0)
        arr = matrix.reindex(index=geneList, columns=geneList, fill_value=0).values
        arr =  arr + arr.T - arr*np.identity(len(arr))
        slice_neighbors.append(arr)
    return np.sum(slice_neighbors, axis = 0)

def _calculate_gene_cell_neighborhood(cell, geneList, distance_threshold):
    points = cell[['absX', 'absY', 'absZ']].values
    point_tree = cKDTree(points)
    pairs = point_tree.query_pairs(distance_threshold)
    genes = cell.gene.values
    pairs = [(genes[i], genes[j]) for (i,j) in pairs]
    pairs = pd.DataFrame(pairs, columns=['gene1', 'gene2'])
    matrix = pd.pivot_table(pairs, index='gene1', columns='gene2', aggfunc='size', fill_value=0)
    arr = matrix.reindex(index=geneList, columns=geneList, fill_value=0).values
    arr =  arr + arr.T - arr*np.identity(len(arr))
    return arr

def _plot_neighborhood_heatmap(knn_matrix_module, knn_matrix_nonmodule, all_genes, module_genes, other_genes, ax):
    knn_matrix_module = np.array(knn_matrix_module)
    knn_matrix_module = np.sum(knn_matrix_module / knn_matrix_module.shape[0], axis=0)
    knn_matrix_nonmodule = np.array(knn_matrix_nonmodule)
    knn_matrix_nonmodule = np.sum(knn_matrix_nonmodule / knn_matrix_nonmodule.shape[0], axis=0)
    pos_k = np.random.permutation([all_genes.index(x) for x in other_genes])[:len(module_genes)]
    indices = [all_genes.index(x) for x in module_genes]
    indices.extend(pos_k)
    module_knn_agg_pos = knn_matrix_module[indices][:, indices] + 1e-32
    module_knn_agg_neg = knn_matrix_nonmodule[indices][:, indices] + 1e-32
    logmat = np.log10(module_knn_agg_pos / module_knn_agg_neg)
    logmat = np.clip(logmat, -1, 1)
    sns.heatmap(logmat, cmap="RdBu_r", ax = ax, center = 0, square=True, cbar_kws={"label": "Log Fold Change in Proximity", "shrink": 0.5, "extend": "both"})
    if len(module_genes) <= 10:
        ax.set_xticks([x + 0.5 for x in list(range(len(indices)))])
        ax.set_yticks([x + 0.5 for x in list(range(len(indices)))])
        ax.set_xticklabels(labels = [all_genes[x].capitalize() for x in indices], rotation=90, fontsize = 13)
        ax.set_yticklabels([all_genes[x].capitalize() for x in indices], rotation = 0, fontsize = 13)
    ax.vlines(len(module_genes), 0, len(indices), color='black', linewidth=2)
    ax.hlines(len(module_genes), 0, len(indices), color='black', linewidth=2)
    ax.set_title("Normalized Sum Ratio")

def visualize_modules(adata_st, mode = ['instant_biclustering', 'sprawl_biclustering'], num_sectors = 10, num_ccircles = 5, distance_threshold = 2, positions = True, performance_flag = True, is_sliced = True, header = False):
    '''
    Model the subcellular patterns found by FSM & LAS using extrapolated scRNA-seq data.
    Arguments
    ----------
    adata_st : AnnData
        Anndata object containing spatial transcriptomics data.
    mode: list
        List of characterizations to model for. Options are 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.
    num_sectors: int
        Number of sectors to divide the circular cell into.
    num_ccircles: int
        Number of concentric circles to divide the circular cell into.
    distance_threshold: float
        Distance threshold used for PP-Test.
    positions: bool
        If True, plot the spatial positions of the module cells.
    '''
    assert type(mode) == list, "mode should be a list"
    print("Visualizing subcellular patterns...")
    for method in mode:
        if method not in ['instant_fsm', 'instant_biclustering', 'sprawl_biclustering']:
            raise ValueError("Invalid mode. Please choose from 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.")
        if method == 'sprawl_biclustering':
            sprawl_results = adata_st.uns[method]
            median_positions = adata_st.uns['transcripts'].groupby('uID')[['absX', 'absY']].median()
            median_positions.reset_index(inplace=True)
            for n, i in sprawl_results.iterrows():
                flag = True
                if performance_flag:
                    if i['tangram'] - i['baseline'] > 0:
                        flag = True
                    else:
                        flag = False
                if flag:
                    density_module = []
                    density_background = []
                    for uID in i.uIDs.split(","):
                        density_cell_module, density_cell_background = _calculate_density(adata_st.uns['transcripts'][adata_st.uns['transcripts'].uID == uID], i.genes.split(","), i.method, num_sectors=num_sectors, num_ccircles=num_ccircles)
                        density_module.append(density_cell_module)
                        density_background.append(density_cell_background)
                    density_background = np.mean(density_background, axis = 0)
                    density_module = np.mean(density_module, axis = 0)
                    pdiff = None
                    if 'tangram' in i.keys() and 'baseline' in i.keys():
                        pdiff = i['tangram'] - i['baseline']
                    _plot_circular_cell(density_module, density_background, median_positions, i.uIDs.split(","), num_sectors, num_ccircles, i.method, i.genes, pdiff, positions)
        else:
            pattern = "clique" if method == 'instant_fsm' else "bicluster"
            instant_results = adata_st.uns[method]
            median_positions = adata_st.uns['transcripts'].groupby('uID')[['absX', 'absY']].median()
            df_transcripts = adata_st.uns['transcripts']
            median_positions.reset_index(inplace=True)
            all_uids = list(adata_st.obs_names.values)
            proximity_matrix = []
            for uID in all_uids:
                if is_sliced:
                    proximity_matrix.append(_calculate_gene_cell_neighborhood_slice(df_transcripts[df_transcripts.uID == uID], adata_st.uns['geneList'], distance_threshold))
                else:
                    proximity_matrix.append(_calculate_gene_cell_neighborhood(df_transcripts[df_transcripts.uID == uID], adata_st.uns['geneList'], distance_threshold))
            proximity_matrix = np.array(proximity_matrix)
            for n, i in instant_results.iterrows():
                flag = True
                if performance_flag:
                    if i['tangram'] - i['baseline'] > 0:
                        flag = True
                    else:
                        flag = False
                if flag:
                    knn_matrix_module = []
                    knn_matrix_nonmodule = []
                    module_genes = [x.lower() for x in i.genes.split(",")]
                    module_cells = [x for x in i.uIDs.split(",")]
                    geneList = list(adata_st.uns['geneList'])
                    geneList = [x.lower() for x in geneList]
                    module_cells_idx = [all_uids.index(x) for x in i.uIDs.split(",")]
                    non_module_cells = list(set(all_uids).difference(set(module_cells)))
                    non_module_cells_idx = [all_uids.index(x) for x in non_module_cells]
                    knn_matrix_module = proximity_matrix[module_cells_idx]
                    non_module_cells = np.unique(df_transcripts[~df_transcripts.uID.isin(module_cells)].uID)
                    knn_matrix_nonmodule = proximity_matrix[non_module_cells_idx]
                    if positions:
                        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5))
                        _plot_spatial_cells(median_positions, module_cells, ax[1])
                    else:
                        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(6,5))
                    _plot_neighborhood_heatmap(knn_matrix_module, knn_matrix_nonmodule, geneList, module_genes, [x for x in geneList if x not in module_genes], ax[0] if positions else ax)
                    if header:
                        fig.suptitle(f"Genes: {i.genes}, #Cells: {len(module_cells)}, Pattern: {pattern}", wrap = True)
                    plt.tight_layout()
                    plt.show()

def visualize_individual_module(adata_st, module_number, filename = None, num_sectors = 10, num_ccircles = 5, mode = 'instant_biclustering', distance_threshold = 2, is_sliced = True, positions = True, header = False, show = True):
    '''
    Model the subcellular patterns for a specific module found by FSM & LAS using extrapolated scRNA-seq data.
    Arguments
    ----------
    adata_st : AnnData
        Anndata object containing spatial transcriptomics data.
    module_number: int
        Index of the module to visualize.
    filename: str
        Name of the file to save the plot at.
    num_sectors: int
        Number of sectors to divide the circular cell into.
    num_ccircles: int
        Number of concentric circles to divide the circular cell into.
    mode: str
        Mode of the selected module. Options are 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.
    distance_threshold: float
        Distance threshold used for PP-Test.
    '''
    print("Visualizing subcellular patterns...")
    if mode == 'sprawl_biclustering':
        sprawl_results = adata_st.uns[mode]
        median_positions = adata_st.uns['transcripts'].groupby('uID')[['absX', 'absY']].median()
        median_positions.reset_index(inplace=True)
        module = sprawl_results.iloc[module_number]
        density_module = []
        density_background = []
        for uID in module.uIDs.split(","):
            density_cell_module, density_cell_background = _calculate_density(adata_st.uns['transcripts'][adata_st.uns['transcripts'].uID == uID], module.genes.split(","), module.method, num_sectors=num_sectors, num_ccircles=num_ccircles)
            density_module.append(density_cell_module)
            density_background.append(density_cell_background)
        density_background = np.mean(density_background, axis = 0)
        density_module = np.mean(density_module, axis = 0)
        pdiff = None
        if 'tangram' in module.keys() and 'baseline' in module.keys():
            pdiff = module['tangram'] - module['baseline']
        _plot_circular_cell(density_module, density_background, median_positions, module.uIDs.split(","), num_sectors, num_ccircles, module.method, module.genes, pdiff, positions, filename)
    else:
        pattern = "clique" if mode == 'instant_fsm' else "bicluster"
        instant_results = adata_st.uns[mode]
        median_positions = adata_st.uns['transcripts'].groupby('uID')[['absX', 'absY']].median()
        df_transcripts = adata_st.uns['transcripts']
        median_positions.reset_index(inplace=True)
        all_uids = list(adata_st.obs_names.values)
        proximity_matrix = []
        for uID in all_uids:
            if is_sliced:
                proximity_matrix.append(_calculate_gene_cell_neighborhood_slice(df_transcripts[df_transcripts.uID == uID], adata_st.uns['geneList'], distance_threshold))
            else:
                proximity_matrix.append(_calculate_gene_cell_neighborhood(df_transcripts[df_transcripts.uID == uID], adata_st.uns['geneList'], distance_threshold))
        proximity_matrix = np.array(proximity_matrix)
        module = instant_results.iloc[module_number]
        knn_matrix_module = []
        knn_matrix_nonmodule = []
        module_genes = [x.lower() for x in module.genes.split(",")]
        module_cells = [x for x in module.uIDs.split(",")]
        geneList = list(adata_st.uns['geneList'])
        geneList = [x.lower() for x in geneList]
        module_cells_idx = [all_uids.index(x) for x in module.uIDs.split(",")]
        non_module_cells = list(set(all_uids).difference(set(module_cells)))
        non_module_cells_idx = [all_uids.index(x) for x in non_module_cells]
        knn_matrix_module = proximity_matrix[module_cells_idx]
        non_module_cells = np.unique(df_transcripts[~df_transcripts.uID.isin(module_cells)].uID)
        knn_matrix_nonmodule = proximity_matrix[non_module_cells_idx]
        if positions:
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5))
            _plot_spatial_cells(median_positions, module_cells, ax[1])
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(6,5))
        _plot_neighborhood_heatmap(knn_matrix_module, knn_matrix_nonmodule, geneList, module_genes, [x for x in geneList if x not in module_genes], ax[1] if positions else ax)
        if header:
            fig.suptitle(f"Genes: {module.genes}, #Cells: {len(module_cells)}, Pattern: {pattern}", wrap = True)
        plt.tight_layout()
        if filename != None:
            plt.savefig(filename, dpi=1000)
        if show:
            plt.show()

def visualize_pattern(adata_st, module_number, pattern, mode = "instant_biclustering", filename = None, num_sectors = 10, num_ccircles = 5, distance_threshold = 2, is_sliced = True, positions = False, header = False, show = True):
    '''
    Model a specific subcellular patterns for a specific module found by FSM & LAS using extrapolated scRNA-seq data.
    Arguments
    ----------
    adata_st : AnnData
        Anndata object containing spatial transcriptomics data.
    module_number: int
        Index of the module to visualize.
    pattern: str
        Pattern to model. Options are 'Proximal', 'Cluster', 'Concentric'.
    mode:
        Mode of the selected module. Options are 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.
    filename: str
        Name of the file to save the plot at.
    num_sectors: int
        Number of sectors to divide the circular cell into.
    num_ccircles: int
        Number of concentric circles to divide the circular cell into.
    distance_threshold: float
        Distance threshold used for PP-Test.
    '''

    print("Visualizing subcellular patterns...")
    results = adata_st.uns[mode]
    adata_st.uns['transcripts']['uID'] = adata_st.uns['transcripts']['uID'].astype(str)
    if pattern in ["Radial", "Concentric"]:
            if pattern == "Cluster":
                pattern = "Radial"
            median_positions = adata_st.uns['transcripts'].groupby('uID')[['absX', 'absY']].median()
            median_positions.reset_index(inplace=True)
            module = results.iloc[module_number]
            density_module = []
            density_background = []
            for uID in module.uIDs.split(","):
                density_cell_module, density_cell_background = _calculate_density(adata_st.uns['transcripts'][adata_st.uns['transcripts'].uID == uID], module.genes.split(","), pattern, num_sectors=num_sectors, num_ccircles=num_ccircles)
                density_module.append(density_cell_module)
                density_background.append(density_cell_background)
            density_background = np.mean(density_background, axis = 0)
            density_module = np.mean(density_module, axis = 0)
            pdiff = None
            if 'tangram' in module.keys() and 'baseline' in module.keys():
                pdiff = module['tangram'] - module['baseline']
            _plot_circular_cell(density_module, density_background, median_positions, module.uIDs.split(","), num_sectors, num_ccircles, pattern, module.genes, pdiff, positions, filename, show)
    else:
        median_positions = adata_st.uns['transcripts'].groupby('uID')[['absX', 'absY']].median()
        df_transcripts = adata_st.uns['transcripts']
        median_positions.reset_index(inplace=True)
        all_uids = list(adata_st.obs_names.values)
        proximity_matrix = []
        for uID in all_uids:
            if is_sliced:
                proximity_matrix.append(_calculate_gene_cell_neighborhood_slice(df_transcripts[df_transcripts.uID == uID], adata_st.uns['geneList'], distance_threshold))
            else:
                proximity_matrix.append(_calculate_gene_cell_neighborhood(df_transcripts[df_transcripts.uID == uID], adata_st.uns['geneList'], distance_threshold))
        proximity_matrix = np.array(proximity_matrix)
        module = results.iloc[module_number]
        knn_matrix_module = []
        knn_matrix_nonmodule = []
        module_genes = [x.lower() for x in module.genes.split(",")]
        module_cells = [x for x in module.uIDs.split(",")]
        geneList = list(adata_st.uns['geneList'])
        geneList = [x.lower() for x in geneList]
        module_cells_idx = [all_uids.index(x) for x in module.uIDs.split(",")]
        non_module_cells = list(set(all_uids).difference(set(module_cells)))
        non_module_cells_idx = [all_uids.index(x) for x in non_module_cells]
        knn_matrix_module = proximity_matrix[module_cells_idx]
        non_module_cells = np.unique(df_transcripts[~df_transcripts.uID.isin(module_cells)].uID)
        knn_matrix_nonmodule = proximity_matrix[non_module_cells_idx]
        if positions:
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5))
            _plot_spatial_cells(median_positions, module_cells, ax[1])
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(6,5))
        _plot_neighborhood_heatmap(knn_matrix_module, knn_matrix_nonmodule, geneList, module_genes, [x for x in geneList if x not in module_genes], ax[1] if positions else ax)
        if header:
            fig.suptitle(f"Genes: {module.genes}, #Cells: {len(module_cells)}, Pattern: {pattern}", wrap = True)
        plt.tight_layout()
        if filename != None:
            plt.savefig(filename, dpi=1000)
        if show:
            plt.show()

