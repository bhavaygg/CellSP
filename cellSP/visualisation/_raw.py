import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull
from pathlib import Path

def plot_module_cells(adata_st, module_number, file_path, mode = "instant_fsm"):
    '''
    Plot the cells in the spatial transcriptomic data.
    Arguments
    ----------
    adata_st : AnnData
        Spatial transcriptomic data.
    '''
    df_transcripts = adata_st.uns['transcripts'].copy()
    df_transcripts['uID'] = df_transcripts['uID'].astype(str)
    results = adata_st.uns[mode]
    module = results.iloc[module_number]
    uIDs = module.uIDs.split(",")
    module_genes = module.genes.split(",")
    for cell in uIDs:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        df_cell = df_transcripts[df_transcripts.uID == str(cell)]
        if 'cell_boundary' in adata_st.uns:
            cell_boundaries = adata_st.uns['cell_boundary']
            cell_boundaries.index = cell_boundaries.index.astype(str)
            cell_boundary = cell_boundaries.loc[cell].values
            ax.plot(cell_boundary[:, 0], cell_boundary[:, 1], '-', color='steelblue', linewidth=1.5)
        else:
            points = df_cell[['absX', 'absY']].values
            cell_boundary = ConvexHull(points)
            for simplex in cell_boundary.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], '-', color='steelblue', linewidth=1.5)
        if "in_nucleus" in df_cell.columns:
            nuc_points = df_cell[df_cell.in_nucleus == 1][['absX', 'absY']].values
            nuc_hull = ConvexHull(nuc_points)
            for simplex in nuc_hull.simplices:
                ax.plot(nuc_points[simplex, 0], nuc_points[simplex, 1], '-.', color = "crimson", linewidth=1)
        ax.scatter(df_cell[df_cell.gene.isin(module_genes)][['absX']].values, df_cell[df_cell.gene.isin(module_genes)][['absY']].values, label="Module", color="forestgreen", alpha=0.9, s = 15, zorder=2)
        ax.scatter(df_cell[~df_cell.gene.isin(module_genes)][['absX']].values, df_cell[~df_cell.gene.isin(module_genes)][['absY']].values, label="Background", color="grey", alpha=0.4, s = 5, zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        #set background white
        ax.set_facecolor('white')
        ax.grid(True)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(Path(file_path) / f"{cell}.png", dpi=1000)
        plt.close()