import pandas as pd
import anndata as ad
import numpy as np
from pathlib import Path

def load_data(sc_adata = None, st_adata = None):
    '''
    Load the spatial and single cell data.
    Arguments
    ----------
    sc_adata : str
        Path to the single cell data.
    st_adata : str
        Path to the spatial transcriptomic data.
    
    Returns
    ----------
    sc_adata : AnnData or None
        Anndata object containing the single cell data.
    st_adata : AnnData or None
        Anndata object containing the spatial transcriptomic data.
    '''
    if sc_adata is not None and st_adata is not None:
        if Path(sc_adata).suffix.lower() == ".h5ad":
            sc_adata = ad.read_h5ad(sc_adata)
        else:
            raise ValueError("Incorrect file format for single cell data. Only .h5ad files are supported.")
        if Path(st_adata).suffix.lower() == ".h5ad":
            st_adata = ad.read_h5ad(st_adata)
        else:
            raise ValueError("Incorrect file format for spatial transcriptomic data. Only .h5ad files are supported.")
        return sc_adata, st_adata
    elif sc_adata is not None:
        if Path(sc_adata).suffix.lower() == ".h5ad":
            sc_adata = ad.read_h5ad(sc_adata)
            return sc_adata
        else:
            raise ValueError("Incorrect file format for single cell data. Only .h5ad files are supported.")
    elif st_adata is not None:
        if Path(st_adata).suffix.lower() == ".h5ad":
            st_adata = ad.read_h5ad(st_adata)
            return st_adata
        else:
            raise ValueError("Incorrect file format for spatial transcriptomic data. Only .h5ad files are supported.")
    else:
        raise ValueError("At least one file path must be provided.")


def sc_to_anndata(expression):
    '''
    Convert single cell csv data to anndata.
    Arguments
    ----------
    expression : str
        Path to the single-cell expression data.
    
    Returns
    ----------
    adata : AnnData
        Anndata object containing the single-cell expression data.
    '''
    if Path(expression).suffix.lower() == ".csv":
        df = pd.read_csv(expression, index_col = 0)
        df.index = df.index.astype(str)
        df = df.loc[:, (df != 0).any(axis=0)]
        adata = ad.AnnData(X=df)
        adata.obs_names = df.index
        adata.obs.index = df.index
        adata.var_names = df.columns
        return adata
    else:
        raise ValueError("Incorrect file format. Only .csv files are supported.")

def st_to_anndata(transcripts, cell_type = None, cell_boundary = None):
    '''
    Convert spatial transcriptomics data to anndata.
    Arguments
    ----------
    transcripts : str
        Path to the spatial transcriptomics data.
    cell_type : str
        Path to the single-cell cell type data.
    cell_boundary : str
        Path to the single-cell cell boundary data.
    
    Returns
    ----------
    adata : AnnData
        Anndata object containing the spatial transcriptomic data.
    '''
    if Path(transcripts).suffix.lower() == ".csv":
        df = pd.read_csv(transcripts)
        df = df[~df['gene'].str.startswith('Blank')]
        df.sort_values(by=['uID', 'gene'], inplace=True)
        df_st = pd.crosstab(df.uID, df.gene)
        df_st.columns.name = None
        df_st.index = df_st.index.astype(str)
        if 'absZ' in df.columns:
            result = pd.pivot_table(df, values=['absX', 'absY', 'absZ'], index=['uID'])
        else:
            result = pd.pivot_table(df, values=['absX', 'absY'], index=['uID'])
        result.index = result.index.astype(str)
        adata = ad.AnnData(X=df_st)
        adata.obs_names = df_st.index
        adata.obs.index = df_st.index
        adata.var_names = df_st.columns
        adata.obsm['spatial'] = result
        adata.uns["transcripts"] = df
        if cell_type != None:
            df_ct = pd.read_csv(cell_type, index_col = 0)
            df_ct.index = df_ct.index.astype(str)
            adata.obs['cell_type'] = df_ct
        if cell_boundary != None:
            if Path(cell_boundary).suffix.lower() == ".csv":
                df_cb = pd.read_csv(cell_boundary)
            else:
                raise ValueError("Incorrect file format. Only .csv files are supported.")
            df_cb = df_cb.set_index('uID')
            df_cb.index = df_cb.index.astype(str)
            adata.uns['cell_boundary'] = df_cb
        return adata
    else:
        raise ValueError("Incorrect file format. Only .csv files are supported.")