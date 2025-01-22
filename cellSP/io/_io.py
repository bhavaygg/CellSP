import pandas as pd
import anndata
from pathlib import Path

def read_h5ad(filename, backed=None):
    """
    Load AnnData object from h5ad.
    Arguments
    ----------
    filename : str
        Name of the file to load.
    backed : str
        If ‘r’, load AnnData in backed mode instead of fully loading it into memory (memory mode). If you want to modify backed attributes of the AnnData object, you need to choose ‘r+’.

    Returns
    ----------
    adata : AnnData
        Anndata object.
    """

    adata = anndata.read_h5ad(filename, backed=backed)
    return adata


def write_h5ad(data, filename):
    """
    Write AnnData to h5ad.
    Arguments
    ----------
    data : AnnData
        Anndata object to save.
    filename : str
        Name of the file to save.
    """
    adata = data.copy()
    adata.write(filename, compression="gzip")

def combine_geo(adata_st, file_path = ".", mode=['instant_fsm', 'instant_biclustering', 'sprawl_biclustering'], setting = "module", filenames = ['instant_fsm', 'instant_biclustering', 'sprawl_biclustering']):
    '''
    Combine the geo analysis and write to an excel file.
    Arguments
    ----------
    adata_st : AnnData
        Spatial transcriptomic data.
    file_path : str
        Path to save the files at.
    mode : list
        List of analysis to combine for. Should be one of 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.
    setting : str
        Setting to perform the analysis. Either 'module' or 'cell'.
    filenames : list
        List of filenames for each mode to save. By default saves as "{mode}_geo_{setting}.xlsx".
    
    Returns
    ----------
    None
    '''
    assert type(mode) == list, "`mode` should be a list"
    for n, method in enumerate(mode):
        if method not in ['instant_fsm', 'instant_biclustering', 'sprawl_biclustering']:
            raise ValueError("Invalid mode. Please choose from 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.")
        if method not in adata_st.uns.keys():
            continue
        # try:
        writer = pd.ExcelWriter(Path(file_path) / f'{filenames[n]}_geo_{setting}.xlsx', engine="xlsxwriter")
        for i in adata_st.uns[f"{method}_geo_{setting}"]:
            adata_st.uns[f"{method}_geo_{setting}"][i].to_excel(writer, sheet_name=f'M{i}', index=False, startrow=0, startcol=0)
            worksheet = writer.sheets[f'M{i}']
            options = {'font': {'color': 'black','size': 14}}
            if setting == "cell":
                if method == "sprawl_biclustering":
                    worksheet.insert_textbox('J1', f"#Genes: {len(adata_st.uns[method].iloc[int(i)]['genes'].split(","))} \
                                                \n#P&C Genes: {adata_st.uns[method].iloc[int(i)]['#pc_genes']} \
                                                \n#Background: {adata_st.n_vars if setting == "cell" else len(adata_st.uns['geneList'])} \
                                                \nPattern: {adata_st.uns[method].iloc[int(i)].method} \
                                                \nGenes: {adata_st.uns[method].iloc[int(i)]['genes']} \
                                                \n#cells: {adata_st.uns[method].iloc[int(i)]['#cells']} \
                                                \nTangram Accuracy: {round(adata_st.uns[method].iloc[int(i)]['tangram'], 3)} \
                                                \nBaseline Accuracy: {round(adata_st.uns[method].iloc[int(i)]['baseline'], 3)}", options)
                else:
                    worksheet.insert_textbox('J1', f"#Genes: {len(adata_st.uns[method].iloc[int(i)]['genes'].split(","))} \
                                                \n#P&C Genes: {adata_st.uns[method].iloc[int(i)]['#pc_genes']} \
                                                \n#Background: {adata_st.n_vars if setting == "cell" else len(adata_st.uns['geneList'])} \
                                                \nGenes: {adata_st.uns[method].iloc[int(i)]['genes']} \
                                                \n#cells: {adata_st.uns[method].iloc[int(i)]['#cells']} \
                                                \nTangram Accuracy: {round(adata_st.uns[method].iloc[int(i)]['tangram'], 3)} \
                                                \nBaseline Accuracy: {round(adata_st.uns[method].iloc[int(i)]['baseline'], 3)}", options)
            else:
                if method == "sprawl_biclustering":
                    worksheet.insert_textbox('J1', f"#Genes: {len(adata_st.uns[method].iloc[int(i)]['genes'].split(","))} \
                                                \n#Background: {adata_st.n_vars if setting == "cell" else len(adata_st.uns['geneList'])} \
                                                \nPattern: {adata_st.uns[method].iloc[int(i)].method} \
                                                \nGenes: {adata_st.uns[method].iloc[int(i)]['genes']} \
                                                \n#cells: {adata_st.uns[method].iloc[int(i)]['#cells']} \
                                                \nTangram Accuracy: {round(adata_st.uns[method].iloc[int(i)]['tangram'], 3)} \
                                                \nBaseline Accuracy: {round(adata_st.uns[method].iloc[int(i)]['baseline'], 3)}", options)
                else:
                    worksheet.insert_textbox('J1', f"#Genes: {len(adata_st.uns[method].iloc[int(i)]['genes'].split(","))} \
                                                \n#Background: {adata_st.n_vars if setting == "cell" else len(adata_st.uns['geneList'])} \
                                                \nGenes: {adata_st.uns[method].iloc[int(i)]['genes']} \
                                                \n#cells: {adata_st.uns[method].iloc[int(i)]['#cells']} \
                                                \nTangram Accuracy: {round(adata_st.uns[method].iloc[int(i)]['tangram'], 3)} \
                                                \nBaseline Accuracy: {round(adata_st.uns[method].iloc[int(i)]['baseline'], 3)}", options)
        writer.close()
