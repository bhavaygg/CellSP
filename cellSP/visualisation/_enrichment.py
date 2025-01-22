import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO
import time
import textwrap
from adjustText import adjust_text

def _run_revigo(module, ds):
    payload = {'cutoff':'0.7', 'valueType':'pvalue', 'speciesTaxon':'0', 'measure':'SIMREL', 'goList': module[['id', 'pValue']].to_csv(sep='\t', index=False, header=False)}
    r = requests.post("http://revigo.irb.hr/StartJob", data=payload)
    jobid = r.json()['jobid']
    if jobid == -1:
        return pd.DataFrame(["None"], columns = ["Representative"])
    # Check job status
    running = 1
    while (running!=0):
        r = requests.get(f"http://revigo.irb.hr/QueryJob?jobid={jobid}&type=jstatus")
        running = r.json()['running']
        time.sleep(2)
    # Fetch results
    r = requests.get(f"http://revigo.irb.hr/QueryJob?jobid={jobid}&namespace={ds}&type=Scatterplot")
    data = StringIO(r.text)
    # Read the tab-separated string into a pandas DataFrame
    df_table = pd.read_csv(data, sep='\t')
    if df_table.empty:
        return pd.DataFrame()
    df_table.drop(columns=['Representative'], inplace=True)
    r = requests.get(f"http://revigo.irb.hr/QueryJob?jobid={jobid}&namespace={ds}&type=TreeMap")
    data = StringIO(r.text[r.text.index("TermID"):])
    # Read the tab-separated string into a pandas DataFrame
    df = pd.read_csv(data, sep='\t')
    if df.empty:
        return pd.DataFrame()
    df_merge = pd.merge(df_table, df[['TermID', 'Representative']], how="left", on="TermID")
    df_merge.dropna(inplace=True, subset=['PC_0', 'PC_1'])
    df_merge['Representative'] = df_merge['Representative'].fillna(df['Name'])
    df_merge['Representative'] = df_merge['Representative'].astype('category')
    df_merge['Representative_ID'] = df_merge['Representative'].cat.codes
    df_merge.sort_values(by='Value', inplace=True)
    return df_merge

def run_revigo(adata_st, module_number, mode = 'instant_fsm', quartile = 80, cell_threshold = 1e-2, dataset = ["BP", "CC", "MF"], backup  = None):
    if backup == None:
        df_module = adata_st.uns[f"{mode}_geo_module"][str(module_number)]
        df_module = df_module[df_module['pValue'] < 0.05]
        df_cell = adata_st.uns[f"{mode}_geo_cell"][str(module_number)]
    else:
        df_module = backup[0]
        df_cell = backup[1]
    if len(df_cell) == 0:
        df_cell = backup
    df_cell = df_cell[df_cell['fdr'] < cell_threshold]
    dict_namespace = {'BP': 1, 'CC': 2, 'MF': 3}
    df_cell_agg = pd.DataFrame()
    df_module_agg = pd.DataFrame()
    for ds in dataset:
        df_rev_module  = _run_revigo(df_module, dict_namespace[ds])
        df_rev_cell = _run_revigo(df_cell, dict_namespace[ds])
        df_module_agg = pd.concat([df_module_agg, df_rev_module])
        df_cell_agg = pd.concat([df_cell_agg, df_rev_cell])
    return df_module_agg, df_cell_agg

def run_revigo_module(df_module, dataset = ["BP", "CC", "MF"]):
    df_module = df_module[df_module['pValue'] < 0.05]
    dict_namespace = {'BP': 1, 'CC': 2, 'MF': 3}
    df_module_agg = pd.DataFrame()
    for ds in dataset:
        df_rev_module  = _run_revigo(df_module, dict_namespace[ds])
        df_module_agg = pd.concat([df_module_agg, df_rev_module])
    return df_module_agg

def run_revigo_cell(df_cell, cell_threshold = 5e-2, dataset = ["BP", "CC", "MF"]):
    df_cell = df_cell[df_cell['fdr'] < cell_threshold]
    dict_namespace = {'BP': 1, 'CC': 2, 'MF': 3}
    df_cell_agg = pd.DataFrame()
    for ds in dataset:
        df_rev_cell = _run_revigo(df_cell, dict_namespace[ds])
        df_cell_agg = pd.concat([df_cell_agg, df_rev_cell])
    return df_cell_agg

def visualize_geo_enrichment(adata_st, module_number, filename = None, mode = 'instant_fsm', quartile = 80, cell_threshold = 1e-2, label = True, adjust = False, fontsize = 9, dataset = ["BP", "CC", "MF"], cmap = "Blues_r"):
    '''
    Visualize the Geo enrichment results.
    Arguments
    ----------
    adata_st : AnnData
        Spatial transcriptomic data.
    module_number : int
        Index of the module to visualize.
    filename : str
        Name of the file to save the plot.
    mode : str
        Type of analysis to visualize. Either 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.
    setting : str
        Setting to perform the analysis. Either 'module' or 'cell'.
    '''
    print("Visualizing subcellular patterns...")
    if mode not in ['instant_fsm', 'instant_biclustering', 'sprawl_biclustering']:
        raise ValueError("Invalid mode. Please choose from 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.")
    df_module = adata_st.uns[f"{mode}_geo_module"][str(module_number)]
    df_module = df_module[df_module['pValue'] < 0.05]
    df_cell = adata_st.uns[f"{mode}_geo_cell"][str(module_number)]
    df_cell = df_cell[df_cell['fdr'] < cell_threshold]
    dict_namespace = {'BP': 1, 'CC': 2, 'MF': 3}
    df_cell_agg = pd.DataFrame()
    df_module_agg = pd.DataFrame()
    for ds in dataset:
        df_rev_module  = _run_revigo(df_module, dict_namespace[ds])
        df_rev_cell = _run_revigo(df_cell, dict_namespace[ds])
        if df_rev_module.empty or df_rev_cell.empty:
            continue
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5), sharey=True)
        im = ax[0].scatter(df_rev_module['PC_0'], df_rev_module['PC_1'], s=df_rev_module['LogSize']*100, c=df_rev_module['Value'], cmap=cmap, alpha=0.8, linewidth=0.5, edgecolors='black', zorder=2)
        cb = fig.colorbar(im, cax=ax[0].inset_axes([0.9, 0.05, 0.03, 0.1]), shrink = 0.2)
        cb.ax.tick_params(labelsize=3) 
        cb.ax.set_title('Log P-value', fontsize=6)
        im = ax[1].scatter(df_rev_cell['PC_0'], df_rev_cell['PC_1'], s=df_rev_cell['LogSize']*100, c=df_rev_cell['Value'], cmap=cmap, alpha=0.8, linewidth=0.5, edgecolors='black', zorder=2)
        cb = fig.colorbar(im, cax = ax[1].inset_axes([0.9, 0.05, 0.03, 0.1]), shrink = 0.2)
        cb.ax.tick_params(labelsize=3) 
        cb.ax.set_title('Log P-value', fontsize=6)
        top_k_indices = np.where(-df_rev_module['Value'].values >= np.percentile(-df_rev_module['Value'].values, quartile))[0]
        texts = []
        for i in top_k_indices:
            texts.append(ax[0].text(df_rev_module['PC_0'].iloc[i], df_rev_module['PC_1'].iloc[i], "\n".join(textwrap.wrap(df_rev_module['Name'].iloc[i].capitalize(), width=20)), fontsize=fontsize, ha='center', wrap = True))
        if adjust:
            adjust_text(texts, ax=ax[0], avoid_self = True) #, force_text = [0.1, 0.1], force_static = [0.1, 0.1]
        top_k_indices = np.where(-df_rev_cell['Value'].values >= np.percentile(-df_rev_cell['Value'].values, quartile))[0]
        texts = []
        for i in top_k_indices:
            texts.append(ax[1].text(df_rev_cell['PC_0'].iloc[i], df_rev_cell['PC_1'].iloc[i], "\n".join(textwrap.wrap(df_rev_cell['Name'].iloc[i].capitalize(), width=20)), fontsize=fontsize, ha='center', wrap = True))
        if adjust:
            adjust_text(texts, ax=ax[1], avoid_self = True) #, force_text = [0.1, 0.1], force_static = [0.1, 0.1]
        if not label:
            fig_nolabel, ax_nolabel = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5), sharey=True)
            im = ax_nolabel[0].scatter(df_rev_module['PC_0'], df_rev_module['PC_1'], s=df_rev_module['LogSize']*100, c=df_rev_module['Value'], cmap=cmap, alpha=0.8, linewidth=0.5, edgecolors='black', zorder=2)
            cb = fig.colorbar(im, cax=ax_nolabel[0].inset_axes([0.9, 0.05, 0.03, 0.1]), shrink = 0.2)
            cb.ax.tick_params(labelsize=6) 
            cb.ax.set_title('Log P-value', fontsize=fontsize)
            im = ax_nolabel[1].scatter(df_rev_cell['PC_0'], df_rev_cell['PC_1'], s=df_rev_cell['LogSize']*100, c=df_rev_cell['Value'], cmap=cmap, alpha=0.8, linewidth=0.5, edgecolors='black', zorder=2)
            cb = fig.colorbar(im, cax = ax_nolabel[1].inset_axes([0.9, 0.05, 0.03, 0.1]), shrink = 0.2)
            cb.ax.tick_params(labelsize=6) 
            cb.ax.set_title('Log P-value', fontsize=fontsize)
            for i in range(len(ax_nolabel)):
                ax_nolabel[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                ax_nolabel[i].set_facecolor('none')
            fig_nolabel.tight_layout()
        ax[0].set_title("Module")
        ax[1].set_title("Cell")
        for i in range(len(ax)):
            ax[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            # ax[i].grid(True, alpha=0.2, zorder=1, color='black')
            ax[i].set_facecolor('none')
        fig.tight_layout()
        df_rev_module['dataset'] = ds
        df_rev_cell['dataset'] = ds
        df_module_agg = pd.concat([df_module_agg, df_rev_module])
        df_cell_agg = pd.concat([df_cell_agg, df_rev_cell])
        if filename != None:
            fig.savefig(filename[:-4] + f"_{ds}.png", dpi=1000)
            if not label:
                fig_nolabel.savefig(filename[:-4] + f"_{ds}_nolabel.png", dpi=1000)
    df_module_agg.sort_values(by='Value', inplace=True)
    df_cell_agg.sort_values(by='Value', inplace=True)
    if filename != None:
        writer = pd.ExcelWriter(filename[:-4] + "_data.xlsx", engine="xlsxwriter")
        df_module_agg.to_excel(writer, sheet_name=f'Module', index=False, startrow=0, startcol=0)
        df_cell_agg.to_excel(writer, sheet_name=f'Cell', index=False, startrow=0, startcol=0)
        writer.close()