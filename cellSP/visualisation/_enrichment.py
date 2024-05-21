import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO
import time
from adjustText import adjust_text

def run_revigo(module):
    payload = {'cutoff':'0.7', 'valueType':'pvalue', 'speciesTaxon':'0', 'measure':'SIMREL', 'goList': module[['id', 'pValue']].to_csv(sep='\t', index=False, header=False)}
    r = requests.post("http://revigo.irb.hr/StartJob", data=payload)
    jobid = r.json()['jobid']
    # Check job status
    running = 1
    while (running!=0):
        r = requests.get(f"http://revigo.irb.hr/QueryJob?jobid={jobid}&type=jstatus")
        running = r.json()['running']
        time.sleep(1)

    # Fetch results
    r = requests.get(f"http://revigo.irb.hr/QueryJob?jobid={jobid}&namespace=1&type=Scatterplot")
    data = StringIO(r.text)
    # Read the tab-separated string into a pandas DataFrame
    df_table = pd.read_csv(data, sep='\t')
    df_table.drop(columns=['Representative'], inplace=True)
    r = requests.get(f"http://revigo.irb.hr/QueryJob?jobid={jobid}&namespace=1&type=TreeMap")
    data = StringIO(r.text[r.text.index("TermID"):])
    # Read the tab-separated string into a pandas DataFrame
    df = pd.read_csv(data, sep='\t')
    df_merge = pd.merge(df_table, df[['TermID', 'Representative']], how="left", on="TermID")
    df_merge.dropna(inplace=True, subset=['PC_0', 'PC_1'])
    df_merge['Representative'] = df_merge['Representative'].astype('category')
    df_merge['Representative_ID'] = df_merge['Representative'].cat.codes
    return df_merge

def visualize_geo_enrichment(adata_st, module_number, filename = None, mode = 'instant_fsm', setting = "module"):
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
    df_cell = df_cell[df_cell['fdr'] < 1e-10]
    df_rev_module  = run_revigo(adata_st.uns[f"{mode}_geo_module"][str(module_number)])
    df_rev_cell = run_revigo(adata_st.uns[f"{mode}_geo_cell"][str(module_number)])
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,5), sharey=True)
    im = ax[0].scatter(df_rev_module['PC_0'], df_rev_module['PC_1'], s=df_rev_module['LogSize']*100, c=df_rev_module['Value'], cmap="autumn_r", alpha=0.7, linewidth=0.5, edgecolors='black')
    # cax = ax[0].inset_axes([0.9, 0.1, 0.05, 0.4])  # [left, bottom, width, height]
    fig.colorbar(im, cax=ax[0].inset_axes([0.9, 0.0, 0.05, 0.1]), shrink = 0.3)
    # fig.colorbar(im, ax=ax[0], shrink=0.5)
    im = ax[1].scatter(df_rev_cell['PC_0'], df_rev_cell['PC_1'], s=df_rev_cell['LogSize']*10, c=df_rev_cell['Value'], cmap="autumn_r", alpha=0.7, linewidth=0.5, edgecolors='black')
    # cax = ax[1].inset_axes([0.9, 0.1, 0.05, 0.4])  # [left, bottom, width, height]
    fig.colorbar(im, cax = ax[1].inset_axes([0.9, 0.0, 0.05, 0.1]), shrink = 0.3)
    # fig.colorbar(im, ax=ax[1], shrink=0.5)

    top_k_indices = np.where(-df_rev_module['Value'].values >= np.percentile(-df_rev_module['Value'].values, 80))[0]
    texts = []
    for i in top_k_indices:
        texts.append(ax[0].text(df_rev_module['PC_0'].iloc[i], df_rev_module['PC_1'].iloc[i], df_rev_module['Name'].iloc[i].capitalize(), fontsize=5, ha='center'))
    adjust_text(texts, ax=ax[0])
    top_k_indices = np.where(-df_rev_cell['Value'].values >= np.percentile(-df_rev_cell['Value'].values, 95))[0]
    texts = []
    for i in top_k_indices:
        texts.append(ax[1].text(df_rev_cell['PC_0'].iloc[i], df_rev_cell['PC_1'].iloc[i], df_rev_cell['Name'].iloc[i].capitalize(), fontsize=5, ha='center'))
    adjust_text(texts, ax=ax[1])
    ax[0].set_title("Module")
    ax[1].set_title("Cell")
    for i in range(len(ax)):
        ax[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_facecolor('none')
    plt.tight_layout()
    # plt.colorbar(im, ax=ax.ravel().tolist(), shrink=0.7)
    plt.savefig(filename, dpi=1000)
    writer = pd.ExcelWriter(filename[:-4] + "_data.xlsx", engine="xlsxwriter")
    df_rev_module.to_excel(writer, sheet_name=f'Module', index=False, startrow=0, startcol=0)
    df_rev_cell.to_excel(writer, sheet_name=f'Cell', index=False, startrow=0, startcol=0)
    writer.close()
    # df_merge['Name'] = df_merge.Name.map(customwrap)
    # df_merge['Representative'] = df_merge.Representative.map(customwrap)
    # fig = px.treemap(
    #     df_merge,
    #     names='Name',
    #     parents='Representative', values = 'LogSize', maxdepth=1
    #     )
    # print(fig.data)
    # fig.update_traces(root_color="lightgrey", textfont=dict(size=10, color="black"))
    # # fig.update_layout(margin = dict(t=10, l=10, r=10, b=10))
    # fig.update_layout(width=500, height=500, uniformtext=dict(minsize=10, mode='hide'))
    # fig.for_each_trace(lambda trace: trace.update(text=[label if parent == "" else "" for label, parent in zip(df['Name'], df['Representative'])]))
    # # fig.update_layout(uniformtext=dict(minsize=15, mode='hide'))
    # fig.write_image(filename, scale=4)