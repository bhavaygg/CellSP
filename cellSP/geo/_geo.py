from datetime import timedelta
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.stats import pearsonr
from scipy import stats

datasets = {"BP": "GO:0008150", "CC": "GO:0005575", "MF": "GO:0003674"}

def _get_panther(geneset, background, dataset, organism, setting = None):
    '''
        Perform GO enrichment analysis using PANTHERDB.
        Arguments
        ----------
        geneset : str
            Geneset to perform the analysis.
        background : str
            Background genes.
        dataset : str
            Dataset to perform the analysis.
        organism : str
            Organism to perform the analysis.
        setting : str
            Setting to perform the analysis. Either 'module' or 'cell'.
    '''
    df = pd.DataFrame()
    r_session = requests.Session()
    retries = Retry(total=2, backoff_factor=1)
    r_session.mount('http://', HTTPAdapter(max_retries=retries))
    for dataset in datasets:
        params = {
            'geneInputList': geneset,
            'organism': organism,
            'refInputList': background,
            'refOrganism': organism,
            'annotDataSet': datasets[dataset],
            'enrichmentTestType': 'FISHER',
            'correction': 'FDR'
        }
        x = r_session.post(f"https://pantherdb.org/services/oai/pantherdb/enrich/overrep", data=params)
        rows = []
        for i in x.json()['results']['result']:
            if i['number_in_list'] > 0:
                try:
                    rows.append([i['number_in_list'], i['fold_enrichment'], i['fdr'], i['expected'], 
                            i['number_in_reference'], i['pValue'], i['term']['label'], i['term']['id']])
                except:
                    rows.append([i['number_in_list'], i['fold_enrichment'], i['fdr'], i['expected'], 
                            i['number_in_reference'], i['pValue'], i['term']['label'], np.nan])
        df_ds = pd.DataFrame(rows, columns = ['number_in_list', 'fold_enrichment', 'fdr', 'expected', 'number_in_reference', 'pValue', 'term', 'id'])
        df_significant = df_ds[df_ds.pValue < 0.05].sort_values(by=['pValue'])
        df_significant['dataset'] = dataset
        df = pd.concat([df, df_significant])
    df.sort_values(by=['pValue'], inplace=True)
    if setting == "cell":
        df = df[df.number_in_reference > 10].copy()
        fdr_new = stats.false_discovery_control(df.pValue.values)
        df['fdr'] = fdr_new
    df['id'] = df['id'].astype('str')
    return df


def geo_analysis(adata_st, mode=['instant_fsm', 'instant_biclustering', 'sprawl_biclustering'], performance_flag = False, organism = 10090, setting = "module", dataset = "GO:0008150"):
    '''
    Perform the geo analysis.
    Arguments
    ----------
    adata_st : AnnData
        Spatial transcriptomic data.
    mode : list
        List of analysis to perform. .
    performance_flag : bool
        Flag to indicate whether to print the performance of the analysis.
    organism : int
        Taxon ID of the organism.
    setting : str
        Setting to perform the analysis. Either 'module' or 'cell'.
    dataset : str
        ID of the annotation dataset to perform enrichment on.
    '''
    assert type(mode) == list, "`mode` should be a list"
    print("Performing GO Enrichment Analysis...")
    start = timeit.default_timer()
    for method in mode:
        if method not in ['instant_fsm', 'instant_biclustering', 'sprawl_biclustering']:
            raise ValueError("Invalid mode. Please choose from 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.")
        if method not in adata_st.uns.keys():
            continue
        results = adata_st.uns[method]
        print(results)
        adata_st.uns[f"{method}_geo_{setting}"] = {}
        for n, i in results.iterrows():
            flag = True
            if performance_flag:
                if i['tangram'] - i['baseline'] > 0:
                    flag = True
                else:
                    flag = False
            if flag:
                if setting == "module":
                    geneset = i['genes']
                    background = ','.join(adata_st.uns['geneList'])
                    result = _get_panther(geneset, background, dataset, organism)
                    if result.shape[0] > 0:
                        adata_st.uns[f"{method}_geo_module"][str(n)] = result
                    else:
                        print(f"No significant GO terms found for module {n} for setting - {setting}")
                elif setting == "cell":
                    geneset = i["shap genes"]
                    geneset_list = geneset.split(",")
                    background = ','.join(adata_st.var_names)
                    gene_expression = adata_st.X
                    correlation_matrix = pd.DataFrame(np.corrcoef(gene_expression, rowvar=False), columns=adata_st.var_names, index=adata_st.var_names)[geneset_list]
                    for gene in correlation_matrix.columns:
                        geneset_list.extend(correlation_matrix[correlation_matrix[gene] > 0.98].index.values)
                    geneset_list = list(set(geneset_list).difference(set(i['genes'].split(","))))
                    geneset = ",".join(geneset_list)
                    adata_st.uns[method].at[n, "#pc_genes"] = len(geneset_list)
                    result = _get_panther(geneset, background, dataset, organism, setting)
                    if result.shape[0] > 0:
                        adata_st.uns[f"{method}_geo_cell"][str(n)] = result
                    else:
                        print(f"No significant GO terms found for module {n} for setting - {setting}")
                else:
                    raise ValueError("Invalid setting. Please choose from 'module' or 'cell'.")
    print("GO Enrichment Analysis Completed in :", timedelta(seconds = timeit.default_timer() - start))
    return adata_st