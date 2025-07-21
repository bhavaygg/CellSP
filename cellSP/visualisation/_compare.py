import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools 
import combinations
from statsmodels.stats.proportion import proportions_ztest

def apply_ztest_pos(c1, c2, n1, n2):
    return proportions_ztest(
        count=[c1 , c2], 
        nobs=[n1, n2], 
        alternative='larger'
      )[1]
def apply_ztest_pos_score(c1, c2, n1, n2):
    return proportions_ztest(
        count=[c1 , c2], 
        nobs=[n1, n2], 
        alternative='larger'
      )[0]

def apply_ztest_neg(c1, c2, n1, n2):
    return proportions_ztest(
        count=[c1 , c2], 
        nobs=[n1, n2], 
        alternative='smaller'
      )[1]
def apply_ztest_neg_score(c1, c2, n1, n2):
    return proportions_ztest(
        count=[c1 , c2], 
        nobs=[n1, n2], 
        alternative='smaller'
      )[0]

def do_proportion_test(adata_st_control, adata_st_condition, file_path = None, show = True):
    '''
    Plot the cells in the spatial transcriptomic data.
    Arguments
    ----------
    adata_st : AnnData
        Spatial transcriptomic data.
    '''
    genes_control = adata_st_control.var_names.tolist()
    genes_condition = adata_st_condition.var_names.tolist()
    assert genes_control == genes_condition, "Gene lists are not the same"
    gene_pair_dict = {}
    gene_dict = {}
    for n, i in adata_st_control.uns['instant_biclustering'].iterrows():
        module_gene_pairs = [tuple(sorted((g1, g2))) for g1, g2 in combinations(i.genes.split(","), 2)]
        for gp in module_gene_pairs:
            if gp not in gene_pair_dict:
                gene_pair_dict[gp] = [[],[]]
            gene_pair_dict[gp][0].extend(i['uIDs'].split(","))
            if gp[0] not in gene_dict:
                gene_dict[gp[0]] = [[],[]]
            if gp[1] not in gene_dict:
                gene_dict[gp[1]] = [[],[]]
            gene_dict[gp[0]][0].extend(i['uIDs'].split(","))
            gene_dict[gp[1]][0].extend(i['uIDs'].split(","))

    for n, i in adata_st_control.uns['sprawl_biclustering'].iterrows():
        module_gene_pairs = [tuple(sorted((g1, g2))) for g1, g2 in combinations(i.genes.split(","), 2)]
        for gp in module_gene_pairs:
            if gp not in gene_pair_dict:
                gene_pair_dict[gp] = [[],[]]
            gene_pair_dict[gp][0].extend(i['uIDs'].split(","))
            if gp[0] not in gene_dict:
                gene_dict[gp[0]] = [[],[]]
            if gp[1] not in gene_dict:
                gene_dict[gp[1]] = [[],[]]
            gene_dict[gp[0]][0].extend(i['uIDs'].split(","))
            gene_dict[gp[1]][0].extend(i['uIDs'].split(","))

    for n, i in adata_st_condition.uns['instant_biclustering'].iterrows():
        module_gene_pairs = [tuple(sorted((g1, g2))) for g1, g2 in combinations(i.genes.split(","), 2)]
        for gp in module_gene_pairs:
            if gp not in gene_pair_dict:
                gene_pair_dict[gp] = [[],[]]
            gene_pair_dict[gp][1].extend(i['uIDs'].split(","))
            if gp[0] not in gene_dict:
                gene_dict[gp[0]] = [[],[]]
            if gp[1] not in gene_dict:
                gene_dict[gp[1]] = [[],[]]
            gene_dict[gp[0]][1].extend(i['uIDs'].split(","))
            gene_dict[gp[1]][1].extend(i['uIDs'].split(","))

    for n, i in adata_st_condition.uns['sprawl_biclustering'].iterrows():
        module_gene_pairs = [tuple(sorted((g1, g2))) for g1, g2 in combinations(i.genes.split(","), 2)]
        for gp in module_gene_pairs:
            if gp not in gene_pair_dict:
                gene_pair_dict[gp] = [[],[]]
            gene_pair_dict[gp][1].extend(i['uIDs'].split(","))
            if gp[0] not in gene_dict:
                gene_dict[gp[0]] = [[],[]]
            if gp[1] not in gene_dict:
                gene_dict[gp[1]] = [[],[]]
            gene_dict[gp[0]][1].extend(i['uIDs'].split(","))
            gene_dict[gp[1]][1].extend(i['uIDs'].split(","))
    
    for i in gene_dict:
        gene_dict[i][0] = len(list(set(gene_dict[i][0])))
        gene_dict[i][1] = len(list(set(gene_dict[i][1])))

    for i in gene_pair_dict:
        gene_pair_dict[i][0] = len(list(set(gene_pair_dict[i][0])))
        gene_pair_dict[i][1] = len(list(set(gene_pair_dict[i][1])))
    
    rows = []
    for i in gene_pair_dict:
        if gene_pair_dict[i][0] > 0 and gene_pair_dict[i][1] > 0:
            c1 = gene_pair_dict[i][0]
            c2 = gene_pair_dict[i][1]
            n1 = gene_dict[i[0]][0] + gene_dict[i[1]][0]
            n2 = gene_dict[i[0]][1] + gene_dict[i[1]][1]
            rows.append([i[0], i[1], c1, c2, n1, n2, apply_ztest_pos(c1,c2,n1,n2), apply_ztest_pos_score(c1,c2,n1,n2), apply_ztest_neg(c1,c2,n1,n2), apply_ztest_neg_score(c1,c2,n1,n2)])

    df_compare = pd.DataFrame(rows, columns=["G1", "G2", "x_ctrl", "x_condition", "n_ctrl", "n_condition", "pval_ctrl", "zscore_ctrl", "pval_condition", "zscore_condition"])
    return df_compare