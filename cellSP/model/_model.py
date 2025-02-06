from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import timeit
import shap
from anndata import AnnData

def _subsample_celltype(positive_cells, negative_cells):
    '''
        Samples cells based on cell type.
    '''
    adjusted_negative_set = pd.DataFrame()
    df_temp = pd.DataFrame()
    proportions = positive_cells['cell_type'].value_counts(normalize=True)
    for cell_type in np.unique(positive_cells['cell_type']):
        subset_negative = negative_cells[negative_cells['cell_type'] == cell_type]
        try:
            subset_adjusted_negative = subset_negative.sample(n=int(proportions[cell_type] * len(positive_cells)))
        except:
            subset_adjusted_negative = subset_negative

        adjusted_negative_set = pd.concat([adjusted_negative_set, subset_adjusted_negative])
    df_temp = pd.concat([positive_cells, adjusted_negative_set])
    X_ct = df_temp['cell_type'].values
    return df_temp, X_ct

def _subsample_random(positive_cells, negative_cells):
    '''
        Randomly sample cells.
    '''
    df_temp = pd.DataFrame()
    negative_cells = negative_cells.sample(n=len(positive_cells))
    df_temp = pd.concat([positive_cells, negative_cells])
    return df_temp

def _gen_corr_matrix(X_extrapolated, X_tangram, module_genes, corr_threshold):
    '''
        Generate correlation matrix for the module genes.
    '''
    if X_extrapolated != None:
        df_corr_ex = X_extrapolated.set_index('uID').corr(method="spearman")
    df_corr_tg = pd.DataFrame(np.corrcoef(X_tangram.values, rowvar=False), columns=X_tangram.columns, index = X_tangram.columns)
    ex_corr_genes = []
    tg_corr_genes = []
    for gene in module_genes:
        if X_extrapolated != None:
            tempdf = df_corr_ex[module_genes].drop(module_genes).sort_values(by=gene,ascending=False)
            ex_corr_genes.extend(list(tempdf[tempdf[gene] > corr_threshold].index.values))
        tempdf = df_corr_tg[module_genes].drop(module_genes).sort_values(by=gene,ascending=False)
        tg_corr_genes.extend(list(tempdf[tempdf[gene] > corr_threshold].index.values))
    return ex_corr_genes, tg_corr_genes

def _run_cv(df_cells, X_extrapolated, X_tangram, all_genes, module_genes, ex_corr_genes, tg_corr_genes, X_ct = None, n_splits = 5, cell_type = True, shap_ = False, **kwargs):
    '''
        Binary classification of modules using gene expression using Random Forest cross validation.
    '''
    geneset1 = list(set(all_genes).difference(module_genes))
    geneset2_ex = list(set(all_genes).difference(module_genes).difference(set(ex_corr_genes)))
    geneset2_tg = list(set(all_genes).difference(module_genes).difference(set(tg_corr_genes)))
    cell_ids = df_cells.uID.values
    y = df_cells['y'].values
    if X_extrapolated != None:
        X = X_extrapolated[X_extrapolated.uID.isin(cell_ids)].copy().set_index('uID').loc[cell_ids][geneset1].values
        X_corr = X_extrapolated[X_extrapolated.uID.isin(cell_ids)].copy().set_index('uID').loc[cell_ids][geneset2_ex].values
    X_tg = X_tangram[X_tangram.uID.isin(cell_ids)].copy().set_index('uID').loc[cell_ids][geneset1].values
    X_tg_corr = X_tangram[X_tangram.uID.isin(cell_ids)].copy().set_index('uID').loc[cell_ids][geneset2_tg].values
    X_baseline = X_tangram[X_tangram.uID.isin(cell_ids)].copy().set_index('uID').loc[cell_ids][module_genes].values
    rskf = StratifiedKFold(n_splits = n_splits, shuffle=True, **kwargs)
    scores = []
    shap_mean_scores = []
    for number, (train_index, test_index) in enumerate(rskf.split(X_tg, y)):
        scores_iter = []
        if X_extrapolated != None:
            X_train, X_test =   X[train_index], X[test_index]
            X_train_corr, X_test_corr = X_corr[train_index], X_corr[test_index]
        X_train_tg, X_test_tg = X_tg[train_index], X_tg[test_index]
        X_train_tg_corr, X_test_tg_corr = X_tg_corr[train_index], X_tg_corr[test_index]
        X_train_baseline, X_test_baseline = X_baseline[train_index], X_baseline[test_index]
        if cell_type:
            X_ct = LabelEncoder().fit_transform(X_ct).reshape(-1, 1)
            X_train_ct, X_test_ct = X_ct[train_index], X_ct[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = RandomForestClassifier(n_jobs=-1, random_state = 0)
        model.fit(X_train_tg, y_train)
        y_pred = model.predict(X_test_tg)
        scores_iter.append(balanced_accuracy_score(y_test, y_pred))
        if shap_:
            explainer = shap.TreeExplainer(model, X_train_tg)
            shap_values = explainer.shap_values(X_test_tg, check_additivity = False).T
            shap_scores = np.mean(np.absolute(shap_values[0]) + np.absolute(shap_values[1]), axis=1)
            shap_mean_scores.append(shap_scores)
        model.fit(X_train_tg_corr, y_train)
        y_pred = model.predict(X_test_tg_corr)
        scores_iter.append(balanced_accuracy_score(y_test, y_pred))
        model.fit(X_train_baseline, y_train)
        y_pred = model.predict(X_test_baseline)
        scores_iter.append(balanced_accuracy_score(y_test, y_pred))
        if X_extrapolated != None:  
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores_iter.append(balanced_accuracy_score(y_test, y_pred))
            scores_iter.append(balanced_accuracy_score(y_test, y_pred))
            model.fit(X_train_corr, y_train)
            y_pred = model.predict(X_test_corr)
        if cell_type:
            model.fit(X_train_ct, y_train)
            y_pred = model.predict(X_test_ct)
            scores_iter.append(balanced_accuracy_score(y_test, y_pred))
        scores.append(scores_iter)
    if shap_:
        shap_scores = np.mean(np.array(shap_mean_scores), axis=0)
        sorted_positive_indices = np.argsort(shap_scores)[::-1][:20]
        top_labels = [geneset1[i] for i in sorted_positive_indices]
        shap_scores = shap_scores[sorted_positive_indices]
        return np.mean(scores, axis = 0), (top_labels, shap_scores)
    return np.mean(scores, axis = 0), None

def model_modules(adata_st, mode = ['instant_biclustering', 'sprawl_biclustering'], n_repeats = 25, subsample = True, do_shap = False, corr_threshold = 0.98, **kwargs):
    '''
    Model the subcellular patterns found by FSM & LAS using extrapolated scRNA-seq data.
    Arguments
    ----------
    adata_st : AnnData
        Anndata object containing spatial transcriptomics data.
    mode: list
        List of characterizations to model for. Options are 'instant_biclustering', 'sprawl_biclustering'.
    n_repeats : int
        Number of repeats of sampling.
    subsample : bool
        If True, subsample the cells based on cell type(if available) or randomly.
    do_shap : bool
        If True, calculate SHAP values.
    '''
    assert type(mode) == list, "`mode` should be a list"
    print("Modeling subcellular patterns...")
    start = timeit.default_timer()
    # if 'X_extrapolated' not in adata_st.uns.keys() or 'X_tangram' not in adata_st.uns.keys():
    #     raise ValueError("Extrapolated scRNA-seq data not found in adata_st.uns. Please run atleast either extrapolation or tangram first.")
    for method in mode:
        if method not in adata_st.uns.keys():
            continue
        if method not in ['instant_fsm', 'instant_biclustering', 'sprawl_biclustering']:
            raise ValueError("Invalid mode. Please choose from 'instant_fsm', 'instant_biclustering', 'sprawl_biclustering'.")
        X_extrapolated = None
        adata_st.uns[method] = adata_st.uns[method].reset_index(drop=True)
        if 'X_extrapolated' in adata_st.uns.keys():
            X_extrapolated = adata_st.uns["X_extrapolated"].copy()
        X_tg = adata_st.to_df().copy().reset_index()
        all_cells = list(X_tg.uID.values)
        all_genes = X_tg.set_index('uID').columns.values
        for nrow, i in adata_st.uns[method].iterrows():
            X_tangram_module = X_tg.copy()
            module_genes = i.genes.split(",")
            module_genes = [x[0].lower() + x[1:] for x in module_genes]
            print(f"Modelling for Module {nrow} with genes:", module_genes, f"in {i['#cells']} cells")
            module_cells = i.uIDs.split(',')
            #sort module cells
            X_tangram_module['y'] = 0
            columns_to_drop = []
            # if 'cell_type' in adata_st.obs.keys() and subsample:
                # columns_to_drop.append('cell_type')
            ex_corr_genes, tg_corr_genes = _gen_corr_matrix(X_extrapolated, X_tangram_module.set_index('uID').copy(), module_genes, corr_threshold)
            scores = []
            X_ct = None
            if 'cell_type' in adata_st.obs.keys() and subsample:
                X_tangram_module['cell_type'] = adata_st.obs['cell_type'].values
                positive_cells = X_tangram_module[X_tangram_module.uID.isin(module_cells)]
                positive_cells['y'] = 1
                negative_cells = X_tangram_module[~X_tangram_module.uID.isin(module_cells)]
                for repeat in range(n_repeats):
                    df_cells, X_ct = _subsample_celltype(positive_cells, negative_cells)
                    score, shap_ = _run_cv(df_cells, X_extrapolated, X_tangram_module, all_genes, module_genes, ex_corr_genes, tg_corr_genes, X_ct = X_ct, n_splits = 5, cell_type = ('cell_type' in adata_st.obs.keys() and subsample), shap_ = do_shap, **kwargs)
                    scores.append(score)
                X_tangram_module.drop(columns=['cell_type'], inplace=True)
            elif subsample:
                positive_cells = X_tangram_module[X_tangram_module.uID.isin(module_cells)]
                positive_cells['y'] = 1
                negative_cells = X_tangram_module[~X_tangram_module.uID.isin(module_cells)]
                # df_cells = pd.concat([positive_cells, negative_cells])
                for repeat in range(n_repeats):
                    df_cells = _subsample_random(positive_cells, negative_cells)
                    score, shap_ = _run_cv(df_cells, X_extrapolated, X_tangram_module, all_genes, module_genes, ex_corr_genes, tg_corr_genes, X_ct = X_ct, n_splits = 5, cell_type = ('cell_type' in adata_st.obs.keys() and subsample), shap_ = do_shap, **kwargs)
                    scores.append(score)
            else:
                positive_cells = X_tangram_module[X_tangram_module.uID.isin(module_cells)]
                positive_cells['y'] = 1
                negative_cells = X_tangram_module[~X_tangram_module.uID.isin(module_cells)]
                df_cells = pd.concat([positive_cells, negative_cells])
                score, shap_ = _run_cv(df_cells, X_extrapolated, X_tangram_module, all_genes, module_genes, ex_corr_genes, tg_corr_genes, X_ct = X_ct, n_splits = 5, cell_type = ('cell_type' in adata_st.obs.keys() and subsample), shap_ = do_shap, **kwargs)
                scores.append(score)
            scores = np.mean(np.array(scores), axis = 0)
            adata_st.uns[method].at[nrow, "tangram"] = scores[0]
            adata_st.uns[method].at[nrow, "tangram_corr"] = scores[1]
            adata_st.uns[method].at[nrow, "baseline"] = scores[2]
            if X_extrapolated != None:
                adata_st.uns[method].at[nrow, "extrapolated"] = scores[3]
                adata_st.uns[method].at[nrow, "extrapolated_corr"] = scores[4]
                if 'cell_type' in adata_st.obs.keys():
                    adata_st.uns[method].at[nrow, "cell_type"] = scores[5]
            else:
                if 'cell_type' in adata_st.obs.keys() and subsample:
                    adata_st.uns[method].at[nrow, "cell_type"] = scores[3]
            if do_shap:
                adata_st.uns[method].at[nrow, "shap genes"] = ','.join(shap_[0])
                # adata_st.uns[method].at[nrow, "shap scores"] = ','.join(map(str, shap_[1]))
            #print the results in formatted manner in 1 line
            # for i in list(set(adata_st.uns[method].columns).intersection(['tangram', 'tangram_corr', 'baseline', 'extrapolated', 'extrapolated_corr', 'cell_type'])):
            #     print(f"\t{i}: {float(adata_st.uns[method].iloc[nrow][i]):.3f}", end = "\t")
            #     print()
    print("Time to model subcellular patterns", timedelta(seconds=timeit.default_timer() - start))
    return adata_st
