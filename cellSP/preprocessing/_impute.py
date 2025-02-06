import magic
import timeit
from datetime import timedelta

def impute(adata_sc, n_jobs=-1, **kwargs):
    '''
    Perform imputation using MAGIC.
    Arguments
    ----------
    adata_sc : AnnData
        Anndata object containing spatial transcriptomics data.
    n_jobs : int
        Number of jobs to run in parallel.
    **kwargs : dict
        Additional keyword arguments for MAGIC.
    '''
    start = timeit.default_timer()
    magic_operator = magic.MAGIC(n_jobs=n_jobs, **kwargs)
    X_magic = magic_operator.fit_transform(adata_sc)
    adata_sc.raw = adata_sc
    adata_sc.X = X_magic.X
    print("Time to impute", timedelta(seconds=timeit.default_timer() - start))
    return adata_sc

