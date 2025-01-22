import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def extrapolate_validation_results(adata):
    size = rcParams["figure.figsize"]
    plt.figure(figsize=(2 * size[0], size[1]))
    plt.subplots_adjust(wspace=0.3)
    plt.grid(True)
    plt.scatter(adata.uns["extrapolate_validation"]["scRNAseq"], adata.uns["extrapolate_validation"]["ST"])
    plt.hlines(y=adata.uns["r2_threshold"], colors='crimson', linestyles='-', label='Spearman Threshold')
    plt.vlines(y=adata.uns["spearman_threshold"], colors='crimson', linestyles='-', label='R2 Threshold')
    plt.xlabel("Extrapolation -> scRNA-seq R2 score")
    plt.ylabel("Extrapolation -> ST spearman correlation")
    return plt.gca()