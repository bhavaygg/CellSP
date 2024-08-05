import plotly as py
import pandas as pd
import numpy as np
# import plotly.plotly as py
import plotly.tools as plotly_tools
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from IPython.display import HTML
from ._enrichment import run_revigo

def create_report(adata_st):
    '''
    Create a report for the results of CellSP.
    Arguments
    ----------
    adata_st : AnnData
        Spatial transcriptomic data.
    '''
    rows_summary = []
    for method in ['instant_biclustering', 'sprawl_biclustering']:
        if method in adata_st.uns:
            results = adata_st.uns[method]
            for n, i in results.iterrows():
                df_rev_module, df_rev_cell = run_revigo(adata_st, module_number = n, mode = method)
                rev_module_term = df_rev_module['Representative'].values[0] if not pd.isnull(df_rev_module['Representative'].values[0]) else df_rev_module['Name'].values[0]
                rev_cell_term = df_rev_cell['Representative'].values[0] if not pd.isnull(df_rev_cell['Representative'].values[0]) else df_rev_cell['Name'].values[0]
                rows_summary.append([method, n, i['genes'], i['#cells'], rev_module_term, rev_cell_term])
    df_summary = pd.DataFrame(rows_summary, columns = ['Method', 'Module Number', 'Genes', '#Cells', 'GO Module Genes', 'GO Module Cells'])
    summary_table_1 = df_summary\
    .to_html()\
    .replace('<table border="1" class="dataframe">','<table class="table table-striped">') # use bootstrap styling
    css_styles = '''
    <style>
        .table {
            word-break: break-word;
            table-layout: fixed;
            width: 100%;
        }
    </style>
    '''
    html_string = '''
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <style>body{ margin:0 100; background:whitesmoke; }</style>
            ''' + css_styles + '''
        </head>
        <body>
            <h1>2014 technology and CPG stock prices</h1>
            <h3>Summary table: 2014 stock statistics</h3>
            ''' + summary_table_1 + '''
        </body>
    </html>'''
    f = open('report.html','w')
    f.write(html_string)
    f.close()