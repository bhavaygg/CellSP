import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from ._enrichment import run_revigo
from ._circularize import visualize_pattern, plot_module_tissue, plot_module_umap
import subprocess
from jinja2 import Template

template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Pandas DataFrame</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Bungee+Spice&family=Rampart+One&family=RocknRoll+One&family=Rowdies:wght@300;400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.3/css/dataTables.bootstrap4.min.css">
    <style>
        table {
            table-layout: fixed;
            width: 100%;
            text-align: center;
            border-collapse:separate;
            border:solid black 1px;
            border-radius:6px;
            
        }
        td, th {
            word-wrap: break-word;
            text-align: center;
        }
        th.description, td.description {
            white-space: normal;
        }
        .container {
            display: flex;
            align-items: flex-start;
            margin-top: 20px;
        }
        .container2 {
            display: flex;
            align-items: flex-start;
            margin-top: 20px;
        }
        .flex-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .left-image {
            flex: 1;
            text-align: left;
        }
        .right-image {
            flex: 1;
            text-align: right;
        }
        .points {
            list-style-type: disc;
            word-wrap: break-word;
            max-width: 300px; /* Set a max width for the bullet points */
            margin: 0 auto; /* Center the bullet points */
        }
        .boxed-section {
            border: 2px solid #000; /* Black border */
            padding: 20px; /* Padding inside the box */
            border-radius: 10px; /* Rounded corners */
            background-color: #f9f9f9; /* Light gray background */
            margin: 20px 0; /* Margin around the box */
        }
        .clickable-row {
            cursor: pointer;
        }
        .clickable-row:hover {
            background-color: #f1f1f1;
        }
        body {
            font-family: 'RocknRoll One', sans-serif;
        }
        .header {
            font-family: "Rampart One", sans-serif;
            font-weight: 800;
            font-style: normal;
            font-size: 50px;
            margin: 20px;
        }
        .geo {
            font-family: "Rowdies", sans-serif;
            align: center;
        }
    </style>
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.min.js"></script>
    <script>
         $(document).ready(function() {
            $('#dataTable').DataTable({
                "order": [],
                "orderClasses": true,
                "columnDefs": [{
                    "targets": "_all",
                    "orderable": true,
                }],
                "language": {
                    "order": "Sort by"
                }
            });
        });
    </script>
</head>
<body>
    <h1 class="header">CellSP Report</h1>
    <div class="container">
        <div class="table-responsive">
            <table id="dataTable" class="table table-striped table-bordered table-hover">
                <thead class="thead-dark">
                    <tr>
                        <th>Mode</th>
                        <th>Pattern</th>
                        <th>#Cells</th>
                        <th>#Genes</th>
                    </tr>
                </thead>
                <tbody>
                    {% for index, row in df.iterrows() %}
                    <tr class="clickable-row" onclick="window.location='#box{{ index+1 }}'">
                        <td>{{ row['Mode'] }}</td>
                        <td>{{ row['Pattern'] }}</td>
                        <td>{{ row['#cells'] }}</td>
                        <td>{{ row['#genes'] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% for images in image_pairs %}
        <div class="boxed-section" id="box{{ loop.index }}">
            <div class="flex-container">
                <div class="left-image">
                    <img src={{ images[0] }} alt="Plot Image" style=" max-width: 50vw; max-height: 50vh; width: auto; height: auto;">
                </div>
                <div class="content">
                    <ul class="points">
                        <h5>{{ images[5][0] }}</h5>
                        {% for point in images[5][1:] %}
                        <li>{{ point }}</li>
                        {% endfor %}
                    </ul>
                </div>
                <div class="right-image">
                    <img src={{ images[1] }} alt="Plot Image" style=" max-width: 50vw; max-height: 50vh; width: auto; height: auto;">
                </div>
            </div>
            <div class="container2">
                <div class="left-image">
                    <img src={{ images[2] }} alt="Plot Image" style=" max-width: 50vw; max-height: 50vh; width: auto; height: auto;">
                </div>
                <div class="table-responsive">
                <h2 class="geo"> GO Genes </h2>
                    <table id="dataTable2" class="table table-striped table-bordered table-hover">
                        <thead class="thead-dark">
                            <tr>
                                <th>Term</th>
                                <th>P-Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for index, row in images[3].iterrows() %}
                            <tr>
                                <td>{{ row['term'] }}</td>
                                <td>{{ row['pValue'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                <div class="table-responsive">
                <h2 class="geo"> GO Cells </h2>
                    <table id="dataTable2" class="table table-striped table-bordered table-hover">
                        <thead class="thead-dark">
                            <tr>
                                <th>Term</th>
                                <th>P-Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for index, row in images[4].iterrows() %}
                            <tr>
                                <td>{{ row['term'] }}</td>
                                <td>{{ row['pValue'] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    {% endfor %}
</body>
</html>
"""

def make_plots(adata_st, n, mode, pattern):
    if mode == "sprawl_biclustering":
        if pattern in ["Radial", "Punctate"]:
            fig = visualize_pattern(adata_st, n, "Radial", mode = mode, show = False, filename = f"plots/{n}_{mode}_{pattern}.png")
        else:
            fig = visualize_pattern(adata_st, n, "Concentric", mode = mode, show = False, filename = f"plots/{n}_{mode}_{pattern}.png")
    else:
        fig = visualize_pattern(adata_st, n, "Colocalization", mode = mode, show = False, filename = f"plots/{n}_{mode}_{pattern}.png")

    fig = plot_module_tissue(adata_st, n, pattern, mode = mode, show = False, filename = f"plots/{n}_{mode}_spatial.png")
    fig = plot_module_umap(adata_st, n, pattern, mode = mode, show = False, filename = f"plots/{n}_{mode}_umap.png")


def create_report(adata_st):
    '''
    Create a report for the results of CellSP.
    Arguments
    ----------
    adata_st : AnnData
        Spatial transcriptomic data.
    '''
    rows_summary = []
    rows = []
    data = []
    subprocess.run(["mkdir", "plots"])
    for n, row in adata_st.uns['instant_biclustering'].iterrows():
        mode = "instant_biclustering"
        pattern = "Colocalization"
        rows.append([mode, pattern, row['#cells'], len(row.genes.split(","))])
        make_plots(adata_st, n, mode, pattern)
        geo_module = adata_st.uns['instant_biclustering_geo_module'][str(n)].iloc[:5][['term', 'pValue']]
        geo_module['pValue'] = geo_module['pValue'].apply(lambda x: f'{x:.2e}' if isinstance(x, float) else x)
        geo_cell = adata_st.uns['instant_biclustering_geo_cell'][str(n)].iloc[:5][['term', 'pValue']]
        geo_cell['pValue'] = geo_cell['pValue'].apply(lambda x: f'{x:.2e}' if isinstance(x, float) else x)
        data.append([f"plots/{n}_{mode}_{pattern}.png", f"plots/{n}_{mode}_spatial.png", f"plots/{n}_{mode}_umap.png", geo_module, geo_cell, [f"Module {n}", f"#Cells: {row['#cells']}", f"#Genes: {len(row.genes.split(","))}", f"Genes: {row.genes}"]])
    for n, row in adata_st.uns['sprawl_biclustering'].iterrows():
        mode = "sprawl_biclustering"
        pattern = row.method
        rows.append([mode, pattern, row['#cells'], len(row.genes.split(","))])
        make_plots(adata_st, n, mode, pattern)
        geo_module = adata_st.uns['sprawl_biclustering_geo_module'][str(n)].iloc[:5][['term', 'pValue']]
        geo_module['pValue'] = geo_module['pValue'].apply(lambda x: f'{x:.2e}' if isinstance(x, float) else x)
        geo_cell = adata_st.uns['sprawl_biclustering_geo_cell'][str(n)].iloc[:5][['term', 'pValue']]
        geo_cell['pValue'] = geo_cell['pValue'].apply(lambda x: f'{x:.2e}' if isinstance(x, float) else x)
        data.append([f"plots/{n}_{mode}_{pattern}.png", f"plots/{n}_{mode}_spatial.png", f"plots/{n}_{mode}_umap.png", geo_module, geo_cell, [f"Module {n}", f"Pattern: {pattern}", f"#Cells: {row['#cells']}", f"#Genes: {len(row.genes.split(","))}", f"Genes: {row.genes}"]])
    df = pd.DataFrame(rows, columns=["Mode", "Pattern", "#cells", "#genes"])
    df_html = df.to_html(classes='table table-striped', escape=False, index=False)
    template = Template(template_str)
    rendered_html = template.render(df=df, image_pairs = data)
    with open('report.html', 'w') as f:
        f.write(rendered_html)