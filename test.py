import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px

# create sample data
df = pd.DataFrame({
    'Year': [2020, 2020, 2021, 2021],
    'Month': ['Jan', 'Feb', 'Jan', 'Feb'],
    'Sales': [100, 200, 150, 250],
    'Profit': [50, 80, 70, 120]
})

# group by Year and Month and aggregate the data
grouped_df = df.groupby(['Year', 'Month']).sum()

# create a multi-index DataFrame
multi_index_df = grouped_df.set_index([grouped_df.index.get_level_values(0), grouped_df.index.get_level_values(1)])

# create function to generate table with clickable headers
def generate_table(dataframe, selected_column=None):
    # header row
    header_row = [html.Th('Year'), html.Th('Month')] + [html.Th(col, id=f'{col}-header', className='clickable-header' + (' selected' if col==selected_column else '')) for col in dataframe.columns]
    # data rows
    data_rows = []
    for i in range(len(dataframe)):
        data_row = [html.Td(dataframe.index[i][0]), html.Td(dataframe.index[i][1])] + [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]
        data_rows.append(html.Tr(data_row))
    # table
    table = html.Table(id='table', children=[
        html.Thead(html.Tr(header_row)),
        html.Tbody(data_rows)
    ])
    return table

# create app
app = dash.Dash(__name__)

# define layout
app.layout = html.Div([
    # table
    generate_table(multi_index_df),
    # plotly graph
    dcc.Graph(id='graph')
])

# define callback for updating the graph
@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input(f'{col}-header', 'n_clicks') for col in multi_index_df.columns]
)
def update_graph(*args):
    # get the index of the clicked column header
    index = args.index(max(args))
    # get the column name from the index
    column_name = multi_index_df.columns[index]
    # create a line chart using Plotly Express
    fig = px.line(multi_index_df.reset_index(), x=['Year', 'Month'], y=column_name, color='Year', line_group='Year')
    return fig

# add CSS stylesheet to style selected column
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

# define callback for highlighting selected column
@app.callback(
    dash.dependencies.Output('table', 'children'),
    [dash.dependencies.Input(f'{col}-header', 'n_clicks') for col in multi_index_df.columns]
)
def highlight_selected_column(*args):
    # get the index of the clicked column header
    index = args.index(max(args))
    # get the column name from the index
    column_name = multi_index_df.columns[index]
    # generate table with selected column highlighted
    table = generate_table(multi_index_df, selected_column=column_name)
    return table
# run app
if __name__ == '__main__':
    app.run_server(debug=True)