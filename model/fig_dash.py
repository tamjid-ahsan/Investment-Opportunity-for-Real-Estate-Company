import dash
import dash_core_components as dcc
import dash_html_components as html
import joblib

fig = joblib.load('viz.joblib')
print(f'{"+"*30}')
print('press `CTRL+C` to exit')
print(f'{"+"*30}')
app = dash.Dash(__name__)
app.layout = html.Div([dcc.Graph(figure=fig)])
# app.run_server(debug=True, use_reloader=False)

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)