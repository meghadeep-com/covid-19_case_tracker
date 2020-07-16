# Created by Meghadeep Roy Chowdhury 7/16/2020
# All rights reserved under GNU AGPLv3
# details: https://www.gnu.org/licenses/agpl-3.0.en.html


import gc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import flask


def convert_tuples_to_dict(tup):
    # Make an empty dictionary
    di = {}
    # Populate the dictionary
    for i, j in tup:
        di.setdefault(i, []).append(j)
    return di


def make_init_df_global(df):
    # Set Country and State as multi-index of the dataframe
    df = df.set_index(['Country/Region', 'Province/State'], drop=True)
    # Keep a separate dataframe for location information
    df_location = df[['Lat', 'Long']].reset_index()
    # Drop location from the main dataframe
    df = df.drop(labels=['Lat', 'Long'], axis=1)
    # Garbage collection for low RAM systems
    gc.collect()

    return df, df_location


def get_transpose(df):
    # Transpose high resolution data and keep the dates in columns
    df = df.transpose().reset_index().rename(columns={'index': 'Date'})
    # Convert Date column items from string to DateTime objects
    df['Date'] = pd.to_datetime(df['Date'])
    # Add a separate column for overall total values in the dataframe
    df.loc[:, ('Total', 'Total')] = df.sum(axis=1)
    # Garbage collection for low RAM systems
    gc.collect()

    return df


def make_dfs_better(df):
    # Fill NaN multi-index column names with Total
    df.columns = pd.MultiIndex.from_frame(df.columns.to_frame().fillna('Total'))
    # Add level[1] total values for each level[0]
    for i in list(df.columns.get_level_values(0).unique()):
        if (i != 'Date') and (i != 'Total'):
            if 'Total' not in list(df[i].columns):
                df[i, 'Total'] = df[i].sum(axis=1)
    # Get separate dataframe for daily increase
    df_daily = df.diff(axis=0)
    # Reinsert Date column in Daily Increase dataframe
    df_daily['Date'] = df['Date']
    # Garbage collection for low RAM systems
    gc.collect()

    return df, df_daily


def level1_validate(level0, level1, df):
    di = convert_tuples_to_dict(np.delete(df.columns.values, 0))
    if level1 == ['Total']:
        flag = set(level1).issubset(set(di[level0[0]]))
        if flag:
            return True
        else:
            raise KeyError
    else:
        return set(level1).issubset(set(di[level0[0]]))


def get_viz_data(level0, level1, df_raw, df_daily, graph_type):
    # Single selection level0
    if len(level0) == 1:
        try:
            if not level1_validate(level0, level1, df_raw):
                level1 = ['Total']
        except KeyError:
            print('What the FUCK?! How can this shit happen?!')
            level0 = ['Total']
            pass
        # Single selection level1
        if len(level1) == 1:
            # Check for daily increase graph type
            if graph_type == 'daily':
                graph_data = go.Scatter(x=df_daily['Date'],
                                        y=df_daily[level0[0]][level1[0]])
            # Daily Increase - 7 Day rolling average
            elif graph_type == 'rolling':
                rolling = df_daily.rolling(7).mean()
                graph_data = go.Scatter(x=df_daily['Date'],
                                        y=rolling[level0[0]][level1[0]])
            # For any other graph type
            else:
                graph_data = go.Scatter(x=df_raw['Date'],
                                        y=df_raw[level0[0]][level1[0]])
            # Get current number
            current_number = f'{int(list(df_raw[level0[0]][level1[0]])[-1]):,}'
            # Get last increase
            last_increase = f'{int(list(df_daily[level0[0]][level1[0]])[-1]):,}'

            return graph_data, current_number, last_increase
        # Multiple selection level1
        else:
            # Make multi-level1-selection raw df
            multi_level1_df_raw = df_raw[level0[0]][level1]
            multi_level1_df_raw['Total_temp'] = multi_level1_df_raw.sum(axis=1)
            # Make multi-level1-selection daily df
            multi_level1_df_daily = df_daily[level0[0]][level1]
            multi_level1_df_daily['Total_temp'] = multi_level1_df_daily.sum(axis=1)

            # # Empty list for storing graph data
            # graph_data = []
            # column_level1 = level1
            # column_level1.append('Total_temp')
            # # Check for daily increase graph type
            # if graph_type == 'daily':
            #     for i in column_level1:
            #         graph_data.append(go.Scatter(x=df_daily['Date'],
            #                                      y=multi_level1_df_daily[i]))
            # # Daily Increase - 7 Day rolling average
            # elif graph_type == 'rolling':
            #     rolling = multi_level1_df_daily.rolling(7).mean()
            #     for i in column_level1:
            #         graph_data.append(go.Scatter(x=df_daily['Date'],
            #                                      y=rolling[i]))
            # # For any other graph type
            # else:
            #     for i in column_level1:
            #         graph_data.append(go.Scatter(x=df_raw['Date'],
            #                                      y=multi_level1_df_raw[i]))

            # Check for daily increase graph type
            if graph_type == 'daily':
                graph_data = go.Scatter(x=df_daily['Date'],
                                        y=multi_level1_df_daily['Total_temp'])
            # Daily Increase - 7 Day rolling average
            elif graph_type == 'rolling':
                rolling = multi_level1_df_daily.rolling(7).mean()
                graph_data = go.Scatter(x=df_daily['Date'],
                                        y=rolling['Total_temp'])
            # For any other graph type
            else:
                graph_data = go.Scatter(x=df_raw['Date'],
                                        y=multi_level1_df_raw['Total_temp'])
            # Get current number
            current_number = f'{int(list(multi_level1_df_raw["Total_temp"])[-1]):,}'
            # Get last increase
            last_increase = f'{int(list(multi_level1_df_daily["Total_temp"])[-1]):,}'

            return graph_data, current_number, last_increase
    # Multi selection level0
    else:
        # Make multi-level0-selection dfs
        multi_level0_df_raw = pd.DataFrame()
        multi_level0_df_daily = pd.DataFrame()
        for i in level0:
            multi_level0_df_raw[i] = df_raw[i]['Total']
            multi_level0_df_daily[i] = df_daily[i]['Total']
        multi_level0_df_raw['Total_temp'] = multi_level0_df_raw.sum(axis=1)
        multi_level0_df_daily['Total_temp'] = multi_level0_df_daily.sum(axis=1)
        # # Empty list for storing graph data
        # graph_data = []
        # column_level0 = level0
        # column_level0.append('Total_temp')
        # # Check for daily increase graph type
        # if graph_type == 'daily':
        #     for i in column_level0:
        #         graph_data.append(go.Scatter(x=df_daily['Date'],
        #                                      y=multi_level0_df_daily[i]))
        # # Daily Increase - 7 Day rolling average
        # elif graph_type == 'rolling':
        #     rolling = multi_level0_df_daily.rolling(7).mean()
        #     for i in column_level0:
        #         graph_data.append(go.Scatter(x=df_daily['Date'],
        #                                      y=rolling[i]))
        # # For any other graph type
        # else:
        #     for i in column_level0:
        #         graph_data.append(go.Scatter(x=df_raw['Date'],
        #                                      y=multi_level0_df_raw[i]))

        # Check for daily increase graph type
        if graph_type == 'daily':
            graph_data = go.Scatter(x=df_daily['Date'],
                                    y=multi_level0_df_daily['Total_temp'])
        # Daily Increase - 7 Day rolling average
        elif graph_type == 'rolling':
            rolling = multi_level0_df_daily.rolling(7).mean()
            graph_data = go.Scatter(x=df_daily['Date'],
                                    y=rolling['Total_temp'])
        # For any other graph type
        else:
            graph_data = go.Scatter(x=df_raw['Date'],
                                    y=multi_level0_df_raw['Total_temp'])

        # Get current number
        current_number = f'{int(list(multi_level0_df_raw["Total_temp"])[-1]):,}'
        # Get last increase
        last_increase = f'{int(list(multi_level0_df_daily["Total_temp"])[-1]):,}'

        return graph_data, current_number, last_increase


"""
############
GET RAW DATA
############
"""
# url_common = '~/PycharmProjects/coronavirus/'
url_common = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
confirmed_global = pd.read_csv(url_common + 'time_series_covid19_confirmed_global.csv')
confirmed_us = pd.read_csv(url_common + 'time_series_covid19_confirmed_US.csv')
death_global = pd.read_csv(url_common + 'time_series_covid19_deaths_global.csv')
death_us = pd.read_csv(url_common + 'time_series_covid19_deaths_US.csv')
recovered_global = pd.read_csv(url_common + 'time_series_covid19_recovered_global.csv')
# us_population = pd.read_csv(r'~/PycharmProjects/coronavirus/co-est2019-alldata.csv', encoding='latin1',
#                                    usecols=['STATE', 'COUNTY', 'POPESTIMATE2019'], dtype=str)
# death_us_fips = death_us

"""
################
MAKE DATA USABLE
################
"""
# Suppress Pandas deep-copy error
pd.options.mode.chained_assignment = None

# Clean up raw global data
confirmed_global, confirmed_global_location = make_init_df_global(confirmed_global)
death_global, death_global_location = make_init_df_global(death_global)
recovered_global, recovered_global_location = make_init_df_global(recovered_global)

# Clean up raw US data
# Set State and County as multi-index of the dataframe and drop the columns we won't use
confirmed_us = confirmed_us.set_index(['Province_State', 'Admin2'], drop=True).drop(
    labels=['Country_Region', 'UID', 'iso2', 'iso3', 'code3', 'Combined_Key'], axis=1)
# Rename Longitude column
confirmed_us = confirmed_us.rename(columns={'Long_': 'Long'})
# Keep a separate dataframe for location and FIPS information
confirmed_us_location = confirmed_us[['Lat', 'Long', 'FIPS']].reset_index()
# Drop the columns we won't use in the main dataframe
confirmed_us = confirmed_us.drop(labels=['Lat', 'Long', 'FIPS'], axis=1)

# Set State and County as multi-index of the dataframe and drop the columns we won't use
death_us = death_us.set_index(['Province_State', 'Admin2'], drop=True).drop(
    labels=['Country_Region', 'UID', 'iso2', 'iso3', 'code3', 'Combined_Key'], axis=1)
# Rename Longitude column
death_us = death_us.rename(columns={'Long_': 'Long'})
# Keep a separate dataframe for location and FIPS information
death_us_location = death_us[['Lat', 'Long', 'FIPS']].reset_index()
# Drop the columns we won't use in the main dataframe
death_us = death_us.drop(labels=['Lat', 'Long', 'Population', 'FIPS'], axis=1)

# # Keep separate dataframe for population information
# us_population['POPESTIMATE2019'] = us_population['POPESTIMATE2019'].apply(int)
# us_population['FIPS'] = us_population['STATE'] + us_population['COUNTY']
# us_population = us_population.drop(labels=['STATE', 'COUNTY'], axis=1)
# fips = []
# k = 0
# while k < len(death_us_fips['FIPS']):
#     try:
#         fips.append(str(int(death_us_fips['FIPS'][k])))
#     except ValueError:
#         print('hmm')
#         fips.append('')
#         pass
#     k += 1
# fips = pd.Series(fips).apply(lambda x: x.zfill(5))
# death_us_fips['FIPS'] = fips

# Transpose high resolution data and get overall totals
confirmed_global = get_transpose(confirmed_global)
death_global = get_transpose(death_global)
recovered_global = get_transpose(recovered_global)
confirmed_us = get_transpose(confirmed_us)
death_us = get_transpose(death_us)

# Add Country-Wise and State-Wise Totals and Daily Increase DF
confirmed_global, confirmed_global_daily = make_dfs_better(confirmed_global)
death_global, death_global_daily = make_dfs_better(death_global)
recovered_global, recovered_global_daily = make_dfs_better(recovered_global)
confirmed_us, confirmed_us_daily = make_dfs_better(confirmed_us)
death_us, death_us_daily = make_dfs_better(death_us)


"""
###################
Actual Dash Stuff
###################
"""
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# Get dictionaries of states in countries
global_dropdown_dict = convert_tuples_to_dict(np.delete(confirmed_global.columns.values, 0))
# Get dictionaries of counties in states
us_dropdown_dict = convert_tuples_to_dict(np.delete(confirmed_us.columns.values, 0))

# Change Dash app title
app.title = 'COVID-19 Case Tracker'

# Graph type options
graph_type_dropdown = [{'label': 'Raw Cumulative', 'value': 'linear'},
                       {'label': 'Logarithmic', 'value': 'log'},
                       {'label': 'Daily Cases', 'value': 'daily'},
                       {'label': 'Daily Cases (7 Day Rolling)', 'value': 'rolling'}]
# Dash drop-down values to names dictionary
fig_graph_title = {'linear': 'Raw Cumulative', 'log': 'Logarithmic', 'daily': 'Daily Cases',
                   None: 'Raw Cumulative', 'rolling': 'Daily Cases (7 Day Rolling Average)'}

# Dash app layout for tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Global Data', children=[
            # Country Dropdown
            html.P([
                html.Label('Country/Region: '),
                dcc.Dropdown(id='country-dropdown',
                             options=[{'label': i, 'value': i}
                                      for i in list(global_dropdown_dict.keys())],
                             value='Total',
                             placeholder='Total',
                             multi=True)
            ], style={'width': '400px',
                      'fontSize': '20px',
                      'padding-left': '100px',
                      'display': 'inline-block'}),
            # State Dropdown
            html.P([
                html.Label('State/Province: '),
                dcc.Dropdown(id='state-dropdown',
                             value='Total',
                             placeholder='Total',
                             multi=True),
            ], style={'width': '400px',
                      'fontSize': '20px',
                      'padding-left': '100px',
                      'display': 'inline-block'}),
            # Graph Type dropdown
            html.P([
                html.Label('Type of graph: '),
                dcc.Dropdown(id='graph-type-global',
                             options=graph_type_dropdown,
                             value='linear',
                             placeholder='Raw Cumulative')
            ], style={'width': '400px',
                      'fontSize': '20px',
                      'padding-left': '100px',
                      'display': 'inline-block'}),
            # The actual tables and graphs
            dcc.Graph(id='current_numbers_global'),
            dcc.Graph(id='increase_numbers_global'),
            dcc.Graph(id='fig_global')
        ]),
        dcc.Tab(label='US Specific Data', children=[
            # US State Dropdown
            html.P([
                html.Label('State: '),
                dcc.Dropdown(id='us-state-dropdown',
                             options=[{'label': i, 'value': i}
                                      for i in list(us_dropdown_dict.keys())],
                             value=['Total'],
                             placeholder='Total',
                             multi=True)
            ], style={'width': '400px',
                      'fontSize': '20px',
                      'padding-left': '100px',
                      'display': 'inline-block'}),
            # County Dropdown
            html.P([
                html.Label('County: '),
                dcc.Dropdown(id='us-county-dropdown',
                             value=['Total'],
                             placeholder='Total',
                             multi=True,
                             disabled=False),
            ], style={'width': '400px',
                      'fontSize': '20px',
                      'padding-left': '100px',
                      'display': 'inline-block'}),
            # Graph type dropdown
            html.P([
                html.Label('Type of graph: '),
                dcc.Dropdown(id='graph-type-us',
                             options=graph_type_dropdown,
                             value='linear',
                             placeholder='Raw Cumulative')
            ], style={'width': '400px',
                      'fontSize': '20px',
                      'padding-left': '100px',
                      'display': 'inline-block'}),
            # The actual tables and graphs
            dcc.Graph(id='current_numbers_us'),
            dcc.Graph(id='increase_numbers_us'),
            dcc.Graph(id='fig_us')
        ]),
    ])
])


# State app drop-down callback
@app.callback([Output('state-dropdown', 'options'),
               Output('state-dropdown', 'disabled')],
              [Input('country-dropdown', 'value')])
def update_global_dropdown(country):
    # Clear State
    if not country:
        country = ['Total']
    # Single-select country
    if len(country) == 1:
        # Overall Total
        if country == ['Total']:
            return [{'label': 'Total', 'value': 'Total'}], True
        # No state in country
        elif len(global_dropdown_dict[country[0]]) == 1:
            return [{'label': 'Total', 'value': 'Total'}], True
        # Regular state with multiple counties
        else:
            return [{'label': i, 'value': i} for i in global_dropdown_dict[country[0]]], False
    # Multi-select us-state
    else:
        return [{'label': 'Total', 'value': 'Total'}], True


# Global visualization update callback
@app.callback([Output('fig_global', 'figure'),
               Output('current_numbers_global', 'figure'),
               Output('increase_numbers_global', 'figure')],
              [Input('country-dropdown', 'value'),
               Input('state-dropdown', 'value'),
               Input('graph-type-global', 'value')])
def update_figure(country, state, graph_type_global):
    if (country == 'Total') or (not country) or (country == ['Total']):
        country = ['Total']
        state = ['Total']
    # Initialize state
    if not level1_validate(country, state, confirmed_global):
        state = ['Total']
    if len(country) == 1:
        if (state == 'Total') or (not state):
            state = ['Total']
        if len(state) == 1:
            # Default option or when user removes state option
            if (country == ['Total']) or (not country):
                country_input = ['Total']
                state_input = ['Total']

                title_confirmed = 'Confirmed Cases: Global'
                confirmed_global_graph_data, confirmed_number_global, confirmed_increase_global = \
                    get_viz_data(country_input, state_input, confirmed_global,
                                 confirmed_global_daily, graph_type_global)

                title_death = 'Number of Deaths: Global'
                death_global_graph_data, death_number_global, death_increase_global = \
                    get_viz_data(country_input, state_input, death_global,
                                 death_global_daily, graph_type_global)

                title_recovery = 'Recovery Numbers: Global'
                recovered_global_graph_data, recovery_number_global, recovered_increase_us = \
                    get_viz_data(country_input, state_input, recovered_global,
                                 recovered_global_daily, graph_type_global)

            # Default option or when user removes state option
            elif not set(state).issubset(set(global_dropdown_dict[country[0]])):
                country_input = country
                state_input = ['Total']

                title_confirmed = 'Confirmed Cases: ' + country_input[0] + ' - ' + state_input[0]
                confirmed_global_graph_data, confirmed_number_global, confirmed_increase_global = \
                    get_viz_data(country_input, state_input, confirmed_global,
                                 confirmed_global_daily, graph_type_global)

                title_death = 'Number of Deaths: ' + country_input[0] + ' - ' + state_input[0]
                death_global_graph_data, death_number_global, death_increase_global = \
                    get_viz_data(country_input, state_input, death_global,
                                 death_global_daily, graph_type_global)

                title_recovery = 'Recovery Numbers: ' + country_input[0] + ' - ' + state_input[0]
                recovered_global_graph_data, recovery_number_global, recovered_increase_us = \
                    get_viz_data(country_input, state_input, recovered_global,
                                 recovered_global_daily, graph_type_global)

            # When user selects everything
            else:
                country_input = country
                state_input = state

                title_confirmed = 'Confirmed Cases: ' + country_input[0] + ' - ' + state_input[0]
                confirmed_global_graph_data, confirmed_number_global, confirmed_increase_global = \
                    get_viz_data(country_input, state_input, confirmed_global,
                                 confirmed_global_daily, graph_type_global)

                # Check if death numbers available state-wise
                if set(state).issubset(set(death_global[country[0]].columns.to_list())):
                    state_input = state

                    title_death = 'Number of Deaths: ' + country_input[0] + ' - ' + state_input[0]
                    death_global_graph_data, death_number_global, death_increase_global = \
                        get_viz_data(country_input, state_input, death_global,
                                     death_global_daily, graph_type_global)
                else:
                    state_input = ['Total']

                    title_death = 'Number of Deaths: ' + country_input[0] + ' - ' + state_input[0]
                    death_global_graph_data, death_number_global, death_increase_global = \
                        get_viz_data(country_input, state_input, death_global,
                                     death_global_daily, graph_type_global)

                # Check if recovery numbers available state-wise
                if set(state).issubset(set(recovered_global[country[0]].columns.to_list())):
                    state_input = state

                    title_recovery = 'Recovery Numbers: ' + country_input[0] + ' - ' + state_input[0]
                    recovered_global_graph_data, recovery_number_global, recovered_increase_us = \
                        get_viz_data(country_input, state_input, recovered_global,
                                     recovered_global_daily, graph_type_global)
                else:
                    state_input = ['Total']

                    title_recovery = 'Recovery Numbers: ' + country_input[0] + ' - ' + state_input[0]
                    recovered_global_graph_data, recovery_number_global, recovered_increase_us = \
                        get_viz_data(country_input, state_input, recovered_global,
                                     recovered_global_daily, graph_type_global)

            # Make an empty subplot figure
            fig_global = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                                       subplot_titles=(title_confirmed, title_recovery, title_death))
            # Insert traces in the figure
            fig_global.add_trace(confirmed_global_graph_data, row=1, col=1)
            fig_global.add_trace(death_global_graph_data, row=2, col=2)
            fig_global.add_trace(recovered_global_graph_data, row=2, col=1)
            # Update size and title
            fig_global.update_layout(showlegend=False, height=800,
                                     title_text=fig_graph_title[graph_type_global] + ': ' +
                                                list(confirmed_global['Date'])[-1].strftime('%x'))
            # Check for logarithmic graph type
            if graph_type_global == 'log':
                fig_global.update_yaxes(type='log')

            # Make current numbers table
            current_numbers_global = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_number_global], [recovery_number_global],
                                                  [death_number_global]]))],
                layout=go.Layout(
                    title=go.layout.Title(text='Current Numbers: ' +
                                               list(confirmed_global['Date'])[-1].strftime('%x'))))

            # Make last increase numbers table
            increase_numbers_global = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_increase_global], [recovered_increase_us],
                                                  [death_increase_global]]))],
                layout=go.Layout(
                    title=go.layout.Title(text='Last Increase: ' +
                                               list(confirmed_global['Date'])[-1].strftime('%x'))))

            # Update sizes of the tables
            current_numbers_global.update_layout(height=250)
            increase_numbers_global.update_layout(height=250)

            return fig_global, current_numbers_global, increase_numbers_global
        # Multi-select state
        else:
            country_input = country
            state_input = state

            title_confirmed = 'Confirmed Cases: ' + country_input[0] + ' - Selected States'
            confirmed_global_graph_data, confirmed_number_global, confirmed_increase_global = \
                get_viz_data(country_input, state_input, confirmed_global,
                             confirmed_global_daily, graph_type_global)

            # Check if death numbers are available state-wise
            if set(state).issubset(set(death_global[country[0]].columns.to_list())):
                title_death = 'Number of deaths: ' + country_input[0] + ' - Selected Counties'
                death_global_graph_data, death_number_global, death_increase_global = \
                    get_viz_data(country_input, state_input, death_global,
                                 death_global_daily, graph_type_global)
            else:
                state_input = ['Total']
                title_death = 'Number of Deaths: ' + country_input[0] + ' - ' + state_input[0]
                death_global_graph_data, death_number_global, death_increase_global = \
                    get_viz_data(country_input, state_input, death_global,
                                 death_global_daily, graph_type_global)

            # Check if recovery numbers available state-wise
            if set(state).issubset(set(recovered_global[country[0]].columns.to_list())):
                state_input = state

                title_recovery = 'Recovery Numbers: ' + country_input[0] + ' - ' + state_input[0]
                recovered_global_graph_data, recovery_number_global, recovered_increase_us = \
                    get_viz_data(country_input, state_input, recovered_global,
                                 recovered_global_daily, graph_type_global)
            else:
                state_input = ['Total']

                title_recovery = 'Recovery Numbers: ' + country_input[0] + ' - ' + state_input[0]
                recovered_global_graph_data, recovery_number_global, recovered_increase_us = \
                    get_viz_data(country_input, state_input, recovered_global,
                                 recovered_global_daily, graph_type_global)

            # Make an empty subplot figure
            fig_global = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                                       subplot_titles=(title_confirmed, title_recovery, title_death))
            # Insert traces in the figure
            # j = 0
            # while j < len(state):
            # fig_global.add_trace(confirmed_global_graph_data[j], row=1, col=1)
            # fig_global.add_trace(death_global_graph_data[j], row=2, col=2)
            # j += 1
            fig_global.add_trace(confirmed_global_graph_data, row=1, col=1)
            fig_global.add_trace(death_global_graph_data, row=2, col=2)
            fig_global.add_trace(recovered_global_graph_data, row=2, col=1)

            # Update size and title
            fig_global.update_layout(showlegend=False, height=800,
                                     title_text=fig_graph_title[graph_type_global] + ': ' +
                                                list(confirmed_global['Date'])[-1].strftime('%x'))
            # Check for logarithmic graph type
            if graph_type_global == 'log':
                fig_global.update_yaxes(type='log')

            # Make current numbers table
            current_numbers_global = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_number_global], [recovery_number_global],
                                                  [death_number_global]]))],
                layout=go.Layout(
                    title=go.layout.Title(
                        text='Current Numbers: ' + list(confirmed_global['Date'])[-1].strftime('%x'))))

            # Make last increase numbers table
            increase_numbers_global = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_increase_global], [recovered_increase_us],
                                                  [death_increase_global]]))],
                layout=go.Layout(
                    title=go.layout.Title(text='Last Increase: ' + list(confirmed_global['Date'])[-1].strftime('%x'))))

            # Update sizes of the tables
            current_numbers_global.update_layout(height=250)
            increase_numbers_global.update_layout(height=250)

            return fig_global, current_numbers_global, increase_numbers_global
    # Multi-Select Countries
    else:
        country_input = country
        state_input = ['Total']

        title_confirmed = 'Confirmed Cases: Selected Countries'
        confirmed_global_graph_data, confirmed_number_global, confirmed_increase_global = \
            get_viz_data(country_input, state_input, confirmed_global,
                         confirmed_global_daily, graph_type_global)

        title_death = 'Number of Deaths: Selected Countries'
        death_global_graph_data, death_number_global, death_increase_global = \
            get_viz_data(country_input, state_input, death_global,
                         death_global_daily, graph_type_global)

        title_recovery = 'Recovery Numbers: Selected Counties'
        recovered_global_graph_data, recovery_number_global, recovered_increase_us = \
            get_viz_data(country_input, state_input, recovered_global,
                         recovered_global_daily, graph_type_global)

        # Make an empty subplot figure
        fig_global = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                                   subplot_titles=(title_confirmed, title_recovery, title_death))
        # Insert traces in the figure
        # j = 0
        # while j < len(state):
        # fig_global.add_trace(confirmed_global_graph_data[j], row=1, col=1)
        # fig_global.add_trace(death_global_graph_data[j], row=2, col=2)
        # j += 1
        fig_global.add_trace(confirmed_global_graph_data, row=1, col=1)
        fig_global.add_trace(death_global_graph_data, row=2, col=2)
        fig_global.add_trace(recovered_global_graph_data, row=2, col=1)
        # Update size and title
        fig_global.update_layout(showlegend=False, height=800,
                                 title_text=fig_graph_title[graph_type_global] + ': ' + list(confirmed_global['Date'])[
                                     -1].strftime(
                                     '%x'))
        # Check for logarithmic graph type
        if graph_type_global == 'log':
            fig_global.update_yaxes(type='log')

        # Make current numbers table
        current_numbers_global = go.Figure(
            data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                           cells=dict(values=[[confirmed_number_global], [recovery_number_global],
                                              [death_number_global]]))],
            layout=go.Layout(
                title=go.layout.Title(text='Current Numbers: ' + list(confirmed_global['Date'])[-1].strftime('%x'))))

        # Make last increase numbers table
        increase_numbers_global = go.Figure(
            data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                           cells=dict(values=[[confirmed_increase_global], [recovered_increase_us],
                                              [death_increase_global]]))],
            layout=go.Layout(
                title=go.layout.Title(text='Last Increase: ' + list(confirmed_global['Date'])[-1].strftime('%x'))))

        # Update sizes of the tables
        current_numbers_global.update_layout(height=250)
        increase_numbers_global.update_layout(height=250)

        return fig_global, current_numbers_global, increase_numbers_global


# County app drop-down callback
@app.callback([Output('us-county-dropdown', 'options'),
               Output('us-county-dropdown', 'disabled')],
              [Input('us-state-dropdown', 'value')])
def update_us_dropdown(us_state):
    # Clear State
    if (not us_state) or (us_state == 'Total'):
        us_state = ['Total']
    # Single-select us-state
    if len(us_state) == 1:
        # Overall Total
        if us_state == ['Total']:
            return [{'label': 'Total', 'value': 'Total'}], True
        # No county in state
        elif len(us_dropdown_dict[us_state[0]]) == 1:
            return [{'label': 'Total', 'value': 'Total'}], True
        # Regular state with multiple counties
        else:
            return [{'label': i, 'value': i} for i in us_dropdown_dict[us_state[0]]], False
    # Multi-select us-state
    else:
        return [{'label': 'Total', 'value': 'Total'}], True


# US visualization update callback
@app.callback([Output('fig_us', 'figure'),
               Output('current_numbers_us', 'figure'),
               Output('increase_numbers_us', 'figure')],
              [Input('us-state-dropdown', 'value'),
               Input('us-county-dropdown', 'value'),
               Input('graph-type-us', 'value')])
def update_us_figure(us_state, county, graph_type_us):
    if (us_state == 'Total') or (not us_state) or (us_state == ['Total']):
        us_state = ['Total']
        county = ['Total']
    # Initialize county
    if not level1_validate(us_state, county, confirmed_us):
        county = ['Total']
    if len(us_state) == 1:
        if (county == 'Total') or (not county):
            county = ['Total']
        if len(county) == 1:
            # Default option or when user removes state option
            if (us_state == ['Total']) or (not us_state):
                state_input = ['Total']
                county_input = ['Total']

                title_confirmed = 'Confirmed Cases: US'
                confirmed_us_graph_data, confirmed_number_us, confirmed_increase_us = \
                    get_viz_data(state_input, county_input, confirmed_us,
                                 confirmed_us_daily, graph_type_us)

                title_death = 'Number of Deaths: US'
                death_us_graph_data, death_number_us, death_increase_us = \
                    get_viz_data(state_input, county_input, death_us,
                                 death_us_daily, graph_type_us)

            # Default option or when user removes county option
            elif not set(county).issubset(set(us_dropdown_dict[us_state[0]])):
                state_input = us_state
                county_input = ['Total']

                title_confirmed = 'Confirmed Cases: ' + state_input[0] + ' - ' + county_input[0]
                confirmed_us_graph_data, confirmed_number_us, confirmed_increase_us = \
                    get_viz_data(state_input, county_input, confirmed_us,
                                 confirmed_us_daily, graph_type_us)

                title_death = 'Number of Deaths: ' + state_input[0] + ' - ' + county_input[0]
                death_us_graph_data, death_number_us, death_increase_us = \
                    get_viz_data(state_input, county_input, death_us,
                                 death_us_daily, graph_type_us)

            # When user selects everything
            else:
                state_input = us_state
                county_input = county

                title_confirmed = 'Confirmed Cases: ' + state_input[0] + ' - ' + county_input[0]
                confirmed_us_graph_data, confirmed_number_us, confirmed_increase_us = \
                    get_viz_data(state_input, county_input, confirmed_us,
                                 confirmed_us_daily, graph_type_us)

                # Check if death numbers available county-wise
                if set(county).issubset(set(death_us[us_state[0]].columns.to_list())):
                    title_death = 'Number of Deaths: ' + state_input[0] + ' - ' + county_input[0]
                    death_us_graph_data, death_number_us, death_increase_us = \
                        get_viz_data(state_input, county_input, death_us,
                                     death_us_daily, graph_type_us)
                else:
                    county_input = ['Total']

                    title_death = 'Number of Deaths: ' + state_input[0] + ' - ' + county_input[0]
                    death_us_graph_data, death_number_us, death_increase_us = \
                        get_viz_data(state_input, county_input, death_us,
                                     death_us_daily, graph_type_us)

            title_recovery = 'Recovery Numbers: US - Total'
            recovered_us_graph_data, recovery_number_us, recovered_increase_us = \
                get_viz_data(['US'], ['Total'], recovered_global,
                             recovered_global_daily, graph_type_us)

            # Make an empty subplot figure
            fig_us = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                                   subplot_titles=(title_confirmed, title_recovery, title_death))
            # Insert traces in the figure
            fig_us.add_trace(confirmed_us_graph_data, row=1, col=1)
            fig_us.add_trace(death_us_graph_data, row=2, col=2)
            fig_us.add_trace(recovered_us_graph_data, row=2, col=1)
            # Update size and title
            fig_us.update_layout(showlegend=False, height=800,
                                 title_text=fig_graph_title[graph_type_us] + ': ' +
                                            list(confirmed_us['Date'])[-1].strftime('%x'))
            # Check for logarithmic graph type
            if graph_type_us == 'log':
                fig_us.update_yaxes(type='log')

            # Make current numbers table
            current_numbers_us = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_number_us], [recovery_number_us],
                                                  [death_number_us]]))],
                layout=go.Layout(
                    title=go.layout.Title(text='Current Numbers: ' +
                                               list(confirmed_us['Date'])[-1].strftime('%x'))))

            # Make last increase numbers table
            increase_numbers_us = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_increase_us], [recovered_increase_us],
                                                  [death_increase_us]]))],
                layout=go.Layout(
                    title=go.layout.Title(text='Last Increase: ' +
                                               list(confirmed_us['Date'])[-1].strftime('%x'))))

            # Update sizes of the tables
            current_numbers_us.update_layout(height=250)
            increase_numbers_us.update_layout(height=250)

            return fig_us, current_numbers_us, increase_numbers_us
        # Multi-select county
        else:
            state_input = us_state
            county_input = county

            title_confirmed = 'Confirmed Cases: ' + state_input[0] + ' - Selected Counties'
            confirmed_us_graph_data, confirmed_number_us, confirmed_increase_us = \
                get_viz_data(state_input, county_input, confirmed_us,
                             confirmed_us_daily, graph_type_us)

            # Check if death numbers are available county-wise
            if set(county).issubset(set(death_us[us_state[0]].columns.to_list())):
                title_death = 'Number of deaths: ' + state_input[0] + ' - Selected Counties'
                death_us_graph_data, death_number_us, death_increase_us = \
                    get_viz_data(state_input, county_input, death_us,
                                 death_us_daily, graph_type_us)
            else:
                county_input = ['Total']
                title_death = 'Number of Deaths: ' + state_input[0] + ' - ' + county_input[0]
                death_us_graph_data, death_number_us, death_increase_us = \
                    get_viz_data(state_input, county_input, death_us,
                                 death_us_daily, graph_type_us)

            title_recovery = 'Recovery Numbers: US - Total'
            recovered_us_graph_data, recovery_number_us, recovered_increase_us = \
                get_viz_data(['US'], ['Total'], recovered_global,
                             recovered_global_daily, graph_type_us)

            # Make an empty subplot figure
            fig_us = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                                   subplot_titles=(title_confirmed, title_recovery, title_death))
            # Insert traces in the figure
            # j = 0
            # while j < len(county):
                # fig_us.add_trace(confirmed_us_graph_data[j], row=1, col=1)
                # fig_us.add_trace(death_us_graph_data[j], row=2, col=2)
                # j += 1
            fig_us.add_trace(confirmed_us_graph_data, row=1, col=1)
            fig_us.add_trace(death_us_graph_data, row=2, col=2)
            fig_us.add_trace(recovered_us_graph_data, row=2, col=1)

            # Update size and title
            fig_us.update_layout(showlegend=False, height=800,
                                 title_text=fig_graph_title[graph_type_us] + ': ' +
                                            list(confirmed_us['Date'])[-1].strftime('%x'))
            # Check for logarithmic graph type
            if graph_type_us == 'log':
                fig_us.update_yaxes(type='log')

            # Make current numbers table
            current_numbers_us = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_number_us], [recovery_number_us],
                                                  [death_number_us]]))],
                layout=go.Layout(
                    title=go.layout.Title(text='Current Numbers: ' + list(confirmed_us['Date'])[-1].strftime('%x'))))

            # Make last increase numbers table
            increase_numbers_us = go.Figure(
                data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                               cells=dict(values=[[confirmed_increase_us], [recovered_increase_us],
                                                  [death_increase_us]]))],
                layout=go.Layout(
                    title=go.layout.Title(text='Last Increase: ' + list(confirmed_us['Date'])[-1].strftime('%x'))))

            # Update sizes of the tables
            current_numbers_us.update_layout(height=250)
            increase_numbers_us.update_layout(height=250)

            return fig_us, current_numbers_us, increase_numbers_us
    # Multi-Select US States
    else:
        state_input = us_state
        county_input = ['Total']

        title_confirmed = 'Confirmed Cases: Selected States'
        confirmed_us_graph_data, confirmed_number_us, confirmed_increase_us = \
            get_viz_data(state_input, county_input, confirmed_us,
                         confirmed_us_daily, graph_type_us)

        title_death = 'Number of Deaths: Selected States'
        death_us_graph_data, death_number_us, death_increase_us = \
            get_viz_data(state_input, county_input, death_us,
                         death_us_daily, graph_type_us)

        title_recovery = 'Recovery Numbers: US - Total'
        recovered_us_graph_data, recovery_number_us, recovered_increase_us = \
            get_viz_data(['US'], ['Total'], recovered_global,
                         recovered_global_daily, graph_type_us)

        # Make an empty subplot figure
        fig_us = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
                               subplot_titles=(title_confirmed, title_recovery, title_death))
        # Insert traces in the figure
        j = 0
        # while j < len(county):
            # fig_us.add_trace(confirmed_us_graph_data[j], row=1, col=1)
            # fig_us.add_trace(death_us_graph_data[j], row=2, col=2)
            # j += 1
        fig_us.add_trace(confirmed_us_graph_data, row=1, col=1)
        fig_us.add_trace(death_us_graph_data, row=2, col=2)
        fig_us.add_trace(recovered_us_graph_data, row=2, col=1)
        # Update size and title
        fig_us.update_layout(showlegend=False, height=800,
                             title_text=fig_graph_title[graph_type_us] + ': ' + list(confirmed_us['Date'])[-1].strftime(
                                 '%x'))
        # Check for logarithmic graph type
        if graph_type_us == 'log':
            fig_us.update_yaxes(type='log')

        # Make current numbers table
        current_numbers_us = go.Figure(
            data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                           cells=dict(values=[[confirmed_number_us], [recovery_number_us],
                                              [death_number_us]]))],
            layout=go.Layout(
                title=go.layout.Title(text='Current Numbers: ' + list(confirmed_us['Date'])[-1].strftime('%x'))))

        # Make last increase numbers table
        increase_numbers_us = go.Figure(
            data=[go.Table(header=dict(values=[title_confirmed, title_recovery, title_death]),
                           cells=dict(values=[[confirmed_increase_us], [recovered_increase_us],
                                              [death_increase_us]]))],
            layout=go.Layout(
                title=go.layout.Title(text='Last Increase: ' + list(confirmed_us['Date'])[-1].strftime('%x'))))

        # Update sizes of the tables
        current_numbers_us.update_layout(height=250)
        increase_numbers_us.update_layout(height=250)

        return fig_us, current_numbers_us, increase_numbers_us


app.run_server(debug=True, use_reloader=False)
