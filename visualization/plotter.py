# visualization/plotter.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import numpy as np

class CalibrationDashboard:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Create the dashboard layout with proper alignment"""
        self.app.layout = html.Div([
            # Hidden storage for the right graph mode and update trigger
            dcc.Store(id='right-graph-mode-store', data='pore_pressure'),  # Changed default to pore_pressure
            dcc.Store(id='update-trigger', data=0),
            
            html.H1("RS2 Material Calibration Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # First Row: Three plots in a flex container
            html.Div([
                # Stress-Strain
                html.Div([
                    dcc.Graph(id='stress-strain-plot',
                             style={'height': '400px'})
                ], style={'flex': '1', 'minWidth': '0', 'padding': '10px'}),
                
                # Stress Path
                html.Div([
                    dcc.Graph(id='stress-path-plot',
                             style={'height': '400px'})
                ], style={'flex': '1', 'minWidth': '0', 'padding': '10px'}),
                
                # Right graph with toggle
                html.Div([
                    html.Div([
                        html.Button("Volumetric Response", 
                                  id='btn-volumetric',
                                  n_clicks=0,
                                  style={'marginRight': '10px'}),  # Remove initial highlight
                        html.Button("Pore Pressure Response", 
                                  id='btn-pore-pressure',
                                  n_clicks=0,
                                  style={'marginRight': '10px', 'backgroundColor': 'lightblue'})  # Add initial highlight
                    ], style={'marginBottom': '10px', 'textAlign': 'center'}),
                    
                    dcc.Graph(id='right-graph',
                             style={'height': '370px'})
                ], style={'flex': '1', 'minWidth': '0', 'padding': '10px'})
            ], style={'display': 'flex', 'marginBottom': '20px', 'overflow': 'hidden'}),
            
            # Second Row: Error plot and parameters
            html.Div([
                # Error Plot
                html.Div([
                    html.H3("Optimization Progress", style={'textAlign': 'center'}),
                    dcc.Graph(id='error-plot',
                             style={'height': '300px'})
                ], style={'flex': '2', 'minWidth': '0', 'padding': '10px'}),
                
                # Parameters
                html.Div([
                    html.H3("Current Parameters", style={'textAlign': 'center'}),
                    html.Div(id='parameter-display',
                            style={'marginBottom': '20px'}),
                    html.H3("Parameter Ranges", style={'textAlign': 'center'}),
                    html.Div(id='parameter-ranges')
                ], style={'flex': '1', 'minWidth': '0', 
                         'padding': '10px', 'marginLeft': '10px'})
            ], style={'display': 'flex'}),
            
            dcc.Interval(id='progress-update', interval=1000, n_intervals=0)
        ], style={'padding': '20px'})

    def _setup_callbacks(self):
        """Setup callbacks with completely separated triggers"""
        
        # 1. Button style updates and mode storage
        @self.app.callback(
            [Output('btn-volumetric', 'style'),
             Output('btn-pore-pressure', 'style'),
             Output('right-graph-mode-store', 'data'),
             Output('update-trigger', 'data')],
            [Input('btn-volumetric', 'n_clicks'),
             Input('btn-pore-pressure', 'n_clicks')],
            [State('update-trigger', 'data')],
            prevent_initial_call=True
        )
        def update_button_styles(vol_clicks, pp_clicks, current_trigger):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'btn-volumetric':
                return (
                    {'marginRight': '10px', 'backgroundColor': 'lightblue'}, 
                    {},
                    'volumetric',
                    current_trigger + 1
                )
            else:
                return (
                    {'marginRight': '10px'}, 
                    {'backgroundColor': 'lightblue'},
                    'pore_pressure',
                    current_trigger + 1
                )

        # [Rest of the callbacks remain exactly the same...]
        # 2. Main plot updates (triggered by interval or button changes)
        @self.app.callback(
            [Output('stress-strain-plot', 'figure'),
             Output('stress-path-plot', 'figure'),
             Output('right-graph', 'figure'),
             Output('error-plot', 'figure')],
            [Input('progress-update', 'n_intervals'),
             Input('update-trigger', 'data')],
            [State('right-graph-mode-store', 'data')],
            prevent_initial_call=True
        )
        def update_plots(n_intervals, trigger, graph_mode):
            if not hasattr(self.calibrator, 'history') or not self.calibrator.history:
                raise PreventUpdate
                
            current = self.calibrator.history[-1]
            test_name = list(current['results'].keys())[0]
            exp_data = self.calibrator.data_loader.tests[test_name]
            num_data = current['results'][test_name]
            
            if graph_mode == 'volumetric':
                right_fig = self._create_volumetric_figure(exp_data, num_data)
            else:
                right_fig = self._create_pore_pressure_figure(exp_data, num_data)
            
            return (
                self._create_stress_strain_figure(exp_data, num_data),
                self._create_stress_path_figure(exp_data, num_data),
                right_fig,
                self._create_error_figure()
            )

        # 3. Parameter updates (triggered by interval only)
        @self.app.callback(
            [Output('parameter-display', 'children'),
             Output('parameter-ranges', 'children')],
            [Input('progress-update', 'n_intervals')],
            prevent_initial_call=True
        )
        def update_parameters(n_intervals):
            if not hasattr(self.calibrator, 'history') or not self.calibrator.history:
                raise PreventUpdate
                
            current = self.calibrator.history[-1]
            return (
                self._create_parameter_display(current['params']),
                self._create_parameter_ranges()
            )

    # [All figure creation methods remain exactly the same...]
    def _create_stress_strain_figure(self, exp_data, num_data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=exp_data['StrainYY'],
            y=exp_data['StressYY'],
            mode='markers',
            name='Experimental',
            marker=dict(color='red', size=8)
        ))
        fig.add_trace(go.Scatter(
            x=num_data['StrainYY'],
            y=num_data['StressYY'],
            mode='lines',
            name='Numerical',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title='Stress-Strain Response',
            xaxis_title='Axial Strain',
            yaxis_title='Deviatoric Stress (kPa)',
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        return fig

    def _create_stress_path_figure(self, exp_data, num_data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=exp_data['p'],
            y=exp_data['q'],
            mode='markers',
            name='Experimental',
            marker=dict(color='red', size=8)
        ))
        fig.add_trace(go.Scatter(
            x=num_data['p'],
            y=num_data['q'],
            mode='lines',
            name='Numerical',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title='Stress Path (p-q space)',
            xaxis_title='Mean Stress (kPa)',
            yaxis_title='Deviatoric Stress (kPa)',
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        return fig

    def _create_volumetric_figure(self, exp_data, num_data):
        fig = go.Figure()
        if 'Volumetric_Strain' in exp_data and 'Volumetric_Strain' in num_data:
            fig.add_trace(go.Scatter(
                x=exp_data['StrainYY'],
                y=exp_data['Volumetric_Strain'],
                mode='markers',
                name='Experimental',
                marker=dict(color='blue', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=num_data['StrainYY'],
                y=num_data['Volumetric_Strain'],
                mode='lines',
                name='Numerical',
                line=dict(color='red', width=2)
            ))
        fig.update_layout(
            title='Volumetric Response',
            xaxis_title='Axial Strain',
            yaxis_title='Volumetric Strain',
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        return fig

    def _create_pore_pressure_figure(self, exp_data, num_data):
        fig = go.Figure()
        
        # Get initial cell pressure from the test configuration
        initial_cell_pressure = exp_data.get('cell_pressure')
        if initial_cell_pressure is None or 'p' not in exp_data or 'StrainYY' not in exp_data:
            return fig  # Return empty figure if data is missing
        
        # Ensure initial_cell_pressure is a numpy array of the same length as p values
        initial_pressures = np.full_like(exp_data['p'], initial_cell_pressure)
        
        # Calculate pore pressure response: initial cell pressure - current p
        fig.add_trace(go.Scatter(
            x=exp_data['StrainYY'],
            y=initial_pressures - exp_data['p'],
            mode='markers',
            name='Experimental',
            marker=dict(color='red', size=8)
        ))
        
        if 'p' in num_data:
            # Create array of initial pressures matching numerical data length
            num_initial_pressures = np.full_like(num_data['p'], initial_cell_pressure)
            fig.add_trace(go.Scatter(
                x=num_data['StrainYY'],
                y=num_initial_pressures - num_data['p'],
                mode='lines',
                name='Numerical',
                line=dict(color='blue', width=2)
            ))
        
        fig.update_layout(
            title='Pore Pressure Response',
            xaxis_title='Axial Strain',
            yaxis_title='Pore Pressure (kPa)',
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
        )
        return fig

    def _create_error_figure(self):
        fig = go.Figure()
        if hasattr(self.calibrator, 'history') and self.calibrator.history:
            errors = [entry['error'] for entry in self.calibrator.history]
            fig.add_trace(go.Scatter(
                x=list(range(len(errors))),
                y=errors,
                mode='lines+markers',
                line=dict(color='red', width=2)
            ))
        fig.update_layout(
            xaxis_title='Iteration',
            yaxis_title='Error (RMSE)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    def _create_parameter_display(self, params):
        rows = []
        for name, value in params.items():
            if isinstance(value, dict):
                rows.append(html.Tr([
                    html.Td(html.B(name)),
                    html.Td(f"{value['base']:.4f}"),
                    html.Td(f"{value['slope']:.4f}" if 'slope' in value else "")
                ]))
            else:
                rows.append(html.Tr([
                    html.Td(html.B(name)),
                    html.Td(f"{value:.4f}"),
                    html.Td("")
                ]))
        return html.Table([html.Tbody(rows)], style={'width': '100%'})

    def _create_parameter_ranges(self):
        if not hasattr(self.calibrator, 'config'):
            return html.Div("No configuration loaded")
            
        ranges = self.calibrator.config.get('parameters', {})
        rows = []
        
        for param, bounds in ranges.items():
            if isinstance(bounds, dict):
                rows.append(html.Tr([
                    html.Td(html.B(param)),
                    html.Td(f"[{bounds['base'][0]:.2f}, {bounds['base'][1]:.2f}]"),
                    html.Td(f"[{bounds['slope'][0]:.2f}, {bounds['slope'][1]:.2f}]" 
                           if 'slope' in bounds else "")
                ]))
            else:
                rows.append(html.Tr([
                    html.Td(html.B(param)),
                    html.Td(f"[{bounds[0]:.2f}, {bounds[1]:.2f}]"),
                    html.Td("")
                ]))
        return html.Table([html.Tbody(rows)], style={'width': '100%'})

    def run(self):
        """Run the dashboard"""
        self.app.run(debug=True, port=8050, use_reloader=False)