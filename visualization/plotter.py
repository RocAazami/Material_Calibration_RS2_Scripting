# visualization/plotter.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import numpy as np

class CalibrationDashboard:
    def __init__(self, calibrator, optimizer=None):
        self.calibrator = calibrator
        self.optimizer = optimizer  # Now accepts the optimizer directly
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Create the dashboard layout with proper alignment and PSO controls"""
        self.app.layout = html.Div([
            # Hidden storage for the right graph mode and update trigger
            dcc.Store(id='right-graph-mode-store', data='pore_pressure'),
            dcc.Store(id='update-trigger', data=0),
            
            html.H1("RS2 Material Calibration Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Optimization Controls
            html.Div([
                html.Button("Pause/Resume", id='btn-pause-resume', n_clicks=0,
                          style={'marginRight': '10px'}),
                html.Button("Stop", id='btn-stop', n_clicks=0,
                          style={'marginRight': '10px'}),
                html.Div(id='optimization-status', style={'display': 'inline-block'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
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
                                  style={'marginRight': '10px'}),
                        html.Button("Pore Pressure Response", 
                                  id='btn-pore-pressure',
                                  n_clicks=0,
                                  style={'marginRight': '10px', 'backgroundColor': 'lightblue'})
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
        """Setup callbacks with PSO control integration"""
        
        # 1. Button style updates and mode storage
        @self.app.callback(
            [Output('btn-volumetric', 'style'),
             Output('btn-pore-pressure', 'style'),
             Output('right-graph-mode-store', 'data'),
             Output('update-trigger', 'data')],
            [Input('btn-volumetric', 'n_clicks'),
             Input('btn-pore-pressure', 'n_clicks')],
            [State('update-trigger', 'data')],
            prevent_initial_call=False
        )
        def update_button_styles(vol_clicks, pp_clicks, current_trigger):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'btn-volumetric':
                return (
                    {'marginRight': '10px', 'backgroundColor': 'lightblue'}, 
                    {'marginRight': '10px'},
                    'volumetric',
                    current_trigger + 1
                )
            else:
                return (
                    {'marginRight': '10px'}, 
                    {'marginRight': '10px', 'backgroundColor': 'lightblue'},
                    'pore_pressure',
                    current_trigger + 1
                )
        
        # 2. PSO Control callbacks
        @self.app.callback(
            [Output('optimization-status', 'children'),
             Output('btn-pause-resume', 'children')],
            [Input('btn-pause-resume', 'n_clicks'),
             Input('btn-stop', 'n_clicks'),
             Input('progress-update', 'n_intervals')],
            prevent_initial_call=False
        )
        def update_optimization_controls(pause_clicks, stop_clicks, n_intervals):
            ctx = dash.callback_context
            
            # Default status
            status = "Optimization status: Not running"
            button_text = "Pause/Resume"
            
            # Check if optimizer exists
            if not hasattr(self, 'optimizer') or self.optimizer is None:
                return status, button_text
            
            # Handle button clicks
            if ctx.triggered and ctx.triggered[0]['prop_id'] != 'progress-update.n_intervals':
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'btn-pause-resume' and hasattr(self.optimizer, 'is_paused'):
                    if self.optimizer.is_paused:
                        self.optimizer.resume()
                    else:
                        self.optimizer.pause()
                
                elif button_id == 'btn-stop' and hasattr(self.optimizer, 'is_running'):
                    self.optimizer.stop()
            
            # Update status based on optimizer state
            if hasattr(self.optimizer, 'is_running') and self.optimizer.is_running:
                if hasattr(self.optimizer, 'is_paused') and self.optimizer.is_paused:
                    status = "Optimization status: Paused"
                    button_text = "Resume"
                else:
                    progress = getattr(self.optimizer, 'current_iteration', 0)
                    max_iter = getattr(self.optimizer, 'pso_params', {}).max_iterations \
                               if hasattr(self.optimizer, 'pso_params') else 100
                    status = f"Optimization status: Running (Iteration {progress}/{max_iter})"
                    button_text = "Pause"
            else:
                status = "Optimization status: Stopped"
                button_text = "Pause/Resume"
            
            return status, button_text

        # 3. Main plot updates (triggered by interval or button changes)
        @self.app.callback(
            [Output('stress-strain-plot', 'figure'),
             Output('stress-path-plot', 'figure'),
             Output('right-graph', 'figure'),
             Output('error-plot', 'figure')],
            [Input('progress-update', 'n_intervals'),
             Input('update-trigger', 'data')],
            [State('right-graph-mode-store', 'data')],
            prevent_initial_call=False
        )
        def update_plots(n_intervals, trigger, graph_mode):
            # 1. Check calibrator state
            if not hasattr(self.calibrator, 'history'):
                return [self._create_empty_figure("Calibrator not initialized")] * 3 + [self._create_error_figure()]
                
            if not self.calibrator.history:
                return [self._create_empty_figure("No calibration runs yet")] * 3 + [self._create_error_figure()]

            current = self.calibrator.history[-1]

            # 2. Get visible tests from config
            try:
                visible_tests = self.calibrator.config['output']['visualization']['visible_tests']
            except (KeyError, TypeError) as e:
                visible_tests = list(current['results'].keys())[:1]  # Fallback to first test

            # 3. Collect data for all visible tests
            test_data = []
            for test_name in visible_tests:
                try:
                    exp_data = self.calibrator.data_loader.tests[test_name]
                    num_data = current['results'][test_name]
                    test_data.append({
                        'exp': exp_data,
                        'num': num_data,
                        'name': test_name,
                        'cell_pressure': exp_data.get('cell_pressure', 'N/A')
                    })
                except KeyError as e:
                    continue
                    
            if not test_data:
                error_msg = "No valid test data available"
                return [self._create_empty_figure(error_msg)] * 3 + [self._create_error_figure()]

            # 4. Create figures with all tests
            try:
                figures = [
                    self._create_stress_strain_figure(test_data),
                    self._create_stress_path_figure(test_data),
                    self._create_volumetric_figure(test_data) if graph_mode == 'volumetric' 
                        else self._create_pore_pressure_figure(test_data),
                    self._create_error_figure()
                ]
                
                return figures
                
            except Exception as e:
                error_msg = f"Figure generation failed: {str(e)}"
                print(f"ERROR: {error_msg}")
                return [self._create_empty_figure(error_msg)] * 3 + [self._create_error_figure()]
    
        # 4. Parameter updates (triggered by interval only)
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

    # [All figure creation methods remain the same as in your original code]
    def _create_stress_strain_figure(self, test_data):
        fig = go.Figure()
        
        for data in test_data:
            fig.add_trace(go.Scatter(
                x=data['exp']['StrainYY'],
                y=data['exp']['StressYY'],
                mode='markers',
                name=f"Exp {data['name']} ({data['cell_pressure']}kPa)",
                marker=dict(color='red', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=data['num']['StrainYY'],
                y=data['num']['StressYY'],
                mode='lines',
                name=f"Num {data['name']} ({data['cell_pressure']}kPa)",
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

    def _create_stress_path_figure(self, test_data):
        fig = go.Figure()
        
        for data in test_data:
            fig.add_trace(go.Scatter(
                x=data['exp']['p'],
                y=data['exp']['q'],
                mode='markers',
                name=f"Exp {data['name']} ({data['cell_pressure']}kPa)",
                marker=dict(color='red', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=data['num']['p'],
                y=data['num']['q'],
                mode='lines',
                name=f"Num {data['name']} ({data['cell_pressure']}kPa)",
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

    def _create_volumetric_figure(self, test_data):
        fig = go.Figure()
        
        for data in test_data:
            if 'Volumetric_Strain' in data['exp'] and 'Volumetric_Strain' in data['num']:
                fig.add_trace(go.Scatter(
                    x=data['exp']['StrainYY'],
                    y=data['exp']['Volumetric_Strain'],
                    mode='markers',
                    name=f"Exp {data['name']} ({data['cell_pressure']}kPa)",
                    marker=dict(color='blue', size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=data['num']['StrainYY'],
                    y=data['num']['Volumetric_Strain'],
                    mode='lines',
                    name=f"Num {data['name']} ({data['cell_pressure']}kPa)",
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

    def _create_pore_pressure_figure(self, test_data):
        fig = go.Figure()
        
        for data in test_data:
            try:
                exp = data['exp']
                num = data['num']
                cp = data['cell_pressure']
                
                # Validate required fields and types
                if ('cell_pressure' not in exp or 
                    'p' not in exp or 
                    'StrainYY' not in exp or
                    not isinstance(exp['p'], (np.ndarray, list))):
                    continue
                    
                # Convert to numpy arrays for safe math operations
                initial_p = float(exp['cell_pressure'])
                exp_p = np.array(exp['p'])
                exp_strain = np.array(exp['StrainYY'])
                
                # Calculate pore pressure response
                exp_pore = initial_p - exp_p
                
                fig.add_trace(go.Scatter(
                    x=exp_strain,
                    y=exp_pore,
                    mode='markers',
                    name=f"Exp {data['name']} ({cp}kPa)",
                    marker=dict(color='red', size=8)
                ))
                
                if 'p' in num and isinstance(num['p'], (np.ndarray, list)):
                    num_p = np.array(num['p'])
                    num_strain = np.array(num['StrainYY'])
                    num_pore = initial_p - num_p
                    
                    fig.add_trace(go.Scatter(
                        x=num_strain,
                        y=num_pore,
                        mode='lines',
                        name=f"Num {data['name']} ({cp}kPa)",
                        line=dict(color='blue', width=2)
                    ))
                    
            except Exception as e:
                print(f"ERROR: Failed to process {data.get('name', 'unknown')} - {str(e)}")
                continue
                
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
        
        # First plot calibrator history if available
        if hasattr(self.calibrator, 'history') and self.calibrator.history:
            errors = [entry['error'] for entry in self.calibrator.history]
            fig.add_trace(go.Scatter(
                x=list(range(len(errors))),
                y=errors,
                mode='lines+markers',
                name='Calibration Error',
                line=dict(color='red', width=2)
            ))
        
        # Then add PSO history if available
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'best_fitness_history') and self.optimizer.best_fitness_history:
            fig.add_trace(go.Scatter(
                x=list(range(len(self.optimizer.best_fitness_history))),
                y=self.optimizer.best_fitness_history,
                mode='lines+markers',
                name='PSO Best Fitness',
                line=dict(color='blue', width=2)
            ))
            
        fig.update_layout(
            title='Optimization Progress',
            xaxis_title='Iteration',
            yaxis_title='Error (RMSE)',
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True
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
        
    def _create_empty_figure(self, message="No data available"):
        """Create a figure with an error message"""
        fig = go.Figure()
        fig.update_layout(
            title=message,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16}
            }]
        )
        return fig

    def run(self):
        """Run the dashboard"""
        self.app.run(debug=True, port=8050, use_reloader=False)