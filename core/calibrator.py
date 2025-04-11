"""
core/calibrator.py
Purpose: Main calibration workflow controller

Key Functions:

evaluate_parameters(): Runs simulations and compares with experimental data

_adjust_for_pressure(): Handles pressure-dependent parameter adjustments

RS2 Adjustments:

Modify _run_simulation() to customize RS2 model updates

Change error metrics in _calculate_error()

"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List
from interfaces.rs2_interface import RS2Interface
from interfaces.data_loader import ExperimentalDataLoader

class MaterialCalibrator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.rs2 = RS2Interface(self.config)   # Pass config to RS2Interface
        self.data_loader = ExperimentalDataLoader()
        self._initialize()
        self.history = []
        self.last_results = {}
        self.last_errors = {}

    def _load_config(self, config_path: str):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize(self):
        """Load data and configure tests"""
        for test in self.config['experimental_data']:
            self.data_loader.load_test(
                test['filepath'],
                test['type'],
                test['cell_pressure']
            )
    def _objective_function(self, x: np.ndarray) -> float:
        try:
            return self.calibrator.evaluate_parameters(self._unpack_parameters(x))
        except Exception as e:
            print(f"🚨 Error evaluating parameters: {str(e)}")
            return np.inf  # Penalize invalid parameters

    def evaluate_parameters(self, params: Dict) -> float:
        """Calculate total error for parameter set"""
        total_error = 0.0
        self.last_results = {}
        self.last_errors = {}
        
        for test_name, test_data in self.data_loader.tests.items():
            adj_params = self._adjust_for_pressure(params, test_data['cell_pressure'])
            numerical = self._run_simulation(adj_params, test_name)
            error = self._calculate_error(test_data, numerical)
            
            self.last_errors[test_name] = error
            self.last_results[test_name] = numerical
            test_config = next(t for t in self.config['experimental_data'] if t['type'] == test_name)
            total_error += error * test_config['weight']
            
        
        self.history.append({
            'params': params,
            'error': total_error,
            'results': self.last_results
        })
        
        return total_error

    def _adjust_for_pressure(self, params: Dict, cell_pressure: float) -> Dict:
        """Apply pressure correction: P = P₀ + k*(σ₃ - σ₃_ref)"""
        adjusted = {}    
        
        for name, value in params.items():
            if isinstance(value, dict):
                adjusted[name] = value['base'] + value['slope']*( cell_pressure)
            else:
                adjusted[name] = value
        return adjusted

    def _run_simulation(self, params: Dict, test_name: str) -> Dict:
        """Run one test simulation"""
        test_data = self.data_loader.tests[test_name]
        self.rs2.create_test_models(
            material=self.config['material_model'],
            parameters=params,
            cell_pressure=test_data['cell_pressure'],
            drainage='drained' if 'drained' in test_name else 'undrained'
        )        
        return self.rs2.get_stress_strain(
            test_type=test_name,  
            parameters=params
        )

    def _calculate_error(self, exp: Dict, num: Dict) -> float:
            # Validate input data
        if not all(key in exp for key in ('StrainYY', 'StressYY')):
            print("❌ Experimental data missing required keys ('StrainYY' or 'StressYY')")
            return np.inf
            
        if not all(key in num for key in ('StrainYY', 'StressYY')):
            print("❌ Numerical results missing required keys ('StrainYY' or 'StressYY')")
            return np.inf
            
        if len(exp['StrainYY']) == 0 or len(exp['StressYY']) == 0:
            print("❌ Experimental data contains empty arrays")
            return np.inf
            
        if len(num['StrainYY']) == 0 or len(num['StressYY']) == 0:
            print("❌ Numerical results contain empty arrays")
            return np.inf

 
        # Create common strain axis
        try:
            min_strain = max(np.min(exp['StrainYY']), np.min(num['StrainYY']))
            max_strain = min(np.max(exp['StrainYY']), np.max(num['StrainYY']))
            common_strain = np.linspace(min_strain, max_strain, 100)  # Fixed grid for safety
        except ValueError as e:
            print(f"❌ Strain axis creation failed: {str(e)}")
            return np.inf
        
        # Interpolation with fallback
        try:
            f_exp = interp1d(exp['StrainYY'], exp['StressYY'], 
                            bounds_error=False, fill_value="extrapolate")
            f_num = interp1d(num['StrainYY'], num['StressYY'],
                            bounds_error=False, fill_value="extrapolate")
        except Exception as e:
            print(f"❌ Interpolation failed: {str(e)}")
            return np.inf
            
        # Get stresses at common strain points
        exp_stress = f_exp(common_strain)
        num_stress = f_num(common_strain)
        
        # Calculate RMSE (optionally normalize by max experimental stress)
        #max_stress = np.max(exp['StressYY'])
        rmse = np.sqrt(np.mean((exp_stress - num_stress)**2)) #/ max_stress
        
        return rmse

    def _load_config(self, path: str) -> Dict:
        """Load YAML configuration"""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)