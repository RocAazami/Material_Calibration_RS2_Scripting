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
        self.rs2 = RS2Interface()
        self.data_loader = ExperimentalDataLoader()
        self._initialize()
        self.history = []
        self.last_results = {}
        self.last_errors = {}

    def _initialize(self):
        """Load data and configure tests"""
        for test in self.config['experimental_data']:
            self.data_loader.load_test(
                test['filepath'],
                test['type'],
                test['cell_pressure']
            )

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
            total_error += error# * self.config['test_weights'][test_name]
        
        self.history.append({
            'params': params,
            'error': total_error,
            'results': self.last_results
        })
        
        return total_error

    def _adjust_for_pressure(self, params: Dict, pressure: float) -> Dict:
        """Apply pressure correction: P = P₀ + k*(σ₃ - σ₃_ref)"""
        adjusted = {}
        σ_ref = self.config['reference_pressure']
        
        for name, value in params.items():
            if isinstance(value, dict):
                adjusted[name] = value['base'] + value['slope']*(pressure - σ_ref)
            else:
                adjusted[name] = value
        return adjusted

    def _run_simulation(self, params: Dict, test_name: str) -> Dict:
        """Run one test simulation"""
        test_data = self.data_loader.tests[test_name]
        self.rs2.create_test_model(
            material=self.config['material_model'],
            parameters=params,
            cell_pressure=test_data['cell_pressure'],
            drainage='drained' if 'drained' in test_name else 'undrained'
        )        
        return self.rs2.get_stress_strain(
            material=self.config['material_model'],
            parameters=params            
        )


    def _calculate_error(self, exp: Dict, num: Dict) -> float:
        """
        Computes error between experimental and numerical StressYY-StrainYY curves
        by interpolating both onto a common strain axis.
        
        Args:
            exp: Experimental data {'StrainYY': [...], 'StressYY': [...], ...}
            num: Numerical results {'StrainYY': [...], 'StressYY': [...]}
        
        Returns:
            Combined error metric (RMSE of StressYY differences)
        """
        # Create common strain axis (union of experimental and numerical points)
        min_strain = min(np.min(exp['StrainYY']), np.min(num['StrainYY']))
        max_strain = min(np.max(exp['StrainYY']), np.max(num['StrainYY']))  # Don't extrapolate
        common_strain = np.union1d(exp['StrainYY'], num['StrainYY'])
        common_strain = common_strain[(common_strain >= min_strain) & (common_strain <= max_strain)]
        
        # Create interpolation functions
        f_exp = interp1d(exp['StrainYY'], exp['StressYY'], 
                        bounds_error=False, fill_value="extrapolate")
        f_num = interp1d(num['StrainYY'], num['StressYY'],
                        bounds_error=False, fill_value="extrapolate")
        
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