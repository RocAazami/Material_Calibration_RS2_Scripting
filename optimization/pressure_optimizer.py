from scipy.optimize import differential_evolution, minimize
"""
optimization/pressure_optimizer.py
Purpose: Handles parameter optimization

Key Methods:

optimize(): Runs hybrid DE + L-BFGS-B

_prepare_bounds(): Configures parameter search space

optimization/sensitivity.py
Purpose: Parameter sensitivity analysis

Key Feature:

python
def calculate_sensitivity(self, params):
    # Returns dict showing ∂(Error)/∂(Parameter)
    
"""


import numpy as np
from typing import Dict, List, Tuple

class PressureDependentOptimizer:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.bounds = self._prepare_bounds()

    def optimize(self) -> Tuple[Dict, float]:
        """Hybrid global/local optimization"""
        # Global phase
        de_result = differential_evolution(
            self._objective_function,
            bounds=self.bounds,
            strategy='best1bin',
            maxiter=50,
            popsize=15,
            tol=0.01,
            disp=True
        )
        
        # Local refinement
        refined = minimize(
            self._objective_function,
            x0=de_result.x,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 50}
        )
        
        return self._unpack_parameters(refined.x), refined.fun

    def _prepare_bounds(self) -> List[Tuple]:
        bounds = []
        for param, config in self.calibrator.config['parameters'].items():
            if isinstance(config, dict):
                bounds.append(config['base'])
                bounds.append(config['slope'])
            else:
                bounds.append(config)
        return bounds

    def _objective_function(self, x: np.ndarray) -> float:
        return self.calibrator.evaluate_parameters(self._unpack_parameters(x))

    def _unpack_parameters(self, x: np.ndarray) -> Dict:
        params = {}
        idx = 0
        for param in self.calibrator.config['parameters']:
            if isinstance(self.calibrator.config['parameters'][param], dict):
                params[param] = {'base': x[idx], 'slope': x[idx+1]}
                idx += 2
            else:
                params[param] = x[idx]
                idx += 1
        return params