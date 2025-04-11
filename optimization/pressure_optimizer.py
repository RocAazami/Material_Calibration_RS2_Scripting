from scipy.optimize import differential_evolution, minimize
import numpy as np
from typing import Dict, List, Tuple

class MaterialCalibrationOptimizer:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.bounds = self._prepare_bounds()

    def optimize(self) -> Tuple[Dict, float]:
        """Hybrid optimization (DE + L-BFGS-B) for simple parameters"""
        de_result = differential_evolution(
            self._objective_function,
            bounds=self.bounds,
            strategy='best1bin',
            maxiter=self.calibrator.config['optimization']['max_iterations'],
            popsize=15,
            tol=0.01,
            disp=True            
        )
        
        refined = minimize(
            self._objective_function,
            x0=de_result.x,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 50}
        )
        
        return self._unpack_parameters(refined.x), refined.fun

    def _prepare_bounds(self) -> List[Tuple]:
        """Bounds now directly from config (no nested base/slope)"""
        return [tuple(param_range) for param_range in self.calibrator.config['parameters'].values()]

    def _objective_function(self, x: np.ndarray) -> float:
        return self.calibrator.evaluate_parameters(self._unpack_parameters(x))

    def _unpack_parameters(self, x: np.ndarray) -> Dict:
        """Simplified: Each parameter is a single value (no slopes)"""
        params = {}
        for i, param_name in enumerate(self.calibrator.config['parameters']):
            params[param_name] = x[i]
        return params