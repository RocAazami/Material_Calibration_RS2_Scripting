from scipy.optimize import differential_evolution, minimize
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from optimization.pso_optimizer import ParticleSwarmOptimizer
from core.worker_pool_manager import WorkerPoolManager

class MaterialCalibrationOptimizer:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self._load_optimization_config()
        self.bounds = self._prepare_bounds()
        
        # State tracking
        self.is_running = False
        self.is_paused = False
        self.current_iteration = 0
        self.best_fitness_history = []
        self.best_position = None
        self.best_fitness = float('inf')
        self.best_params = {}
        self.stagnation_counter = 0
        
        # Create PSO optimizer if needed
        if self.method == 'pso':
            self.pso_optimizer = ParticleSwarmOptimizer(calibrator, calibrator.config)
        else:
            self.pso_optimizer = None
            

    def _load_optimization_config(self):
        """Load optimization method and parameters from config"""
        opt_config = self.calibrator.config.get('optimization', {})
        self.method = opt_config.get('method', 'original')  # default to original

    def optimize(self):
        """Run optimization using the configured method"""
        self.is_running = True
        self.is_paused = False
        
        if self.method == 'pso':
            # Use dedicated PSO optimizer
            if self.pso_optimizer:
                result = self.pso_optimizer.optimize()
                # Copy PSO state to this optimizer for dashboard compatibility
                self.best_fitness_history = self.pso_optimizer.best_fitness_history
                self.best_position = self.pso_optimizer.best_position
                self.best_fitness = self.pso_optimizer.best_fitness
                self.best_params = self.pso_optimizer.best_params
                self.current_iteration = self.pso_optimizer.current_iteration
                self.is_running = self.pso_optimizer.is_running
                self.is_paused = self.pso_optimizer.is_paused
                return result
            else:
                print("PSO optimizer not initialized, falling back to original method")
                return self.optimize_original()
        else:
            return self.optimize_original()

    def optimize_original(self):
        """Hybrid optimization (DE + L-BFGS-B) for simple parameters"""
        print("\nStarting original optimization method (DE + L-BFGS-B)...")
        
        de_result = differential_evolution(
            self._objective_function,
            bounds=self.bounds,
            strategy='best1bin',
            maxiter=self.calibrator.config['optimization'].get('max_iterations', 20),
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
        """Bounds for original method (simple parameters)"""
        bounds = []
        for param_config in self.calibrator.config['parameters'].values():
            if isinstance(param_config, dict):
                bounds.append(tuple(param_config.get('base', [0, 1])))
            else:
                bounds.append(tuple(param_config))
        return bounds

    def _objective_function(self, x: np.ndarray) -> float:
        """Objective function for original method"""
        try:
            return self.calibrator.evaluate_parameters(self._unpack_parameters(x))
        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            return float('inf')

    def _unpack_parameters(self, x: np.ndarray) -> Dict:
        """Simplified: Each parameter is a single value (no slopes)"""
        params = {}
        for i, param_name in enumerate(self.calibrator.config['parameters']):
            params[param_name] = x[i]
        return params
    
    # Methods for dashboard integration - delegate to PSO optimizer if available
    def pause(self):
        """Pause the optimization process"""
        if self.pso_optimizer and self.method == 'pso':
            self.pso_optimizer.pause()
        self.is_paused = True
        print("Optimization paused")
    
    def resume(self):
        """Resume the optimization process"""
        if self.pso_optimizer and self.method == 'pso':
            self.pso_optimizer.resume()
        self.is_paused = False
        print("Optimization resumed")
    
    def stop(self):
        """Stop the optimization process"""
        if self.pso_optimizer and self.method == 'pso':
            self.pso_optimizer.stop()
        self.is_running = False
        print("Optimization stopped")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current optimization progress for dashboard"""
        if self.pso_optimizer and self.method == 'pso':
            return self.pso_optimizer.get_progress()
        
        return {
            'iteration': self.current_iteration,
            'max_iterations': self.calibrator.config['optimization'].get('max_iterations', 100),
            'best_fitness': self.best_fitness,
            'best_params': self.best_params,
            'history': self.best_fitness_history,
            'is_running': self.is_running,
            'is_paused': self.is_paused
        }
