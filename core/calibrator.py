import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Any, Optional
import os
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
        self.current_params = {}
        self.evaluation_count = 0
        self.output_dir = os.path.abspath(self.config['output']['directory'])
        os.makedirs(self.output_dir, exist_ok=True)

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

    def update_model_parameters(self, params: Dict):
        """
        Formats parameters for RS2Interface and updates the model
        Handles both simple parameters and base/slope parameters
        """
        # Convert parameters to format expected by RS2Interface
        formatted_params = {}
        for param_name, value in params.items():
            if isinstance(value, dict):
                # Handle base/slope parameters
                formatted_params[param_name] = value['base']  # RS2Interface expects simple values
            else:
                # Handle simple parameters
                formatted_params[param_name] = value
        
        # Store current parameters for later use
        self.current_params = formatted_params

    def evaluate_parameters(self, params: Dict) -> float:
        """
        Evaluate parameters by running all configured tests and computing aggregate error
        Returns: Total error across all tests
        """
        self.evaluation_count += 1
        print(f"\nEvaluation #{self.evaluation_count} - Parameters: {params}")
        
        try:
            all_results = {}
            all_errors = {}
            total_error = 0.0
            total_weight = 0.0
            
            # Get visible tests from config
            try:
                visible_tests = self.config['output']['visualization']['visible_tests']
            except (KeyError, TypeError):
                # Fallback to all experimental data if not specified
                visible_tests = [test['type'] for test in self.config['experimental_data']]
            
            # Get weights for each test
            test_weights = {}
            for test in self.config['experimental_data']:
                test_name = test['type']
                test_weights[test_name] = test.get('weight', 1.0)
            
            # First update model parameters once
            self.update_model_parameters(params)
            
            # Create all test models in one batch
            model_paths = self.create_all_test_models(params)
            if not model_paths:
                print("❌ Failed to create test models")
                return float('inf')
            
            # Then run all tests
            for test_name in visible_tests:
                try:
                    # Get model path for this test
                    if test_name not in model_paths:
                        print(f"❌ Model path for test {test_name} not found")
                        all_errors[test_name] = 1e6
                        total_error += 1e6 * test_weights.get(test_name, 1.0)
                        total_weight += test_weights.get(test_name, 1.0)
                        continue
                    
                    # Extract results for this test
                    results = self.extract_test_results(params, test_name, model_paths[test_name])
                    all_results[test_name] = results
                    
                    # Calculate error for this test
                    exp_data = self.data_loader.tests[test_name]
                    test_error = self._calculate_single_test_error(exp_data, results)
                    all_errors[test_name] = test_error
                    
                    # Add weighted error to total
                    weight = test_weights.get(test_name, 1.0)
                    total_error += test_error * weight
                    total_weight += weight
                    
                    print(f"  Test {test_name}: Error = {test_error:.6f}, Weight = {weight}")
                    
                except Exception as e:
                    print(f"Error evaluating test {test_name}: {str(e)}")
                    # Instead of failing completely, assign a high error and continue
                    all_errors[test_name] = 1e6
                    total_error += 1e6 * test_weights.get(test_name, 1.0)
                    total_weight += test_weights.get(test_name, 1.0)

            # Normalize total error by total weight
            if total_weight > 0:
                total_error /= total_weight
            
            # Store results and return total error
            self.last_results = all_results
            self.last_errors = all_errors
            self.add_history_entry(params, all_results, total_error)
            
            print(f"  Total error: {total_error:.6f}")
            return total_error
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return float('inf')

    def create_all_test_models(self, params: Dict) -> Dict[str, str]:
        """
        Create all test models in one batch
        Returns: Dictionary mapping test names to model paths
        """
        try:
            # Convert parameters to simple format
            simple_params = {}
            for param_name, value in params.items():
                if isinstance(value, dict):
                    simple_params[param_name] = value['base']
                else:
                    simple_params[param_name] = value
            
            # Create models for all tests
            print(f"Creating models for all tests with parameters: {simple_params}")
            model_paths = self.rs2.create_test_models(
                material=self.config['material_model'],
                parameters=simple_params,
                cell_pressure=0,  # Not used directly (handled per test)
                drainage="drained"  # Not used directly (handled per test)
            )
            
            print(f"Created models: {model_paths}")
            return model_paths
            
        except Exception as e:
            print(f"❌ Error creating test models: {str(e)}")
            return {}

    def extract_test_results(self, params: Dict, test_name: str, model_path: str) -> Dict:
        """
        Extract results from a computed model
        Returns: Dictionary of results (same structure as experimental data)
        """
        try:
            # Convert parameters to simple format
            simple_params = {}
            for param_name, value in params.items():
                if isinstance(value, dict):
                    simple_params[param_name] = value['base']
                else:
                    simple_params[param_name] = value
            
            # Verify model file exists
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return {'StrainYY': [], 'StressYY': []}
            
            # Extract results using RS2Interface
            print(f"Extracting results for {test_name} from {model_path}")
            results = self.rs2.get_stress_strain(
                test_type=test_name,
                parameters=simple_params
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error extracting results for {test_name}: {str(e)}")
            return {'StrainYY': [], 'StressYY': []}

    def run_single_test(self, params: Dict, test_name: str) -> Dict:
        """
        Run a single test simulation with the given parameters
        Returns: Dictionary of results (same structure as experimental data)
        """
        try:
            # Find test configuration
            test_config = next(
                (t for t in self.config['experimental_data'] if t['type'] == test_name),
                None
            )
            
            if test_config is None:
                raise ValueError(f"Test configuration for {test_name} not found")
            
            # Convert parameters to simple format
            simple_params = {}
            for param_name, value in params.items():
                if isinstance(value, dict):
                    simple_params[param_name] = value['base']
                else:
                    simple_params[param_name] = value
            
            # Create parameter string for filename
            param_str = "_".join(f"{k}={v:.3f}" for k, v in simple_params.items())
            model_filename = f"{test_name}_{param_str}.fez"
            model_path = os.path.join(self.output_dir, model_filename)
            
            # Check if model already exists
            if os.path.exists(model_path):
                print(f"Model already exists: {model_path}")
            else:
                # Find base model path
                base_model_ref = test_config['base_model']
                base_model_path = self._resolve_path_reference(base_model_ref)
                
                # Create and compute model
                print(f"Creating model for {test_name} with parameters: {simple_params}")
                
                # Get test-specific parameters
                cell_pressure = test_config['cell_pressure']
                drainage = test_config.get('drainage', 'drained')
                
                # Create model using RS2Interface
                self.rs2.create_test_models(
                    material=self.config['material_model'],
                    parameters=simple_params,
                    cell_pressure=cell_pressure,
                    drainage=drainage
                )
            
            # Extract results
            return self.rs2.get_stress_strain(
                test_type=test_name,
                parameters=simple_params
            )
            
        except Exception as e:
            print(f"Error running test {test_name}: {str(e)}")
            return {'StrainYY': [], 'StressYY': []}  # Return empty to avoid crash

    def _resolve_path_reference(self, path_ref: str) -> str:
        """
        Resolve a path reference from the config
        Example: "$base_models.drained_100" -> "RS2_Models/NorSand - UTC - p0=100-Base Model.fez"
        """
        if not path_ref.startswith('$'):
            return path_ref
            
        # Remove $ and split by dot
        parts = path_ref[1:].split('.')
        
        # Navigate through config
        current = self.config
        for part in parts:
            if part not in current:
                raise ValueError(f"Path reference {path_ref} not found in config")
            current = current[part]
            
        return current

    def _calculate_single_test_error(self, exp_data: Dict, num_data: Dict) -> float:
        """
        Calculate error for a single test case
        """
        # Validate input data
        if not all(key in exp_data for key in ('StrainYY', 'StressYY')):
            print("❌ Experimental data missing required keys ('StrainYY' or 'StressYY')")
            return np.inf
            
        if not all(key in num_data for key in ('StrainYY', 'StressYY')):
            print("❌ Numerical results missing required keys ('StrainYY' or 'StressYY')")
            return np.inf
            
        if len(exp_data['StrainYY']) == 0 or len(exp_data['StressYY']) == 0:
            print("❌ Experimental data contains empty arrays")
            return np.inf
            
        if len(num_data['StrainYY']) == 0 or len(num_data['StressYY']) == 0:
            print("❌ Numerical results contain empty arrays")
            return np.inf
        
        # Create interpolation functions for numerical data
        try:
            # Use safer interpolation with bounds handling
            num_stress_interp = interp1d(
                num_data['StrainYY'], 
                num_data['StressYY'],
                bounds_error=False, 
                fill_value="extrapolate"
            )
            
            # Interpolate numerical results to experimental strain points
            num_stress_at_exp_strain = num_stress_interp(exp_data['StrainYY'])
            
            # Calculate RMSE for stress-strain
            squared_errors = (exp_data['StressYY'] - num_stress_at_exp_strain)**2
            stress_rmse = np.sqrt(np.mean(squared_errors))
            
            # Initialize total error with stress-strain error
            total_error = stress_rmse
            
            # Add volumetric strain error if available
            if ('Volumetric_Strain' in exp_data and len(exp_data['Volumetric_Strain']) > 0 and
                'Volumetric_Strain' in num_data and len(num_data['Volumetric_Strain']) > 0):
                
                vol_interp = interp1d(
                    num_data['StrainYY'], 
                    num_data['Volumetric_Strain'],
                    bounds_error=False, 
                    fill_value="extrapolate"
                )
                
                vol_at_exp_strain = vol_interp(exp_data['StrainYY'])
                vol_squared_errors = (exp_data['Volumetric_Strain'] - vol_at_exp_strain)**2
                vol_rmse = np.sqrt(np.mean(vol_squared_errors))
                
                # Add volumetric error with weight
                total_error += vol_rmse * 0.5
            
            # Add p-q path error if available
            if ('p' in exp_data and 'q' in exp_data and 
                'p' in num_data and 'q' in num_data):
                
                # Interpolate q vs p
                q_interp = interp1d(
                    num_data['p'], 
                    num_data['q'],
                    bounds_error=False, 
                    fill_value="extrapolate"
                )
                
                # Find common p range
                min_p = max(min(exp_data['p']), min(num_data['p']))
                max_p = min(max(exp_data['p']), max(num_data['p']))
                
                # Create common p points
                common_p = np.linspace(min_p, max_p, 50)
                
                # Get experimental q at common p
                exp_q_interp = interp1d(
                    exp_data['p'], 
                    exp_data['q'],
                    bounds_error=False, 
                    fill_value="extrapolate"
                )
                
                exp_q_at_common_p = exp_q_interp(common_p)
                num_q_at_common_p = q_interp(common_p)
                
                # Calculate RMSE for p-q path
                pq_squared_errors = (exp_q_at_common_p - num_q_at_common_p)**2
                pq_rmse = np.sqrt(np.mean(pq_squared_errors))
                
                # Add p-q path error with weight
                total_error += pq_rmse * 0.3
            
            return total_error
            
        except Exception as e:
            print(f"❌ Error calculation failed: {str(e)}")
            return np.inf

    def add_history_entry(self, params: Dict, results: Dict, error: float):
        """Add an entry to the optimization history"""
        self.history.append({
            'params': params.copy(),
            'results': results.copy(),
            'error': error
        })

    def get_best_parameters(self) -> Dict:
        """Get the best parameters found so far"""
        if not self.history:
            return {}
            
        best_entry = min(self.history, key=lambda entry: entry['error'])
        return best_entry['params']

    def get_best_error(self) -> float:
        """Get the lowest error found so far"""
        if not self.history:
            return float('inf')
            
        return min(entry['error'] for entry in self.history)

    def save_results(self, output_path: str):
        """Save optimization results to file"""
        import json
        
        if not self.history:
            print("No optimization history to save")
            return
            
        best_entry = min(self.history, key=lambda entry: entry['error'])
        
        # Create a simplified history for saving
        simplified_history = []
        for entry in self.history:
            simplified_entry = {
                'params': entry['params'],
                'error': entry['error']
            }
            simplified_history.append(simplified_entry)
        
        # Create output dictionary
        output = {
            'best_parameters': best_entry['params'],
            'best_error': best_entry['error'],
            'history': simplified_history,
            'config': self.config
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Results saved to {output_path}")