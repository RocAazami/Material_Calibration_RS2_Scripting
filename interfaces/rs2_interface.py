from typing import Dict, List
import os
from enum import Enum
import clr
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
# Rest of your existing imports
from rs2.modeler.RS2Modeler import RS2Modeler
from rs2.interpreter.RS2Interpreter import RS2Interpreter
from rs2.interpreter.InterpreterEnums import *

def sanitize_path(path):
    """Convert path to string and replace backslashes"""
    return str(path).replace('\\', '/')

class _RS2Worker:
    """Internal worker class that handles a single RS2 model"""
    
    def __init__(self, config, worker_id):
        self.config = config
        self.worker_id = worker_id
        self.output_dir = os.path.abspath(config['output']['directory'])
        
        # Get port configuration from config or use defaults
        modeler_base_port = 60054
        interpreter_base_port = 60154
        
        if 'parallel_processing' in config:
            modeler_base_port = config['parallel_processing'].get('modeler_base_port', 60054)
            interpreter_base_port = config['parallel_processing'].get('interpreter_base_port', 60154)
            
        self.modeler_port = modeler_base_port + worker_id
        self.interpreter_port = interpreter_base_port + worker_id
        self.modeler = None
        self.interpreter = None
    
    def initialize(self):
        """Initialize RS2 applications for this worker"""
        try:
            print(f"Worker {self.worker_id}: Starting RS2 Modeler on port {self.modeler_port}...")
            RS2Modeler.startApplication(port=self.modeler_port)
            self.modeler = RS2Modeler(port=self.modeler_port)
            
            print(f"Worker {self.worker_id}: Starting RS2 Interpreter on port {self.interpreter_port}...")
            RS2Interpreter.startApplication(port=self.interpreter_port)
            self.interpreter = RS2Interpreter(port=self.interpreter_port)
            
            return True
        except Exception as e:
            print(f"Worker {self.worker_id}: Initialization failed: {str(e)}")
            self.close()
            return False
    
    def process_model(self, test_type, parameters, base_model_ref, cell_pressure, drainage):
        """Process a single model: create, compute, and extract results"""
        try:
            # 1. Create and compute the model
            model_path = self._create_and_compute_model(
                test_type, parameters, base_model_ref, cell_pressure, drainage
            )
            
            # 2. Extract results
            results = self._extract_results(test_type, parameters, model_path)
            
            return {
                'test_type': test_type,
                'model_path': model_path,
                'results': results,
                'success': True
            }
        except Exception as e:
            print(f"Worker {self.worker_id}: Error processing {test_type}: {str(e)}")
            return {
                'test_type': test_type,
                'success': False,
                'error': str(e)
            }
    
    def _create_and_compute_model(self, test_type, parameters, base_model_ref, cell_pressure, drainage):
        """Create and compute a single model"""
        print(f"Worker {self.worker_id}: Creating model for {test_type}")
        
        # Resolve base model path
        base_path = self._resolve_path(base_model_ref)
        
        # Open and configure model
        model = self.modeler.openFile(base_path)
        material = model.getMaterialPropertyByName("Sand")
        if material is None:
            raise ValueError(f"Material 'Sand' not found in {base_path}")
        
        # Set parameters
        material.Strength.NorSandStrength.setMTCCriticalFrictionRatio(float(parameters['M_tc']))
        material.Strength.NorSandStrength.setH0PlasticHardeningModulus(float(parameters['H_0']))
        material.Strength.NorSandStrength.setPsi0InitialStateParameter(float(parameters['psi_0']))
        
        # Save parameterized model
        param_str = "_".join(f"{k}={v:.3f}" for k, v in parameters.items())
        output_path = os.path.join(
            self.output_dir,
            sanitize_path(f"{test_type}_{param_str}.fez")
        )
        
        print(f"Worker {self.worker_id}: Saving to {output_path}")
        model.saveAs(output_path)
        
        print(f"Worker {self.worker_id}: Computing model {test_type}")
        model.compute()
        
        return output_path
    
    def _extract_results(self, test_type, parameters, model_path):
        """Extract results from a computed model"""
        print(f"Worker {self.worker_id}: Extracting results for {test_type}")
        
        if not os.path.exists(model_path):
            print(f"Worker {self.worker_id}: Model file not found: {model_path}")
            return {'StrainYY': [], 'StressYY': []}
        
        model_results = self.interpreter.openFile(model_path)
        model_results.AddMaterialQuery([[0.5, 0.5]])
        
        extracted_data = {'StrainYY': [], 'StressYY': [], 'p': [], 'q': [], 'Volumetric_Strain': []}
        
        result_types = [
            ('StrainYY', ExportResultType.SOLID_STRAIN_STRAIN_YY),
            ('StressYY', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_SIGMA_YY),
            ('p', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_MEAN_STRESS),
            ('q', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_VON_MISES_STRESS),
            ('Volumetric_Strain', ExportResultType.SOLID_STRAIN_VOLUMETRIC_STRAIN)
        ]
        
        for key, result_type in result_types:
            model_results.SetResultType(result_type)
            for stageNum in range(1, 300):
                try:
                    model_results.SetActiveStage(stageNum)
                    query_point = model_results.GetMaterialQueryResults()[0]
                    value = query_point.GetAllValues()[0].value
                    extracted_data[key].append(value)
                except Exception:
                    break
        
        model_results.close()
        return extracted_data
    
    def _resolve_path(self, path_ref: str) -> str:
        """Resolve paths without $output.directory placeholder"""
        if path_ref.startswith('$'):
            components = path_ref[1:].split('.')
            current = self.config
            for component in components:
                current = current.get(component)
                if current is None:
                    raise ValueError(f"Path reference {path_ref} not found in config")
            return os.path.abspath(current)
        return os.path.abspath(path_ref)
    
    def close(self):
        """Close RS2 applications"""
        if hasattr(self, 'modeler') and self.modeler is not None:
            try:
                self.modeler.closeProgram()
            except:
                pass
            self.modeler = None
            
        if hasattr(self, 'interpreter') and self.interpreter is not None:
            try:
                self.interpreter.closeProgram()
            except:
                pass
            self.interpreter = None


class RS2Interface:
    """Interface for RS2 modeling and result extraction with parallel processing"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.abspath(config['output']['directory'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For parallel processing - get from config
        self.max_workers = 3  # Default
        if 'parallel_processing' in config:
            self.max_workers = config['parallel_processing'].get('workers', 3)
            
        self.workers = []
        self.results = {}
        
        # For backward compatibility
        self.modeler = None
        self._initialize_single_modeler()
    
    def _initialize_single_modeler(self):
        """Initialize a single modeler for backward compatibility"""
        try:
            port = 60054
            print(f"Starting RS2 on port {port} (for backward compatibility)...")
            RS2Modeler.startApplication(port=port)
            self.modeler = RS2Modeler(port=port)
        except Exception as e:
            print(f"Warning: Failed to initialize backward compatibility modeler: {str(e)}")
    
    def _initialize_workers(self):
        """Initialize worker processes for parallel execution"""
        self.workers = []
        for i in range(self.max_workers):
            worker = _RS2Worker(self.config, i)
            if worker.initialize():
                self.workers.append(worker)
            else:
                print(f"Failed to initialize worker {i}")
        
        if not self.workers:
            raise RuntimeError("Failed to initialize any workers")
    
    def create_test_models(self, material: str, parameters: Dict, cell_pressure: float, drainage: str) -> Dict:
        """Create models for ALL test types with parallel processing"""
        print("🟢 Entered create_test_models with parallel processing")
        
        # Initialize workers if not already done
        if not self.workers:
            self._initialize_workers()
        
        # Prepare the list of models to process
        model_tasks = []
        for test in self.config['experimental_data']:
            model_tasks.append({
                'test_type': test['type'],
                'parameters': parameters,
                'base_model_ref': test['base_model'],
                'cell_pressure': test['cell_pressure'],
                'drainage': test['drainage']
            })
        
        # Process models in parallel
        outputs = {}
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            # Submit tasks
            future_to_task = {}
            for i, task in enumerate(model_tasks):
                worker_id = i % len(self.workers)
                worker = self.workers[worker_id]
                
                future = executor.submit(
                    worker.process_model,
                    task['test_type'],
                    task['parameters'],
                    task['base_model_ref'],
                    task['cell_pressure'],
                    task['drainage']
                )
                future_to_task[future] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result['success']:
                        outputs[result['test_type']] = result['model_path']
                        self.results[result['test_type']] = result['results']
                    else:
                        print(f"Failed to process {task['test_type']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"Exception processing {task['test_type']}: {str(e)}")
        
        return outputs
    
    def _create_single_model(self, test_type: str, parameters: Dict, 
                            base_model_ref: str, cell_pressure: float, 
                            drainage: str) -> str:
        """Create individual test model (backward compatibility method)"""
        print("🟢 Entered _create_single_model (legacy method)")
        
        # Resolve base model path
        base_path = self._resolve_path(base_model_ref)
        print(f"  Resolved base path: {base_path}")
        print(f"🔄 Opening base model: {base_path}")
        
        # Configure model
        model = self.modeler.openFile(base_path)
        print("  Model opened successfully")
        print("  Fetching material 'Sand'...")
        material = model.getMaterialPropertyByName("Sand")            
        if material is None:
            raise ValueError(f"Material 'Sand' not found in {base_path}")
            
        # Set parameters
        print(f"⚙️ Setting parameters: {parameters}")
        material.Strength.NorSandStrength.setMTCCriticalFrictionRatio(
            float(parameters['M_tc']))
        material.Strength.NorSandStrength.setH0PlasticHardeningModulus(
            float(parameters['H_0']))
        material.Strength.NorSandStrength.setPsi0InitialStateParameter(
            float(parameters['psi_0']))

        # Save parameterized model
        param_str = "_".join(f"{k}={v:.3f}" for k,v in parameters.items())
        output_path = os.path.join(
            self.output_dir,
            sanitize_path(f"{test_type}_{param_str}.fez")
        )
        print(f"Resolved output_path: {output_path}")

        print(f"💾 Saving to: {output_path}")
        model.saveAs(output_path)
        print("⚡ Computing model...")
        model.compute()
        return output_path
    
    def get_stress_strain(self, test_type: str, parameters: Dict[str, float]):
        """Extracts results from RS2 model file with parameter-based naming"""
        # Check if we already have results from parallel processing
        if test_type in self.results:
            return self.results[test_type]
        
        # If not, extract results using the traditional method
        port_Interpreter = 60005

        RS2Interpreter.startApplication(port=port_Interpreter)
        interpreter = RS2Interpreter(port=port_Interpreter)

        # Construct the filename based on how the model was saved
        param_str = "_".join(f"{k}={v:.3f}" for k, v in parameters.items())
        filename = f"{test_type}_{param_str}.fez"
        output_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(output_path):
            print(f"⚠️ Model file not found: {output_path}")
            return {'StrainYY': [], 'StressYY': []}  # Return empty to avoid crash    

        model_results = interpreter.openFile(output_path)       
        model_results.AddMaterialQuery([[0.5, 0.5]])        
        
        extracted_data = {'StrainYY': [], 'StressYY': [], 'p': [], 'q': [], 'Volumetric_Strain': []}

        result_types = [
            ('StrainYY', ExportResultType.SOLID_STRAIN_STRAIN_YY),
            ('StressYY', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_SIGMA_YY),
            ('p', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_MEAN_STRESS),
            ('q', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_VON_MISES_STRESS),
            ('Volumetric_Strain', ExportResultType.SOLID_STRAIN_VOLUMETRIC_STRAIN)
        ]

        for key, result_type in result_types:
            model_results.SetResultType(result_type)
            for stageNum in range(1, 300):
                try:
                    model_results.SetActiveStage(stageNum)
                    query_point = model_results.GetMaterialQueryResults()[0]
                    value = query_point.GetAllValues()[0].value
                    extracted_data[key].append(value)
                except Exception:
                    break

        model_results.close()
        interpreter.closeProgram()
        return extracted_data
    
    def _resolve_path(self, path_ref: str) -> str:
        """Resolve paths without $output.directory placeholder"""
        if path_ref.startswith('$'):
            components = path_ref[1:].split('.')
            current = self.config
            for component in components:
                current = current.get(component)
                if current is None:
                    raise ValueError(f"Path reference {path_ref} not found in config")
            return os.path.abspath(current)
        return os.path.abspath(path_ref)
    
    def close(self):
        """Clean up resources"""
        # Close workers
        for worker in self.workers:
            worker.close()
        self.workers = []
        
        # Close backward compatibility modeler
        if hasattr(self, 'modeler') and self.modeler is not None:
            try:
                self.modeler.closeProgram()
            except:
                pass
            self.modeler = None