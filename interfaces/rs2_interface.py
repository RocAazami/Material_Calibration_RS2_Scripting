# interfaces/rs2_interface.py
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from typing import Dict, List, Tuple, Optional

# Import RS2 libraries (assuming these are available in your environment)
try:
    import clr
    from rs2.modeler.RS2Modeler import RS2Modeler
    from rs2.interpreter.RS2Interpreter import RS2Interpreter
    from rs2.interpreter.InterpreterEnums import *
except ImportError as e:
    print(f"Warning: RS2 libraries not imported: {str(e)}")

def sanitize_path(path):
    """Convert path to string and replace backslashes with forward slashes for consistency"""
    return str(path).replace('\\', '/')

def is_port_available(port: int) -> bool:
    """Check if a port is available by attempting to bind to it"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except socket.error:
            return False

def find_available_port(start_port: int, end_port: int) -> Optional[int]:
    """Find an available port in the given range"""
    for port in range(start_port, end_port + 1):
        if is_port_available(port):
            return port
    return None

class RS2Worker:
    """Worker class that handles a single RS2 model instance with dedicated ports"""
    
    def __init__(self, config, worker_id):
        """
        Initialize worker with configuration and unique ID
        Args:
            config: Full configuration dictionary
            worker_id: Unique integer identifier for this worker
        """
        self.config = config
        self.worker_id = worker_id
        self.output_dir = os.path.abspath(config['output']['directory'])
        
        # Get port configuration from config or use defaults
        port_range = config['parallel_processing'].get('port_range', [60000, 61000])
        
        # Find available ports dynamically
        self.modeler_port = None
        self.interpreter_port = None
        self.modeler = None
        self.interpreter = None
        
        # Track usage count for reuse
        self.use_count = 0
        self.max_reuse_count = config['parallel_processing'].get('max_reuse_count', 10)
        self.in_use = False
        self.lock = threading.Lock()
    
    def initialize(self) -> bool:
        """
        Initialize RS2 applications for this worker with dedicated ports
        Returns: True if initialization succeeded, False otherwise
        """
        try:
            # Find available ports
            port_range = self.config['parallel_processing'].get('port_range', [60000, 61000])
            start_port, end_port = port_range
            
            # Try to find available modeler port
            self.modeler_port = find_available_port(start_port, end_port)
            if self.modeler_port is None:
                print(f"Worker {self.worker_id}: No available port found for modeler in range {start_port}-{end_port}")
                return False
                
            # Try to find available interpreter port (different from modeler port)
            self.interpreter_port = find_available_port(self.modeler_port + 1, end_port)
            if self.interpreter_port is None:
                print(f"Worker {self.worker_id}: No available port found for interpreter in range {self.modeler_port+1}-{end_port}")
                return False
            
            # Start RS2 applications with the available ports
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
    
    def acquire(self) -> bool:
        """
        Acquire this worker for use
        Returns: True if worker was acquired, False if worker is already in use
        """
        with self.lock:
            if self.in_use:
                return False
            
            self.in_use = True
            self.use_count += 1
            
            # Check if worker needs to be restarted
            if self.use_count > self.max_reuse_count:
                print(f"Worker {self.worker_id}: Restarting after {self.use_count} uses")
                self.close()
                if not self.initialize():
                    print(f"Worker {self.worker_id}: Failed to restart")
                    return False
                self.use_count = 1
            
            return True
    
    def release(self):
        """Release this worker"""
        with self.lock:
            self.in_use = False
    
    def process_model(self, test_type, parameters, base_model_ref, cell_pressure, drainage):
        """
        Process a single model: create, compute, and extract results
        Args:
            test_type: Name/type of the test (e.g., 'drained_100')
            parameters: Dictionary of material parameters
            base_model_ref: Reference to base model path
            cell_pressure: Cell pressure for this test
            drainage: Drainage condition ('drained'/'undrained')
        Returns: Dictionary with results and status
        """
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
        """
        Create and compute a single model with given parameters
        Args:
            test_type: Name of the test case
            parameters: Dictionary of material parameters
            base_model_ref: Path to base model file
            cell_pressure: Confining pressure for test
            drainage: Drainage condition
        Returns: Path to saved model file
        """
        print(f"Worker {self.worker_id}: Creating model for {test_type}")
        
        # Resolve base model path from config references
        base_path = self._resolve_path(base_model_ref)
        
        # Verify base model exists
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base model not found: {base_path}")
        
        # Open and configure model
        model = self.modeler.openFile(base_path)
        material = model.getMaterialPropertyByName("Sand")
        if material is None:
            raise ValueError(f"Material 'Sand' not found in {base_path}")
        
        # Set NorSand parameters from the parameters dictionary
        material.Strength.NorSandStrength.setMTCCriticalFrictionRatio(float(parameters['M_tc']))
        material.Strength.NorSandStrength.setH0PlasticHardeningModulus(float(parameters['H_0']))
        material.Strength.NorSandStrength.setPsi0InitialStateParameter(float(parameters['psi_0']))
        
        # Save parameterized model with unique name based on parameters
        param_str = "_".join(f"{k}={v:.3f}" for k, v in parameters.items())
        output_path = os.path.join(
            self.output_dir,
            sanitize_path(f"{test_type}_{param_str}.fez")
        )
        
        print(f"Worker {self.worker_id}: Saving to {output_path}")
        model.saveAs(output_path)
        
        print(f"Worker {self.worker_id}: Computing model {test_type}")
        model.compute()
        
        # Verify the model was computed successfully
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Computed model file not found: {output_path}")
        
        return output_path
    
    def _extract_results(self, test_type, parameters, model_path):
        """
        Extract stress-strain results from computed model
        Args:
            test_type: Name of the test case
            parameters: Dictionary of parameters used
            model_path: Path to computed model file
        Returns: Dictionary of extracted results
        """
        print(f"Worker {self.worker_id}: Extracting results for {test_type}")
        
        if not os.path.exists(model_path):
            print(f"Worker {self.worker_id}: Model file not found: {model_path}")
            return {'StrainYY': [], 'StressYY': []}
        
        # Wait a moment to ensure file is fully written
        time.sleep(0.5)
        
        # Try multiple times to open the file (in case of file access issues)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                model_results = self.interpreter.openFile(model_path)
                break
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Worker {self.worker_id}: Attempt {attempt+1} failed to open {model_path}: {str(e)}")
                    time.sleep(1)  # Wait before retrying
                else:
                    print(f"Worker {self.worker_id}: Failed to open {model_path} after {max_attempts} attempts")
                    return {'StrainYY': [], 'StressYY': []}
        
        model_results.AddMaterialQuery([[0.5, 0.5]])  # Query at center of sample
        
        # Initialize dictionary to store all result types
        extracted_data = {
            'StrainYY': [], 
            'StressYY': [], 
            'p': [], 
            'q': [], 
            'Volumetric_Strain': []
        }
        
        # Map of our names to RS2 result types
        result_types = [
            ('StrainYY', ExportResultType.SOLID_STRAIN_STRAIN_YY),
            ('StressYY', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_SIGMA_YY),
            ('p', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_MEAN_STRESS),
            ('q', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_VON_MISES_STRESS),
            ('Volumetric_Strain', ExportResultType.SOLID_STRAIN_VOLUMETRIC_STRAIN)
        ]
        
        # Extract each result type through all stages
        for key, result_type in result_types:
            model_results.SetResultType(result_type)
            for stageNum in range(1, 300):  # Scan through stages until we fail
                try:
                    model_results.SetActiveStage(stageNum)
                    query_point = model_results.GetMaterialQueryResults()[0]
                    value = query_point.GetAllValues()[0].value
                    extracted_data[key].append(value)
                except Exception:
                    break  # No more stages
        
        model_results.close()
        return extracted_data
    
    def _resolve_path(self, path_ref: str) -> str:
        """
        Resolve paths that may contain config references (starting with $)
        Args:
            path_ref: Path string that may contain config references
        Returns: Absolute resolved path
        """
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
        """Cleanly shutdown RS2 applications for this worker"""
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
    """Main interface for RS2 modeling and result extraction with parallel processing support"""
    
    def __init__(self, config):
        """
        Initialize the RS2 interface with configuration
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.output_dir = os.path.abspath(config['output']['directory'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parallel processing configuration
        self.max_workers = config['parallel_processing'].get('workers', 1)
        self.workers = []
        self.results = {}
        
        # Worker pool management
        self.worker_lock = threading.Lock()
        
        # Initialize workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize all worker processes for parallel execution"""
        self.workers = []
        
        # Try to initialize the requested number of workers
        for i in range(self.max_workers):
            worker = RS2Worker(self.config, i)
            if worker.initialize():
                self.workers.append(worker)
            else:
                print(f"Failed to initialize worker {i}")
        
        # If no workers were initialized, try one more time with random ports
        if not self.workers and self.max_workers > 0:
            print("Retrying worker initialization with random ports...")
            worker = RS2Worker(self.config, 0)
            if worker.initialize():
                self.workers.append(worker)
        
        print(f"Initialized {len(self.workers)} workers out of {self.max_workers} requested")
    
    def get_available_worker(self):
        """
        Get an available worker from the pool
        Returns: Worker instance or None if no workers are available
        """
        with self.worker_lock:
            for worker in self.workers:
                if worker.acquire():
                    return worker
            return None
    
    def release_worker(self, worker):
        """Release a worker back to the pool"""
        worker.release()
    
    def create_test_models(self, material: str, parameters: Dict, cell_pressure: float, drainage: str) -> Dict:
        """
        Create models for ALL test types using parallel processing
        Args:
            material: Material model name
            parameters: Dictionary of material parameters
            cell_pressure: Not used directly (handled per test)
            drainage: Not used directly (handled per test)
        Returns: Dictionary of created model paths by test type
        """
        print("Starting parallel model creation...")
        
        # Initialize workers if not already done
        if not self.workers:
            self._initialize_workers()
        
        # If still no workers, try to create a single worker on demand
        if not self.workers:
            print("Creating worker on demand...")
            worker = RS2Worker(self.config, 0)
            if worker.initialize():
                self.workers.append(worker)
            else:
                raise RuntimeError("Failed to initialize any workers")
        
        # Prepare task list from experimental data config
        model_tasks = []
        for test in self.config['experimental_data']:
            model_tasks.append({
                'test_type': test['type'],
                'parameters': parameters,
                'base_model_ref': test['base_model'],
                'cell_pressure': test['cell_pressure'],
                'drainage': test['drainage']
            })
        
        # Process models in parallel using ThreadPoolExecutor
        outputs = {}
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            # Submit all tasks to the executor
            future_to_task = {}
            for task in model_tasks:
                # Get an available worker
                worker = self.get_available_worker()
                if worker is None:
                    print(f"No workers available for task {task['test_type']}, waiting...")
                    # Wait for a worker to become available
                    while worker is None:
                        time.sleep(0.5)
                        worker = self.get_available_worker()
                
                future = executor.submit(
                    worker.process_model,
                    task['test_type'],
                    task['parameters'],
                    task['base_model_ref'],
                    task['cell_pressure'],
                    task['drainage']
                )
                future_to_task[future] = (task, worker)
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task, worker = future_to_task[future]
                try:
                    result = future.result()
                    if result['success']:
                        outputs[result['test_type']] = result['model_path']
                        self.results[result['test_type']] = result['results']
                    else:
                        print(f"Failed to process {task['test_type']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"Exception processing {task['test_type']}: {str(e)}")
                finally:
                    # Release the worker back to the pool
                    self.release_worker(worker)
        
        return outputs
    
    def get_stress_strain(self, test_type: str, parameters: Dict[str, float]) -> Dict:
        """
        Extract stress-strain results for a single test
        Args:
            test_type: Name of the test case
            parameters: Dictionary of material parameters
        Returns: Dictionary of extracted results
        """
        # First check if we have cached results from parallel processing
        if test_type in self.results:
            return self.results[test_type]
        
        # If no cached results, try to extract from file
        try:
            # Build filename from parameters
            param_str = "_".join(f"{k}={v:.3f}" for k, v in parameters.items())
            filename = f"{test_type}_{param_str}.fez"
            output_path = os.path.join(self.output_dir, filename)
            
            if not os.path.exists(output_path):
                print(f"⚠️ Model file not found: {output_path}")
                return {'StrainYY': [], 'StressYY': []}
            
            # Try to use an available worker
            worker = self.get_available_worker()
            if worker:
                try:
                    results = worker._extract_results(test_type, parameters, output_path)
                    return results
                finally:
                    self.release_worker(worker)
            
            # If no workers, create a temporary interpreter
            print("No workers available, creating temporary interpreter...")
            
            # Find an available port
            port_range = self.config['parallel_processing'].get('port_range', [60000, 61000])
            start_port, end_port = port_range
            interpreter_port = find_available_port(start_port, end_port)
            
            if interpreter_port is None:
                print(f"No available port found for interpreter in range {start_port}-{end_port}")
                return {'StrainYY': [], 'StressYY': []}
            
            print(f"Starting RS2 Interpreter on port {interpreter_port}")
            RS2Interpreter.startApplication(port=interpreter_port)
            interpreter = RS2Interpreter(port=interpreter_port)

            # Open model and set up query point
            model_results = interpreter.openFile(output_path)
            model_results.AddMaterialQuery([[0.5, 0.5]])  # Center point query
            
            # Initialize dictionary for all result types
            extracted_data = {
                'StrainYY': [], 
                'StressYY': [], 
                'p': [], 
                'q': [], 
                'Volumetric_Strain': []
            }

            # Define all result types we want to extract
            result_types = [
                ('StrainYY', ExportResultType.SOLID_STRAIN_STRAIN_YY),
                ('StressYY', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_SIGMA_YY),
                ('p', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_MEAN_STRESS),
                ('q', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_VON_MISES_STRESS),
                ('Volumetric_Strain', ExportResultType.SOLID_STRAIN_VOLUMETRIC_STRAIN)
            ]

            # Extract each result type through all stages
            for key, result_type in result_types:
                model_results.SetResultType(result_type)
                for stageNum in range(1, 300):  # Scan through stages
                    try:
                        model_results.SetActiveStage(stageNum)
                        query_point = model_results.GetMaterialQueryResults()[0]
                        value = query_point.GetAllValues()[0].value
                        extracted_data[key].append(value)
                    except Exception:
                        break  # No more stages
                        
            # Clean up
            model_results.close()
            interpreter.closeProgram()
            
            return extracted_data

        except Exception as e:
            print(f"Error extracting results: {str(e)}")
            return {'StrainYY': [], 'StressYY': []}
    
    def close(self):
        """Clean up all RS2 resources"""
        # Close all workers
        for worker in self.workers:
            worker.close()
        self.workers = []
