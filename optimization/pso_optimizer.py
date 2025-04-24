import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel
import time
import matplotlib.pyplot as plt
import threading
from core.worker_pool_manager import WorkerPoolManager

class PSOParameters(BaseModel):
    """Parameters for Particle Swarm Optimization"""
    num_particles: int = 20
    particles_per_worker: int = 1  # New field
    max_iterations: int = 100
    cognitive_weight: float = 1.5
    social_weight: float = 1.5
    inertia_weight: float = 0.7
    min_inertia: float = 0.4
    max_inertia: float = 0.9
    tolerance: float = 1e-6
    max_stagnation: int = 10
    velocity_damping: float = 0.5

class ParticleSwarmOptimizer:
    """PSO implementation with worker-particle coordination"""


    def __init__(self, calibrator, config):
        """
        Initialize PSO optimizer
        
        Args:
            calibrator: MaterialCalibrator instance
            config: Configuration dictionary
        """
        self.calibrator = calibrator
        self.config = config
         
        # Initialize pso_params FIRST
        self.pso_params = PSOParameters()  # <-- Add this line before loading config
        
        # Then load configuration
        self._load_optimization_config()  # Now this can safely access self.pso_params
        
        # Initialize worker pool
        num_workers = self.config['parallel_processing'].get('workers', 1)
        particles_per_worker = self.pso_params.particles_per_worker
        self.worker_pool = WorkerPoolManager(num_workers, particles_per_worker)


        # Rest of your initialization code...
        self.is_running = False
        self.is_paused = False
        self.current_iteration = 0
        self.best_fitness_history = []
        self.best_position = None
        self.best_fitness = float('inf')
        self.best_params = {}
        self.stagnation_counter = 0
        
        # Worker coordination
        self.available_workers = []
        self.worker_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.pending_evaluations = 0
        self.evaluation_results = {}

    def _load_optimization_config(self):
        """Load PSO parameters from config"""
        opt_config = self.config.get('optimization', {})
        pso_config = opt_config.get('pso', {})
        
        # Update all parameters from config
        self.pso_params.cognitive_weight = pso_config.get('cognitive_weight', 1.5)
        self.pso_params.social_weight = pso_config.get('social_weight', 1.5)
        self.pso_params.inertia_weight = pso_config.get('inertia_weight', 0.7)
        self.pso_params.min_inertia = pso_config.get('min_inertia', 0.4)
        self.pso_params.max_inertia = pso_config.get('max_inertia', 0.9)
        self.pso_params.tolerance = opt_config.get('tolerance', 1e-6)
        self.pso_params.max_iterations = opt_config.get('max_iterations', 100)
        
        # Calculate number of particles based on workers
        num_workers = self.config['parallel_processing'].get('workers', 1)
        particles_per_worker = pso_config.get('particles_per_worker', 1)
        
        self.pso_params.particles_per_worker = particles_per_worker
        self.pso_params.num_particles = num_workers * particles_per_worker
        
        print(f"PSO Configuration:")
        print(f"  - Workers: {num_workers}")
        print(f"  - Particles: {self.pso_params.num_particles}")
        print(f"  - Particles per worker: {self.pso_params.particles_per_worker}")

    def optimize(self):
        """Run PSO optimization"""
        self.is_running = True
        self.is_paused = False
        start_time = time.time()
        
        # Get parameter bounds
        bounds = self._get_parameter_bounds()
        param_names = list(bounds.keys())
        
        # Initialize particles and velocities
        particles = self._initialize_particles(bounds)
        velocities = self._initialize_velocities(bounds)
        
        # Initialize personal and global best
        personal_best = particles.copy()
        personal_best_scores = np.full(self.pso_params.num_particles, np.inf)
        global_best = None
        global_best_score = np.inf
        
        # Reset history and counters
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.current_iteration = 0
        
        # Main optimization loop
        for iteration in range(self.pso_params.max_iterations):
            if not self.is_running:
                print("Optimization stopped.")
                break
                
            # Handle pause state
            while self.is_paused and self.is_running:
                time.sleep(0.5)
                
            if not self.is_running:
                break
                
            self.current_iteration = iteration
            iteration_start = time.time()
            
            # Calculate dynamic inertia weight
            w = self._calculate_inertia_weight(iteration)
            
            # Evaluate all particles in parallel
            self._evaluate_particles_parallel(particles, param_names)
            
            # Process evaluation results
            current_best_score = float('inf')
            for i in range(self.pso_params.num_particles):
                # Get evaluation result
                error = self.evaluation_results.get(i, np.inf)
                
                # Update personal best
                if error < personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = error
                
                # Update global best
                if error < global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = error
                    self.best_position = global_best.copy()
                    self.best_fitness = global_best_score
                    self.best_params = self._array_to_params(global_best, param_names)
                    self.stagnation_counter = 0  # Reset stagnation counter
                    print(f"Iter {iteration}: New best error = {global_best_score:.6f}")
                
                current_best_score = min(current_best_score, error)
            
            # Record history
            self.best_fitness_history.append(global_best_score)
            
            # Update velocities and positions
            for i in range(self.pso_params.num_particles):
                r1, r2 = np.random.random(len(param_names)), np.random.random(len(param_names))
                
                # Velocity update with cognitive and social components
                cognitive = self.pso_params.cognitive_weight * r1 * (personal_best[i] - particles[i])
                social = self.pso_params.social_weight * r2 * (global_best - particles[i])
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Position update with boundary handling
                new_position = particles[i] + velocities[i]
                
                # Improved boundary handling with reflection
                for j in range(len(param_names)):
                    min_val, max_val = list(bounds.values())[j]
                    
                    # If out of bounds, reflect and dampen velocity
                    if new_position[j] < min_val:
                        new_position[j] = min_val + (min_val - new_position[j]) * self.pso_params.velocity_damping
                        velocities[i][j] *= -self.pso_params.velocity_damping
                    elif new_position[j] > max_val:
                        new_position[j] = max_val - (new_position[j] - max_val) * self.pso_params.velocity_damping
                        velocities[i][j] *= -self.pso_params.velocity_damping
                
                # Final safety check to ensure within bounds
                particles[i] = np.clip(
                    new_position,
                    [b[0] for b in bounds.values()],
                    [b[1] for b in bounds.values()]
                )
            
            # Check for convergence
            if iteration > 0:
                improvement = abs(self.best_fitness_history[-2] - global_best_score)
                if improvement < self.pso_params.tolerance:
                    self.stagnation_counter += 1
                    print(f"Small improvement: {improvement:.8f} < {self.pso_params.tolerance}")
                    print(f"Stagnation counter: {self.stagnation_counter}/{self.pso_params.max_stagnation}")
                    
                    if self.stagnation_counter >= self.pso_params.max_stagnation:
                        print(f"Converged after {iteration+1} iterations (stagnation limit reached)")
                        break
                else:
                    self.stagnation_counter = 0
            
            iteration_time = time.time() - iteration_start
            print(f"Iteration {iteration+1}/{self.pso_params.max_iterations} completed in {iteration_time:.2f}s")
        
        # Optimization complete
        total_time = time.time() - start_time
        print(f"PSO optimization completed in {total_time:.2f} seconds")
        print(f"Best parameters: {self.best_params}")
        print(f"Best fitness: {self.best_fitness:.6f}")
        
        self.is_running = False
        return self.best_params, self.best_fitness
    
    def _evaluate_particles_parallel(self, particles, param_names):
        """
        Evaluate all particles in parallel using available workers
        
        Args:
            particles: Array of particle positions
            param_names: List of parameter names
        """
        # Reset evaluation results
        self.evaluation_results = {}
        
        # Get number of workers
        num_workers = self.config['parallel_processing'].get('workers', 1)
        
        if num_workers <= 1:
            # Sequential evaluation if only one worker
            for i in range(self.pso_params.num_particles):
                params = self._array_to_params(particles[i], param_names)
                try:
                    error = self.calibrator.evaluate_parameters(params)
                except Exception as e:
                    print(f"Evaluation failed for particle {i}: {str(e)}")
                    error = np.inf
                self.evaluation_results[i] = error
        else:
            # Parallel evaluation using worker pool
            self._evaluate_particles_with_worker_pool(particles, param_names)
    
    def _evaluate_particles_with_worker_pool(self, particles, param_names):
        """
        Evaluate particles using a pool of workers
        
        Args:
            particles: Array of particle positions
            param_names: List of parameter names
        """
        # Initialize worker pool if not already done
        if not hasattr(self.calibrator.rs2, 'workers') or not self.calibrator.rs2.workers:
            print("No workers available, initializing workers...")
            self.calibrator.rs2._initialize_workers()
        
        # Get available workers
        available_workers = len(self.calibrator.rs2.workers)
        if available_workers == 0:
            print("WARNING: No workers available, falling back to sequential evaluation")
            for i in range(self.pso_params.num_particles):
                params = self._array_to_params(particles[i], param_names)
                try:
                    error = self.calibrator.evaluate_parameters(params)
                except Exception as e:
                    print(f"Evaluation failed for particle {i}: {str(e)}")
                    error = np.inf
                self.evaluation_results[i] = error
            return
        
        # Use ThreadPoolExecutor for parallel evaluation
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Calculate batch size based on available workers
        batch_size = min(available_workers, self.pso_params.num_particles)
        
        # Process particles in batches
        for batch_start in range(0, self.pso_params.num_particles, batch_size):
            batch_end = min(batch_start + batch_size, self.pso_params.num_particles)
            batch = range(batch_start, batch_end)
            
            # Create parameter dictionaries for this batch
            batch_params = [self._array_to_params(particles[i], param_names) for i in batch]
            
            # Evaluate batch in parallel
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Submit all tasks
                future_to_idx = {}
                for idx, params in zip(batch, batch_params):
                    future = executor.submit(self._safe_evaluate, params)
                    future_to_idx[future] = idx
                
                # Process results as they complete
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        error = future.result()
                        self.evaluation_results[idx] = error
                    except Exception as e:
                        print(f"Evaluation failed for particle {idx}: {str(e)}")
                        self.evaluation_results[idx] = np.inf
    
    def _safe_evaluate(self, params):
        """Safely evaluate parameters, returning infinity on failure"""
        try:
            return self.calibrator.evaluate_parameters(params)
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return float('inf')
    
    def _get_parameter_bounds(self) -> Dict[str, tuple]:
        """Get parameter bounds from config"""
        bounds = {}
        for param, config in self.config['parameters'].items():
            if isinstance(config, dict):
                bounds[f"{param}_base"] = tuple(config['base'])
                if 'slope' in config:
                    bounds[f"{param}_slope"] = tuple(config['slope'])
            else:
                bounds[param] = tuple(config)
        return bounds
    
    def _initialize_particles(self, bounds: Dict[str, tuple]) -> np.ndarray:
        """Initialize particle positions randomly within bounds"""
        num_params = len(bounds)
        particles = np.zeros((self.pso_params.num_particles, num_params))
        
        for i, (param, (lower, upper)) in enumerate(bounds.items()):
            particles[:, i] = np.random.uniform(lower, upper, self.pso_params.num_particles)
        
        return particles
    
    def _initialize_velocities(self, bounds: Dict[str, tuple]) -> np.ndarray:
        """Initialize particle velocities"""
        num_params = len(bounds)
        velocities = np.zeros((self.pso_params.num_particles, num_params))
        
        for i, (lower, upper) in enumerate(bounds.values()):
            range_val = (upper - lower) * 0.1  # Initialize to 10% of parameter range
            velocities[:, i] = np.random.uniform(-range_val, range_val, self.pso_params.num_particles)
        
        return velocities
    
    def _array_to_params(self, array: np.ndarray, param_names: List[str]) -> Dict:
        """Convert numpy array to parameter dictionary"""
        params = {}
        for i, name in enumerate(param_names):
            if '_base' in name:
                param = name.replace('_base', '')
                if param not in params:
                    params[param] = {}
                params[param]['base'] = array[i]
            elif '_slope' in name:
                param = name.replace('_slope', '')
                if param not in params:
                    params[param] = {}
                params[param]['slope'] = array[i]
            else:
                params[name] = array[i]
        return params
    
    def _calculate_inertia_weight(self, iteration: int) -> float:
        """Calculate dynamic inertia weight that decreases linearly"""
        progress = iteration / self.pso_params.max_iterations
        return (self.pso_params.max_inertia - self.pso_params.min_inertia) * (1 - progress) + self.pso_params.min_inertia
    
    # Methods for dashboard integration
    def pause(self):
        """Pause the optimization process"""
        self.is_paused = True
        print("Optimization paused")
    
    def resume(self):
        """Resume the optimization process"""
        self.is_paused = False
        print("Optimization resumed")
    
    def stop(self):
        """Stop the optimization process"""
        self.is_running = False
        print("Optimization stopped")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current optimization progress for dashboard"""
        return {
            'iteration': self.current_iteration,
            'max_iterations': self.pso_params.max_iterations,
            'best_fitness': self.best_fitness,
            'best_params': self.best_params,
            'history': self.best_fitness_history,
            'is_running': self.is_running,
            'is_paused': self.is_paused
        }
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot the convergence history"""
        if not self.best_fitness_history:
            print("No optimization history to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.best_fitness_history)), self.best_fitness_history, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Error)')
        plt.title('PSO Convergence History')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
