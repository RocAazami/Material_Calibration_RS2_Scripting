import threading
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

class WorkerPoolManager:
    """
    Manages a pool of workers for parallel processing,
    coordinating between PSO particles and RS2 instances
    """
    
    def __init__(self, num_workers: int, particles_per_worker: int):
        """
        Initialize with particles per worker ratio
        
        Args:
            num_workers: Number of available workers
            particles_per_worker: How many particles each worker handles
        """
        self.num_workers = num_workers
        self.particles_per_worker = particles_per_worker
        self.max_particles = num_workers * particles_per_worker
        
        # Worker tracking
        self.available_workers = []
        self.worker_lock = threading.Lock()
        
        print(f"Worker Pool initialized: {num_workers} workers, {self.max_particles} particles")

    
    def register_worker(self, worker_id: int, worker_obj: Any):
        """
        Register a worker with the pool
        
        Args:
            worker_id: Unique worker ID
            worker_obj: Worker object reference
        """
        with self.worker_lock:
            self.available_workers.append({
                'id': worker_id,
                'obj': worker_obj,
                'in_use': False,
                'particles': []
            })
    
    def get_worker_for_particle(self, particle_id: int) -> Optional[Dict]:
        """
        Get a worker for a specific particle
        
        Args:
            particle_id: Particle ID to assign to a worker
            
        Returns:
            Worker dictionary or None if no workers available
        """
        with self.worker_lock:
            # First try to find a worker that already has this particle
            for worker in self.available_workers:
                if particle_id in worker['particles'] and not worker['in_use']:
                    worker['in_use'] = True
                    return worker
            
            # Then try to find a worker with capacity
            for worker in self.available_workers:
                if not worker['in_use'] and len(worker['particles']) < self.particles_per_worker:
                    worker['in_use'] = True
                    worker['particles'].append(particle_id)
                    return worker
            
            # Finally, try any available worker
            for worker in self.available_workers:
                if not worker['in_use']:
                    worker['in_use'] = True
                    worker['particles'] = [particle_id]  # Replace existing particles
                    return worker
            
            return None
    
    def release_worker(self, worker: Dict):
        """
        Release a worker back to the pool
        
        Args:
            worker: Worker dictionary to release
        """
        with self.worker_lock:
            worker['in_use'] = False
    
    def execute_parallel(self, tasks: List[Dict], task_function: Callable, max_workers: Optional[int] = None):
        """
        Execute tasks in parallel using the worker pool
        
        Args:
            tasks: List of task dictionaries
            task_function: Function to execute for each task
            max_workers: Maximum number of workers to use (defaults to self.num_workers)
            
        Returns:
            Dictionary of results by task ID
        """
        if max_workers is None:
            max_workers = self.num_workers
        
        results = {}
        
        # Reset task tracking
        with self.task_lock:
            self.pending_tasks = len(tasks)
            self.task_complete_event.clear()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                # Get worker for this task
                worker = self.get_worker_for_particle(task.get('particle_id', 0))
                if worker is None:
                    # Wait for a worker to become available
                    while worker is None:
                        time.sleep(0.1)
                        worker = self.get_worker_for_particle(task.get('particle_id', 0))
                
                # Submit task with worker
                future = executor.submit(task_function, task, worker['obj'])
                future_to_task[future] = (task, worker)
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                task, worker = future_to_task[future]
                try:
                    result = future.result()
                    task_id = task.get('id', task.get('particle_id', 0))
                    results[task_id] = result
                except Exception as e:
                    print(f"Task execution failed: {str(e)}")
                finally:
                    # Release worker
                    self.release_worker(worker)
                    
                    # Update task tracking
                    with self.task_lock:
                        self.pending_tasks -= 1
                        if self.pending_tasks <= 0:
                            self.task_complete_event.set()
        
        return results
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        return self.task_complete_event.wait(timeout)
