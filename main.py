# main.py
from core.calibrator import MaterialCalibrator
from optimization.material_calibration_optimizer import MaterialCalibrationOptimizer
from visualization.plotter import CalibrationDashboard
import threading
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Material Calibration Tool')
    parser.add_argument('--optimizer', type=str, default='pso',
                        choices=['original', 'pso'],
                        help='Optimization method (original or pso)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (overrides config)')
    parser.add_argument('--particles', type=int, default=None,
                        help='Number of PSO particles (overrides config)')
    args = parser.parse_args()
    
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Get the absolute path to the GitRepo directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gitrepo_dir = os.path.dirname(script_dir)  # Goes up one level from core/ to GitRepo/
    rs2_models_path = os.path.join(gitrepo_dir, 'RS2_Models')

    print(f"Absolute path to RS2_Models: {rs2_models_path}")
    print(f"RS2_Models exists: {os.path.exists(rs2_models_path)}")
    if os.path.exists(rs2_models_path):
        print(f"Contents: {os.listdir(rs2_models_path)}")
    else:
        print("Could not find RS2_Models directory. Please ensure it exists at:", rs2_models_path)   
        
    # Initialize components
    calibrator = MaterialCalibrator("config.yaml")
    print(f"Using optimizer: {args.optimizer}")
    
    # Override config settings if specified via command line
    if args.optimizer in ['original', 'pso']:
        calibrator.config['optimization']['method'] = args.optimizer
    
    if args.workers is not None and args.workers > 0:
        print(f"Overriding workers: {args.workers}")
        calibrator.config['parallel_processing']['workers'] = args.workers
    
    if args.particles is not None and args.particles > 0:
        print(f"Overriding particles: {args.particles}")
        if 'pso' not in calibrator.config['optimization']:
            calibrator.config['optimization']['pso'] = {}
        calibrator.config['optimization']['pso']['max_particles'] = args.particles
    
    # Create optimizer
    optimizer = MaterialCalibrationOptimizer(calibrator)
    
    # Create dashboard with optimizer reference
    dashboard = CalibrationDashboard(calibrator, optimizer)
    
    # Run optimization in separate thread
    opt_thread = threading.Thread(
        target=optimizer.optimize,
        daemon=True
    )
    opt_thread.start()
    
    # Start dashboard (this will block)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard closed by user")
        optimizer.stop()  # Ensure optimization is stopped
    except Exception as e:
        print(f"Dashboard error: {e}")
        optimizer.stop()  # Ensure optimization is stopped

if __name__ == "__main__":
    main()
