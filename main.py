# main.py
from core.calibrator import MaterialCalibrator
from optimization.pressure_optimizer import MaterialCalibrationOptimizer
from visualization.plotter import CalibrationDashboard
import threading
import os
# 2025-April-09 first push to Git Hub
def main():
    
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"RS2 Models directory exists: {os.path.exists('RS2_Models')}")  # Should be True
    print(f"Base model in RS2_Models: {os.listdir('RS2_Models')}")  # Should list .fez files
    
    # Initialize components
    calibrator = MaterialCalibrator("config.yaml")
    print(calibrator.config['base_models']['drained_100'])  # Should show the filename
    optimizer = MaterialCalibrationOptimizer(calibrator)
    
    # Create dashboard - this must happen before starting optimization thread
    dashboard = CalibrationDashboard(calibrator)
    
    # Run optimization in separate thread
    opt_thread = threading.Thread(
        target=optimizer.optimize,
        daemon=True  # Thread will exit when main program exits
    )
    opt_thread.start()
    
    # Start dashboard (this will block)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard closed by user")
    except Exception as e:
        print(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()