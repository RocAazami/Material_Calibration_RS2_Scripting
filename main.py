# main.py
from core.calibrator import MaterialCalibrator
from optimization.pressure_optimizer import PressureDependentOptimizer
from visualization.plotter import CalibrationDashboard
import threading
import os

def main():
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Initialize components
    calibrator = MaterialCalibrator("config.yaml")
    optimizer = PressureDependentOptimizer(calibrator)
    
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