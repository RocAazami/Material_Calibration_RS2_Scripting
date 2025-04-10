# Material Calibration with RS2 Scripting

This repository contains scripts and tools developed for the automated calibration of material models using RS2 (Finite Element software by Rocscience). The goal is to streamline the process of optimizing material parameters through scripting and intelligent automation.

## 📁 Project Structure

- `Scripts/`: Python scripts for running calibration, optimization routines, and RS2 interactions.
- `InputModels/`: Contains initial RS2 model files to be used in the calibration.
- `Output/`: Calibrated model outputs, logs, and result files.
- `Utils/`: Utility functions and helpers.
- `Plots/`: Scripts and outputs for visualization of calibration results.

## ⚙️ Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, SciPy (you can install using `pip install -r requirements.txt`)
- RS2 installed with scripting capabilities enabled

## 🚀 How to Run

1. Prepare your input model and define material parameters.
2. Run the main calibration script (e.g. `main.py`).
3. Check output folder for calibrated model and plots.

## 📌 Notes

- These tools are designed to support continuous development and research in geotechnical FEM calibration.
- For internal use by the Rocscience team or academic collaborators.

---

