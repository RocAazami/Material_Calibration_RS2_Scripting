"""
interfaces/data_loader.py
Purpose: Load experimental data

Key Function:

python
def load_test(self, filepath, test_type, cell_pressure):
    # Returns dict with keys: strain, stress, pore_pressure

"""


import pandas as pd
from typing import Dict

class ExperimentalDataLoader:
    def __init__(self):
        self.tests = {}
        self.test_weights = {}

    def load_test(self, filepath: str, test_type: str, cell_pressure: float):
        """Load test data from CSV"""
        df = pd.read_csv(filepath)
        self.tests[test_type] = {
            'StrainYY': df['StrainYY'].values,
            'StressYY': df['StressYY'].values,
            'p': df['p'].values,
            'q': df['q'].values,
            'Volumetric_Strain': df['Volumetric_Strain'].values,            
            'cell_pressure': cell_pressure,
            'failure_strain': df['StrainYY'].iloc[-1]  # Last strain as failure
        }