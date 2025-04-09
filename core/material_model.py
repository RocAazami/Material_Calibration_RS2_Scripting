"""
core/material_model.py
Purpose: Material model definitions

Key Classes:

NorSand: Default implementation

Customization:

Add new models (e.g., MohrCoulomb, HardeningSoil)

Modify parameter bounds/ranges

"""

class MaterialModel:
    """Base class for material models"""
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}

class NorSand(MaterialModel):
    """NorSand constitutive model"""
    def __init__(self):
        super().__init__("NorSand")
        self.parameters = {
            'M_tc': 1.5,
            'H_0': 100.0,            
            'psi_0': 0.00
        }