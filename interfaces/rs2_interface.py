"""
Critical RS2 Functions:

python
def create_test_model(self, material, parameters, cell_pressure, drainage):
    # Create triaxial test model in RS2
    self.model.MaterialProperties.AddMaterial().Behavior = MaterialType.NorSand
    self.model.Stages[0].CellPressure = cell_pressure
    self.model.Stages[0].Drainage = DrainageType.Drained if drainage == "drained" else DrainageType.Undrained

def get_stress_strain(self):
    # Extract results from RS2
    results = self.model.GetResults()
    return {
        'strain': results.GetColumn("Axial Strain"),
        'stress': results.GetColumn("Deviator Stress"),
        'pore_pressure': results.GetColumn("Pore Pressure")
    }
Adjustment Points:

Change stage configuration in create_test_model()

Modify result extraction in get_stress_strain()

"""

from typing import Dict, List
import os
from enum import Enum
import clr
import socket
# Rest of your existing imports and code...
#RocAA_Start
from rs2.modeler.RS2Modeler import RS2Modeler
from rs2.interpreter.RS2Interpreter import RS2Interpreter
from rs2.interpreter.InterpreterEnums import *

def sanitize_path(path):
    """Convert path to string and replace backslashes"""
    return str(path).replace('\\', '/')
    
# General input
filePath = r"C:\\Director of B&R\\Calibration-AI-Scripting-RS2\\NorSand - UTC - p0=100-Base Model.fez"
output_dir = r"C:\Director of B&R\Calibration-AI-Scripting-RS2"
#json_filename = r"C:\Director of B&R\Linkedin\Scripting Wall\liner_database_results.json"
#monitoring_path = r"C:\Director of B&R\Linkedin\Scripting Wall\monitoring_data.json"  # Windows
#RocAA_End

class RS2Interface:

    def create_test_model(self, material: str, parameters: Dict[str, float], cell_pressure: float, drainage: str):
        
        """Create triaxial test model in RS2"""
        port_Modeller = 60054
        # First try to start a new instance
        try:
            print(f"Attempting to start new RS2 instance on port {port_Modeller}...")
            RS2Modeler.startApplication(port=port_Modeller)
            modeler = RS2Modeler(port=port_Modeller)
            print("Successfully started new RS2 instance")
            #return modeler
        except Exception as start_error:
            print(f"Failed to start new instance: {start_error}")
            print("Attempting to connect to existing instance...")
            try:
                modeler = RS2Modeler(port=port_Modeller)
                print("Successfully connected to existing RS2 instance")
                #return modeler
            except Exception as connect_error:
                print(f"Failed to connect to existing instance: {connect_error}")
                raise RuntimeError("Could not start or connect to RS2 Modeler")        
            
        model = modeler.openFile(filePath)
     
        # Get material model , based on the name and the one we need to Adjust and Calibrate 
        material_to_adjust = model.getMaterialPropertyByName("Sand")  
        if material_to_adjust is None:
            raise ValueError("Material 'Sand' not found. Available materials: " 
                       f"{[mat.Name for mat in model.getMaterialPropertyByName]}")
       
        # Apply parameters        
        material_to_adjust.Strength.NorSandStrength.setMTCCriticalFrictionRatio(float(parameters['M_tc']))
        material_to_adjust.Strength.NorSandStrength.setH0PlasticHardeningModulus(float(parameters['H_0']))
        material_to_adjust.Strength.NorSandStrength.setPsi0InitialStateParameter(float(parameters['psi_0']))
    
        filename = f"M_tc={parameters['M_tc']:.3f}_H_0={parameters['H_0']:.3f}_psi_0={parameters['psi_0']:.3f}.fez"
        output_path = os.path.join(output_dir, filename)             
        model.saveAs(output_path) 
        model.compute()
         
    def run_analysis(self) -> bool:
        """Run the analysis"""
        return True

    def get_stress_strain(self, material: str, parameters: Dict[str, float]) -> Dict[str, List[float]]:    
        """Extracts results from RS2 model file with parameter-based naming"""
#        RS2Interpreter.startApplication(port=60005)   
#        interpreter = RS2Interpreter(port=60005)        
        port_Interpreter = 60005
        """
        # First try to start a new instance
        try:
            print(f"Attempting to start new RS2 Interpreter instance on port {port_Interpreter}...")
            RS2Interpreter.startApplication(port=port_Interpreter)
            interpreter = RS2Interpreter(port=port_Interpreter)
            print("Successfully started new RS2 Interpreter instance")
            #return modeler
        except Exception as start_error:
            print(f"Failed to start new Interpreter instance: {start_error}")
            print("Attempting to connect to existing Interpreter instance...")
            try:
                interpreter = RS2Interpreter(port=port_Interpreter)
                print("Successfully connected to existing RS2 Interpreter instance")
                #return modeler
            except Exception as connect_error:
                print(f"Failed to connect to existing Interpreter instance: {connect_error}")
                raise RuntimeError("Could not start or connect to RS2 Modeler")        
        """
        RS2Interpreter.startApplication(port=port_Interpreter)
        interpreter = RS2Interpreter(port=port_Interpreter)
  
        filename = (f"M_tc={parameters['M_tc']:.3f}_H_0={parameters['H_0']:.3f}_psi_0={parameters.get('psi_0', 0.0):.3f}.fez")
        output_path = os.path.join(output_dir, filename)
        model_results = interpreter.openFile(output_path)
        model_results.AddMaterialQuery([[0.5,0.5]]) #centre
        
        extracted_data = {'StrainYY': [], 'StressYY': [],'p': [], 'q': [],'Volumetric_Strain': []}
      
        # Extract all result types in single pass
        result_types = [
            ('StrainYY', ExportResultType.SOLID_STRAIN_STRAIN_YY),
            ('StressYY', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_SIGMA_YY),
            ('p', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_MEAN_STRESS),
            ('q', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_VON_MISES_STRESS),
            ('Volumetric_Strain', ExportResultType.SOLID_STRAIN_VOLUMETRIC_STRAIN)
        ]

        for key, result_type in result_types:
            model_results.SetResultType(result_type)
            for stageNum in range (1, 300, 1) :  # Iterate through stages
                try:
                    model_results.SetActiveStage(stageNum)
                    query_point = model_results.GetMaterialQueryResults()[0]                
                    Value = query_point.GetAllValues()[0].value
                    extracted_data[key].append(Value)
                except:
                    #print("no more stages")
                    break
        
        model_results.close()
        interpreter.closeProgram()
        return extracted_data 