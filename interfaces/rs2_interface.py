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
#filePath = r"C:\\Director of B&R\\Calibration-AI-Scripting-RS2\\NorSand - UTC - p0=100-Base Model.fez"
#output_dir = r"C:\Director of B&R\Calibration-AI-Scripting-RS2"


class RS2Interface:
    def __init__(self, config):
        self.config = config
        # Resolve output directory to absolute path
        self.output_dir = os.path.abspath(config['output']['directory'])
        os.makedirs(self.output_dir, exist_ok=True)  # Create if missing
        self.modeler = self._initialize_rs2()

    def _initialize_rs2(self):
        try:
            port = 60054  # Change to 60055 if conflicts occur
            print(f"Starting RS2 on port {port}...")
            RS2Modeler.startApplication(port=port)
            return RS2Modeler(port=port)
        except Exception as e:
            raise RuntimeError(f"RS2 initialization failed: {str(e)}")

    def create_test_models(self, material: str, parameters: Dict, cell_pressure: float, drainage: str) -> Dict:  # ✅ Add all parameter
        """Create models for ALL test types"""
        print("🟢 Entered create_test_models")
        outputs = {}
        try:
            for test in self.config['experimental_data']:
                print(f"🔵 Processing test: {test['type']}")
                model_path = self._create_single_model(
                    test_type=test['type'],
                    parameters=parameters,
                    base_model_ref=test['base_model'],
                    cell_pressure=test['cell_pressure'],
                    drainage=test['drainage']
                )
                outputs[test['type']] = model_path
            return outputs
        except Exception as e:
            print(f"🔴 create_test_models failed: {str(e)}")
            raise
   

    def _create_single_model(self, test_type: str, parameters: Dict, 
                            base_model_ref: str, cell_pressure: float, 
                            drainage: str) -> str:
        """Create individual test model"""
        
        print("🟢 Entered _create_single_model")
        # Resolve base model path
        base_path = self._resolve_path(base_model_ref)
        print(f"  Resolved base path: {base_path}")
        print(f"🔄 Opening base model: {base_path}")
        
        # Configure model
        model = self.modeler.openFile(base_path)
        print("  Model opened successfully")
        print("  Fetching material 'Sand'...")
        material = model.getMaterialPropertyByName("Sand")            
        if material is None:
            raise ValueError(f"Material 'Sand' not found in {base_path}")
            
        # Set parameters
        print(f"⚙️ Setting parameters: {parameters}")
        material.Strength.NorSandStrength.setMTCCriticalFrictionRatio(
            float(parameters['M_tc']))
        material.Strength.NorSandStrength.setH0PlasticHardeningModulus(
            float(parameters['H_0']))
        material.Strength.NorSandStrength.setPsi0InitialStateParameter(
            float(parameters['psi_0']))

        # Save parameterized model
        param_str = "_".join(f"{k}={v:.3f}" for k,v in parameters.items())
        output_path = os.path.join(
            self.output_dir,
            sanitize_path(f"{test_type}_{param_str}.fez")
        )
        print(f"Resolved output_path: {output_path}")  # Should show valid path without "$output.directory"

        print(f"💾 Saving to: {output_path}")
        model.saveAs(output_path)
        print("⚡ Computing model...")
        model.compute()  # Assume compute() returns a boolean
        return output_path
    

    def _resolve_path(self, path_ref: str) -> str:
        """Resolve paths without $output.directory placeholder"""
        if path_ref.startswith('$'):
            # Example: "$base_models.drained_100" → config['base_models']['drained_100']
            components = path_ref[1:].split('.')
            current = self.config
            for component in components:
                current = current.get(component)
                if current is None:
                    raise ValueError(f"Path reference {path_ref} not found in config")
            return os.path.abspath(current)
        return os.path.abspath(path_ref)

    def sanitize_path(path: str) -> str:
        """Make paths RS2-friendly"""
        return (
            path
            .replace(" ", "_")
            .replace("&", "and")
            .replace("(", "")
            .replace(")", "")
        )

    def get_stress_strain(self, test_type: str, parameters: Dict[str, float]):
        # Sanitize to match _create_single_model naming
        sanitized_test_type = test_type.replace(" ", "_").replace("&", "and")
        param_str = "_".join(f"{k}={v:.3f}" for k, v in parameters.items())
        filename = f"{sanitized_test_type}_{param_str}.fez"  # Now matches saved files
    
        """Extracts results from RS2 model file with parameter-based naming"""
        port_Interpreter = 60005

        RS2Interpreter.startApplication(port=port_Interpreter)
        interpreter = RS2Interpreter(port=port_Interpreter)

        # Construct the filename based on how the model was saved
        param_str = "_".join(f"{k}={v:.3f}" for k, v in parameters.items())
        filename = f"{test_type}_{param_str}.fez"
        output_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(output_path):
            print(f"⚠️ Model file not found: {output_path}")
            return {'StrainYY': [], 'StressYY': []}  # Return empty to avoid crash    

        model_results = interpreter.openFile(output_path)       
        model_results.AddMaterialQuery([[0.5, 0.5]])        
        
        extracted_data = {'StrainYY': [], 'StressYY': [], 'p': [], 'q': [], 'Volumetric_Strain': []}

        result_types = [
            ('StrainYY', ExportResultType.SOLID_STRAIN_STRAIN_YY),
            ('StressYY', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_SIGMA_YY),
            ('p', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_MEAN_STRESS),
            ('q', ExportResultType.SOLID_EFFECTIVE_STRESS_EFFECTIVE_VON_MISES_STRESS),
            ('Volumetric_Strain', ExportResultType.SOLID_STRAIN_VOLUMETRIC_STRAIN)
        ]

        for key, result_type in result_types:
            model_results.SetResultType(result_type)
            for stageNum in range(1, 300):
                try:
                    model_results.SetActiveStage(stageNum)
                    query_point = model_results.GetMaterialQueryResults()[0]
                    value = query_point.GetAllValues()[0].value
                    extracted_data[key].append(value)
                except Exception:
                    break

        model_results.close()
        interpreter.closeProgram()
        return extracted_data
