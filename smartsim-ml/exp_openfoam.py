# PyFoam for OpenFOAM input file and folder manipulation 
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import os
import subprocess
from smartsim import Experiment
from smartredis import Client 
import numpy as np
import jinja2 as jj

from of_common import connect, get_field_name
client, of_case_name, fn_name = connect()

polyMesh_dir = os.path.join(of_case_name, "constant", "polyMesh")

if not os.path.isdir(polyMesh_dir):
    print("[INFO] No mesh found -> running blockMesh and renumberMesh")

    # blockMesh settings & model
    blockMesh_settings = exp.create_run_settings(exe="blockMesh", exe_args=f"-case {of_case_name}")
    blockMesh_model = exp.create_model(name="blockMesh", run_settings=blockMesh_settings)
    # Mesh with blockMesh, and wait for meshing to finish before running the next model
    exp.start(blockMesh_model, summary=True, block=True) 

    # renumberMesh
    renumberMesh_settings = exp.create_run_settings(exe="renumberMesh", exe_args=f"-case {of_case_name}")
    renumberMesh_model = exp.create_model(name="renumberMesh", run_settings=renumberMesh_settings)
    exp.start(renumberMesh_model, summary=True, block=True) 

    # writeCellCentres
    writeCellCentres_settings = exp.create_run_settings(exe="postProcess", exe_args=f"-func writeCellCentres -case {of_case_name}")
    writeCellCentres_model = exp.create_model(name="writeCellCentres", run_settings=writeCellCentres_settings)
    exp.start(writeCellCentres_model, summary=True, block=True) 
    
    # writeMeshObj (face centres)
    writeMeshObj_settings = exp.create_run_settings(exe="writeMeshObj", exe_args=f"-case {of_case_name}")
    writeMeshObj_model = exp.create_model(name="writeMeshObj", run_settings=writeMeshObj_settings)
    exp.start(writeMeshObj_model, summary=True, block=True)

    # writeCellVolumes
    writeCellVolumes_settings = exp.create_run_settings(exe="postProcess", exe_args=f"-func writeCellVolumes -case {of_case_name}")
    writeCellVolumes_model = exp.create_model(name="writeCellVolumes", run_settings=writeCellVolumes_settings)
    exp.start(writeCellVolumes_model, summary=True, block=True)

#initialize
#subprocess.run(["python", "initialize_T.py"], check=True, cwd=of_case_name)
subprocess.run(["python", "initialize_T2.py"], check=True, cwd=of_case_name)
#subprocess.run(["python", "initialize_U.py"], check=True, cwd=of_case_name)
subprocess.run(["python", "initialize_U2.py"], check=True, cwd=of_case_name)

else:
    print("[INFO] Existing mesh detected -> skip blockMesh / renumberMesh")


# buoyantBoussinesqPimpleFoam_ML settings & model
solve_settings = exp.create_run_settings(exe="buoyantBoussinesqPimpleFoam_ML", exe_args=f"-case {of_case_name}")
solve_model = exp.create_model(name="solve", run_settings=solve_settings)
# Run simpleFoam solver
exp.start(solve_model, summary=True, block=True) 

# Get the names of OpenFOAM fiels from controlDict.functionObject 
#control_dict = ParsedParameterFile(os.path.join(of_case_name, "system/controlDict"))
#field_names = list(control_dict["functions"][fn_name]["fields"])

