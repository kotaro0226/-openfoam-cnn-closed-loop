# PyFoam for OpenFOAM input file and folder manipulation 
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import os
import subprocess
from smartsim import Experiment
from smartredis import Client 
import numpy as np
import jinja2 as jj

from of_common import connect, get_field_name
client, exp, of_case_name, fn_name = connect()

polyMesh_dir = os.path.join(of_case_name, "constant", "polyMesh")

CASED = os.environ["CASED"]


# --- preprocess ---
if not os.path.isdir(polyMesh_dir):
    subprocess.run(["blockMesh", "-case", f"{of_case_name}"], check=True)
    subprocess.run(["renumberMesh", "-case", f"{of_case_name}"], check=True)
    subprocess.run(["postProcess", "-func", "writeCellCentres", "-case", f"{of_case_name}"], check=True)
    subprocess.run(["writeMeshObj", "-case", f"{of_case_name}"], check=True)
    subprocess.run(["postProcess", "-func", "writeCellVolumes", "-case", f"{of_case_name}"], check=True)

    # initialize scripts
    subprocess.run(["python", "initialize_T2.py"], cwd=f"{CASED}", check=True)
    subprocess.run(["python", "initialize_U2.py"], cwd=f"{CASED}", check=True)

    # decompose for parallel run
    subprocess.run(["decomposePar", "-force", "-case", f"{of_case_name}"], check=True)

else:
    print("[INFO] Existing mesh detected -> skip blockMesh / renumberMesh")

# --- solver (parallel) ---

solve_settings = exp.create_run_settings(
    exe="buoyantBoussinesqPimpleFoam_ML",
    exe_args=f"-case {of_case_name} -parallel"
)

solve_settings.set_nodes(2)
solve_settings.set_tasks(112)
solve_settings.set_tasks_per_node(56)

solve_model = exp.create_model("solve", solve_settings)
exp.start(solve_model, summary=True, block=True)


#subprocess.run([
#    "srun", "-N", "2", "-n", "112",
#    "buoyantBoussinesqPimpleFoam_ML", "-case", f"{of_case_name}", "-parallel"
#], check=True)



