# PyFoam for OpenFOAM input file and folder manipulation 
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import os
from smartsim import Experiment
from smartredis import Client 
import numpy as np
import jinja2 as jj

#connect
PORT = os.environ["PORT"]
IFACE = os.environ["IFACE"]
client = Client(address=f"localhost:{PORT}", cluster=False)
client.ping()
exp = Experiment("ofrml", launcher="local")
db = exp.get_database()

env = jj.Environment()

of_case_name = os.environ["CASED"]
fn_name = "pUPhiTest"

def get_field_name(fn_name, field_name, processor=0, timestep=None):
    """
    Get the name of the field from the database. This function uses
    a metadata dataset posted by the function object itself to determine
    how things are named through Jinja2 templates

    Args:
        fn_name (str): The name of the function object
        field_name (str): The name of the OpenFOAM field
        processor (int): The MPI rank
        timestep (int): The target timestep index
    """
    client.poll_dataset(fn_name+"_metadata", 10, 1000)
    meta = client.get_dataset(fn_name+"_metadata")
    ds_naming = env.from_string(str(meta.get_meta_strings("dataset")[0]))
    ds_name = ds_naming.render(time_index=timestep, mpi_rank=processor)
    f_naming = env.from_string(str(meta.get_meta_strings("field")[0]))
    f_name = f_naming.render(name=field_name, patch="internal")
    return f"{{{ds_name}}}.{f_name}"



# blockMesh settings & model
blockMesh_settings = exp.create_run_settings(exe="blockMesh", exe_args=f"-case {of_case_name}")
blockMesh_model = exp.create_model(name="blockMesh", run_settings=blockMesh_settings)
# Mesh with blockMesh, and wait for meshing to finish before running the next model
exp.start(blockMesh_model, summary=True, block=True) 
    
# simpleFoam settings & model
simpleFoam_settings = exp.create_run_settings(exe="simpleFoam", exe_args=f"-case {of_case_name}")
simpleFoam_model = exp.create_model(name="simpleFoam", run_settings=simpleFoam_settings)
# Run simpleFoam solver
exp.start(simpleFoam_model, summary=True, block=True) 

# Get the names of OpenFOAM fiels from controlDict.functionObject 
control_dict = ParsedParameterFile(os.path.join(of_case_name, "system/controlDict"))
field_names = list(control_dict["functions"][fn_name]["fields"])

