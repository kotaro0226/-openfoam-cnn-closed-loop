# PyFoam for OpenFOAM input file and folder manipulation 
import os
import subprocess

from of_common import connect
client, of_case_name, fn_name = connect()

polyMesh_dir = os.path.join(of_case_name, "constant", "polyMesh")
proc0_polyMesh_dir = os.path.join(of_case_name, "processor0", "constant", "polyMesh")

CASED = os.environ["CASED"]
OF_NTASKS = int(os.environ.get("OF_NTASKS", os.environ.get("SLURM_NTASKS", "1")))

def _build_solver_cmd():
    launcher = os.environ.get("OF_LAUNCHER", "srun").strip().lower()
    solver = os.environ.get("OF_SOLVER", "buoyantBoussinesqPimpleFoam_ML")
    force_parallel = os.environ.get("OF_FORCE_PARALLEL", "0").strip().lower() in {"1", "true", "yes", "on"}
    ntasks = os.environ.get("OF_NTASKS", os.environ.get("SLURM_NTASKS", "1"))
    is_parallel = force_parallel or int(ntasks) > 1

    if launcher == "none":
        cmd = [solver, "-case", of_case_name]
        if is_parallel:
            cmd.append("-parallel")
        return cmd

    cmd = [launcher]

    nodes = os.environ.get("OF_NODES")
    tasks_per_node = os.environ.get("OF_TASKS_PER_NODE", os.environ.get("SLURM_NTASKS_PER_NODE"))

    if launcher == "srun":
        if nodes:
            cmd += ["--nodes", str(nodes)]
        if ntasks:
            cmd += ["--ntasks", str(ntasks)]
        if tasks_per_node:
            cmd += ["--ntasks-per-node", str(tasks_per_node)]

        extra = os.environ.get("OF_SRUN_ARGS", "").strip()
        if extra:
            cmd.extend(extra.split())
    elif launcher in {"mpirun", "mpiexec"}:
        if ntasks:
            cmd += ["-np", str(ntasks)]

        extra_key = "OF_MPIRUN_ARGS" if launcher == "mpirun" else "OF_MPIEXEC_ARGS"
        extra = os.environ.get(extra_key, "").strip()
        if extra:
            cmd.extend(extra.split())

    cmd += [solver, "-case", of_case_name]
    if is_parallel:
        cmd.append("-parallel")
    return cmd


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
    if OF_NTASKS > 1:
        subprocess.run(["decomposePar", "-force", "-case", f"{of_case_name}"], check=True)

else:
    print("[INFO] Existing mesh detected -> skip blockMesh / renumberMesh")

    if OF_NTASKS > 1 and not os.path.isdir(proc0_polyMesh_dir):
        print("[INFO] Existing serial mesh detected -> running decomposePar for parallel run")
        subprocess.run(["decomposePar", "-force", "-case", f"{of_case_name}"], check=True)

# --- solver (parallel) ---
subprocess.run(_build_solver_cmd(), check=True)

#solve_settings = exp.create_run_settings(
#    exe="buoyantBoussinesqPimpleFoam_ML",
#    exe_args=f"-case {of_case_name} -parallel"
#)
#
#solve_settings.set_nodes(2)
#solve_settings.set_tasks(112)
#solve_settings.set_tasks_per_node(56)
#
#solve_model = exp.create_model("solve", solve_settings)
#exp.generate(solve_model, overwrite=True)
#exp.start(solve_model, summary=True, block=True)


