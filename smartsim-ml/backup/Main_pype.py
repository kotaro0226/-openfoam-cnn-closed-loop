# Pype.py

from smartsim import Experiment
from smartredis import Client
import os
PORT = os.environ["PORT"]
IFACE = os.environ["IFACE"]

exp = Experiment("ofrml", launcher="local")

db = exp.create_database(
    port=PORT,
    interface=IFACE
)

try:
    # --- start orchestrator ---
    exp.start(db)
    print(f"[INFO] Orchestrator started at localhost:{PORT}")

    # check orchestrator
    client = Client(address=f"localhost:{PORT}", cluster=False)
    client.ping()

    # --- Main ---
    
    import subprocess

    #1 calc & put flow field
    subprocess.run(["python", "exp_openfoam.py"], check=True)

    #2 train(randum) & put model
    subprocess.run(["python", "train.py", "medium_cnn"], check=True)

    #3 get model , flow field and Inference
    subprocess.run(["python", "benchmark_cfd.py", "medium_cnn"], check=True)

    # --- End ---

finally:
    # --- stop orchestrator ---
    exp.stop(db)
    print("[INFO] Orchestrator stopped")
