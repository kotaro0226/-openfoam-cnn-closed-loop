# Pype.py

# =========================
# Switches
# =========================
EXE1 = 1        # 1: Exe exp_openfoam.py
EXE2 = 1        # 1: Exe make dataset & maps
MKMAP = 0       # 1: EXE make maps
EXE3 = 0        # 1: Exe subsampling
EXE4 = 0        # 1: Exe train.py
EXE5 = 0        # 1: Exe inference

# NAMES
DATA1 = "TGV_0-2"  #make dataset
DATA2 = "TGV_0-2"  #subsampling target data
DATA3 = "subsampled_ofCell_random_random"   #Train data

from smartsim import Experiment
from smartredis import Client
import os
import subprocess
#PORT = int(os.environ["PORT"])
#IFACE = os.environ["IFACE"]
HOMED = os.environ["HOMED"]
DATAD = os.environ["DATAD"]
NODES = int(os.environ.get("NODES", "1"))
WORKD = os.environ["WORKD"]
CASED = os.environ["CASED"]
FOAM_RUN = os.environ["FOAM_RUN"]
SICKLE_ROOT = os.environ["SICKLE_ROOT"]

#exp = Experiment("ofrml", launcher="local", exp_path=WORKD)
#db = exp.create_database(
#    port=PORT,
#    interface=IFACE
#)
exp = Experiment("ofrml", launcher="slurm", exp_path=WORKD)
db = exp.create_database(
    db_nodes=1,
#    db_nodes=NODES,
    interface=["hsn0","hsn1","hsn2","hsn3"],
    single_cmd=True,
)


try:
    #SmartSIM
    exp.generate(db, overwrite=True)
    exp.start(db, summary=True, block=True)
    address = db.get_address()[0]
    env = os.environ.copy()
    env["SSDB"] = address
    #client = Client(address=address, cluster=(NODES > 1))
    client = Client(address=address, cluster=False)
    print("DB address:", address)

    # --- Main ---
    import subprocess

    #1 calc & put flow field
    if EXE1:
        subprocess.run(["python", "exp_openfoam.py"], check=True, env=env)
    

    #2 Make dataset
    if EXE2:
        subprocess.run([
            "python", f"{WORKD}/utils/make_cfd_dataset.py",
            "--outdir", f"{DATAD}/of-raw-ranks",
            "--start", "1",
            "--nproc", "112",
            "--workers", "112"
        ], check=True, env=env)

        if MKMAP:
            # ---- make maps ----
            subprocess.run([
                "python", f"{WORKD}/utils/assemble_global_xy_from_procAddressing.py",
                "--case", f"{FOAM_RUN}",
                "--rank-dir", f"{DATAD}/of-raw-ranks",
                "--nproc", "4",
                "--out", f"{DATAD}/of-raw/{DATA1}.npz",
            ], check=True, env=env)

            # ---- make face maps ----
            subprocess.run([
                "python", f"{WORKD}/utils/make_face_maps_from_C.py",
                "--case", f"{FOAM_RUN}",
                "--npz", f"{DATAD}/of-raw/{DATA1}.npz",
                "--C", f"{FOAM_RUN}/0/C",
                "--out", f"{DATAD}/maps/face_maps_rank0_internal.npz",
            ], check=True)
    
            # ---- make cell map ----
            subprocess.run([
                "python", f"{WORKD}/utils/make_cell_maps.py",
                "--case", f"{FOAM_RUN}",
                "--face-maps", f"{DATAD}/maps/face_maps_rank0_internal.npz",
                "--time", "0",
                "--out", f"{DATAD}/maps/cell_maps_rank0_internal.npz",
            ], check=True)


    #3 Sampling
    if EXE3:
        # ============================================================
        # Cell-wise subsampling (for continuity-aware training)
        # - Directly calls SICKLE subsample.py (no directional split)
        # - Samples cells and stores local face connectivity
        # - Output NPZ contains:
        #     X, Y                      : (T*num_hypercubes, Ns, C)
        #     face_ids_list             : per-block unique face IDs
        #     cell_faces_local_idx      : local face indices (Ns,6)
        #     cell_face_sign, V         : for divergence loss
        # ============================================================
        subprocess.run([
            "python", "-u", f"{SICKLE_ROOT}/subsample.py",
            # Data type: OpenFOAM cell-wise dataset with connectivity
            "--dtype", "of-cell-npz",
            # Face-wise CFD features/labels (rAUf, phiHbyA, divPhiHbyA, pEqnFlux)
            "--xy_npz", f"{DATAD}/of_raw/{DATA2}.npz",
            # Cell-to-face connectivity, signs, and cell volumes
            "--cell_npz", f"{DATAD}/maps/cell_maps_rank0_internal.npz",
            # Index of clustering variable in X (e.g., div(phiHbyA))
            "--cv_index", "2",
            # Hypercube selection strategy (random blocks in space)
            "--hypercubes", "random",
            # Subsampling method inside each hypercube
            "--method", "random",
            # Number of hypercubes per timestep
            "--num_hypercubes", "16",
            # Number of sampled cells per hypercube
            "--num_samples", "1000",
            # Temporal window size (1 = independent snapshots)
            "--window", "1",
            # Output directory for subsampled NPZ
            "--output_dir", f"{DATAD}/sampled",
            # File name template (expanded inside SICKLE)
            "--fileprefix", f"{DATA2}-cell_{hypercubes}_{method}",
        ], check=True)
    
  
    #4 Train
    if EXE4:
        # ==========================================================
        # SAMPLE_MODE = 1
        # Cell-wise training:
        #   - One training sample corresponds to one cell
        #   - Each cell contains 6 faces
        #   - A shared MLP is applied independently to each face
        #   - Optional physics-informed continuity loss is enabled
        # ==========================================================
        subprocess.run([
            "python", "-u", "train.py", "mlp",
            "--src", "sickle",
    
            # Input dataset:
            # Cell-based NPZ where each sample packs 6 faces per cell
            "--cell_npz", f"{DATAD}/sampled/{DATA3}.npz",
    
            # Training control
            "--epochs", "3",            # Number of training epochs
            "--lr", "1e-3",              # Learning rate for Adam optimizer
            "--batch", "8192",           # Batch size in terms of number of cells
    
            # Physics-informed continuity loss (∇·φ ≈ 0)
            "--lambda-cont", "1.0",      # Weight of continuity (divergence) loss; 0 disables physics term
    
            # Feature layout inside one face
            "--phi-in-face", "1",        # Index of phiHbyA within per-face feature vector
            "--fields-per-face", "3",    # Number of input features per face (e.g., rAUf, phiHbyA, divPhiHbyA)
    
            # MLP architecture
            "--hidden", "32", "32",       # Hidden layer sizes (shared across all faces)
            "--act", "tanh",              # Activation function
            # Output
            "--save-jit", f"{DATAD}/jit/mlp_cell_h32-32.jit",  # Saved TorchScript model
 
            "--dropout", "0.05",
            
            "--split", "0.8,0.1,0.1",
            "--seed", "1234",
            "--split-save", f"{DATAD}/splits/split_cell_seed1234.npz",
            
            "--fit-norm",
            "--norm-path", f"{DATAD}/stats/norm_cell_h32-32.npz",
            #"--split-load", f"{DATAD}/splits/split_cell_seed1234.npz",
            #"--load-norm",
            #"--norm-path", f"{DATAD}/stats/norm_cell_h64-64.npz",

        ], check=True)


    #5 get model , flow field and Inference
    if EXE5:

        #5 cell-wise evaluation (Train/Val/Test split based)
        jit = f"{DATAD}/jit/mlp_cell_h32-32.jit"
    
        # subsampled cell dataset (label may change: random_random etc.)
        cell_npz = f"{DATAD}/sampled/subsampled_ofCell_random_random.npz"
    
        # split + norm produced by train.py
        split_npz = f"{DATAD}/splits/split_cell_seed1234.npz"
        norm_path = f"{DATAD}/stats/norm_cell_h32-32.npz"
    
        metrics_csv = f"{DATAD}/infer/eval_cell_metrics.csv"
        pred_npz    = f"{DATAD}/pred/eval_cell_test_pred_h32-32.npz"  # optional
    
        # switches (edit here)
        WHICH = "test"          # train / val / test
        SAVE_PRED = 1           # 1: save pred_npz
    
        cmd = [
            "python", "-u", "eval_cell.py",
            "--jit", jit,
            "--cell_npz", cell_npz,
            "--split_npz", split_npz,
            "--which", WHICH,
            "--norm_path", norm_path,
            "--batch_cells", "8192",
            "--fields_per_face", "3",
            "--phi_in_face", "1",
            "--metrics_csv", metrics_csv,
        ]
        if SAVE_PRED:
            cmd += ["--save_pred_npz", pred_npz]
    
        subprocess.run(cmd, check=True)


        ## ---- hardcode for now (edit here when sweeping arch) ----
        #jit = f"{DATAD}/jit/mlp_cell_h64-64.jit"
        ##jit = f"{DATAD}/jit/mlp_face_h64-64.jit"
        #raw_npz  = f"{DATAD}/of_raw/cfd_xy_2.0-2.4.npz"          # Before sub-sampling
        #cell_npz = f"{DATAD}/maps/cell_maps_rank0_internal.npz"
        #out_npz  = f"{DATAD}/pred/pred_pEqnFlux_allfaces_64-64.npz"
        #metrics_csv = f"{DATAD}/infer/infer_metrics.csv"
    
        ## switches (edit here)
        #DO_CONT_EVAL = 1
        #TSPEC = "0:100"              # "all", "0,1,2", "10:20"
        #BATCH_FACES = 1048576
    
        ## ---- optional: safety check (recommended) ----
        #if DO_CONT_EVAL:
        #    import numpy as np
        #    z = np.load(raw_npz, allow_pickle=True)
        #    X = z["X"]
        #    cm = np.load(cell_npz, allow_pickle=True)
        #    mx = int(cm["cell_faces_idx"].max())
        #    nf = int(X.shape[1])
        #    print(f"[INFO] RAW X.shape={X.shape}")
        #    print(f"[INFO] cell_faces_idx max={mx}  raw Nf={nf}")
        #    if mx >= nf:
        #        raise RuntimeError(
        #            "Face indexing mismatch: cell_maps refers to face id outside raw dataset.\n"
        #            "Check make_cfd_dataset.py output (internal faces only?) and face_maps/cell_maps consistency."
        #        )
    
        ## ---- run inference ----
        #if DO_CONT_EVAL:
        #    subprocess.run([
        #        "python", "-u", "infer.py",
        #        "--jit", jit,
        #        "--dataset_npz", raw_npz,
        #        "--timesteps", TSPEC,
        #        "--batch_faces", str(BATCH_FACES),
        #        "--out", out_npz,
        #        "--metrics_csv", metrics_csv,
        #        "--cont_eval",
        #        "--cell_maps_npz", f"{DATAD}/maps/cell_maps_rank0_internal.npz",
        #        "--debug_stats",
        #    ], check=True)
        #else:
        #    subprocess.run([
        #        "python", "-u", "infer.py",
        #        "--jit", jit,
        #        "--dataset_npz", raw_npz,
        #        "--timesteps", TSPEC,
        #        "--batch_faces", str(BATCH_FACES),
        #        "--out", out_npz,
        #        "--metrics_csv", metrics_csv,
        #        "--cont_eval",
        #    ], check=True)


    # --- End ---

finally:
    # --- stop orchestrator ---
    exp.stop(db)
    print("[INFO] Orchestrator stopped")
