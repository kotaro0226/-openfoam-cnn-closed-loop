# Pype.py

# =========================
# Switches
# =========================
EXE1 = 0        # 1: Exe exp_openfoam.py
EXE2 = 0        # 1: Exe make dataset & maps
MKMAP = 0       # 1: EXE make maps
SAMPLE_MODE = 1 # 1: Cell-wise / 2: Face-wise
EXE3 = 0        # 1: Exe sampling
EXE4 = 0        # 1: Exe train.py
EXE5 = 1        # 1: inference/benchmark


from smartsim import Experiment
from smartredis import Client
import os
import subprocess
PORT = int(os.environ["PORT"])
HOMED = os.environ["HOMED"]
DATAD = os.environ["DATAD"]
IFACE = os.environ["IFACE"]
WORKD = os.environ["WORKD"]
CASED = os.environ["CASED"]
FOAM_RUN = os.environ["FOAM_RUN"]
SICKLE_ROOT = os.environ["SICKLE_ROOT"]

exp = Experiment("ofrml", launcher="local", exp_path=WORKD)

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

    # --- Main ---
    
    import subprocess

    #1 calc & put flow field
    if EXE1:
        subprocess.run(["python", "exp_openfoam.py"], check=True)
    

    #2 Make dataset
    if EXE2:
        subprocess.run([
            "python", "make_cfd_dataset.py",
            "--out", f"{DATAD}/of-raw/cfd_xy.npz"
        ], check=True)

        if MKMAP:
            # ---- make maps ----
            # ---- make face maps ----
            subprocess.run([
                "python", "make_face_maps_from_C.py",
                "--case", f"{FOAM_RUN}",
                "--npz", f"{DATAD}/of-raw/cfd_xy.npz",
                "--out", f"{DATAD}/maps/face_maps_rank0_internal.npz",
            ], check=True)
    
    
            if SAMPLE_MODE == 1:
                # ---- make cell map ----
                subprocess.run([
                    "python", "make_cell_maps.py",
                    "--case", f"{FOAM_RUN}",
                    "--face-maps", f"{DATAD}/maps/face_maps_rank0_internal.npz",
                    "--time", "0",
                    "--out", f"{DATAD}/maps/cell_maps_rank0_internal.npz",
                ], check=True)

    
    #3 Sampling
    if EXE3:
        if SAMPLE_MODE == 1:
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
                "--xy_npz", f"{DATAD}/of-raw/cfd_xy_2.4-2.8.npz",
                # Cell-to-face connectivity, signs, and cell volumes
                "--cell_npz", f"{DATAD}/maps/cell_maps_rank0_internal.npz",
                # Index of clustering variable in X (e.g., div(phiHbyA))
                "--cv_index", "2",
                # Hypercube selection strategy (random blocks in space)
                "--hypercubes", "random",
                # Subsampling method inside each hypercube
                "--method", "random",
                # Number of hypercubes per timestep
                "--num_hypercubes", "32",
                # Number of sampled cells per hypercube
                "--num_samples", "1000",
                # Temporal window size (1 = independent snapshots)
                "--window", "1",
                # Output directory for subsampled NPZ
                "--output_dir", f"{DATAD}/sampled",
                # File name template (expanded inside SICKLE)
                "--fileprefix", "ofCell_{hypercubes}_{method}",
            ], check=True, cwd=SICKLE_ROOT)
    
    
        elif SAMPLE_MODE == 2:
            # ============================================================
            # Face-wise subsampling (directional, legacy pipeline)
            # - Uses run_sickle_subsample.py wrapper
            # - Executed separately for x / y / z face directions
            # - Output NPZ contains independent face samples only
            #   (no cell connectivity information)
            # ============================================================
            subprocess.run([
                "python", "run_sickle_subsample.py",
                # Root directory of SICKLE repository
                "--sickle_root", f"{SICKLE_ROOT}",
                # Data type: OpenFOAM face-wise dataset
                "--dtype", "of-face-npz",
                # Face-wise CFD features/labels
                "--xy_npz", f"{DATAD}/of-raw/cfd_xy_2.4-2.8.npz",
                # Face center maps for structured indexing
                "--maps_npz", f"{DATAD}/maps/face_maps_rank0_internal.npz",
                # Index of clustering variable in X
                "--cv_index", "2",
                # Hypercube selection strategy
                "--hypercubes", "random",
                # Subsampling method inside hypercube
                "--method", "random",
                # Number of hypercubes per timestep
                "--num_hypercubes", "32",
                # Number of sampled faces per hypercube
                "--num_samples", "1000",
                # Structured grid size for each sampled cube
                "--nxsl", "16", "--nysl", "16", "--nzsl", "16",
                # Temporal window size
                "--window", "1",
                # Output directory for subsampled NPZ
                "--output_dir", f"{DATAD}/sampled",
                # File name template (direction expanded by wrapper)
                "--fileprefix", "ofFace_{direction}_{hypercubes}_{method}",
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
        if SAMPLE_MODE == 1:
            subprocess.run([
                "python", "-u", "train.py", "mlp",
                "--src", "sickle",
    
                # Input dataset:
                # Cell-based NPZ where each sample packs 6 faces per cell
                "--cell_npz", f"{DATAD}/sampled/subsampled_ofCell_random_random.npz",
    
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
                "--hidden", "64", "64",       # Hidden layer sizes (shared across all faces)
                "--act", "tanh",              # Activation function
    
                # Output
                "--save-jit", f"{DATAD}/jit/mlp_cell_h64-64.jit",  # Saved TorchScript model
            ], check=True)
    
        # ==========================================================
        # SAMPLE_MODE = 2
        # Face-wise training:
        #   - Each face is treated as an independent sample
        #   - No explicit cell-level physics constraints
        #   - Equivalent to standard face-wise regression
        # ==========================================================
        elif SAMPLE_MODE == 2:
            subprocess.run([
                "python", "train.py", "mlp",
                "--src", "sickle",
    
                # Input datasets:
                # Face-based NPZ files sampled along x, y, and z orientations
                "--sickle_x", f"{DATAD}/sampled/subsampled_ofFace_x_random_random.npz",
                "--sickle_y", f"{DATAD}/sampled/subsampled_ofFace_y_random_random.npz",
                "--sickle_z", f"{DATAD}/sampled/subsampled_ofFace_z_random_random.npz",
    
                # Training control
                "--batch", "16384",           # Batch size in terms of number of faces
                "--epochs", "3",              # Number of training epochs
                "--lr", "1e-3",               # Learning rate for Adam optimizer
    
                # MLP architecture
                "--hidden", "16", "16",        # Hidden layer sizes
                "--act", "tanh",               # Activation function
    
                # Output
                "--save-jit", f"{DATAD}/jit/mlp_face_h16-16.jit",  # Saved TorchScript model
            ], check=True)


    #5 get model , flow field and Inference
    if EXE5:
        # ---- hardcode for now (edit here when sweeping arch) ----
        jit = f"{DATAD}/jit/mlp_cell_h64-64.jit"
        #jit = f"{DATAD}/jit/mlp_face_h64-64.jit"
        raw_npz  = f"{DATAD}/of_raw/cfd_xy_2.0-2.4.npz"          # Before sub-sampling
        cell_npz = f"{DATAD}/maps/cell_maps_rank0_internal.npz"
        out_npz  = f"{DATAD}/pred/pred_pEqnFlux_allfaces_64-64.npz"
        metrics_csv = f"{DATAD}/infer/infer_metrics.csv"
    
        # switches (edit here)
        DO_CONT_EVAL = 1
        TSPEC = "0:100"              # "all", "0,1,2", "10:20"
        BATCH_FACES = 1048576
    
        # ---- optional: safety check (recommended) ----
        if DO_CONT_EVAL:
            import numpy as np
            z = np.load(raw_npz, allow_pickle=True)
            X = z["X"]
            cm = np.load(cell_npz, allow_pickle=True)
            mx = int(cm["cell_faces_idx"].max())
            nf = int(X.shape[1])
            print(f"[INFO] RAW X.shape={X.shape}")
            print(f"[INFO] cell_faces_idx max={mx}  raw Nf={nf}")
            if mx >= nf:
                raise RuntimeError(
                    "Face indexing mismatch: cell_maps refers to face id outside raw dataset.\n"
                    "Check make_cfd_dataset.py output (internal faces only?) and face_maps/cell_maps consistency."
                )
    
        # ---- run inference ----
        if DO_CONT_EVAL:
            subprocess.run([
                "python", "-u", "infer.py",
                "--jit", jit,
                "--dataset_npz", raw_npz,
                "--timesteps", TSPEC,
                "--batch_faces", str(BATCH_FACES),
                "--out", out_npz,
                "--metrics_csv", metrics_csv,
                "--cont_eval",
                "--cell_maps_npz", f"{DATAD}/maps/cell_maps_rank0_internal.npz",
                "--debug_stats",
            ], check=True)
        else:
            subprocess.run([
                "python", "-u", "infer.py",
                "--jit", jit,
                "--dataset_npz", raw_npz,
                "--timesteps", TSPEC,
                "--batch_faces", str(BATCH_FACES),
                "--out", out_npz,
                "--metrics_csv", metrics_csv,
                "--cont_eval",
            ], check=True)


    # --- End ---

finally:
    # --- stop orchestrator ---
    exp.stop(db)
    print("[INFO] Orchestrator stopped")
