# streaming TG initialization
import numpy as np
import re
from tqdm import tqdm

def read_mesh_dimensions(bmd_path="system/blockMeshDict"):
    with open(bmd_path) as f:
        text = f.read()

    # Look for the "hex ( ... ) (nx ny nz)" line in the blocks section
    match = re.search(r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", text)
    if not match:
        raise RuntimeError("Could not find hex block with grid dimensions in blockMeshDict")

    nx, ny, nz = map(int, match.groups())
    return nx, ny, nz

nx, ny, nz = read_mesh_dimensions()
print(f"Detected mesh: nx={nx}, ny={ny}, nz={nz}")
#nx, ny, nz = 128, 64, 128

Lx = Ly = Lz = 2*np.pi
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
U0 = 1.0   # sets KE(0) ≈ 0.125

total_cells = nx * ny * nz

with open("0/U", "w") as f:
    # --- header ---
    f.write("""FoamFile
{
    version 2.0;
    format ascii;
    class volVectorField;
    location "0";
    object U;
}
dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector>
""")
    f.write(f"{total_cells}\n(\n")

    # --- main loop with progress bar ---
    for k in tqdm(range(nz), desc="Writing U field", ncols=80):
        z = (k + 0.5) * dz
        cz = np.cos(z)
        for j in range(ny):
            y = (j + 0.5) * dy
            sy, cy = np.sin(y), np.cos(y)
            for i in range(nx):
                x = (i + 0.5) * dx
                sx, cx = np.sin(x), np.cos(x)
                Ux =  U0 * sx * cy * cz
                Uy = -U0 * cx * sy * cz
                f.write(f"({Ux} {Uy} 0)\n")

    # --- footer ---
    f.write(""")\n;
boundaryField
{
    xMin { type cyclic; }
    xMax { type cyclic; }
    yMin { type cyclic; }
    yMax { type cyclic; }
    zMin { type cyclic; }
    zMax { type cyclic; }
}
""")

print(f"Wrote 0/U for {total_cells:,} cells (U0={U0})")
