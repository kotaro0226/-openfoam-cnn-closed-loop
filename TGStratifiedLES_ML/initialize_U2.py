import numpy as np
import re

U0 = 1.0  # sets KE(0) ≈ 0.125

def read_internal_field_vectors(path):
    """
    Read 'internalField nonuniform List<vector>' from an OpenFOAM field file.
    Returns: (N, 3) float array
    """
    with open(path, "r") as f:
        text = f.read()

    # Find "internalField nonuniform List<vector>" then N then "( ... )"
    m = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s*?\n\s*(\d+)\s*\(\s*(.*?)\s*\)\s*;",
        text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError(f"Could not parse internalField from {path}")

    n = int(m.group(1))
    body = m.group(2)

    # Extract all (x y z)
    vecs = re.findall(r"\(\s*([eE0-9\+\-\.]+)\s+([eE0-9\+\-\.]+)\s+([eE0-9\+\-\.]+)\s*\)", body)
    if len(vecs) != n:
        raise RuntimeError(f"Vector count mismatch in {path}: header N={n}, parsed={len(vecs)}")

    arr = np.array(vecs, dtype=float)
    return arr

def write_U(path, U):
    n = U.shape[0]
    with open(path, "w") as f:
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
        f.write(f"{n}\n(\n")
        for ux, uy, uz in U:
            f.write(f"({ux} {uy} {uz})\n")
        f.write(""")
;
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

# --- main ---
C = read_internal_field_vectors("0/C")  # uses OpenFOAM's cell ordering
x = C[:, 0]
y = C[:, 1]
z = C[:, 2]

Ux =  U0 * np.sin(x) * np.cos(y) * np.cos(z)
Uy = -U0 * np.cos(x) * np.sin(y) * np.cos(z)
Uz = np.zeros_like(Ux)

U = np.column_stack([Ux, Uy, Uz])
write_U("0/U", U)

print(f"Wrote 0/U from 0/C ordering: Ncells={U.shape[0]}, U0={U0}")
print(f"U range: Ux[{Ux.min():.3e},{Ux.max():.3e}] Uy[{Uy.min():.3e},{Uy.max():.3e}]")
