#!/usr/bin/env python3
import re, numpy as np
from pathlib import Path

def read_scalar_list(path):
    """Return numpy array of internalField values from a volScalarField file."""
    vals = []
    with open(path) as f:
        text = f.read()
    # find 'internalField ... List<...>\nNNN\n(\n ... \n)\n'
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", text, re.S)
    if not m:
        # try uniform
        m2 = re.search(r"internalField\s+uniform\s+([eE0-9\.\+\-]+)\s*;", text)
        if m2:
            raise ValueError(f"{path} has uniform internalField; need nonuniform cell-centre list.")
        raise ValueError(f"Could not parse internalField from {path}")
    n = int(m.group(1))
    body = m.group(2).strip().split()
    if len(body) != n:
        raise ValueError(f"{path}: found {len(body)} values, header says {n}")
    return np.array([float(v) for v in body], dtype=float)

def read_vector_list_C(path):
    """Return numpy array of y from a volVectorField C (0/C)."""
    ys = []
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line.startswith("(") and line.endswith(")") and " " in line:
                x,y,z = [float(v) for v in line.strip("()").split()]
                ys.append(y)
    if not ys:
        raise ValueError(f"No vector tuples found in {path}")
    return np.array(ys, dtype=float)

case = Path(".")
paths_to_try = [
    case/"0/Cy",                         # component file (preferred if present)
    case/"0/C",                          # vector field
    case/"constant/polyMesh/C",          # legacy
]

y = None
if paths_to_try[0].exists():
    y = read_scalar_list(paths_to_try[0])
elif paths_to_try[1].exists():
    y = read_vector_list_C(paths_to_try[1])
elif paths_to_try[2].exists():
    # legacy vector list in polyMesh/C: lines "(x y z)"
    ys = []
    with open(paths_to_try[2]) as f:
        for line in f:
            line=line.strip()
            if line.startswith("(") and line.endswith(")"):
                x,yv,z = [float(v) for v in line.strip("()").split()]
                ys.append(yv)
    y = np.array(ys, dtype=float)
else:
    raise SystemExit("Could not find cell centres. Run:\n  postProcess -func writeCellCentres\nor check that 0/Cy exists.")

# physics
g = 9.81
beta = 3.33e-3
N = 1.0e-2
grad = -(N**2)/(g*beta)   # K/m  (negative for stable stratification)
T0 = 300.0

print(f"Loaded {y.size} cell centres; using grad = {grad:.6f} K/m")

# write volScalarField T
out = case/"0/T"
with open(out, "w") as f:
    f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      T;
}
dimensions      [0 0 0 1 0 0 0];
internalField   nonuniform List<scalar>
""")
    f.write(f"{y.size}\n(\n")
    for yi in y:
        f.write(f"{T0 + grad*yi}\n")
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
print(f"Wrote stratified T field for {y.size} cells to {out}")
