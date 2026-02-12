""" 
Read SST data and save as .npy file. Example usage:

    python -u sst_to_npy.py --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

"""
import numpy as np
from args import args
from dataloaders import load_data

# Load the data
X, Y, cv, x, y, z = load_data(args.path, args)

X_time0 = X[0, :, :]
print(X_time0.shape)

outfile = 'data/P1F4R32.npy'
print(f"Saving as {outfile}")
np.save(outfile, X_time0)
