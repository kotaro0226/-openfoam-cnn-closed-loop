
# Preliminary requirement: Create data/P1F4R32.npy file

    source /lustre/orion/proj-shared/gen150/dsml/venv/sst/bin/activate

Note: run the following command from main sickle directory

    python -u sst_to_npy.py --dims 3 --dtype sst-binary --path /lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/ --noseed --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64

# Perform subsampling analysis

Note: run following from `uips` directory
 
    python subsample.py -i input4D

Or in parallel:

    srun -np 4 python subsample.py -i input2D

This will generate the following files with downsampled results for 1k, 10k, and 100k samples:

    downSampledData_100000_it0.npz  downSampledData_10000_it0.npz  downSampledData_1000_it0.npz  scaler.npz
    downSampledData_100000_it1.npz  downSampledData_10000_it1.npz  downSampledData_1000_it1.npz

# Visualize results

    python visualize.py -i input4D

This will create Figures/*.png

# To plot loss

    python plotLoss.py -i input4D

Which will output csv and pt files in TrainingLog/*

# To compare MaxEnt methods with Phase-space-sampling approach

From the [Phase-space-sampling repo](https://github.com/NREL/Phase-space-sampling), run

    srun -np 4 python tests/main_from_input.py -i input2D

This will output several files. The only one used now is:

    downSampledData_1000_it1.npz

copy this one to sickle repo. Also, copy `combustion2DToDownsampleSmall.npy` to the `data` directory in the sickle repo.

Then run:

    python eval-maxent.py

This will output three `.png` images: 

    scatter_maxent_downsampled.png  
    scatter_random_downsampled.png  
    scatter_phasespace_downsampled.png

These two images can be compared to compare the two methods. 

To cluster on 2D, change the following line in `algorithms.py` from:

    data = cv[timestep, :].reshape(-1, 1)

to:

    data = cv[timestep, :]

