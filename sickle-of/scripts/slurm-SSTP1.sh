#!/bin/bash
#SBATCH -A XYZ123
#SBATCH -J sickle
#SBATCH -p extended
##SBATCH -p batch  # partition
##SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 06:00:00

# Setup environment
SRC="/path/to/sickle"
. $SRC/contrib/environment

# Define the list of cases
CASES=("Hmaxent-Xmaxent-32" \
       "Hmaxent-Xrandom-32" \
       "Hmaxent-Xuips-32" \
       "Hrandom-Xfull" \
       "Hrandom-Xmaxent-32" \
       "Hrandom-Xrandom-32" \
       "Hrandom-Xuips-32")

# Define the run directory and source path
RUNDIR="$MEMBERWORK/xyz123/sickle/${SLURM_JOB_ID}"
mkdir -p $RUNDIR "$RUNDIR/snapshots" "$RUNDIR/plots"

# Copy all case files to the run directory
for CASE in "${CASES[@]}"; do
    echo "Copying case file: $CASE.yaml"
    cp $SRC/contrib/configs/SST/P1/$CASE.yaml $RUNDIR
done

# Copy the slurm.sh script for reproducibility
echo "Copying slurm.sh to $RUNDIR"
cp "${BASH_SOURCE[0]}" "$RUNDIR"

# Change directory to the run directory once
cd "$RUNDIR" || exit

# Initialize the counter variable
count=0

# Loop over each case and execute the commands
for CASE in "${CASES[@]}"; do
    echo "Processing case: $CASE"
    
    ### SUBSAMPLING
    time srun -N "$SLURM_NNODES" -n 32 python -u "$SRC/subsample.py" "$CASE.yaml" \
              --output_dir "$RUNDIR/snapshots" >& "$RUNDIR/subsample${count}.out"

    ### TRAINING
    time srun -N "$SLURM_NNODES" --ntasks-per-node=8 python -u "$SRC/train.py" --plot \
              --output_dir "$RUNDIR/snapshots" "$CASE.yaml" >& "$RUNDIR/train${count}.out"

    echo "Finished processing case: $CASE"

    # Increment the counter
    count=$((count + 1))
done
