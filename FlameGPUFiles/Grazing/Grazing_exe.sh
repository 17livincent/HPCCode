#!/bin/bash

#SBATCH -o Grazing.out             #STDOUT (%j = JobId)
#SBATCH -e Grazing.err             #STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # send a notification when the job starts, stops, or fails
#SBATCH --mail-user=vincentl@asu.edu # send-to address

#SBATCH -N 1            # number of compute node
#SBATCH -n 1            # number of CPU cores to reserve on the nodes
#SBATCH --gres=gpu:1    # request # of GPUs

##SBATCH -C GTX1080     # calls for specific GPU
##SBATCH -C V100

##SBATCH -p physicsgpu1  # GTX1080
#SBATCH -p cidsegpu1    # V100
##SBATCH -p rcgpu1      # V100 32Gb RAM
#SBATCH -q wildfire     # run job under wildfire QOS queue

#SBATCH -t 0-00:15      # wall time (D-HH:MM)

######################

module purge

module load cuda/9.2.148

module load flamegpu/1.5.0

# flamegpu/1.5.0/ copied into home directory
EXECUTABLE_PATH=~/FlameGPUFiles/Grazing
EXECUTABLE=~/FlameGPUFiles/Grazing
INPUT_PATH=~/FlameGPUFiles/Grazing/iterations/0.xml
NUM_ITERATIONS=500
OUTPUT_FREQ=5

cd $EXECUTABLE_PATH
make clean
make console
cd ~


for counter in {1..1}
do
	time	$EXECUTABLE 	$INPUT_PATH 	$NUM_ITERATIONS		0	$OUTPUT_FREQ
done
