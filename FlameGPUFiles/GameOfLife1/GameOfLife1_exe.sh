#!/bin/bash

#SBATCH -o GameOfLife1_exe.out             #STDOUT (%j = JobId)
#SBATCH -e GameOfLife1_exe.err             #STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # send a notification when the job starts, stops, or fails
#SBATCH --mail-user=vincentl@asu.edu # send-to address

#SBATCH -N 1            # number of compute node
#SBATCH -n 1            # number of CPU cores to reserve on the nodes
#SBATCH --gres=gpu:2    # request # of GPUs

##SBATCH -p physicsgpu1  # GTX1080
#SBATCH -p cidsegpu1   # V100
#SBATCH -q wildfire     # run job under wildfire QOS queue

#SBATCH -t 0-00:15      # wall time (D-HH:MM)

######################

module purge

module load cuda/9.2.148

module load flamegpu/1.5.0

# flamegpu/1.5.0/ copied into home directory
EXECUTABLE_PATH=~/1.5.0/examples/GameOfLife1
EXECUTABLE=~/1.5.0/bin/linux-x64/Release_Console/GameOfLife1
INPUT_PATH=~/1.5.0/examples/GameOfLife1/iterations/0.xml
NUM_ITERATIONS=100
OUTPUT_FREQ=5

cd $EXECUTABLE_PATH
make clean
make console
cd ~


for counter in {1..1}
do
	time	$EXECUTABLE 	$INPUT_PATH 	$NUM_ITERATIONS		0	$OUTPUT_FREQ
done
