#!/bin/bash

#SBATCH -o Boids_2D_exe.out             #STDOUT (%j = JobId)
#SBATCH -e Boids_2D_exe.err             #STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # notification for start, stop, or fail
#SBATCH --mail-user=vincentl@asu.edu

#SBATCH --gres=gpu:4    # request # of GPUs

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
EXECUTABLE_PATH=~/FlameGPUFiles/Boids_2D/Boids_2D
EXECUTABLE=~/1.5.0/bin/linux-x64/Release_Console/Boids_2D
INPUT_PATH=~/FlameGPUFiles/Boids_2D/iterations/0.xml
NUM_ITERATIONS=200
OUTPUT_FREQ=5

cd $EXECUTABLE_PATH
make clean
make console

for counter in {1..1}
do
	time $EXECUTABLE $INPUT_PATH $NUM_ITERATIONS 0 $OUTPUT_FREQ
done
