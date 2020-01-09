#!bin/bash

# SHOULD NOT BE SUBMITTED VIA BATCH SCRIPT 

interactive -p physicsgpu1 -q wildfire -gres=gpu:2 -t 0-00:10

module purge

module load cuda/9.2.148

module load flamegpu/1.5.0

# running Sugarscape visualisation
cd 1.5.0/bin/linux-x64/Release_Visualisation
./Sugarscape make visualisation
