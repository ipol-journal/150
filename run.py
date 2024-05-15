#!/usr/bin/env python3

import subprocess
import argparse
import PIL.Image
import math

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("sigma", type=int)
args = ap.parse_args()

#run algo
def merge_files(input_files, output_file):
    with open(output_file, 'w') as output:
        for input_file in input_files:
            with open(input_file, 'r') as file:
                # Read the RMSE data from the current input file
                data_RMSE = float(file.readline().strip())
                parts = input_file.split('.')[0].split('_')
                RMSE = parts[1]+' '+parts[0]                
                # Write the float number to the output file
                output.write(f"{RMSE.upper()} : {data_RMSE}\n")

# Denoise image
subprocess.run(['denoise_'+str(args.sigma), '-c', 'input_0.png'])

# Compute image differences
subprocess.run(['imdiff', 'clean.png', 'nn.png', 'diffnn.png'])
subprocess.run(['imdiff', 'clean.png', 'bm3d.png', 'diffbm3d.png'])
subprocess.run(['imdiff', 'clean.png', 'ssann.png', 'diffssann.png'])

# Compute image rmse
with open('rmse_nn.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'clean.png', 'nn.png'], stderr=stdout, stdout=stdout)

with open('rmse_bm3d.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'clean.png', 'bm3d.png'], stderr=stdout, stdout=stdout)

with open('rmse_ssann.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'clean.png', 'ssann.png'], stderr=stdout, stdout=stdout)

# List of input files
input_files = ['rmse_nn.txt', 'rmse_bm3d.txt', 'rmse_ssann.txt']

# Merge RMSE data into one file
merge_files(input_files, 'output.txt')

# Resize for visualization (new size of the smallest dimension = 200)
(sizeX, sizeY) = PIL.Image.open('input_0.png').size
zoomfactor = max(1, int(math.ceil(200.0/min(sizeX, sizeY))))
(sizeX, sizeY) = (zoomfactor*sizeX, zoomfactor*sizeY)

for filename in ['input_0', 'noisy', 'mark', 'bm3d', 'clean',
    'nn', 'ssann', 'diffnn', 'diffssann', 'diffbm3d']:
    im = PIL.Image.open(filename + '.png')
    im = im.resize((sizeX, sizeY))
    im.save(filename + '_zoom.png')
