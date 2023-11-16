#!/usr/bin/env python3

import subprocess
import argparse
import PIL.Image
import math

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("sigma", type=int)
args = ap.parse_args()

with open('rmse_IHS.txt', 'w') as file:               
    p = ['imdiff_ipol', 'input_0.png', 'ihs.png', 'diffInputIHS.png']
    subprocess.run(p, stdout=file)

#run algo
# Denoise image
subprocess.run(['denoise_'+str(args.sigma), '-c', 'input_0.png'])

# Compute image differences
subprocess.run(['imdiff', 'clean.png', 'nn.png', 'diffnn.png'])
subprocess.run(['imdiff', 'clean.png', 'bm3d.png', 'diffbm3d.png'])
subprocess.run(['imdiff', 'clean.png', 'ssann.png', 'diffssann.png'])

# Compute image rmse
with open('rmse_nn.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'clean.png', 'nn.png'], stderr=stdout)

with open('rmse_bm3d.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'clean.png', 'bm3d.png'], stderr=stdout)

with open('rmse_ssann.txt', 'w') as stdout:
    subprocess.run(['imdiff', '-mrmse', 'clean.png', 'ssann.png'], stderr=stdout)

# Resize for visualization (new size of the smallest dimension = 200)
(sizeX, sizeY) = PIL.Image.open('input_0.png').size
zoomfactor = max(1, int(math.ceil(200.0/min(sizeX, sizeY))))
(sizeX, sizeY) = (zoomfactor*sizeX, zoomfactor*sizeY)

for filename in ['input_0_sel', 'noisy', 'mark', 'bm3d', 'clean',
    'nn', 'ssann', 'diffnn', 'diffssann', 'diffbm3d']:
    im = PIL.Image.open(filename + '.png')
    im.resize((sizeX, sizeY), method='nearest')
    im.save(filename + '_zoom.png')


'cp src/NeuralStruct.cpp.10 src/NeuralStruct.cpp && make denoise -j4 -C src -f makefile && mv src/denoise src/denoise_10'