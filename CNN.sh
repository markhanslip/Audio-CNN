#!/bin/bash

cd && . ./anaconda3/etc/profile.d/conda.sh && conda activate PyTorch && cd ./Audio-CNN && python ./inference.py --runtime 100 --model_path ./models/bigpvt_cqt_10epochs_densenet_44100_batch64_shuf.pth
