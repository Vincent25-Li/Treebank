#!/bin/bash

if [ "$1" = "train" ]; then
    python train.py --name=baseline --smoothing=True
elif [ "$1" = "test" ]; then
    python test.py --name=large --load_path=./save/train/large-04/best.pth.tar \
    --batch_size=8 --pce_model=albert-large-v2
elif [ "$1" = "setup" ]; then
    python setup.py --binary=True
else
    echo "Invalid Option Selected"
fi