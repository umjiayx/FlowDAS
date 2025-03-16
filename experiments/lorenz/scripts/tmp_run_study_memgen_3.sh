#! /bin/bash

cd ../

python generate.py --config=generate_Lorenz_data_G_memgen_13

python train.py --config=train_win8_G_memgen_13

python eval.py --config=eval_win8_G_memgen_13 --N_trajectory=64 --LT=15