#! /bin/bash

cd ../
python generate.py --config=generate_Lorenz_data --num_datasets=1 --num_particles=1024

python train.py --config=train_win1 --epochs=3000

python eval.py --config=eval_win1 --N_trajectory=32 --LT=15