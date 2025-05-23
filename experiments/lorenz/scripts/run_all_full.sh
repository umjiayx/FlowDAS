#! /bin/bash

cd ../
python generate.py --config=generate_Lorenz_data --num_datasets=1 --num_particles=1024
python generate.py --config=generate_Lorenz_data_G --num_datasets=50 --num_particles=128

python train.py --config=train_win1 --epochs=3000
python train.py --config=train_win2 --epochs=3000
python train.py --config=train_win3 --epochs=3000

python train.py --config=train_win1_G --epochs=3000
python train.py --config=train_win2_G --epochs=3000
python train.py --config=train_win3_G --epochs=3000

python eval.py --config=eval_win1 --N_trajectory=64 --LT=15
python eval.py --config=eval_win2 --N_trajectory=64 --LT=15
python eval.py --config=eval_win3 --N_trajectory=64 --LT=15

python eval.py --config=eval_win1_G --N_trajectory=64 --LT=15
python eval.py --config=eval_win2_G --N_trajectory=64 --LT=15
python eval.py --config=eval_win3_G --N_trajectory=64 --LT=15

python eval.py --config=eval_win1_G_on_data --N_trajectory=64 --LT=15
python eval.py --config=eval_win2_G_on_data --N_trajectory=64 --LT=15
python eval.py --config=eval_win3_G_on_data --N_trajectory=64 --LT=15

python eval.py --config=eval_win1_on_Gdata --N_trajectory=64 --LT=15
python eval.py --config=eval_win2_on_Gdata --N_trajectory=64 --LT=15
python eval.py --config=eval_win3_on_Gdata --N_trajectory=64 --LT=15

echo "Run all complete!"