Run the following commands to execute and train the models:

## Q 2.1
python train_model.py --type 'vox' --max_iter 5000 --save_freq 10 --lr 1e-4
python train_model.py --load_checkpoint --type 'vox' --max_iter 10000 --save_freq 10 --lr 1e-5
python train_model.py --load_checkpoint --type 'vox' --max_iter 15000 --save_freq 10 --lr 1e-6

## Q 2.2
python train_model.py --type 'point' --max_iter 5000

## Q 2.3
python train_model.py --type 'mesh' --max_iter 10000