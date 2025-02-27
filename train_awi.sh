#!/bin/bash
mkdir -p checkpoints
#python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
#python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001
#python -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85

#python -u train.py --name raft-dubbo --stage dubbo --validation awi_uv --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 --num_steps 50000 --batch_size 2 --lr 0.00025 --image_size 600 1232 --wdecay 0.0001 --gamma=0.85 #--wandb --save
python -u train.py --name raft-deniliquin --stage deniliquin --validation awi_uv --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 --num_steps 50000 --batch_size 2 --lr 0.00025 --image_size 1028 1232 --wdecay 0.0001 --gamma=0.85 --wandb --save
