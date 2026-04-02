#!/usr/bin/env bash

# Launch ACT BC training for Can task from Robomimic

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-mh-can-image \
    --policy act \
    --batch_size 128 \
    --wandb_project robomimic-can-bc \
    --wandb_enable \
    --eval_env Can \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 10 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 1000 