python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-can-final \
    wandb.name=residual-rl \
    debug=false \
    algo.scratch_dir=/home/atkesonlab4/Desktop/scratch