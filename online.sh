# don't forget to replace wandb_id in config!
task=can
export PYTHONPATH=$(pwd):$PYTHONPATH
bash ./resfit/rl_finetuning/shell/paper_runs/${task}/1_can_residual_rl.sh

