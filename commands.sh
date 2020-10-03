# Train the agent
python train.py \
--project_name car1d \
--run_name scratch \
--env envs:Car1DEnv-v0 \
--eval_every 10 \
--log_std_init -3.0 \
--total_timesteps 1000000 \
--n_steps 1000 \
--policy_dims 256 256

# After training, appropriate directories are created under checkpoints
python enjoy.py \
--env envs:Car1DEnv-v0 \
--policy_path checkpoints/car1d_scratch/latest.zip \
--stats_path checkpoints/car1d_scratch/latest_stats.pth