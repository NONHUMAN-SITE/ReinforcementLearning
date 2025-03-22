python ../../train.py \
    env=lunarlander \
    algorithm=ppo \
    train.total_timesteps=500 \
    algorithm.K_epochs=30 \
    algorithm.T_steps=300 \
    algorithm.N_actors=8 \
    algorithm.eps_clip=0.2 \
    algorithm.entropy_coef=0.01 \
    algorithm.vf_coef=1.0 \
    algorithm.gamma=0.98 \
    algorithm.gae_lambda=0.95 \
    train.save_frequency=10 \
    train.validate_frequency=10 \
    train.validate_episodes=5 \
    train.seed=42 \
    train.save_path=../../output \
    buffer.batch_size=512
    