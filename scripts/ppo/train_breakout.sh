python ../../train.py \
    env=breakout \
    algorithm=ppo \
    train.total_timesteps=1000000 \
    algorithm.K_epochs=3 \
    algorithm.T_steps=128 \
    algorithm.N_actors=4 \
    algorithm.eps_clip=0.1 \
    algorithm.entropy_coef=0.01 \
    algorithm.vf_coef=1.0 \
    algorithm.gamma=0.99 \
    algorithm.gae_lambda=0.95 \
    train.save_frequency=10000 \
    train.validate_frequency=10 \
    train.validate_episodes=5 \
    train.seed=42 \
    train.save_path=../../output \
    buffer.batch_size=256
    