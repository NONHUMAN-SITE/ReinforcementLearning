import os
import torch
from omegaconf import OmegaConf
from rl.utils.logger import logger
from rl.utils.environment import load_env, load_algorithm, load_model

def main():
    # Folder where the checkpoints and configurations are
    checkpoint_dir = "output/ppo-bipedalwalker"
    
    # Path to the eval_config.yaml file
    eval_config_path = os.path.join(checkpoint_dir, "eval_config.yaml")
    
    # Load the YAML file with OmegaConf
    cfg_eval = OmegaConf.load(eval_config_path)
    
    env_cfg = cfg_eval.env
    algo_cfg = cfg_eval.algorithm

    logger.info("Loading environment...")
    env = load_env(env_cfg)
    print(env_cfg.render_mode)
    logger.success(f"Environment loaded")

    logger.info("Loading model...")
    model = load_model(env_cfg, algo_cfg)
    logger.success(f"Model loaded")

    logger.info("Evaluating model...")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.set_device(device)
    model.load_model(checkpoint_dir)

    if env.is_continuous:
        model.set_std(env_cfg.min_std)

    epochs = 10
    episodes_reward = []

    for epoch in range(epochs):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, logprob = model.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            state = next_state
            total_reward += reward
            env.render()

        episodes_reward.append(total_reward)
        
        logger.info(f"Epoch {epoch} - Reward: {total_reward}")

    logger.info(f"Average reward: {sum(episodes_reward) / len(episodes_reward)}")

if __name__ == "__main__":
    main()