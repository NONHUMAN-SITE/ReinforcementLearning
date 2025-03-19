from rl.utils.logger import logger
from huggingface_hub import snapshot_download


# Model ID in Hugging Face
repo_id = "NONHUMAN-RESEARCH/PPO-BipedalWalker-v3"

# Folder where the model will be downloaded
dest_folder = "./output_hf/ppo-bipedalwalker"

logger.info(f"Downloading model {repo_id}...")

snapshot_download(repo_id=repo_id, local_dir=dest_folder)

logger.success(f"Model downloaded in '{dest_folder}'")

