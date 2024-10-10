# Common Classes and Helpers

## Installation

1. Download the pre-converted MLX embedding model:
   - Go to the Hugging Face repository: [Jaward/mlx-bge-small-en](https://huggingface.co/Jaward/mlx-bge-small-en)
   - Download the "bge-small-en.npz" file
   - Place the downloaded file in the `mlx-bge-small-en` directory of your project

   This file is a pre-converted MLX format embedding file, which is necessary for running the BERT embedding model in this project.

## File Structure

- `common/emb_model.py`: Custom BERT embedding model implementation
- `common/mlx-bge-small-en/`: Directory containing the MLX embedding model file
- `common/bge-small-en.npz`: Pre-converted MLX format embedding file (to be downloaded)

