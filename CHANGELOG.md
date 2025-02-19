# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [0.3.8] - 2025/2/19

### Updated

- fixed some issues in fastapi-server and langchain pipeline.

## [0.3.7] - 2025/2/18

### Updated

- synchronized with mlx==0.23.0 and mlx-lm==0.21.4

## [0.3.6] - 2025/1/15

### Updated

- created async_generate_step in fast-api
- added token usage information in fast-api
- extended libra router types

## [0.3.5] - 2024/11/25

### Updated

- improved fastAPI server
- support libra confidence router

## [0.3.4] - 2024/10/18

### Updated

- improved the hidden states generation method
- project structure refactoring

## [0.3.3] - 2024/10/10

### Added

- langchain integration
- local_rag and graph_rag example

### Updated

- generate method to support hidden states output

## [0.3.2] - 2024/09/09

### Added

- model management, FastAPI-server
- unit test

### Updated

- synchronized with the mlx-lm
- simplified README

## [0.3.1] - 2024/14/06

### Updated

- updated mlx_fastchat_worker for supporting mlx >= 0.14.
- updated conda config.

## [0.3.0] - 2024/11/06

### Added

- Lora support for GBA low-bit models.

## [0.2.1] - 2024/01/05

### Added

- support for Phi-3

## [0.2.0] - 2024/22/04

### Added

- Conversion: Utilize **gba2mlx.py** to convert models from GBA format to a format compatible with the MLX framework, ensuring smooth integration and optimal performance.
- Generation: Includes scripts for generating content using GBA quantized models within the MLX environment, empowering users to leverage the advanced capabilities of GBA models for natural language content creation.
- Fully support [GreenBitAI's MLX Model Collection](https://huggingface.co/collections/GreenBitAI/greenbitai-mlx-llm-6614eb6ceb8da657c2b4ed58)

## [0.1.0] - 2024/08/04

### Added

- Initial commit