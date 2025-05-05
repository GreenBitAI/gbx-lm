import sys
from pathlib import Path

from setuptools import setup, find_packages

package_dir = Path(__file__).parent / "gbx_lm"
sys.path.append(str(package_dir))
from version import __version__

# Define core dependencies
core_requirements = [
    "mlx>=0.25.0",
    "numpy",
    "transformers[sentencepiece]>=4.39.3",  # Include sentencepiece for tokenizer support
    "huggingface_hub",
    "torch>=2.0.0",
    "protobuf",
    "pyyaml",
    "datasets"
]

# FastAPI server related dependencies - included by default
server_requirements = [
    "fastapi",
    "uvicorn",
    "pydantic"
]

# Define optional dependencies
extras_require = {
    # Server dependencies are included in the base requirements
    # but also defined here for flexibility
    "server": server_requirements,

    # LangChain integration
    "langchain": [
        "langchain-core"
    ],

    # If we want to enable the MLX-LM model support in the FastAPI server.
    "mlx-lm": [
        "mlx-lm>=0.24.0"
    ],

    # Development dependencies
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.20.0",
        "httpx>=0.24.0"
    ],
    "evaluate": ["lm-eval", "tqdm"]
}

# Add "all" option to include all extra dependencies
extras_require["all"] = []
for group_name, deps in extras_require.items():
    if group_name != "all":  # Avoid recursion
        extras_require["all"].extend(deps)

# Define the packages to include
packages = find_packages(include=["gbx_lm", "gbx_lm.*"])

# Define directories and files that should be excluded from MANIFEST.in
# NOTE: This does not affect wheel builds, but does affect sdist builds
# Use the package_data parameter to determine which files will be included in the package
package_data = {
    "": ["*.txt", "*.md", "*.json"],
}

setup(
    name="gbx-lm",
    version=__version__,
    description="GBA Model Toolkit for MLX",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="team@greenbit.ai",
    author="GreenBitAI and MLX Contributors",
    url="https://github.com/GreenBitAI/gbx-lm",
    license="Apache-2.0",
    install_requires=core_requirements + server_requirements,
    extras_require=extras_require,
    packages=packages,
    package_data=package_data,
    # Explicitly specify which files should be included in the source distribution
    include_package_data=False,
    # Explicitly exclude examples and tests directories
    exclude_package_data={"": ["examples/*", "tests/*"]},
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="mlx, apple, large language models, llm, language models",
    entry_points={
        "console_scripts": [
            "gbx_lm.chat = gbx_lm.chat:main",
            "gbx_lm.evaluate = gbx_lm.evaluate:main",
            "gbx_lm.generate = gbx_lm.generate:main",
            "gbx_lm.lora = gbx_lm.lora:main",
            "gbx_lm.manage = gbx_lm.manage:main"
        ]
    },
)