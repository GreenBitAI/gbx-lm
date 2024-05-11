import sys
from pathlib import Path

from setuptools import setup, find_packages

package_dir = Path(__file__).parent / "gbx_lm"
with open(Path(__file__).parent / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from version import __version__

setup(
    name="gbx-lm",
    version=__version__,
    description="GBA Model Toolkit for MLX",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="team@greenbit.ai",
    author="GreenBitAI and MLX Contributors",
    url="https://github.com/GreenBitAI/gbx-lm",
    license="Apache-2.0",
    install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.9",
)