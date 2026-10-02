from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="agrofood-emissions-regression",
    version="0.1.0",
    description="Per-Area regression analysis of agrofood emissions",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<3",
        "pandas>=2.2,<3",
        "scikit-learn>=1.5,<2",
    ],
    extras_require={"test": ["pytest>=8,<10"]},
    entry_points={
        "console_scripts": [
            "agrofood-regression=src.cli:main",
        ],
    },
)
