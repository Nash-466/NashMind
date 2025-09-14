from __future__ import annotations
"""
Setup script for Burhan ARC Project
"""
from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="burhan-arc-project",
    version="1.0.0",
    author="Nabil Alagi",
    description="Advanced AI system for solving ARC (Abstraction and Reasoning Corpus) problems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/burhan-arc-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "full": read_requirements(),
        "ml": [
            "scikit-learn>=1.0.0",
            "scipy>=1.7.0",
        ],
        "deep": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
        "optimization": [
            "optuna>=2.10.0",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "burhan-arc=main:main",
            "burhan-test=integration_tests:run_integration_tests",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/burhan-arc-project/issues",
        "Source": "https://github.com/your-username/burhan-arc-project",
        "Documentation": "https://github.com/your-username/burhan-arc-project/blob/main/README.md",
    },
)
