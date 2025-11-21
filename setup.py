"""
Setup configuration for Best-of-N generation system.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bestofn",
    version="0.2.0",
    author="Patrick",
    author_email="",
    description="Best-of-N candidate generation with secure verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eous/bestofn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bestofn-generate=generate_best_of_n:main",
            "bestofn-inspect=inspect_experiment:main",
        ],
    },
)
