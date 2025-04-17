import os

import pkg_resources
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chem-xlstm",
    py_modules=["chemxlstm"], 
    version="1.0",
    author="Philipp Seidl",
    author_email="ph.seidl92@gmail.com",
    packages=find_packages(exclude=["tests*"]),
    install_requires="""s4dd@git+https://github.com/molML/s4-for-de-novo-drug-design.git
        xlstm
        einops
        opt_einsum
        numpy
        tqdm
        scikit-learn
        scipy
        pandas
        rdkit-pypi
        transformers
        matplotlib
        torch
        torchvision
        wandb
        fcd_torch   
        omegaconf   
        dacite  
        fvcore
        swifter""".split(), #        mamba_ssm@git+https://github.com/state-spaces/mamba.git
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml-jku/Chem-xLSTM",
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-2-Clause License",
        "Operating System :: linux-64",
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)