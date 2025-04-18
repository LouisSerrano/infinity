from setuptools import find_packages, setup

setup(
    name="infinity",
    version="0.1.0",
    description="Package for Airfoil Geometric Design with INR",
    author="Louis Serrano",
    author_email="louis.serrano@isir.upmc.fr",
    install_requires=[
        "airfrans",
        "einops",
        "hydra-core",
        "wandb",
        "torch",
        "pandas",
        "matplotlib",
        "xarray",
        "scipy",
        "h5py",
        "torch_geometric",
    ],
    package_dir={"infinity": "infinity"},
    packages=find_packages(),
)
