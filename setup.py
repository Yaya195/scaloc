from setuptools import setup, find_packages

setup(
    name="scaloc",
    version="0.1.0",
    description="Scalable indoor localization with graph + federated learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "networkx",
        "torch",
        "torch-geometric",
        "pyyaml",
        "matplotlib",
        "jupyter",
        "tensorboard",
        "setuptools",
    ],
)
